"""
ONNX-based shuttlecock detection. Runs YOLOv8 ONNX inference on a background
thread so the CV pipeline never blocks. Same interface as PersonFilter.
"""

import logging
import threading
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ONNXShuttleDetector:
    """Runs ONNX shuttle detection asynchronously on a background thread.

    detect(frame) queues the frame and returns the last cached result immediately.
    A background thread runs inference every *interval* frame submissions.
    shuttle_present reflects the most recent inference result.
    """

    def __init__(self, model_path: str = "data/weights/best.onnx",
                 confidence: float = 0.25,
                 interval: int = 5,
                 input_size: int = 640):
        self.confidence = confidence
        self.interval = interval
        self.input_size = input_size
        self._shuttle_present = False
        self._session = None
        self._input_name = None
        self._output_name = None
        self._lock = threading.Lock()
        self._pending_frame = None
        self._frame_ready = threading.Event()
        self._count = 0

        try:
            import onnxruntime as ort
        except ImportError:
            logger.info("onnxruntime not installed — ONNX shuttle detection disabled")
            return

        try:
            self._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            _, _, h, w = self._session.get_inputs()[0].shape
            self.input_size = h
            logger.info("ONNXShuttleDetector loaded (%s, conf=%.2f, every %d frames)",
                        model_path, confidence, interval)
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
        except Exception as e:
            logger.warning("Failed to load ONNX model %s: %s", model_path, e)
            self._session = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = self.input_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        dx = (self.input_size - new_w) // 2
        dy = (self.input_size - new_h) // 2
        canvas[dy:dy + new_h, dx:dx + new_w] = resized
        blob = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(blob, axis=0)

    def _run(self):
        """Background thread: waits for frames, runs inference every N frames."""
        while True:
            self._frame_ready.wait()
            self._frame_ready.clear()

            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                continue

            self._count += 1
            if self._count % self.interval != 0:
                continue

            try:
                inp = self._preprocess(frame)
                out = self._session.run([self._output_name], {self._input_name: inp})[0]
                # Output shape: (1, 4+num_classes, N) — already sigmoided by Ultralytics
                scores = out[0, 4:, :]  # class confidences in [0, 1]
                max_conf = float(scores.max())
                with self._lock:
                    self._shuttle_present = max_conf >= self.confidence
            except Exception as e:
                logger.warning("ONNX inference failed: %s", e)
                with self._lock:
                    self._shuttle_present = False

    def detect(self, frame: np.ndarray) -> bool:
        """Queue frame for inference, return last cached result immediately."""
        if self._session is None:
            return False
        with self._lock:
            self._pending_frame = frame
        self._frame_ready.set()
        return self._shuttle_present

    @property
    def shuttle_present(self) -> bool:
        return self._shuttle_present
