"""
ONNX-based shuttlecock detection. Runs a YOLOv8 ONNX model on the Pi.
Same interface as PersonFilter — detect(frame) returns bool, caches every N frames.
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ONNXShuttleDetector:
    """Periodically runs ONNX shuttle detection on the frame.

    ONNXShuttleDetector.detect(frame) runs inference every `interval` frames
    and caches the result.  shuttle_present reflects the most recent result.
    Gracefully disables when the model file is missing or onnxruntime is not installed.
    """

    def __init__(self, model_path: str = "data/weights/best.onnx",
                 confidence: float = 0.25,
                 interval: int = 5,
                 input_size: int = 640):
        self.confidence = confidence
        self.interval = interval
        self.input_size = input_size
        self._count = 0
        self._shuttle_present = False
        self._session = None
        self._input_name = None
        self._output_name = None

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
            self.input_size = h  # use model's expected size
            logger.info("ONNXShuttleDetector loaded (%s, conf=%.2f, every %d frames)",
                        model_path, confidence, interval)
        except Exception as e:
            logger.warning("Failed to load ONNX model %s: %s", model_path, e)
            self._session = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox resize to input_size x input_size, return (1,3,H,W) float32 [0,1]."""
        h, w = frame.shape[:2]
        scale = self.input_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create square canvas and paste
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        dx = (self.input_size - new_w) // 2
        dy = (self.input_size - new_h) // 2
        canvas[dy:dy + new_h, dx:dx + new_w] = resized

        # HWC → CHW, normalize, add batch dim
        blob = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(blob, axis=0)

    def detect(self, frame: np.ndarray) -> bool:
        """Returns True if a shuttle is detected in the frame.

        Runs inference every *interval* frames; caches result in between.
        """
        if self._session is None:
            return False

        self._count += 1
        if self._count % self.interval != 0:
            return self._shuttle_present

        try:
            inp = self._preprocess(frame)
            out = self._session.run([self._output_name], {self._input_name: inp})[0]
            # out shape: (1, num_classes+4, N) or (1, 5, N)
            # Confidence scores are in the last channel(s)
            scores = out[0, 4:, :]  # (num_classes, N) or (1, N)
            max_conf = float(scores.max())
            self._shuttle_present = max_conf >= self.confidence
        except Exception as e:
            logger.warning("ONNX inference failed: %s", e)
            self._shuttle_present = False

        return self._shuttle_present

    @property
    def shuttle_present(self) -> bool:
        return self._shuttle_present
