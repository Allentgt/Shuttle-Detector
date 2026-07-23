"""
TFLite-based person detection. Uses a COCO pre-trained model to check
whether a person is present in the frame. Landing events near a person
are suppressed to reduce false positives.
"""

import os
import urllib.request
import zipfile
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = "data/models"
MODEL_ZIP_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/models/"
    "tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
)
MODEL_FILE = "detect.tflite"
LABEL_FILE = "labelmap.txt"

# COCO class IDs (0-indexed from model output)
PERSON_CLASS = 1  # model uses 0 for background, 1 = person


def _download_model(model_dir: str):
    """Download and extract the TFLite COCO model."""
    os.makedirs(model_dir, exist_ok=True)
    zip_path = os.path.join(model_dir, "model.zip")
    logger.info("Downloading COCO person detection model (~7MB)...")
    urllib.request.urlretrieve(MODEL_ZIP_URL, zip_path)
    logger.info("Extracting model...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(model_dir)
    os.remove(zip_path)
    logger.info("Model ready at %s", model_dir)


class PersonFilter:
    """Periodically runs person detection on the frame.
    
    PersonFilter.detect(frame) returns True if a person is found.
    The detector runs every `interval` frames for performance.
    """

    def __init__(self, model_path: str | None = None,
                 confidence: float = 0.4,
                 interval: int = 10):
        self.confidence = confidence
        self.interval = interval
        self._count = 0
        self._person_present = False
        self._interp = None
        self._input_shape = None
        self._output_details = None
        self._input_details = None

        # Check if tflite is available before downloading
        try:
            import tflite_runtime.interpreter as tflite  # noqa: F401
        except ImportError:
            logger.info("tflite-runtime not available — person detection disabled")
            return

        if model_path is None:
            model_dir = MODEL_DIR
            model_path = os.path.join(model_dir, MODEL_FILE)
            if not os.path.exists(model_path):
                _download_model(model_dir)

        self._load(model_path)
        logger.info("PersonFilter loaded (conf=%.1f, every %d frames)",
                     confidence, interval)

    def _load(self, path: str):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            logger.info(
                "tflite-runtime not available on this platform. "
                "Person detection disabled. (Install on Pi: pip install tflite-runtime)"
            )
            return

        try:
            self._interp = tflite.Interpreter(model_path=path)
            self._interp.allocate_tensors()
            self._input_details = self._interp.get_input_details()
            self._output_details = self._interp.get_output_details()
            self._input_shape = self._input_details[0]["shape"]
        except Exception:
            logger.warning("Failed to init TFLite interpreter (numpy version mismatch?). "
                           "Person detection disabled.", exc_info=True)
            self._interp = None

    def detect(self, frame: np.ndarray) -> bool:
        """Returns True if a person is detected in the frame.
        
        Runs inference every `interval` frames; caches result in between.
        """
        if self._interp is None:
            return False

        self._count += 1
        if self._count % self.interval != 0:
            return self._person_present

        h, w = self._input_shape[1], self._input_shape[2]
        resized = cv2.resize(frame, (w, h))
        input_data = np.expand_dims(resized.astype(np.uint8), axis=0)

        self._interp.set_tensor(self._input_details[0]["index"], input_data)
        self._interp.invoke()

        # SSD MobileNet V1 COCO output:
        #   [0]: boxes [1, N, 4]  (ymin, xmin, ymax, xmax normalized)
        #   [1]: classes [1, N]   (0-indexed, 1 = person)
        #   [2]: scores [1, N]
        #   [3]: num_detections [1]
        classes = self._interp.get_tensor(self._output_details[1]["index"])
        scores = self._interp.get_tensor(self._output_details[2]["index"])
        num = int(self._interp.get_tensor(self._output_details[3]["index"])[0])

        self._person_present = False
        for i in range(num):
            cls = int(classes[0][i])
            score = float(scores[0][i])
            if cls == PERSON_CLASS and score >= self.confidence:
                self._person_present = True
                break

        return self._person_present

    @property
    def person_present(self) -> bool:
        return self._person_present
