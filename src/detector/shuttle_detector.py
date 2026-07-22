"""
ML-based shuttle detection via Roboflow inference package.

Mirrors the PersonFilter pattern: runs object-detection inference every N
frames, caches the result in between.  Model weights are downloaded on the
first call and cached locally (default /tmp/cache; set $MODEL_CACHE_DIR
for persistence across reboots).  Subsequent runs are fully offline.

Gracefully disables when the inference package or API key is missing.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "shuttlecock-detection-r9upv/1"


class ShuttleDetector:
    """Periodically runs shuttle ML detection on the frame.

    shuttle_detector.detect(frame) runs inference every `interval` frames
    and caches the result.  shuttle_detector.shuttle_present reflects the
    most recent inference result.
    """

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        confidence: float = 0.3,
        interval: int = 10,
    ):
        self.confidence = confidence
        self.interval = interval
        self._count = 0
        self._shuttle_present = False
        self._model = None

        if model_id is None:
            model_id = DEFAULT_MODEL_ID

        if api_key is None:
            api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            logger.info(
                "ROBOFLOW_API_KEY not set — shuttle ML detection disabled. "
                "Set the env var or pass --roboflow-api-key"
            )
            return

        try:
            from inference import get_model

            self._model = get_model(model_id=model_id, api_key=api_key)
            logger.info(
                "ShuttleDetector loaded (model=%s, conf=%.1f, every %d frames)",
                model_id,
                confidence,
                interval,
            )
        except ImportError:
            logger.info(
                "inference package not installed — shuttle ML detection disabled. "
                "Install: pip install inference"
            )
        except Exception as e:
            logger.warning("Failed to load shuttle model: %s", e)

    def detect(self, frame: np.ndarray) -> bool:
        """Returns True if a shuttle is detected in the frame.

        Runs inference every *interval* frames; caches result in between.
        """
        if self._model is None:
            return False

        self._count += 1
        if self._count % self.interval != 0:
            return self._shuttle_present

        try:
            results = self._model.infer(frame)
            # Object-detection output: {'predictions': [{...}, ...]}
            predictions = results.get("predictions", []) if isinstance(results, dict) else []
            self._shuttle_present = any(
                p.get("confidence", 0) >= self.confidence for p in predictions
            )
        except Exception as e:
            logger.warning("Shuttle inference call failed: %s", e)
            self._shuttle_present = False

        return self._shuttle_present

    @property
    def shuttle_present(self) -> bool:
        """Most recently cached shuttle detection result."""
        return self._shuttle_present
