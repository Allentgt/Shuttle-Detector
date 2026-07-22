import logging
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class Camera(ABC):
    @abstractmethod
    def start(self): ...

    @abstractmethod
    def stop(self): ...

    @abstractmethod
    def read_frame(self) -> np.ndarray | None: ...


class PicameraCamera(Camera):
    def __init__(self, size=(640, 480)):
        self.size = size
        self._cam = None

    def start(self):
        from picamera2 import Picamera2  # will raise ImportError if missing

        self._cam = Picamera2()
        config = self._cam.create_video_configuration(
            main={"size": self.size, "format": "RGB888"}
        )
        self._cam.configure(config)
        self._cam.start()
        logger.info("Pi camera started (%dx%d)", *self.size)

    def stop(self):
        if self._cam:
            self._cam.stop()
            self._cam = None
        logger.info("Pi camera stopped")

    def read_frame(self) -> np.ndarray | None:
        if self._cam is None:
            return None
        return self._cam.capture_array()


class MockCamera(Camera):
    def __init__(self, source: int | str = 0):
        self.source = source
        self._cap = None

    def start(self):
        import cv2
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

    def stop(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def read_frame(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None
