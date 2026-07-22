from abc import ABC, abstractmethod
import numpy as np

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
        self._cap = None
        self._use_v4l2 = False

    def start(self):
        try:
            from picamera2 import Picamera2

            self._cam = Picamera2()
            config = self._cam.create_video_configuration(
                main={"size": self.size, "format": "RGB888"}
            )
            self._cam.configure(config)
            self._cam.start()
            self._use_v4l2 = False
        except ImportError:
            # Fall back to OpenCV V4L2 on /dev/video0
            import cv2

            self._cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not self._cap.isOpened():
                raise RuntimeError(
                    "picamera2 missing and /dev/video0 not accessible. "
                    "Install: sudo apt install python3-picamera2 --no-install-recommends"
                )
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            self._use_v4l2 = True

    def stop(self):
        if self._use_v4l2:
            if self._cap:
                self._cap.release()
            self._cap = None
        elif self._cam:
            self._cam.stop()
            self._cam = None

    def read_frame(self) -> np.ndarray | None:
        if self._use_v4l2:
            ret, frame = self._cap.read()
            return frame if ret else None
        elif self._cam:
            return self._cam.capture_array()
        return None


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
