import threading
import time
import cv2
import os
import logging
from dataclasses import dataclass
from .camera import Camera
from .pipeline import BackgroundSubtractor, BlobFilter, LandingDetector

logger = logging.getLogger(__name__)


@dataclass
class DetectionEvent:
    timestamp: float
    snapshot_path: str | None = None


class SharedState:
    def __init__(self):
        self.armed: bool = True
        self.sound_path: str = "data/sounds/default.wav"
        self.last_event: DetectionEvent | None = None
        self.latest_frame = None
        self.test_sound_requested: bool = False
        self.floor_y: int | None = None
        self._lock = threading.Lock()

    def get_latest_frame(self):
        with self._lock:
            return self.latest_frame

    def set_latest_frame(self, frame):
        with self._lock:
            self.latest_frame = frame

    def get_last_event(self):
        with self._lock:
            return self.last_event

    def set_last_event(self, event):
        with self._lock:
            self.last_event = event

    def is_armed(self):
        with self._lock:
            return self.armed

    def toggle_arm(self):
        with self._lock:
            self.armed = not self.armed
            return self.armed

    def set_armed(self, val: bool):
        with self._lock:
            self.armed = val

    def get_sound_path(self):
        with self._lock:
            return self.sound_path

    def set_sound_path(self, path: str):
        with self._lock:
            self.sound_path = path


class Detector:
    def __init__(
        self,
        camera: Camera,
        state: SharedState,
        min_area=200,
        max_area=5000,
        floor_ratio=0.6,
        min_aspect=0.3,
        max_aspect=3.0,
        debug=False,
    ):
        self.camera = camera
        self.state = state
        self.bg_sub = BackgroundSubtractor()
        self.blob_filter = BlobFilter(min_area, max_area, floor_ratio, min_aspect, max_aspect)
        self.landing = LandingDetector()
        self._thread: threading.Thread | None = None
        self._running = False
        self.sound_callback = None
        self.person_filter = None
        self.shuttle_detector = None
        self.debug = debug

    def start(self):
        self.camera.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Detector started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self.camera.stop()
        logger.info("Detector stopped")

    def _loop(self):
        # Warm up background model
        for _ in range(10):
            frame = self.camera.read_frame()
            if frame is not None:
                self.bg_sub.apply(frame)

        while self._running:
            # Check for test-sound requests before frame read
            if self.state.test_sound_requested:
                self.state.test_sound_requested = False
                if self.sound_callback:
                    self.sound_callback()

            frame = self.camera.read_frame()
            if frame is None:
                logger.warning("Empty frame from camera — check camera connection")
                time.sleep(0.03)
                continue

            # Run person detection (cached; runs inference every N frames)
            if self.person_filter:
                self.person_filter.detect(frame)
            # Run shuttle ML detection (cached; runs inference every N frames)
            if self.shuttle_detector:
                self.shuttle_detector.detect(frame)

            # Push floor calibration from state to pipeline
            if self.state.floor_y is not None:
                self.blob_filter.floor_y = self.state.floor_y
            else:
                self.blob_filter.floor_y = None

            mask = self.bg_sub.apply(frame)
            blobs = self.blob_filter.filter(mask, frame.shape)

            display = frame.copy()
            if self.debug:
                h = display.shape[0]
                fy = self.state.floor_y if self.state.floor_y is not None else int(h * self.blob_filter.floor_ratio)
                cv2.line(display, (0, fy), (display.shape[1], fy), (0, 255, 255), 2)
                for bx, by, bw, bh in blobs:
                    cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(display, f"blobs: {len(blobs)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                tracks = self.landing.get_track_info()
                cv2.putText(display, f"tracks: {tracks}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            self.state.set_latest_frame(display)

            if self.landing.update(blobs, frame.shape):
                ts = time.time()
                snap_dir = "data/snapshots"
                os.makedirs(snap_dir, exist_ok=True)
                snap_path = os.path.join(snap_dir, "last.jpg")
                cv2.imwrite(snap_path, frame)
                self.state.set_last_event(
                    DetectionEvent(timestamp=ts, snapshot_path=snap_path)
                )
                if self.state.is_armed() and self.sound_callback:
                    self.sound_callback()
