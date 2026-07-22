import cv2
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class BackgroundSubtractor:
    def __init__(self, history=500, var_threshold=16, detect_shadows=False):
        self._sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return self._sub.apply(frame)


class BlobFilter:
    def __init__(self, min_area=200, max_area=5000, floor_ratio=0.6,
                 min_aspect=0.3, max_aspect=3.0):
        self.min_area = min_area
        self.max_area = max_area
        self.floor_ratio = floor_ratio
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.floor_y: int | None = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def filter(self, mask: np.ndarray, frame_shape) -> list:
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height = frame_shape[0]
        floor_threshold = self.floor_y if self.floor_y is not None else int(height * self.floor_ratio)
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            center_y = y + h // 2
            if center_y < floor_threshold:
                continue
            aspect = w / h if h > 0 else 99
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue
            results.append((x, y, w, h))
        return results


class LandingDetector:
    def __init__(self, persistence_frames=5, cooldown_seconds=2.0,
                 fall_check_pixels=20, fall_check_history=15):
        self.persistence_frames = persistence_frames
        self.cooldown_seconds = cooldown_seconds
        self.fall_check_pixels = fall_check_pixels
        self.fall_check_history = fall_check_history
        self._tracks: dict = {}
        self._last_event_time = 0.0
        self._match_distance = 30
        self._blob_history: list = []
        self.person_filter = None    # optional PersonFilter instance
        self.shuttle_detector = None  # optional ShuttleDetector instance

    def _fell_from_above(self, cx, cy) -> bool:
        """Check if any recent blob was above this position (shuttle fell down)."""
        for hist in self._blob_history[:-1]:  # exclude current frame
            for hx, hy in hist:
                if (abs(hx - cx) < 50 and  # X tolerance (shuttle may drift)
                    hy < cy - self.fall_check_pixels):
                    return True
        return False

    def update(self, blobs: list, frame_shape) -> bool:
        now = time.monotonic()
        if now - self._last_event_time < self.cooldown_seconds:
            return False

        # Record current blob centroids for fall-from-above check
        cur_centroids = [(bx + bw // 2, by + bh // 2) for bx, by, bw, bh in blobs]
        self._blob_history.append(cur_centroids)
        if len(self._blob_history) > self.fall_check_history:
            self._blob_history.pop(0)

        matched = set()
        for cx, cy in cur_centroids:
            best_key = None
            for key, (tx, ty, _, _) in self._tracks.items():
                if abs(cx - tx) < self._match_distance and abs(cy - ty) < self._match_distance:
                    best_key = key
                    break
            if best_key is not None:
                ox, oy, cnt, _ = self._tracks[best_key]
                self._tracks[best_key] = (cx, cy, cnt + 1, 0)
                matched.add(best_key)
            else:
                self._tracks.setdefault(len(self._tracks), (cx, cy, 1, 0))

        # Prune unmatched tracks
        self._tracks = {k: v for k, v in self._tracks.items() if k in matched}

        for key, (cx, cy, count, _) in self._tracks.items():
            if count >= self.persistence_frames:
                # Person in frame → likely false positive
                if self.person_filter and self.person_filter.person_present:
                    logger.debug("Suppressed trigger — person in frame")
                    self._tracks.clear()
                    return False
                # No shuttle confirmed by ML → suppress
                if self.shuttle_detector and not self.shuttle_detector.shuttle_present:
                    logger.debug("Suppressed trigger — no shuttle in frame (ML)")
                    self._tracks.clear()
                    return False
                # Only trigger if the object appears to have fallen from above
                if self._fell_from_above(cx, cy):
                    self._last_event_time = now
                    self._tracks.clear()
                    return True
        return False

    def get_track_info(self) -> str:
        if not self._tracks:
            return "none"
        parts = [f"{v[2]}" for v in self._tracks.values()]
        return ",".join(parts)
