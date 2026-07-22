"""
Shuttle Detector — entry point.

Usage:
    uv run python src/main.py            # dev mode (webcam mock + log sound)
    uv run python src/main.py --prod     # Pi mode (picamera + aplay)
"""

import argparse
import logging
import sys
import os

# Ensure src/ is importable when running from project root
_src = os.path.join(os.path.dirname(__file__))
if _src not in sys.path:
    sys.path.insert(0, os.path.dirname(_src))

from src.detector.camera import MockCamera, PicameraCamera
from src.detector.detector import Detector, SharedState
from src.detector.classifier import PersonFilter
from src.detector.shuttle_detector import ShuttleDetector
from src.sound.player import SoundPlayer
from src.web.server import create_app
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def main():
    parser = argparse.ArgumentParser(description="Shuttle Detector")
    parser.add_argument("--prod", action="store_true", help="Enable Pi hardware (picamera + aplay)")
    parser.add_argument("--source", default=0, help="Video source for mock mode (file path or camera index)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--min-area", type=int, default=200, help="Min blob area in pixels (lower = more sensitive)")
    parser.add_argument("--max-area", type=int, default=5000, help="Max blob area in pixels")
    parser.add_argument("--floor-ratio", type=float, default=0.6, help="Lower portion of frame to monitor (0-1)")
    parser.add_argument("--persistence", type=int, default=5, help="Frames a blob must persist to trigger")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Min seconds between triggers")
    parser.add_argument("--fall-pixels", type=float, default=20, help="Min Y distance to confirm object fell from above")
    parser.add_argument("--min-aspect", type=float, default=0.3, help="Min blob aspect ratio (w/h)")
    parser.add_argument("--max-aspect", type=float, default=3.0, help="Max blob aspect ratio (w/h)")
    parser.add_argument("--debug", action="store_true", help="Show CV debug overlay on stream")
    parser.add_argument("--person-model", default=None,
                        help="Path to TFLite COCO model (default: auto-download to data/models/)")
    parser.add_argument("--shuttle-model", default=None,
                        help="Roboflow model ID for shuttle detection "
                             "(default: shuttlecock-detection-r9upv/1)")
    parser.add_argument("--roboflow-api-key", default=None,
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    args = parser.parse_args()

    state = SharedState()

    if args.prod:
        logger.info("Production mode: Pi camera + aplay sound")
        camera = PicameraCamera()
        player = SoundPlayer("data/sounds/current.wav")
    else:
        logger.info("Dev mode: mock camera (source=%s) + log sound", args.source)
        try:
            src = int(args.source)
        except ValueError:
            src = args.source
        camera = MockCamera(src)
        player = SoundPlayer("data/sounds/current.wav")

    detector = Detector(
        camera, state,
        min_area=args.min_area,
        max_area=args.max_area,
        floor_ratio=args.floor_ratio,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        debug=args.debug,
    )
    detector.landing.persistence_frames = args.persistence
    detector.landing.cooldown_seconds = args.cooldown
    detector.landing.fall_check_pixels = args.fall_pixels
    detector.sound_callback = player.play

    # Person detection filter (COCO TFLite model)
    pf = PersonFilter(model_path=args.person_model, confidence=0.4, interval=10)
    detector.person_filter = pf
    detector.landing.person_filter = pf

    # Shuttle ML detection filter (Roboflow inference)
    sd = ShuttleDetector(
        model_id=args.shuttle_model,
        api_key=args.roboflow_api_key,
        confidence=0.3,
        interval=10,
    )
    detector.shuttle_detector = sd
    detector.landing.shuttle_detector = sd

    app = create_app(state)

    logger.info("Starting detector...")
    detector.start()

    logger.info("Starting web server on http://0.0.0.0:%d", args.port)
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    finally:
        detector.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
