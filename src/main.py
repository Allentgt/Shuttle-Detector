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

# Ensure project root is on sys.path for `src.` imports
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.detector.camera import MockCamera, PicameraCamera
from src.detector.detector import Detector, SharedState
from src.detector.classifier import PersonFilter
from src.detector.onnx_shuttle import ONNXShuttleDetector
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
    parser.add_argument("--prod", action="store_true", help="Pi hardware mode (picamera or USB webcam + aplay)")
    parser.add_argument("--usb", type=int, default=None,
                        help="USB webcam index (omit to use picamera2 with --prod)")
    parser.add_argument("--cam-width", type=int, default=640, help="Camera capture width (lower = faster USB)")
    parser.add_argument("--cam-height", type=int, default=480, help="Camera capture height")
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
    parser.add_argument("--bg-learning-rate", type=float, default=None,
                        help="Background model learning rate (default=auto, 0=never adapt)")
    parser.add_argument("--debug", action="store_true", help="Show CV debug overlay on stream")
    parser.add_argument("--person-model", default=None,
                        help="Path to TFLite COCO model (default: auto-download to data/models/)")
    parser.add_argument("--shuttle-onnx", default=None,
                        help="Path to YOLOv8 ONNX model for shuttle detection "
                             "(e.g. data/weights/best.onnx)")
    args = parser.parse_args()

    state = SharedState()

    if args.prod:
        if args.usb is not None:
            logger.info("USB webcam mode: /dev/video%d + aplay sound", args.usb)
            camera = MockCamera(args.usb, width=args.cam_width, height=args.cam_height)
        else:
            logger.info("Pi camera mode: picamera2 + aplay sound")
            camera = PicameraCamera()
    else:
        logger.info("Dev mode: mock camera (source=%s) + log sound", args.source)
        try:
            src = int(args.source)
        except ValueError:
            src = args.source
        camera = MockCamera(src, width=args.cam_width, height=args.cam_height)
    player = SoundPlayer("data/sounds/current.wav")

    detector = Detector(
        camera, state,
        min_area=args.min_area,
        max_area=args.max_area,
        floor_ratio=args.floor_ratio,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        debug=args.debug,
        bg_learning_rate=args.bg_learning_rate,
    )
    detector.landing.persistence_frames = args.persistence
    detector.landing.cooldown_seconds = args.cooldown
    detector.landing.fall_check_pixels = args.fall_pixels
    detector.sound_callback = player.play

    # Person detection filter (COCO TFLite model) — gracefully degrades if tflite missing
    pf = PersonFilter(model_path=args.person_model, confidence=0.4, interval=10)
    detector.person_filter = pf
    detector.landing.person_filter = pf

    # Shuttle ML detection filter (ONNX YOLOv8 model)
    if args.shuttle_onnx is not None:
        sd = ONNXShuttleDetector(model_path=args.shuttle_onnx, confidence=0.25, interval=5)
        detector.shuttle_detector = sd
        detector.landing.shuttle_detector = sd
        logger.info("ONNX shuttle detection enabled (%s)", args.shuttle_onnx)
    else:
        logger.info("No --shuttle-onnx given — shuttle ML gate disabled, CV-only mode")

    app = create_app(state, detector=detector)

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
