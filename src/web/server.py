import asyncio
import cv2
import os
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
SNAPSHOT_DIR = "data/snapshots"
SOUND_DIR = "data/sounds"


def create_app(state, detector=None) -> FastAPI:
    app = FastAPI(title="Shuttle Detector")

    @app.get("/")
    async def dashboard():
        index_path = os.path.join(STATIC_DIR, "index.html")
        with open(index_path, encoding="utf-8") as f:
            return HTMLResponse(f.read())

    @app.get("/stream")
    async def mjpeg_stream():
        async def generate():
            while True:
                frame = state.get_latest_frame()
                if frame is not None:
                    ret, jpeg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                    )
                    if ret:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                        )
                await asyncio.sleep(0.05)

        return StreamingResponse(
            generate(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/api/status")
    async def status():
        last = state.get_last_event()
        return {
            "armed": state.is_armed(),
            "last_event_timestamp": last.timestamp if last else None,
            "last_event_snapshot": (
                "/api/snapshots/last.jpg"
                if last and last.snapshot_path and os.path.exists(last.snapshot_path)
                else None
            ),
            "sound_file": state.get_sound_path(),
        }

    def _read_config():
        """Read live config from detector objects."""
        cfg = {}
        if detector:
            bf = detector.blob_filter
            ld = detector.landing
            cfg["min_area"] = bf.min_area
            cfg["max_area"] = bf.max_area
            cfg["floor_ratio"] = bf.floor_ratio
            cfg["min_aspect"] = bf.min_aspect
            cfg["max_aspect"] = bf.max_aspect
            cfg["persistence"] = ld.persistence_frames
            cfg["cooldown"] = ld.cooldown_seconds
            cfg["fall_pixels"] = ld.fall_check_pixels
            cfg["match_distance"] = ld._match_distance
        cfg["floor_y"] = state.floor_y
        return cfg

    @app.get("/api/config")
    async def get_config():
        return _read_config()

    @app.post("/api/config")
    async def set_config(body: dict):
        if not detector:
            return JSONResponse({"ok": False, "error": "Detector not available"}, status_code=503)
        bf = detector.blob_filter
        ld = detector.landing
        if "min_area" in body:
            bf.min_area = int(body["min_area"])
        if "max_area" in body:
            bf.max_area = int(body["max_area"])
        if "floor_ratio" in body:
            bf.floor_ratio = float(body["floor_ratio"])
        if "min_aspect" in body:
            bf.min_aspect = float(body["min_aspect"])
        if "max_aspect" in body:
            bf.max_aspect = float(body["max_aspect"])
        if "persistence" in body:
            ld.persistence_frames = int(body["persistence"])
        if "cooldown" in body:
            ld.cooldown_seconds = float(body["cooldown"])
        if "fall_pixels" in body:
            ld.fall_check_pixels = float(body["fall_pixels"])
        if "match_distance" in body:
            ld._match_distance = float(body["match_distance"])
        logger.info("Config updated: %s", body)
        return _read_config()

    @app.get("/api/arm")
    async def get_arm():
        return {"armed": state.is_armed()}

    @app.post("/api/arm")
    async def set_arm(body: dict):
        state.set_armed(bool(body.get("armed", False)))
        return {"armed": state.is_armed()}

    @app.post("/api/sound")
    async def upload_sound(file: UploadFile = File(...)):
        if not file.filename:
            return JSONResponse({"ok": False, "error": "No file provided"}, status_code=400)
        ext = os.path.splitext(file.filename)[1].lower()
        if ext != ".wav":
            return JSONResponse(
                {"ok": False, "error": "Only .wav files are accepted (aplay requirement)"},
                status_code=400,
            )
        os.makedirs(SOUND_DIR, exist_ok=True)
        dest = os.path.join(SOUND_DIR, "current.wav")
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
        state.set_sound_path(dest)
        return {"ok": True, "path": dest}

    @app.get("/api/snapshots/{filename:path}")
    async def get_snapshot(filename: str):
        if ".." in filename or "/" in filename:
            return JSONResponse({"error": "Invalid path"}, status_code=400)
        path = os.path.normpath(os.path.join(SNAPSHOT_DIR, filename))
        if not path.startswith(os.path.normpath(SNAPSHOT_DIR)):
            return JSONResponse({"error": "Access denied"}, status_code=403)
        if not os.path.exists(path):
            return JSONResponse({"error": "Not found"}, status_code=404)
        return FileResponse(path)

    @app.post("/api/play-test")
    async def play_test():
        state.test_sound_requested = True
        return {"ok": True}

    @app.get("/api/calibrate-frame")
    async def calibrate_frame():
        """Return the latest frame as a static JPEG for calibration clicking."""
        frame = state.get_latest_frame()
        if frame is None:
            return JSONResponse({"error": "No frame yet"}, status_code=503)
        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ret:
            return JSONResponse({"error": "Encoding failed"}, status_code=500)
        from fastapi.responses import Response
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")

    @app.post("/api/calibrate")
    async def calibrate(body: dict):
        y = body.get("y")
        if not isinstance(y, (int, float)):
            return JSONResponse({"error": "Missing or invalid 'y'"}, status_code=400)
        state.floor_y = int(y)
        logger.info("Floor calibrated at y=%d", int(y))
        return {"ok": True, "floor_y": int(y)}

    @app.delete("/api/calibrate")
    async def reset_calibrate():
        state.floor_y = None
        logger.info("Floor calibration reset")
        return {"ok": True}

    return app
