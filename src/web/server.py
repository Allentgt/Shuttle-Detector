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


def create_app(state) -> FastAPI:
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
