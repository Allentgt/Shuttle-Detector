# Implementation Plan

## 1. Project scaffold
- `pyproject.toml` — deps: fastapi, uvicorn, opencv-python-headless, numpy
- `.python-version` — 3.11
- Directory structure: `src/{detector,web,sound}/`, `data/{sounds,snapshots}/`

## 2. Detector module (`src/detector/`)
- **camera.py** — `Camera` base + `PicameraCamera` (picamera2) + `MockCamera` (OpenCV from file)
- **pipeline.py** — `BackgroundSubtractor` (MOG2) + `BlobFilter` (size/position gate) + `LandingDetector` (stable-blob → event)
- **detector.py** — async event loop: grab frame → process → emit events to shared state

## 3. Sound module (`src/sound/`)
- **player.py** — `SoundPlayer`: subprocess `aplay WAV` on Pi, log on dev. Single-sound, no queue, newest wins.

## 4. Web server (`src/web/`)
- **server.py** — FastAPI routes: `/` dashboard, `/stream` MJPEG, `/api/status` JSON, `/api/arm` toggle, `/api/sound` upload, `/api/snapshots/{name}` serve
- **static/index.html** — Dashboard: MJPEG feed, status panel, arm toggle (CSS switch), sound upload + test-play, last-landing timestamp

## 5. Entry point (`src/main.py`)
- Wires camera → detector → shared state → web server
- FastAPI lifespan: start detector thread, clean up on shutdown
- `--prod` flag: PiCamera + aplay. Default: MockCamera + log.

## Dependency graph

```
pyproject.toml          (independent)
camera.py               (independent)
pipeline.py             (needs camera frame format)
detector.py             (needs camera + pipeline)
player.py               (independent)
server.py               (needs shared state dataclass shape)
index.html              (independent)
main.py                 (needs everything)
```

## Execution order
1. Project config + dirs (me)
2. Implement modules (parallel):
   - Detector (fixer)
   - Sound player (fixer)
   - Web server (fixer)
   - Dashboard HTML (designer)
3. Wire up main.py (me)
4. Verify `uv sync` + structure
