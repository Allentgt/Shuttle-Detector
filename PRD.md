# Shuttle Detector — PRD

## Problem
A badminton shuttle landing on the court floor is a discrete event. In practice sessions, drills, or casual play, there's no automated way to trigger sound on landing without a person monitoring each rally.

## Goal
A headless Raspberry Pi 4 system with camera that detects when a shuttle lands on the court floor and plays a custom sound. Controlled via web dashboard — no SSH or display needed after setup.

## Requirements

### Functional
1. **Detection**: Camera continuously monitors a full badminton court (both halves, ~13.4m). When a shuttle enters frame and stops moving on the floor, a landing event fires.
2. **Sound playback**: On landing event, plays a user-uploaded audio file through the 3.5mm jack to a powered speaker.
3. **Arm toggle**: Web switch that controls whether sound plays. Disarmed = detection runs + feed visible, but no sound.
4. **Dashboard**: Web UI showing live camera feed, recent landing info (timestamp + snapshot), sound upload/swap, test-play button, arm toggle.
5. **Sound upload**: User uploads a WAV/MP3 file via dashboard. Replaces active sound immediately. No SSH required.

### Non-functional
6. **Single process**: FastAPI serves web + streams video. Detection runs in a background thread. One Python process.
7. **CV-first**: Background subtraction + contour filtering. No ML model. Upgrade path to ML if needed.
8. **Dev/Prod split**: Mock camera on desktop (video file or test image). PiCamera via picamera2 on actual Pi.
9. **Cooldown**: Minimum 2s between landing events to prevent double-triggers.

## Platform
- **Hardware**: Raspberry Pi 4 (2GB), Camera Module v1.3, powered speaker via 3.5mm jack
- **OS**: Raspberry Pi OS Lite 64-bit (Bookworm+)
- **Language**: Python 3.11
- **Package manager**: uv
- **Camera lib**: picamera2 (via apt) with `--system-site-packages` venv

## Constraints
- No ML model (CV only)
- No cloud dependency
- No display — dashboard is the only UI
- Sound via `aplay` subprocess (pre-installed, zero deps)
