# Shuttle Detector

A Raspberry Pi 4 based system that uses a camera module and computer vision to detect when a badminton shuttlecock lands on the court floor, then plays a sound. Headless setup with a web interface for control and monitoring.

## Language

**Shuttle / Shuttlecock**:
A badminton shuttlecock — the projectile being hit during play.
_Avoid_: Birdie, ball, projectile

**Landing Event**:
The moment a shuttle makes contact with the court floor and remains stationary. The system detects this visually through CV and triggers the configured sound.
_Avoid_: Hit, bounce, score

**Court**:
A full regulation badminton court (13.4m × 6.1m). The system monitors both halves — both the player's side and the opponent's side.

**Detection**:
The computer vision process that continuously analyzes the camera feed to identify landing events. Uses background subtraction and blob filtering — no ML model.

**Arm Toggle**:
A software switch that controls whether sound plays. When armed, landing events trigger sound. When disarmed, detection still runs and the web feed still shows activity, but no sound is output.
_Avoid_: Enable, mute

**Dashboard**:
The web interface served by the Pi. Shows live camera feed, last landing timestamp + snapshot, sound upload/playback controls, and the arm toggle.

**Sound File**:
A user-uploaded audio file played through the Pi's 3.5mm jack to a powered speaker when a landing event is detected. Swappable via the dashboard without SSH.
