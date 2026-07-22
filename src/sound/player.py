import subprocess
import os
import sys
import struct
import wave
import logging

logger = logging.getLogger(__name__)

_DEFAULT_SOUND_PATH = "data/sounds/default.wav"


def _make_beep_wav(path: str, freq=660, duration=0.2, volume=0.3, sample_rate=22050):
    """Write a short sine-wave beep WAV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_samples = int(sample_rate * duration)
    data = b"".join(
        struct.pack("<h", int(volume * 32767 * 0.5 * (
            1.0 + __import__("math").sin(2 * __import__("math").pi * freq * t / sample_rate)
        )))
        for t in range(n_samples)
    )
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)


class SoundPlayer:
    def __init__(self, sound_path: str = _DEFAULT_SOUND_PATH):
        self.sound_path = sound_path
        self._is_pi = (
            sys.platform.startswith("linux") and os.path.exists("/usr/bin/aplay")
        )
        # Create a default beep WAV if no sound file exists
        if not os.path.exists(sound_path):
            _make_beep_wav(sound_path)
            logger.info("Generated default beep sound at %s", sound_path)

    def set_sound(self, path: str):
        self.sound_path = path
        logger.info("Sound set to %s", path)

    def play(self):
        if not os.path.exists(self.sound_path):
            logger.warning("Sound file not found: %s", self.sound_path)
            return
        if self._is_pi:
            subprocess.Popen(
                ["aplay", self.sound_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform == "win32":
            import winsound
            winsound.PlaySound(self.sound_path, winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        else:
            logger.info("SOUND: %s", self.sound_path)
