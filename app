# app.py

import time
import threading
from typing import Optional, Tuple

import cv2

from car_controller import CarController
from emotion_heuristics import EmotionHeuristics, FaceMetrics, EmotionFerPlus
from voice_control import VoiceThread
from motion_controller import MotionController

# SCRIPTED REACTIONS (keycodes)
# W=87, S=83, A=65, D=68

EMOTION_SCRIPTS = {
    "happy": [
        ("DOWN 87", 0.12),  # W down
        ("UP 87",   0.05),  # W up
        ("DOWN 68", 0.12),  # D down
        ("UP 68",   0.05),  # D up
        ("STOP",    0.00),
    ],
    "angry": [
        ("DOWN 83", 0.15),  # S down
        ("UP 83",   0.05),  # S up
        ("DOWN 65", 0.10),  # A down
        ("UP 65",   0.05),
        ("STOP",    0.00),
    ],
    "neutral": [
        ("STOP", 0.00),
    ],
    "no_face": [
        ("STOP", 0.00),
    ],
}

VOICE_SCRIPTS = {
    "forward": [
        ("DOWN 87", 0.20),
        ("UP 87",   0.05),
    ],
    "back": [
        ("DOWN 83", 0.20),
        ("UP 83",   0.05),
    ],
    "left": [
        ("DOWN 65", 0.15),
        ("UP 65",   0.05),
    ],
    "right": [
        ("DOWN 68", 0.15),
        ("UP 68",   0.05),
    ],
    "spin": [
        ("DOWN 68", 0.25),
        ("UP 68",   0.05),
        ("DOWN 65", 0.25),
        ("UP 65",   0.05),
        ("STOP",    0.00),
    ],
}


class App:
    def __init__(self):
        # Low-level car control (real or SIM)
        self.car = CarController()

        # Unified motion controller (keycode + logical actions)
        self.motion = MotionController(self.car, speed=18, turn_angle=20)

        # Emotion detector (you can later swap to EmotionFerPlus if you want)
        self.detector = EmotionHeuristics()
        self.mode_emotion = False

        # Script lock so reactions don’t overlap too wildly
        self._script_lock = threading.Lock()

        # Voice thread; will call on_voice_command(cmd: str)
        self.voice = VoiceThread(self.on_voice_command)
        self.voice.start()

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Camera not found. Make sure the camera is enabled and attached."
            )

    def on_voice_command(self, cmd: str):
        """
        Map voice recognition results to scripted keycode reactions.
        """
        cmd = cmd.strip().lower()
        print(f"[Voice CMD] {cmd}")

        # Toggle emotion mode via voice if you want
        if cmd in ("emotion_on", "emotion on"):
            self.mode_emotion = True
            print("[Voice] Emotion mode ON")
            return
        if cmd in ("emotion_off", "emotion off"):
            self.mode_emotion = False
            print("[Voice] Emotion mode OFF")
            # optional: stop when leaving emotion mode
            # self.motion.handle_line("STOP")
            return

        # If we have a scripted reaction for this phrase, run it
        if cmd in VOICE_SCRIPTS:
            print(f"[Voice] Running script for '{cmd}'")
            self.run_script(VOICE_SCRIPTS[cmd])
            return

        # Else: ignore or add more direct mappings here if you want
        # e.g., "start" / "stop" could map to long holds instead

    # ------------------------------------------------------------------
    # EMOTION → KEYCODE SCRIPTS
    # ------------------------------------------------------------------

    def emotion_to_action(self, label: str):
        """
        Map emotion label -> scripted sequence of keycode commands.
        """
        script = EMOTION_SCRIPTS.get(label)
        if not script:
            return

        print(f"[Emotion] Running script for label={label}")
        self.run_script(script)

    def draw_overlay(
        self,
        frame,
        label: str,
        metrics: Optional[FaceMetrics],
        bbox: Optional[Tuple[int, int, int, int]],
    ):
        h, w = frame.shape[:2]

        # Face box
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Mode + emotion label
        text = f"mode: {'EMOTION' if self.mode_emotion else 'IDLE'} | emotion: {label}"
        cv2.putText(
            frame,
            text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 220, 50),
            2,
        )

        # Metrics (only if using EmotionHeuristics; FER+ returns None)
        if metrics:
            cv2.putText(
                frame,
                f"smile={metrics.smile:.2f}  brow={metrics.brow_furrow:.2f}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

        # Help text
        cv2.putText(
            frame,
            "[E] toggle emotion mode    [Q] quit",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
        )
        return frame

    def loop(self):
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    continue

                label, metrics, bbox = self.detector.estimate_emotion(frame)

                if self.mode_emotion:
                    self.emotion_to_action(label)

                out = self.draw_overlay(frame, label, metrics, bbox)
                cv2.imshow("PiCar-X Emotion Sensing (debug view)", out)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("e"), ord("E")):
                    self.mode_emotion = not self.mode_emotion

        finally:
            self.cleanup()

    def cleanup(self):
        print("[App] shutting down…")
        try:
            self.motion.shutdown()
        except Exception:
            pass
        try:
            self.car.shutdown()
        except Exception:
            pass
        try:
            if self.voice:
                self.voice.stop()
        except Exception:
            pass
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    def run_script(self, script):
        """
        Run a scripted reaction: a list of (command_string, delay_seconds) pairs.
        """
        def _worker():
            # prevent overlapping reaction scripts (optional but nice)
            if not self._script_lock.acquire(blocking=False):
                print("[App] Script already running, ignoring new one")
                return
            try:
                for cmd, delay in script:
                    self.motion.handle_line(cmd)
                    if delay > 0:
                        time.sleep(delay)
            finally:
                self._script_lock.release()

        threading.Thread(target=_worker, daemon=True).start()


if __name__ == "__main__":
    App().loop()
