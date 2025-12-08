# app.py

import time
import threading
from typing import Optional, Tuple

import cv2

from car_controller import CarController
from emotion_heuristics import EmotionHeuristics, FaceMetrics, EmotionFerPlus
from voice_control import VoiceThread
from motion_controller import MotionController

# Try to import Picamera2 (preferred on Raspberry Pi)
try:
    from picamera2 import Picamera2
    HAVE_PICAM2 = True
except ImportError:
    HAVE_PICAM2 = False

# SCRIPTED REACTIONS (keycodes)
# W=87, S=83, A=65, D=68

EMOTION_SCRIPTS = {
    # Happy: quick hop forward + playful wiggle
    "happy": [
        ("DOWN 87", 0.15),  # small forward hop
        ("UP 87",   0.05),
        ("DOWN 68", 0.10),  # right wiggle
        ("UP 68",   0.05),
        ("DOWN 65", 0.10),  # left wiggle
        ("UP 65",   0.05),
        ("STOP",    0.00),
    ],

    # Sad: reverse then forward back to position, small hesitant wiggle
    "sad": [
        ("DOWN 83", 0.18),   # reverse
        ("UP 83",   0.05),

        ("DOWN 87", 0.18),   # forward return
        ("UP 87",   0.05),

        # tiny hesitant wiggle
        ("DOWN 65", 0.10),   # left
        ("UP 65",   0.05),
        ("DOWN 68", 0.10),   # right
        ("UP 68",   0.05),

        ("STOP",    0.00),
    ],

    # Angry: strong retreat + aggressive spin
    "angry": [
        ("DOWN 83", 0.20),  # reverse a bit
        ("UP 83",   0.05),
        ("DOWN 83", 0.20),
        ("UP 83",   0.05),

        ("DOWN 68", 0.20),  # spin right
        ("UP 68",   0.05),
        ("DOWN 68", 0.20),
        ("UP 68",   0.05),

        ("STOP",    0.00),
    ],

    # No face: just stop
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
        # Low-level car control (real PiCar-X HW)
        self.car = CarController()

        # Unified motion controller (keycode + logical actions)
        self.motion = MotionController(self.car, speed=18, turn_angle=20)

        # Emotion detector: you can swap between heuristics and FER+ if you like
        # self.detector = EmotionFerPlus()     # ONNX FER+ (requires onnxruntime + haarcascade)
        self.detector = EmotionHeuristics()    # MediaPipe heuristics
        self.mode_emotion = False

        # Emotion cooldown: minimum time between auto emotion reactions
        self.emotion_cooldown = 3.0  # seconds
        self._last_emotion_time = 0.0
        self._last_emotion_label = "no_face"

        # Voice control toggle
        self.voice_enabled = True

        # Script lock so reactions don’t overlap too wildly
        self._script_lock = threading.Lock()

        # Voice thread; will call on_voice_command(cmd: str)
        self.voice = VoiceThread(self.on_voice_command)
        self.voice.start()

        # Camera: prefer Picamera2, fall back to cv2.VideoCapture, else None
        self.picam2 = None
        self.cap = None

        if HAVE_PICAM2:
            print("[App] Using Picamera2 for video capture")
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"format": "XRGB8888", "size": (640, 480)}
            )
            self.picam2.configure(config)
            self.picam2.start()
        else:
            print("[App] Picamera2 not available; trying cv2.VideoCapture(0)")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self.cap = cap
                print("[App] Using cv2.VideoCapture(0) for video capture")
            else:
                print("[App] WARNING: No camera available; running without video/emotion.")

    # ------------------------------------------------------------------
    # VOICE → KEYCODE SCRIPTS
    # ------------------------------------------------------------------

    def on_voice_command(self, cmd: str):
        """
        Map voice recognition results to scripted keycode reactions.
        """
        cmd = cmd.strip().lower()
        print(f"[Voice CMD] {cmd}")

        if cmd in ("voice_off", "voice off"):
            self.voice_enabled = False
            print("[Voice] Voice control DISABLED")
            return

        if not self.voice_enabled:
            print("[Voice] Ignoring command (voice disabled)")
            return

        if cmd in ("emotion_on", "emotion on"):
            self.mode_emotion = True
            print("[Voice] Emotion mode ON")
            # reset gating so next emotion can react immediately
            self._last_emotion_time = 0.0
            self._last_emotion_label = "no_face"
            return
        if cmd in ("emotion_off", "emotion off"):
            self.mode_emotion = False
            print("[Voice] Emotion mode OFF")
            return

        if cmd in ("happy", "be happy"):
            print("[Voice] Triggering HAPPY emotion reaction (force)")
            self.emotion_to_action("happy", force=True)
            return

        if cmd in ("sad", "be sad"):
            print("[Voice] Triggering SAD emotion reaction (force)")
            self.emotion_to_action("sad", force=True)
            return

        if cmd in ("angry", "be angry"):
            print("[Voice] Triggering ANGRY emotion reaction (force)")
            self.emotion_to_action("angry", force=True)
            return

        # If we have a scripted movement reaction for this phrase, run it
        if cmd in VOICE_SCRIPTS:
            print(f"[Voice] Running script for '{cmd}'")
            self.run_script(VOICE_SCRIPTS[cmd])
            return

    # ------------------------------------------------------------------
    # EMOTION → KEYCODE SCRIPTS (with cooldown + change check)
    # ------------------------------------------------------------------

    def emotion_to_action(self, label: str, force: bool = False):
        """
        Map emotion label -> scripted sequence of keycode commands.

        - Automatic calls (force=False):
            * At most one reaction every self.emotion_cooldown seconds
            * Only when label != last reacted label
        - Manual test (force=True, via keys 1–4):
            * Ignores cooldown + label check, does NOT affect cooldown state
        """
        script = EMOTION_SCRIPTS.get(label)
        if not script:
            return

        # Manual test path: ignore cooldown + last label, and don't disturb them.
        if force:
            print(f"[Emotion] (FORCE) Running script for label={label}")
            self.run_script(script)
            return

        now = time.monotonic()

        # Only react if the *new* label is different from the last reacted label
        if label == self._last_emotion_label:
            # same emotion as last reaction → skip
            return

        dt = now - self._last_emotion_time
        if dt < self.emotion_cooldown:
            # still in cooldown window → skip
            return

        # Passed both checks: update gating state and run
        self._last_emotion_time = now
        self._last_emotion_label = label
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

        # Mode + emotion label + voice mode
        text = (
            f"emotion_mode: {'ON' if self.mode_emotion else 'OFF'} | "
            f"voice: {'ON' if self.voice_enabled else 'OFF'} | "
            f"emotion: {label}"
        )
        cv2.putText(
            frame,
            text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 220, 50),
            2,
        )

        # Metrics (only if using EmotionHeuristics; FER+ returns None)
        if metrics:
            cv2.putText(
                frame,
                f"smile={metrics.smile:.2f}  brow={metrics.brow_furrow:.2f}  mouth={metrics.mouth_open:.2f}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

        # Help text
        cv2.putText(
            frame,
            "[E] emotion mode  [V] voice toggle  [1-4] test scripts  [Q] quit",
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
                frame = None
                label = "no_face"
                metrics: Optional[FaceMetrics] = None
                bbox: Optional[Tuple[int, int, int, int]] = None

                # Grab a frame from Picamera2 or cv2.VideoCapture if available
                if self.picam2 is not None:
                    # Picamera2 returns BGRA; convert to BGR for OpenCV
                    arr = self.picam2.capture_array()
                    frame = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                elif self.cap is not None:
                    ok, f = self.cap.read()
                    if not ok:
                        key = cv2.waitKey(1) & 0xFF
                        continue
                    frame = f

                if frame is not None:
                    # Run emotion detector and optional reaction
                    label, metrics, bbox = self.detector.estimate_emotion(frame)

                    if self.mode_emotion:
                        self.emotion_to_action(label)

                    out = self.draw_overlay(frame, label, metrics, bbox)
                    cv2.imshow("PiCar-X Emotion Sensing (debug view)", out)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    # No camera available: keep UI responsive; voice still runs in background
                    key = cv2.waitKey(50) & 0xFF

                # Keyboard controls
                if key in (ord("q"), ord("Q")):
                    break

                # Toggle emotion mode
                if key in (ord("e"), ord("E")):
                    self.mode_emotion = not self.mode_emotion
                    print(f"[App] Emotion mode set to {self.mode_emotion}")
                    # reset gating when toggling mode
                    self._last_emotion_time = 0.0
                    self._last_emotion_label = "no_face"

                # Toggle voice control
                if key in (ord("v"), ord("V")):
                    self.voice_enabled = not self.voice_enabled
                    print(f"[App] Voice control set to {self.voice_enabled}")

                # Manual test triggers for emotion scripts (bypass cooldown/label gating)
                if key == ord("1"):
                    print("[Key] Trigger HAPPY script (force)")
                    self.emotion_to_action("happy", force=True)
                if key == ord("2"):
                    print("[Key] Trigger ANGRY script (force)")
                    self.emotion_to_action("angry", force=True)
                if key == ord("3"):
                    print("[Key] Trigger SAD script (force)")
                    self.emotion_to_action("sad", force=True)
                if key == ord("4"):
                    print("[Key] Trigger NO_FACE script (force)")
                    self.emotion_to_action("no_face", force=True)

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
            if self.picam2 is not None:
                self.picam2.stop()
        except Exception:
            pass
        try:
            if self.cap is not None:
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
                    # send keycode-style command into MotionController
                    self.motion.handle_line(cmd)
                    if delay > 0:
                        time.sleep(delay)
            finally:
                self._script_lock.release()

        threading.Thread(target=_worker, daemon=True).start()


if __name__ == "__main__":
    App().loop()
