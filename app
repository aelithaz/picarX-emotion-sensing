# app.py

import time
from typing import Optional, Tuple

import cv2

from car_controller import CarController
from emotion_heuristics import EmotionHeuristics, FaceMetrics
from voice_control import VoiceThread
from motion_controller import MotionController


class App:
    def __init__(self):
        # Low-level car control (real or SIM)
        self.car = CarController()

        # Unified motion controller (keycode + logical actions)
        self.motion = MotionController(self.car, speed=18, turn_angle=20)

        # Emotion detector
        self.detector = EmotionHeuristics()
        self.mode_emotion = False

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
        Skeleton: map voice recognition results to motion via keycodes.

        Later hardcode how specific voice phrases map to keycodes, e.g.:

            if cmd == "go forward":
                self.motion.handle_line("DOWN 87")  # W down
            elif cmd == "stop":
                self.motion.handle_line("STOP")

        For now, just prints the command.
        """
        print(f"[Voice CMD] {cmd}")

        # Toggle emotion mode via voice if you want
        if cmd == "emotion_on":
            self.mode_emotion = True
            return
        if cmd == "emotion_off":
            self.mode_emotion = False
            # optional: stop when leaving emotion mode
            # self.motion.handle_line("STOP")
            return

        # TODO: replace with actual mappings from recognized text -> keycodes.
        #
        # Example:
        #
        # if cmd == "go forward":
        #     self.motion.handle_line("DOWN 87")   # W down
        # elif cmd == "stop":
        #     self.motion.handle_line("STOP")
        #
        # if cmd == "reverse":
        #     self.motion.handle_line("DOWN 83")   # S down
        # elif cmd == "reverse stop":
        #     self.motion.handle_line("UP 83")     # S up
        #
        # if cmd == "turn left":
        #     self.motion.handle_line("DOWN 65")   # A down
        # elif cmd == "left stop":
        #     self.motion.handle_line("UP 65")     # A up
        #
        # if cmd == "turn right":
        #     self.motion.handle_line("DOWN 68")   # D down
        # elif cmd == "right stop":
        #     self.motion.handle_line("UP 68")     # D up

    # ------------------------------------------------------------------
    # EMOTION → KEYCODE SKELETON
    # ------------------------------------------------------------------

    def emotion_to_action(self, label: str):
        """
        Skeleton: map emotion label -> sequences of keycode commands.

        Here you can later hardcode your "reactions" as scripts of keycode
        strings, e.g.:

            self.motion.handle_line("DOWN 87")
            time.sleep(0.1)
            self.motion.handle_line("UP 87")

        For now, everything is left as TODOs.
        """
        if label == "happy":
            # TODO: hardcode a sequence of keycode commands for "happy"
            # Example:
            # self.car.set_light_rgb(255, 215, 0)
            # self.motion.handle_line("DOWN 87")  # W
            # time.sleep(0.12)
            # self.motion.handle_line("UP 87")
            pass

        elif label == "angry":
            # TODO: sequence for "angry"
            # Example:
            # self.car.set_light_rgb(30, 144, 255)
            # self.motion.handle_line("DOWN 83")  # S
            # time.sleep(0.12)
            # self.motion.handle_line("UP 83")
            pass

        elif label == "neutral":
            # TODO: neutral behavior (maybe soft stop / white LED)
            # Example:
            # self.car.set_light_rgb(255, 255, 255)
            # self.motion.handle_line("STOP")
            pass

        elif label == "no_face":
            # TODO: when no face is detected (dim + stop?)
            # Example:
            # self.car.set_light_rgb(10, 10, 10)
            # self.motion.handle_line("STOP")
            pass

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

        # Metrics
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


if __name__ == "__main__":
    App().loop()
