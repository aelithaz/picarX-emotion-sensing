# motion_controller.py

import re
import threading
import time
from typing import Dict

from car_controller import CarController

# Numeric key codes only: W = 87, S = 83, A = 65, D = 68
KEY_TO_ACTION: Dict[str, str] = {
    "87": "forward",   # W
    "83": "backward",  # S
    "65": "left",      # A
    "68": "right",     # D
}

TAP_MAX = 0.18  # seconds: <= TAP_MAX is treated as a "tap burst"

DOWN_RE = re.compile(r"^(?:DOWN|KEYDOWN)\s+(\d+)$", re.IGNORECASE)
UP_RE   = re.compile(r"^(?:UP|KEYUP)\s+(\d+)$", re.IGNORECASE)


class MotionController:
    """
    Unified motion controller.

    Drive in two ways:

    1) Keycode-style strings (old Bluetooth style):
         handle_line("DOWN 87")
         handle_line("UP 87")
         handle_line("STOP")

    2) Direct logical actions:
         set_action("forward", True)   # hold forward
         set_action("forward", False)  # release
         tap("left", 0.2)              # quick left tap
    """

    def __init__(self, car: CarController, speed: int = 20, turn_angle: int = 30):
        self.car = car
        self.speed = speed
        self.turn_angle = turn_angle

        # state (logical actions)
        self._lock = threading.Lock()
        self._held: Dict[str, bool] = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
        }
        self._ts: Dict[str, float] = {
            "forward": 0.0,
            "backward": 0.0,
            "left": 0.0,
            "right": 0.0,
        }

        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()


    def handle_line(self, msg: str):
        """
        Process a single keycode-style message, e.g.:

            "DOWN 87"
            "UP 87"
            "STOP"  or "0"

        Used for hardcoding reactions as sequences of keycode strings.
        """
        msg = msg.strip()
        if not msg:
            return

        # DOWN / KEYDOWN
        m = DOWN_RE.match(msg)
        if m:
            code = m.group(1)
            logical = KEY_TO_ACTION.get(code)
            if logical:
                self._set_key(logical, True)
            return

        # UP / KEYUP
        m = UP_RE.match(msg)
        if m:
            code = m.group(1)
            logical = KEY_TO_ACTION.get(code)
            if logical:
                self._on_key_up(logical)
            return

        # STOP or "0"
        if msg.upper() == "STOP" or msg == "0":
            self.stop_all()
            return

        # unknown message → ignore


    def set_action(self, action: str, is_down: bool):
        """
        Hold/release a logical action directly:
          action ∈ {"forward","backward","left","right"}.
        """
        if action not in self._held:
            return
        self._set_key(action, is_down)

    def tap(self, action: str, duration: float = 0.25):
        """
        Quick "tap" burst in a given direction.
        Implemented using the same internal state as keycodes.
        """
        if action not in self._held:
            return

        def _worker():
            self._set_key(action, True)
            time.sleep(duration)
            self._on_key_up(action)

        threading.Thread(target=_worker, daemon=True).start()

    def stop_all(self):
        # Release all actions and stop the car.
        with self._lock:
            for k in self._held:
                self._held[k] = False
        self.car.drive(0)
        self.car.steer(0)

    def shutdown(self):
        # Stop control loop and neutralize the car.
        self._stop_evt.set()
        self._thread.join(timeout=1.0)
        self.stop_all()


    def _set_key(self, logical: str, is_down: bool):
        now = time.monotonic()
        with self._lock:
            self._held[logical] = is_down
            if is_down:
                self._ts[logical] = now

    def _on_key_up(self, logical: str):
        now = time.monotonic()
        with self._lock:
            was_down = self._held[logical]
            pressed_at = self._ts[logical]
            self._held[logical] = False

        dt = now - pressed_at

        # quick tap → burst behavior
        if was_down and dt <= TAP_MAX:
            if logical == "forward":
                self._safe_forward()
            elif logical == "backward":
                self._safe_backward()
            elif logical == "left":
                self._safe_turn(-self.turn_angle)
            elif logical == "right":
                self._safe_turn(+self.turn_angle)
        else:
            # on steering release after hold, center wheel
            if logical in ("left", "right"):
                self.car.steer(0)
            # drive axis is handled by control loop on next tick

    # "safe" bursts used for taps
    def _safe_forward(self, duration: float = 0.25):
        self.car.drive(self.speed)
        time.sleep(duration)
        self.car.drive(0)

    def _safe_backward(self, duration: float = 0.25):
        self.car.drive(-self.speed)
        time.sleep(duration)
        self.car.drive(0)

    def _safe_turn(self, angle: int, duration: float = 0.25):
        self.car.steer(angle)
        time.sleep(duration)
        self.car.steer(0)


    def _control_loop(self):
        last_speed = None
        last_angle = None

        while not self._stop_evt.is_set():
            with self._lock:
                fwd = self._held["forward"]
                back = self._held["backward"]
                left = self._held["left"]
                right = self._held["right"]
                ts = self._ts.copy()

            # resolve drive
            if fwd and not back:
                speed = +self.speed
            elif back and not fwd:
                speed = -self.speed
            elif fwd and back:
                speed = +self.speed if ts["forward"] >= ts["backward"] else -self.speed
            else:
                speed = 0

            # resolve steer
            if left and not right:
                angle = -self.turn_angle
            elif right and not left:
                angle = +self.turn_angle
            elif left and right:
                angle = -self.turn_angle if ts["left"] >= ts["right"] else +self.turn_angle
            else:
                angle = 0

            if speed != last_speed:
                self.car.drive(speed)
                last_speed = speed
            if angle != last_angle:
                self.car.steer(angle)
                last_angle = angle

            time.sleep(0.03)  # ~33 Hz update
