from __future__ import annotations
from dataclasses import dataclass

try:
    from picarx.picarx import Picarx
except Exception:
    Picarx = None


@dataclass
class CarConfig:
    max_speed: int = 100        # clamp |speed| to this
    steer_trim: int = 0         # degrees to add to steering servo
    max_steer_angle: int = 30   # for clamping, in degrees


class CarController:
    def __init__(self, config: CarConfig | None = None):
        self.config = config or CarConfig()

        self._sim = False
        self._px = None

        if Picarx is None:
            # No hardware library available -> SIM mode
            print("[CarController] Picarx import failed; running in SIM mode")
            self._sim = True
        else:
            try:
                self._px = Picarx()
                # Try to center steering and stop motion at startup
                try:
                    self._px.set_dir_servo_angle(0 + self.config.steer_trim)
                    self._px.forward(0)
                except Exception:
                    pass
                print("[CarController] Picarx initialized (HW mode)")
            except Exception as e:
                print("[CarController] Failed to create Picarx instance:", e)
                print("[CarController] Falling back to SIM mode")
                self._sim = True

        # Start in a neutral state
        self.drive(0)
        self.steer(0)


    def drive(self, speed: int):
        """
        speed > 0  → forward
        speed < 0  → backward
        speed == 0 → stop
        """
        # clamp speed to [-max_speed, max_speed]
        max_s = self.config.max_speed
        s = int(max(-max_s, min(max_s, speed)))

        if self._sim or self._px is None:
            print(f"[CarController] DRIVE {s} (SIM)")
            return

        try:
            if s > 0:
                self._px.forward(s)
            elif s < 0:
                self._px.backward(-s)
            else:
                self._px.forward(0)
        except Exception as e:
            print("[CarController] Error driving:", e)

    def steer(self, angle: int):
        """
        angle < 0 → turn left
        angle > 0 → turn right
        angle = 0 → center
        """
        max_a = self.config.max_steer_angle
        a = int(max(-max_a, min(max_a, angle)))  # clamp

        if self._sim or self._px is None:
            print(f"[CarController] STEER {a} (SIM)")
            return

        try:
            # Same pattern as your old code: servo + camera pan
            self._px.set_dir_servo_angle(a + self.config.steer_trim)
            self._px.set_cam_pan_angle(a)
        except Exception as e:
            print("[CarController] Error steering:", e)

    def shutdown(self):
        """
        Stop the car and put it in a safe neutral state.
        """
        print("[CarController] shutdown()")
        # Stop motion + center steering
        self.drive(0)
        self.steer(0)

        if not self._sim and self._px is not None:
            try:
                self._px.set_dir_servo_angle(0 + self.config.steer_trim)
            except Exception:
                pass
