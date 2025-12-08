# test_motion.py

import time
from car_controller import CarController
from motion_controller import MotionController

car = CarController()
motion = MotionController(car, speed=18, turn_angle=20)

print("Forward tap…")
motion.handle_line("DOWN 87")  # W down
time.sleep(0.2)
motion.handle_line("UP 87")    # W up

time.sleep(1.0)

print("Right tap…")
motion.handle_line("DOWN 68")  # D down
time.sleep(0.2)
motion.handle_line("UP 68")    # D up

time.sleep(1.0)

print("Stopping + shutdown")
motion.stop_all()
motion.shutdown()
car.shutdown()
