from math import floor, copysign
from adafruit_servokit import ServoKit
import RPi.GPIO as GPIO
import RpiMotorLib
import time

# Stepper
STEP_PIN = 24
DIR_PIN = 23
ENABLE_PIN = 16
M1_PIN = 17
M2_PIN = 27
M3_PIN = 22
STEP_DEG = 360 / 200 / 16
CCW_SIGN = 1

# Servo
OE_PIN = 26
DROP_ANGLE = -90
CHANNEL = 0

# Hardcoded bin angles
BIN_ANGLES = {
    0: 0,      # black
    1: 120,    # blue
    2: 240,    # green
}


class Motors:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(ENABLE_PIN, GPIO.OUT)
        GPIO.setup(OE_PIN, GPIO.OUT)
        self.enable_servo(True)
        self.enable_stepper(False)

        self.servo = ServoKit(channels=16).servo[CHANNEL]
        self.stepper = RpiMotorLib.A4988Nema(DIR_PIN, STEP_PIN, (M1_PIN, M2_PIN, M3_PIN), "A4988")
        self.current_angle = 0

    def cleanup(self):
        self.enable_servo(False)
        self.enable_stepper(False)
        GPIO.cleanup()

    def enable_stepper(self, enable):
        GPIO.output(ENABLE_PIN, GPIO.LOW if enable else GPIO.HIGH)

    def enable_servo(self, enable):
        GPIO.output(OE_PIN, GPIO.LOW if enable else GPIO.HIGH)

    def _rotate(self, target_angle):
        """Rotate stepper from current angle to target angle (blocking)."""
        diff = target_angle - self.current_angle
        num_steps = abs(int(round(diff / STEP_DEG)))
        if num_steps == 0:
            return
        # True = clockwise, False = counter-clockwise (adjust with CCW_SIGN)
        direction = (CCW_SIGN * copysign(1, diff)) < 0
        self.stepper.motor_go(direction, "1/16", num_steps, 0.0005, False)
        self.current_angle = target_angle

    def sort(self, class_idx):
        """Full blocking sort: rotate to bin, drop, return to 0."""
        angle = BIN_ANGLES[class_idx]

        self.enable_stepper(True)

        self._rotate(angle)             # go to bin
        self.servo.angle = DROP_ANGLE   # drop item
        time.sleep(1.0)                 # wait for drop
        self.servo.angle = 0            # reset servo
        self._rotate(0)                 # return home

        self.enable_stepper(False)