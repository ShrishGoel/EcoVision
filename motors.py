from math import floor, copysign
import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
from adafruit_servokit import ServoKit
import time

# Stepper Pins
STEP_PIN = 24
DIR_PIN = 23
ENABLE_PIN = 16
M1_PIN = 17
M2_PIN = 27
M3_PIN = 22

# Servo Pins
OE_PIN = 26
ANGLE = 90
CHANNEL = 0

# MATH: 1.8 for Full Step, 0.1125 for 1/16 step
STEP_DEG = 1.8 
CCW_SIGN = 1 # Change to -1 if motor spins opposite of intended

# Using signed angles for -40 to 40 range
BIN_ANGLES = {
    0: 40,      # Black
    1: 0,       # Blue
    2: -40,     # Green
}

class Motors:
    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(ENABLE_PIN, GPIO.OUT)
        GPIO.setup(OE_PIN, GPIO.OUT)
        self.enable_stepper(True)
        self.enable_servo(True)

        self.servo = ServoKit(channels = 16).servo[CHANNEL]
        self.stepper = RpiMotorLib.A4988Nema(DIR_PIN, STEP_PIN, (M1_PIN, M2_PIN, M3_PIN), "A4988")
        self.current_angle = 0

    def cleanup(self):
        # 1. Force the driver OFF
        self.enable_stepper(False)
        self.enable_servo(False)
        
        # 2. Force STEP and DIR to a solid LOW state
        GPIO.output(STEP_PIN, GPIO.LOW)
        GPIO.output(DIR_PIN, GPIO.LOW)
        
        # 3. Small delay to let pins settle
        time.sleep(0.1)
        
        # 4. Cleanup
        GPIO.cleanup()
        print("Motors safely disabled.")

    def enable_stepper(self, enable):
        GPIO.output(ENABLE_PIN, GPIO.LOW if enable else GPIO.HIGH)

    def enable_servo(enable):
        GPIO.output(OE_PIN, GPIO.LOW if enable else GPIO.HIGH)

    def _rotate(self, target_angle):
        """Rotate to an absolute angle position."""
        diff = target_angle - self.current_angle
        num_steps = abs(int(round(diff / STEP_DEG)))
        
        if num_steps == 0:
            return
            
        # Direction logic based on the sign of the difference
        direction = (CCW_SIGN * copysign(1, diff)) < 0
        
        # Using 0.005 delay for stability. If vibrating, increase to 0.01
        self.stepper.motor_go(direction, "Full", num_steps, 0.005, False)
        
        self.current_angle = target_angle
        print(f"Current Position: {self.current_angle}°")

    def sort(self, class_idx):
        """Standard sort: Go to bin and return home."""
        if class_idx not in BIN_ANGLES: return
        target = BIN_ANGLES[class_idx]

        self._rotate(target)
        self.servo.angle = ANGLE             
        time.sleep(0.5)
        self.servo.angle = 0             
        self._rotate(0) 