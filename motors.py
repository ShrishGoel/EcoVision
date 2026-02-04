# pip install adafruit-circuitpython-servokit

from math import floor, copysign, sin, pi
from adafruit_servokit import ServoKit
import GPIOEmulator as GPIO
import RpiMotorLib
import time

# Stepper
STEP_PIN = 17
DIR_PIN = 22
ENABLE_PIN = 27
M1_PIN = 23
M2_PIN = 24
M3_PIN = 25
STEP_DEG = 360/200/16
CCW_SIGN = 1

# Servo
OE_PIN = 4
DROP_ANGLE = -90
CHANNEL = 0

class Motors():
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(ENABLE_PIN, GPIO.OUT)
        GPIO.setup(OE_PIN, GPIO.OUT)
        self.enable_servo(True)
        self.enable_stepper(False)

        self.servo = ServoKit(channels = 16).servo[CHANNEL]
        self.stepper = RpiMotorLib.A4988Nema(DIR_PIN, STEP_PIN, (M1_PIN, M2_PIN, M3_PIN), "A4988")
        self.stepper_angle = 0
        self.servo_angle = 0

    def cleanup(self):
        self.enable_servo(False)
        self.enable_stepper(False)
        GPIO.cleanup()

    def enable_stepper(enable):
        if enable:
              GPIO.output(ENABLE_PIN, GPIO.LOW)
        else: GPIO.output(ENABLE_PIN, GPIO.HIGH)

    def enable_servo(enable):
        if enable:
              GPIO.output(OE_PIN, GPIO.LOW)
        else: GPIO.output(OE_PIN, GPIO.HIGH)

    def rotate_to(self, angle, move_time, hold_time):
        self.command = {
            "angle": angle,
            "move_time": move_time,
            "hold_time": hold_time,
            "start": time.time()
        }
        self.enable_stepper(True)
    
    def periodic(self):
        if self.command:
            t = time.time() - self.command["start"]
            final_angle = self.command["angle"]
            move = self.command["move_time"]
            hold = self.command["hold_time"]

            alpha = 1
            if t < move: alpha = t / move
            if t > move + hold: alpha = (t - hold) / move - 1
            alpha = max(0, min(1, alpha))
            alpha = sin(alpha * pi / 2) ** 2

            num_steps = floor((alpha * final_angle - self.stepper_angle) / STEP_DEG)
            self.stepper_angle += num_steps * STEP_DEG
            if num_steps > 0:
                dir = CCW_SIGN * copysign(num_steps)
                dir = True if dir == 1 else False
                self.stepper.motor_go(dir, "1/16", num_steps)
            
            servo_angle = 0
            if alpha == 1: servo_angle = DROP_ANGLE
            if servo_angle != self.servo_angle:
                self.servo_angle = servo_angle
                self.servo.angle = servo_angle

            if t > 2*move + hold:
                del self.command
                self.enable_stepper(False)