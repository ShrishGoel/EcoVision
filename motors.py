import math
import time
import sys
import signal
import RPi.GPIO as GPIO
from adafruit_servokit import ServoKit

STEP_PIN = 24
DIR_PIN = 23
ENABLE_PIN = 16
MODE_PINS = (17, 27, 22)
OE_PIN = 26

MICROSTEP_MODE = "1/4"
STEP_DEG = 1.8 / 4

BIN_ANGLES = {
    0: 35,
    1: 0,
    2: -35,
}

class Motors:
    def __init__(self):
        try:
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup([STEP_PIN, DIR_PIN, ENABLE_PIN, OE_PIN], GPIO.OUT)
            GPIO.setup(MODE_PINS, GPIO.OUT)

            GPIO.output(MODE_PINS, (GPIO.LOW, GPIO.HIGH, GPIO.LOW))

            signal.signal(signal.SIGINT, self._signal_handler)

            self.enable_stepper(True)
            self.enable_servo(True)

            self.servo = ServoKit(channels=16).servo[0]
            self.current_angle = 0
            print("Hardware initialized. Trapezoidal Profiling Active.")
        except Exception as e:
            print(f"Initialization Error: {e}")
            self.cleanup()

    def _signal_handler(self, sig, frame):
        print("\n[STOP] Emergency Interrupt!")
        self.cleanup()
        sys.exit(0)

    def enable_stepper(self, enable):
        GPIO.output(ENABLE_PIN, GPIO.LOW if enable else GPIO.HIGH)

    def enable_servo(self, enable):
        GPIO.output(OE_PIN, GPIO.LOW if enable else GPIO.HIGH)

    def _rotate(self, target_angle):
        diff = target_angle - self.current_angle
        num_steps = abs(int(round(diff / STEP_DEG)))
        if num_steps == 0:
            return

        GPIO.output(DIR_PIN, GPIO.HIGH if diff > 0 else GPIO.LOW)

        cruise_delay = 0.008
        start_delay = 0.15
        ramp_time = 2

        start_time = time.time()

        for i in range(num_steps):
            elapsed = time.time() - start_time
            remaining_steps = num_steps - i
            steps_to_brake = ramp_time / (cruise_delay * 2)

            if elapsed < ramp_time:
                factor = elapsed / ramp_time
                delay = start_delay - (factor * (start_delay - cruise_delay))
            elif remaining_steps < steps_to_brake:
                factor = 1 - (remaining_steps / steps_to_brake)
                delay = cruise_delay + (factor * (start_delay - cruise_delay))
            else:
                delay = cruise_delay

            GPIO.output(STEP_PIN, GPIO.HIGH)
            time.sleep(delay)
            GPIO.output(STEP_PIN, GPIO.LOW)
            time.sleep(delay)

        self.current_angle = target_angle

    def sort(self, class_idx):
        if class_idx not in BIN_ANGLES:
            return
        target = BIN_ANGLES[class_idx]

        self._rotate(target)
        time.sleep(0.5)

        print("Dumping...")
        self.servo.angle = 10
        time.sleep(3)

        self.servo.angle = 160
        time.sleep(1)

        print("Returning Home...")
        self._rotate(0)
        print("Cycle Complete.")

    def cleanup(self):
        print("Cleaning up GPIO...")
        try:
            self.enable_stepper(False)
            GPIO.cleanup()
        except:
            pass

if __name__ == "__main__":
    m = Motors()
    try:
        print("Starting Loop Test. Press Ctrl+C to stop.")
        while True:
            m.sort(0)
            time.sleep(2)
            m.sort(2)
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        m.cleanup()
