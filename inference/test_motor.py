"""Sweep test for stepper motor range of motion."""

import time
from motors import Motors


def run_sweep():
    """Continuously sweep the stepper between -40, 0, and +40 degrees."""
    m = Motors()
    try:
        print("Starting sweep test: -40 -> 0 -> +40 -> 0")
        m.enable_stepper(True)
        time.sleep(0.5)

        while True:
            print("\nMoving to -40...")
            m._rotate(-40)
            time.sleep(1)

            print("Moving to 0...")
            m._rotate(0)
            time.sleep(1)

            print("Moving to +40...")
            m._rotate(40)
            time.sleep(1)

            print("Moving to 0...")
            m._rotate(0)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        m.cleanup()
        print("Done.")


if __name__ == "__main__":
    run_sweep()