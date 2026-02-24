import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from motors import Motors

def run_sweep(): 
    m = Motors()
    try:
        print("Starting Sweep Test: -40 to 0 to 40 to 0")
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