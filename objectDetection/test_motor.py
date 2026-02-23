import cv2
import numpy as np
import onnxruntime as ort
import time
import threading
from motors import Motors


class Test:
    def __init__(self):
        self.motors = Motors()

    def test(self):
        print("Starting motor test loop. Press 'q' in the display window to quit.")
        
        # Create a small window to capture key presses
        cv2.namedWindow("Motor Test")
        display = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(display, "Press 'q' to quit", (80, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        i = 0
        try:
            while True:
                print(f"Iter {i}: Sorting class {i % 3}")
                self.motors.sort(i % 3)
                i += 1
                
                # Wait 5 seconds, but check for 'q' frequently
                start_wait = time.time()
                while time.time() - start_wait < 5:
                    cv2.imshow("Motor Test", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quit signal received.")
                        return
        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:    
            self.motors.cleanup()
            cv2.destroyAllWindows()
            print("Cleanup complete.")

if __name__ == "__main__":
    Test().test()
