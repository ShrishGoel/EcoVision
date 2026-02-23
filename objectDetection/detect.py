import cv2
import numpy as np
import onnxruntime as ort
import time
import threading
from motors import Motors

MODEL_PATH = 'model.onnx'
LABELS = ["black", "blue", "green"]
EDGE_THRESHOLD = 500
COOLDOWN_SECONDS = 3
TARGET_SIZE = 224
TEMPERATURE = 0.5

MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STD = (1.0 / (0.229 * 255), 1.0 / (0.224 * 255), 1.0 / (0.225 * 255))


def load_optimized_model(model_path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(model_path, options, providers=['CPUExecutionProvider'])
    return session, session.get_inputs()[0].name


class EcoVisionTurbo:
    def __init__(self):
        self.session, self.input_name = load_optimized_model(MODEL_PATH)
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.motors = Motors()

        self.latest_frame = None
        self.res_text, self.conf_text = "Ready", 0.0
        self.is_running = True
        self.last_trigger_time = 0

    def inference_thread(self):
        while self.is_running:
            if self.latest_frame is not None:
                h, w = self.latest_frame.shape[:2]
                min_dim = min(h, w)
                cx, cy = w // 2, h // 2
                roi = self.latest_frame[cy-min_dim//2:cy+min_dim//2,
                                        cx-min_dim//2:cx+min_dim//2]

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

                if cv2.countNonZero(edges) > EDGE_THRESHOLD:
                    if (time.time() - self.last_trigger_time) > COOLDOWN_SECONDS:

                        self.res_text, self.conf_text = "Waiting...", 0.0
                        time.sleep(3)
                        
                        self.res_text = "Analyzing..."
                        frames_to_process = []
                        for _ in range(5):
                            if self.latest_frame is not None:
                                curr_h, curr_w = self.latest_frame.shape[:2]
                                curr_min = min(curr_h, curr_w)
                                c_x, c_y = curr_w // 2, curr_h // 2
                                current_roi = self.latest_frame[c_y-curr_min//2:c_y+curr_min//2, 
                                                                c_x-curr_min//2:c_x+curr_min//2]
                                frames_to_process.append(current_roi)
                            
                            time.sleep(0.1) 

                        if frames_to_process:
                            self.process_inference(frames_to_process)
                            
                        self.last_trigger_time = time.time()
            time.sleep(0.01)

    def process_inference(self, frames):
        total_probs = np.zeros(len(LABELS))
        
        for roi in frames:
            blob = cv2.dnn.blobFromImage(roi, 1.0/255.0,
                                         (TARGET_SIZE, TARGET_SIZE), swapRB=True)
            for i in range(3):
                blob[0, i] = (blob[0, i] - (MEAN[i]/255.0)) * (STD[i]*255.0)

            logits = self.session.run(None,
                        {self.input_name: blob.astype(np.float32)})[0]
            exp_logits = np.exp(logits[0] / TEMPERATURE)
            probs = exp_logits / np.sum(exp_logits)
            total_probs += probs

        avg_probs = total_probs / len(frames)
        idx = int(np.argmax(avg_probs))
        
        self.res_text, self.conf_text = LABELS[idx], avg_probs[idx]

        if avg_probs[idx] > 0.7:   
            self.motors.sort(idx)        # <<< rotate + drop + return

    def run(self):
        threading.Thread(target=self.inference_thread, daemon=True).start()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.latest_frame = frame

            color = (0, 255, 0) if isinstance(self.conf_text, float) and self.conf_text > 0.7 else (0, 165, 255)
            
            if self.conf_text > 0.0:
                display_text = f"{self.res_text.upper()} ({self.conf_text:.1%})"
            else:
                display_text = f"{self.res_text.upper()}"
                
            cv2.putText(frame, display_text,
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow('EcoVision Turbo', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                break

        self.motors.cleanup()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    EcoVisionTurbo().run()