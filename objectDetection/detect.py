import cv2
import numpy as np
import onnxruntime as ort
from collections import Counter
import time

# ==================== CONFIGURATION ====================
MODEL_PATH = 'model_resnet.onnx'
LABELS = ["black", "blue", "green"]
EDGE_THRESHOLD = 500
NUM_FRAMES = 5
COOLDOWN_SECONDS = 7
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
DETECTION_SCALE = 0.5

# ==================== MODEL SETUP ====================
def load_model(model_path):
    """Load ONNX model optimized for ARM"""
    # Use CPU execution provider
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name
    print(f"Model input: {input_name}")
    print(f"Model loaded successfully")
    return session, input_name

# ==================== PREPROCESSING ====================
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

def preprocess_frame(frame):
    """Preprocess single frame for ONNX"""
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img

def preprocess_batch(frames):
    """Batch preprocessing with normalization"""
    batch = np.stack([preprocess_frame(f) for f in frames])
    batch = (batch - MEAN) / STD  # Normalization happens here
    return batch.astype(np.float32)

# ==================== DETECTION ====================
def detect_object_presence(frame, scale=0.5):
    """Edge detection for object presence"""
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    v = np.median(blurred)
    edges = cv2.Canny(blurred, int(0.67 * v), int(1.33 * v))
    
    edge_count = cv2.countNonZero(edges) / (scale * scale)
    return int(edge_count)

# ==================== INFERENCE ====================
def predict_batch(session, input_name, frames):
    """ONNX batch inference"""
    batch = preprocess_batch(frames)
    outputs = session.run(None, {input_name: batch})[0]
    predictions = np.argmax(outputs, axis=1)
    return predictions

def get_consensus_prediction(predictions, labels):
    """Majority voting"""
    counts = Counter(predictions.tolist())
    most_common_idx, count = counts.most_common(1)[0]
    confidence = count / len(predictions)
    return labels[most_common_idx], confidence

# ==================== MAIN LOOP ====================
def main():
    print("Loading ONNX model...")
    session, input_name = load_model(MODEL_PATH)
    
    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print(f"Monitoring... Edge threshold: {EDGE_THRESHOLD}")
    
    # State variables
    collection_mode = False
    captured_frames = []
    final_prediction = "Waiting..."
    confidence = 0.0
    last_trigger_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            edge_count = detect_object_presence(frame, DETECTION_SCALE)
            current_time = time.time()
            
            can_trigger = (current_time - last_trigger_time) > COOLDOWN_SECONDS
            should_trigger = edge_count > EDGE_THRESHOLD and can_trigger
            
            if should_trigger and not collection_mode:
                print(f"\n[TRIGGERED] Edges: {edge_count}")
                collection_mode = True
                captured_frames = []
            
            if collection_mode:
                captured_frames.append(frame.copy())
                
                if len(captured_frames) >= NUM_FRAMES:
                    start_time = time.time()
                    
                    predictions = predict_batch(session, input_name, captured_frames)
                    final_prediction, confidence = get_consensus_prediction(
                        predictions, LABELS
                    )
                    
                    inference_time = (time.time() - start_time) * 1000
                    print(f"Result: {final_prediction} ({confidence:.0%})")
                    print(f"Inference: {inference_time:.1f}ms")
                    
                    collection_mode = False
                    last_trigger_time = current_time
            
            # Display
            color = (0, 0, 255) if collection_mode else (0, 255, 0)
            cv2.putText(frame, f"Edges: {edge_count}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Bin: {final_prediction} ({confidence:.0%})", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('EcoVision', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()