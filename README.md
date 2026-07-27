# EcoVision

AI-powered waste sorting system that classifies objects into three categories — **black** (landfill), **blue** (recyclable), and **green** (compostable) — and autonomously directs them into the correct bin using a stepper-servo mechanism.

## Architecture

```
Camera → Edge Detection → MobileNetV3 (ONNX) → Motor Controller → Sorting Bins
```

1. **Object detection** — Canny edge density on the center crop triggers classification when a new object appears.
2. **Classification** — A fine-tuned MobileNetV3-Large runs inference via ONNX Runtime, averaging predictions across 5 frames for robustness.
3. **Sorting** — A stepper motor rotates a chute to the target bin; a servo dumps the object and returns home.

## Project Structure

```
EcoVision/
├── data/
│   ├── process.py        # SAM 3.0 segmentation + background compositing
│   ├── split.py          # Train/val split utility
│   └── background.jpg    # Clean background for compositing
├── training/
│   ├── train.py          # MobileNetV3 distributed training
│   ├── export.py         # PyTorch → ONNX export
│   └── results.txt       # Classification metrics
├── inference/
│   ├── detect.py         # Real-time detection + sorting pipeline
│   ├── motors.py         # Stepper + servo hardware controller
│   └── test_motor.py     # Motor sweep test
├── .gitignore
└── README.md
```

## Data Pipeline

1. **Capture** raw images of waste objects per category into `data/rawData/{black,blue,green}/`.
2. **Segment** objects using SAM 3.0 with category-specific text prompts:
   ```bash
   cd data && python process.py
   ```
3. **Split** into train/val sets (90/10):
   ```bash
   python split.py
   ```

## Training

MobileNetV3-Large was selected for deployment due to its balance of accuracy and inference speed on Raspberry Pi.

```bash
cd training && python train.py
```

Key training techniques:
- Balanced sampling via inverse-frequency oversampling
- Differential learning rates (backbone 10× slower than head)
- Progressive unfreezing of backbone layers
- Label smoothing (0.1) and aggressive augmentation

### Preliminary Results

Results from an early training run on ~1,000 validation images (see `training/results.txt`):

```
              precision    recall  f1-score   support
   black bin       0.83      0.88      0.86       305
    blue bin       0.95      0.91      0.93       611
   green bin       0.92      0.99      0.95       127

    accuracy                           0.91      1043
```

> **Note:** These metrics are from prototype development, not a rigorous evaluation. The project focus was end-to-end system integration — from data pipeline through real-time inference to physical sorting hardware.

## Deployment

### Export to ONNX

```bash
cd training && python export.py
```

### Run on Raspberry Pi

```bash
cd inference && python detect.py
```

Press `q` to quit. The system:
- Monitors the camera for new objects via edge detection
- Waits for the object to settle, then captures 5 frames
- Averages softmax predictions (temperature-scaled) across frames
- Triggers motor sorting when confidence exceeds 70%

## Hardware

- **Stepper motor** — NEMA 17 with A4988 driver (1/4 microstepping)
- **Servo** — PCA9685-driven, controls the dump mechanism
- **Camera** — USB webcam (index 1)
- **Controller** — Raspberry Pi (BCM GPIO)

### GPIO Pin Map

| Function       | GPIO Pin |
|---------------|----------|
| Step          | 24       |
| Direction     | 23       |
| Enable        | 16       |
| Microstep M0  | 17       |
| Microstep M1  | 27       |
| Microstep M2  | 22       |
| Servo OE      | 26       |

## Dependencies

```
torch
torchvision
opencv-python
onnxruntime
numpy
scikit-learn
ultralytics
RPi.GPIO
adafruit-circuitpython-servokit
```

## License

This project is for educational and research purposes.
