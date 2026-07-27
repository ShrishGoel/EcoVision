"""Export trained MobileNetV3-Large checkpoint to ONNX format."""

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 3
LOAD_PATH = "mobilenetv3_finetuned_best.pth"
EXPORT_PATH = "model_mobilenetv3.onnx"

# Reconstruct model architecture and load weights
model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

state_dict = torch.load(LOAD_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

print(f"Exporting {LOAD_PATH} to ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    EXPORT_PATH,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"Success! Exported to {EXPORT_PATH}")