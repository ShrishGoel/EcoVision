import torch
from torchvision.models import mobilenet_v3_small 

model = mobilenet_v3_small(weights=None)

# 2. MobileNetV3 Surgery
# The classifier is a Sequence of [Linear, Hardswish, Dropout, Linear]
# We need to grab the input features of the very last layer
in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, 3)

# 3. Load your weights
# Ensure the .pth file contains a state_dict for MobileNetV3
model.load_state_dict(torch.load('mobilenetv3_balanced.pth', map_location='cpu'))
model.eval()

# 4. Export logic
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model_mobilenetv3.onnx",
    input_names=['input'],
    output_names=['output'],
    verbose=True,
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("MobileNetV3 successfully exported to model_mobilenetv3.onnx")