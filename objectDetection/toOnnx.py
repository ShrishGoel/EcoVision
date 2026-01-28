import torch
from torchvision.models import resnet18  # 1. New Import

# 2. Build the ResNet18 skeleton
model = resnet18(weights=None)

# 3. ResNet surgery is simpler! 
# It uses 'model.fc' instead of 'model.classifier'
model.fc = torch.nn.Linear(model.fc.in_features, 3)

# 4. Load your weights
# Make sure this file name matches your actual .pth file
model.load_state_dict(torch.load('resnet18_optimized.pth', map_location='cpu'))
model.eval()

# 5. Export logic
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model_resnet.onnx",
    input_names=['input'],
    output_names=['output'],
    verbose=True,
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("ResNet18 successfully exported to model_resnet.onnx")