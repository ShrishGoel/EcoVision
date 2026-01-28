import os
from pathlib import Path
import random
from PIL import Image
from torchvision import transforms


RAW_DIR = "rawData"
PROCESSED_DIR = "processedData"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
IMAGE_SIZE = 224
random.seed(42)


save_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])


def save_transformed_image(img, transform, save_path):
    """Apply transform to PIL image and save as a PIL image."""
    processed_img = transform(img)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    processed_img.save(save_path)

classes = [d.name for d in os.scandir(RAW_DIR) if d.is_dir()]

for cls in classes:
    cls_path = Path(RAW_DIR) / cls
    images = [f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(images)

    n = len(images)
    n_train = int(TRAIN_SPLIT * n)
    n_val = int(VAL_SPLIT * n)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    for split, split_images in splits.items():
        transform = save_transform 
        for img_name in split_images:
            img_path = cls_path / img_name
            try:
                img = Image.open(img_path).convert("RGB")
                out_path = Path(PROCESSED_DIR) / split / cls / img_name
                save_transformed_image(img, transform, out_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print("Processing complete! Images saved to:", PROCESSED_DIR)