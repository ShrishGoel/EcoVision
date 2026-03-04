import os
import shutil
import random
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path("processedData")
CLASSES = ["blue"]
TRAIN_SPLIT = 0.90
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def split_dataset():
    for cls in CLASSES:
        src_path = BASE_DIR / cls
        train_path = BASE_DIR / "train" / cls
        val_path = BASE_DIR / "val" / cls

        if not src_path.exists():
            print(f"Skipping {cls}: Folder not found in {BASE_DIR}")
            continue

        images = [f for f in src_path.iterdir() if f.suffix.lower() in EXTENSIONS]
        random.shuffle(images)

        split_idx = int(len(images) * TRAIN_SPLIT)
        train_files = images[:split_idx]
        val_files = images[split_idx:]

        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)

        print(f"Processing {cls}: {len(images)} total images...")

        for f in train_files:
            shutil.move(str(f), str(train_path / f.name))

        for f in val_files:
            shutil.move(str(f), str(val_path / f.name))

        try:
            src_path.rmdir()
        except OSError:
            print(f"Note: {src_path} was not empty after move, keeping it.")

    print("\nReorganization Complete!")
    print(f"Structure is now: {BASE_DIR}/train and {BASE_DIR}/val")

if __name__ == "__main__":
    split_dataset()