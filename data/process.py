import os
import sys
import numpy as np
import cv2
import logging
import torch
from pathlib import Path
from ultralytics.models.sam import SAM3SemanticPredictor

# --- CONFIGURATION ---
RAW_DIR = Path("rawData")
PROCESSED_DIR = Path("processedData")
BG_IMAGE_PATH = Path("background.jpg")
SAM_MODEL_PATH = "sam3.pt"

SEG_SIZE = 1036
OUTPUT_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_MASK_PERCENT = 0.005

# --- MASTER PROMPTS ---
WASTE_PROMPTS = {
    "black": [
        "man-made shiny metallic foil snack bag, crinkled silver chip packet, reflective wrapper (not a rock, not a stone)",
        "black plastic garbage bag, crumpled synthetic plastic wrapper, disposable face mask (not a pebble, not pavement)"
    ],
    "green": [
        "organic food waste, banana peel, apple core, vegetable scraps, fruit peelings",
        "brown garden leaves, wilted flowers, grass clippings, organic yard waste",
        "compostable food scraps, eggshells, leftover food"
    ],
    "blue": [
        "matte brown cardboard piece, corrugated cardboard, shipping box fragment",
        "recyclable plastic bottle, empty aluminum soda can",
        "white paper, newspaper, paper cup, non-shiny paper"
    ]
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("Segmenter")

class SmartSegmenter:
    def __init__(self):
        log.info(f"Initializing SAM 3.0 on H200 ({DEVICE})...")
        overrides = {
            "model": SAM_MODEL_PATH,
            "device": DEVICE,
            "conf": 0.20,
            "imgsz": SEG_SIZE,
            "task": "segment",
            "mode": "predict",
            "half": True,
            "max_det": 1,
            "retina_masks": True 
        }
        try:
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
        except Exception as e:
            log.error(f"Failed to load SAM 3: {e}")
            sys.exit(1)

    def process(self, img_path, color_label):
        try:
            prompts = WASTE_PROMPTS.get(color_label.lower(), ["one single piece of litter"])
            results = None
            
            for p in prompts:
                results = self.predictor(source=img_path, text=[p])
                if results and results[0].masks and len(results[0].masks.data) > 0:
                    break

            if not results or not results[0].masks or len(results[0].masks.data) == 0:
                results = self.predictor(source=img_path, text=["one single piece of man-made trash (not a rock)"], conf=0.12)

            if not results or not results[0].masks or len(results[0].masks.data) == 0:
                return None

            mask = results[0].masks.data[0].cpu().numpy().astype(np.float32)
            if mask.shape[:2] != (SEG_SIZE, SEG_SIZE):
                mask = cv2.resize(mask, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_LINEAR)
                
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask
        except Exception as e:
            log.debug(f"Segmentation Error: {e}")
            return None

def main():
    if not BG_IMAGE_PATH.exists():
        log.error(f"Critical Error: {BG_IMAGE_PATH.name} not found in current directory.")
        return
        
    segmenter = SmartSegmenter()
    bg_raw = cv2.imread(str(BG_IMAGE_PATH))
    if bg_raw is None: return
    bg = cv2.resize(bg_raw, (SEG_SIZE, SEG_SIZE))

    tasks = []
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    
    for color_dir in sorted([d for d in RAW_DIR.iterdir() if d.is_dir()]):
        color_name = color_dir.name.lower()
        imgs = sorted([f for f in color_dir.iterdir() if f.suffix.lower() in valid_exts])
        for img_p in imgs:
            out_p = PROCESSED_DIR / color_name / img_p.name
            tasks.append((img_p, out_p, color_name))

    if not tasks:
        log.warning("No images found in rawData folders.")
        return

    # Counters
    stats = {"processed": 0, "too_small": 0, "failed": 0}
    total = len(tasks)
    
    log.info(f"Pipeline Started: Processing {total} images across all categories...")

    for i, (img_p, out_p, color) in enumerate(tasks, 1):
        mask = segmenter.process(img_p, color)
        
        if mask is not None:
            pixel_ratio = np.sum(mask > 0.5) / (SEG_SIZE * SEG_SIZE)
            if pixel_ratio < MIN_MASK_PERCENT:
                log.warning(f"[{i}/{total}] SKIPPED: Object too small ({pixel_ratio:.4f})")
                stats["too_small"] += 1
                continue

            raw_img = cv2.imread(str(img_p))
            if raw_img is None: continue
            img_work = cv2.resize(raw_img, (SEG_SIZE, SEG_SIZE))
            
            alpha = cv2.GaussianBlur(mask, (9, 9), 0)[:, :, None]
            comp = (img_work * alpha + bg * (1.0 - alpha)).astype(np.uint8)
            
            final = cv2.resize(comp, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
            
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), final, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            log.info(f"[{i}/{total}] Saved: {img_p.name} ({color})")
            stats["processed"] += 1
        else:
            log.warning(f"[{i}/{total}] FAILED: No mask found for {img_p.name}")
            stats["failed"] += 1

    print("\n" + "="*45)
    print(f"H200 SAM 3.0 COMPLETE PIPELINE SUMMARY")
    print("="*45)
    print(f"Successfully Created:  {stats['processed']}")
    print(f"Excluded (Too Small):   {stats['too_small']}")
    print(f"Failed (No Detection):  {stats['failed']}")
    print(f"Total Images Processed: {total}")
    print("="*45)

if __name__ == "__main__":
    main()