import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

# ==================== CONFIGURATION ====================
TRAIN_DIR = "../data/processedData/train"
VAL_DIR = "../data/processedData/val"
LOAD_PATH = "mobilenetv3_large_final.pth" # Custom weights to attempt loading
SAVE_PATH = "mobilenetv3_finetuned_best.pth"

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3               # Increased for better convergence
WEIGHT_DECAY = 1e-2     # Standard for AdamW
NUM_WORKERS = 8
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== REPRODUCIBILITY ====================
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==================== DATASET WRAPPER ====================
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super(SafeImageFolder, self).__getitem__(index)
        except Exception as e:
            import random
            return self.__getitem__(random.randint(0, len(self) - 1))

# ==================== TRANSFORMS ====================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), 
    transforms.RandomHorizontalFlip(),
    
    # 1. Higher Jitter: Simulates shadows and different light bulbs
    transforms.ColorJitter(
        brightness=0.4, 
        contrast=0.4, 
        saturation=0.4, 
        hue=0.1
    ),
    
    # 2. Grayscale: Forces model to learn texture (CRITICAL for cardboard)
    transforms.RandomGrayscale(p=0.2), 
    
    # 3. Sharpness: Cardboard is sharp/flat; plastic is crinkly/blurry
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== BALANCED SAMPLING ====================
def get_balanced_indices(dataset):
    targets = np.array(dataset.targets)
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(weight[targets]).double()
    
    g = torch.Generator()
    g.manual_seed(SEED)
    return torch.multinomial(samples_weight, len(dataset), replacement=True, generator=g).tolist()

# ==================== MAIN ====================
def main():
    # 1. Datasets
    full_train = SafeImageFolder(TRAIN_DIR, transform=train_transform)
    val_data = SafeImageFolder(VAL_DIR, transform=val_transform)
    num_classes = len(full_train.classes)

    print(f"Device: {DEVICE} | Classes: {full_train.classes}")

    train_indices = get_balanced_indices(full_train)
    train_loader = DataLoader(
        Subset(full_train, train_indices), 
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 2. Model: Start with IMAGENET1K_V2 Foundation
    print("Initializing MobileNetV3 with ImageNet-V2 weights...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    
    # Modify the classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    # OPTIONAL: Load previous best custom weights if they exist
    if os.path.exists(LOAD_PATH):
        try:
            state_dict = torch.load(LOAD_PATH, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f">>> Resuming from custom weights: {LOAD_PATH}")
        except:
            print(">>> Custom weights incompatible/corrupt. Proceeding with ImageNet weights.")

    model = model.to(DEVICE)

    # 3. Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Differential LR: Backbone 1e-4, Head 1e-3
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': LR / 10},
        {'params': model.classifier.parameters(), 'lr': LR}
    ], weight_decay=WEIGHT_DECAY)

    # StepLR: More stable than Plateau for recovering bad convergence
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_acc = 0.0

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = (correct / total) * 100
        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        curr_lr = optimizer.param_groups[1]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {val_acc:.2f}% | LR: {curr_lr:.7f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  [SAVE] New Best Accuracy: {val_acc:.2f}%")

    # 5. Final Evaluation
    print("\n" + "="*40)
    print("FINAL EVALUATION REPORT")
    print("="*40)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(DEVICE))
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=val_data.classes))
    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    main()