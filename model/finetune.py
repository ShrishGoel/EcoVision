import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import os
import numpy as np
from PIL import Image

# ==================== CONFIGURATION ====================
TRAIN_DIR = "data/processedData/train"
VAL_DIR = "data/processedData/val"
LOAD_PATH = "mobilenetv3_large_final.pth" 
BATCH_SIZE_PER_GPU = 64
EPOCHS = 30
LR = 5e-5               
WEIGHT_DECAY = 5e-2     
NUM_WORKERS = 8

# Dataset Safety Wrapper
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super(SafeImageFolder, self).__getitem__(index)
        except Exception as e:
            import random
            return self.__getitem__(random.randint(0, len(self)-1))

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_balanced_indices(dataset, rank, world_size):
    targets = np.array(dataset.targets)
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(weight[targets]).double()
    num_samples = len(dataset)
    balanced_indices = torch.multinomial(samples_weight, num_samples, replacement=True).tolist()
    shard_size = num_samples // world_size
    start = rank * shard_size
    return balanced_indices[start : start + shard_size]

# ==================== WORKER FUNCTION ====================
def train_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357' 
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    DEVICE = f"cuda:{rank}"

    full_train_dataset = SafeImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = SafeImageFolder(VAL_DIR, transform=val_transform)
    num_classes = len(full_train_dataset.classes)

    train_indices = get_balanced_indices(full_train_dataset, rank, world_size)
    train_subset = Subset(full_train_dataset, train_indices)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE_PER_GPU, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_PER_GPU, sampler=val_sampler, num_workers=NUM_WORKERS)

    # Model Architecture
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    # Load Previous Weights
    if os.path.exists(LOAD_PATH):
        state_dict = torch.load(LOAD_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
        if rank == 0: print(f">>> Successfully loaded weights from {LOAD_PATH}")
    
    model = model.to(DEVICE)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    optimizer = torch.optim.AdamW([
        {'params': model.module.features.parameters(), 'lr': LR * 0.1},
        {'params': model.module.classifier.parameters(), 'lr': LR}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_acc = 0.0

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
        
        dist.all_reduce(torch.tensor(correct).to(DEVICE), op=dist.ReduceOp.SUM)
        dist.all_reduce(torch.tensor(total).to(DEVICE), op=dist.ReduceOp.SUM)
        val_acc = (correct / total) * 100
        scheduler.step(val_acc)

        if rank == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | LR: {optimizer.param_groups[1]['lr']:.7f} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.module.state_dict(), "mobilenetv3_finetuned_best.pth")

    # ==================== FINAL EVALUATION (RANK 0 ONLY) ====================
    if rank == 0:
        print("\n" + "="*40)
        print("RUNNING FINAL CLASSIFICATION REPORT")
        print("="*40)
        
        model.eval()
        all_preds = []
        all_labels = []
        
        eval_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_PER_GPU, shuffle=False, num_workers=NUM_WORKERS)
        
        with torch.no_grad():
            for images, labels in eval_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        print(classification_report(all_labels, all_preds, target_names=val_dataset.classes))
        
        print("\nCONFUSION MATRIX")
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Starting Fine-tuning on {world_size} GPUs...")
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)