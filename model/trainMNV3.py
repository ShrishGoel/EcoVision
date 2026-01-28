import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
import os

# Paths and hyperparameters
TRAIN_DIR = "data/processedData/train"
VAL_DIR = "data/processedData/val"
BATCH_SIZE_PER_GPU = 64
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-2 
NUM_WORKERS = 8

# Modernized Augmentation
train_transform = transforms.Compose([
    transforms.TrivialAugmentWide(), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_worker(rank, world_size):
    # --- Initialize distributed training ---
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    DEVICE = f"cuda:{rank}"

    # --- Datasets and loaders ---
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    num_classes = len(train_dataset.classes)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_PER_GPU, sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_PER_GPU, sampler=val_sampler,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # --- Model ---
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    # Freeze feature extractor initially
    for param in model.features.parameters():
        param.requires_grad = False

    model = model.to(DEVICE)
    
    # find_unused_parameters=True prevents crashes when unfreezing layers mid-run
    model = nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[rank], 
        find_unused_parameters=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # --- Training loop ---
    for epoch in range(EPOCHS):
        
        if epoch == 5:
            for param in model.module.features.parameters():
                param.requires_grad = True
            
            # Reduce LR for the backbone fine-tuning phase
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR * 0.1
            
            if rank == 0:
                print(f"\n>>> Epoch {epoch+1}: Unfreezing backbone and reducing LR to {LR*0.1}")

        train_sampler.set_epoch(epoch)
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
        
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = (correct / total) * 100
        scheduler.step()

        if rank == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if rank == 0:
        torch.save(model.module.state_dict(), "mobilenetv3_optimized.pth")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 8
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)