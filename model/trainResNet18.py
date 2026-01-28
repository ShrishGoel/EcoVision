import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
import os

TRAIN_DIR = "data/processedData/train"
VAL_DIR = "data/processedData/val"
BATCH_SIZE_PER_GPU = 64
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-2 
NUM_WORKERS = 8

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    DEVICE = f"cuda:{rank}"

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    num_classes = len(train_dataset.classes)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_PER_GPU, sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_PER_GPU, sampler=val_sampler,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Replace the 'fc' (fully connected) layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze everything initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze ONLY the new fc head
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)
    # find_unused_parameters=True is critical when unfreezing layers mid-run
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # --- Training loop ---
    for epoch in range(EPOCHS):
        # --- Unfreeze Logic at Epoch 5 ---
        if epoch == 15:
            # Unfreeze every parameter in the model (the backbone)
            for param in model.module.parameters():
                param.requires_grad = True
            
            # Reduce LR for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4 
            
            if rank == 0:
                print(">>> ResNet Backbone unfrozen. Fine-tuning mode active.")

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
        torch.save(model.module.state_dict(), "resnet18_optimized.pth")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)