import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import datasets, transforms, models
import os
import numpy as np

# ==================== CONFIGURATION ====================
TRAIN_DIR = "data/processedData/train"
VAL_DIR = "data/processedData/val"
BATCH_SIZE_PER_GPU = 64
EPOCHS = 40 
LR = 1e-3
WEIGHT_DECAY = 1e-2 
NUM_WORKERS = 8

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), 
    transforms.RandomHorizontalFlip(),
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

# ==================== BALANCING LOGIC ====================
def get_balanced_indices(dataset, rank, world_size):
    targets = np.array(dataset.targets)
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(weight[targets]).double()

    num_samples = len(dataset)
    balanced_indices = torch.multinomial(samples_weight, num_samples, replacement=True).tolist()

    shard_size = num_samples // world_size
    start = rank * shard_size
    end = start + shard_size
    return balanced_indices[start:end]

# ==================== WORKER FUNCTION ====================
def train_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    DEVICE = f"cuda:{rank}"

    full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    num_classes = len(full_train_dataset.classes)

    train_indices = get_balanced_indices(full_train_dataset, rank, world_size)
    train_subset = Subset(full_train_dataset, train_indices)
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=BATCH_SIZE_PER_GPU, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_PER_GPU, sampler=val_sampler, num_workers=NUM_WORKERS)

    model = models.mobilenet_v3_large(weights='IMAGENET1K_V2')
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    for param in model.features.parameters():
        param.requires_grad = False

    model = model.to(DEVICE)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    targets = np.array(full_train_dataset.targets)
    class_counts = torch.tensor([len(np.where(targets == t)[0]) for t in np.unique(targets)], dtype=torch.float)
    cw = 1. / class_counts
    cw = (cw / cw.sum()) * num_classes 
    criterion = nn.CrossEntropyLoss(weight=cw.to(DEVICE), label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        if epoch == 10:
            for param in model.module.features.parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.AdamW([
                {'params': model.module.features.parameters(), 'lr': LR * 0.1},
                {'params': model.module.classifier.parameters(), 'lr': LR}
            ], weight_decay=WEIGHT_DECAY)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(EPOCHS - 10))
            
            if rank == 0:
                print(f"\n>>> Epoch {epoch+1}: Backbone unfrozen with Differential LR.")

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
        
        dist.all_reduce(torch.tensor(correct).to(DEVICE), op=dist.ReduceOp.SUM)
        dist.all_reduce(torch.tensor(total).to(DEVICE), op=dist.ReduceOp.SUM)
        val_acc = (correct / total) * 100
        scheduler.step()

        if rank == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

    if rank == 0:
        torch.save(model.module.state_dict(), "mobilenetv3_large_final.pth")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)