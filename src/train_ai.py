import os
import torch
import torch.nn.functional as F
import numpy as np
import wandb
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from tqdm import tqdm
from training_models import SimCLR

# --- HYPERPARAMETERS (From PDF) ---
EPOCHS = 200
BATCH_SIZE = 512  # Try 512 if your GPU allows, otherwise 256 or 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHKPT_PATH = "./models/best_simclr_model_ablation.pth"

CONFIG = {
    "learning_rate": 3e-4,
    "weight_decay": 1e-6,
    "temperature": 0.5,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "knn_k": 20
}

# --- DATA AUGMENTATION PIPELINE ---
class SimCLRTransform:
    def __init__(self, transform_pipeline):
        self.transform_pipeline = transform_pipeline
    def __call__(self, x):
        return self.transform_pipeline(x), self.transform_pipeline(x)

# CIFAR-10 standard mean and std
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

# Training Transforms (Two Twins)
s = 0.5
color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
train_transforms = T.Compose([
    T.RandomResizedCrop(size=32),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([color_jitter], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(cifar10_mean, cifar10_std)
])
simclr_transform = SimCLRTransform(train_transforms)

# kNN Testing Transforms (No random crops/colors, just standard normalization)
test_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(cifar10_mean, cifar10_std)
])

# --- KNN MONITOR FUNCTION ---
@torch.no_grad()
def knn_monitor(model, memory_loader, test_loader, device, k=20):
    """
    Evaluates the representation quality using a k-Nearest Neighbor classifier.
    Extracts the 'h' vectors from the train set (memory) and test set.
    """
    model.eval()
    
    # 1. Extract Memory Bank (Train Set)
    memory_features = []
    memory_labels =[]
    for x, y in memory_loader:
        x = x.to(device)
        h, _ = model(x)  # We want 'h' (encoder output), NOT 'z'
        memory_features.append(h)
        memory_labels.append(y)
    
    memory_features = torch.cat(memory_features, dim=0).t() # Shape: [512, N_train]
    memory_features = F.normalize(memory_features, dim=0)   # Normalize for cosine similarity
    memory_labels = torch.cat(memory_labels, dim=0).to(device)

    # 2. Extract Test Queries and Compute kNN
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        h, _ = model(x)
        features = F.normalize(h, dim=1) # Shape:[batch_size, 512]
        
        # Cosine similarity is just a matrix multiplication because vectors are normalized
        similarity = torch.mm(features, memory_features) # Shape: [batch_size, N_train]
        
        # Get indices of the top 'k' most similar training images
        _, topk_indices = similarity.topk(k, dim=1)
        
        # Gather the labels of those top 'k' images
        topk_labels = memory_labels[topk_indices]
        
        # Majority vote
        predictions, _ = torch.mode(topk_labels, dim=1)
        
        correct += (predictions == y).sum().item()
        total += y.size(0)
        
    return (correct / total) * 100.0

# --- TRAINING LOOP ---
def train(config_override):
    wandb.init(project="NYCU-AI-Capstone-Project2", config=config_override, name=f"SimCLR-temp{config_override['temperature']}-batch{config_override['batch_size']}")
    os.makedirs(os.path.dirname(CHKPT_PATH), exist_ok=True)
    print(f"Device: {DEVICE}")

    # 1. Prepare Datasets (We need 3 loaders!)
    # Loader A: SimCLR Training (returns twin images, no labels)
    train_dataset = CIFAR10(root='./dataset', train=True, download=True, transform=simclr_transform)
    train_loader = DataLoader(train_dataset, batch_size=config_override['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    
    # Loader B: kNN Memory (Train set, but no twins, no crazy augmentations, returns labels)
    memory_dataset = CIFAR10(root='./dataset', train=True, download=True, transform=test_transforms)
    memory_loader = DataLoader(memory_dataset, batch_size=config_override['batch_size'], shuffle=False, num_workers=4)

    # Loader C: kNN Test (Test set, returns labels)
    test_dataset = CIFAR10(root='./dataset', train=False, download=True, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=config_override['batch_size'], shuffle=False, num_workers=4)
    
    # 2. Initialize Model
    model = SimCLR(config=config_override, chkpt_file_pth=CHKPT_PATH, device=DEVICE)
    
    best_knn_acc = 0.0

    for epoch in range(EPOCHS):
        # -- TRAIN --
        model.train()
        train_loss = 0.0
        
        # Note: dataset returns ((view_1, view_2), label). We ignore the label '_'.
        for (view_1, view_2), _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            loss = model.learn(view_1, view_2)
            train_loss += loss
            
        avg_train_loss = train_loss / len(train_loader)

        log_dict = {
            "epoch": epoch + 1,
            "train_nt_xent_loss": avg_train_loss,
            "learning_rate": config_override["learning_rate"]
        }

        # -- KNN MONITOR (Every 5 Epochs per PDF) --
        # We also do it on epoch 0 just to see the baseline random accuracy (~10%)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            knn_acc = knn_monitor(model, memory_loader, test_loader, DEVICE, k=config_override["knn_k"])
            log_dict["knn_accuracy"] = knn_acc
            print(f"Epoch {epoch+1:03d} | Loss: {avg_train_loss:.4f} | kNN Acc: {knn_acc:.2f}%")
            
            # Checkpoint based on kNN accuracy (The true measure of representation quality)
            if knn_acc > best_knn_acc:
                best_knn_acc = knn_acc
                model.save_model()
                wandb.run.summary["best_knn_acc"] = best_knn_acc
                print("  --> Saved Best Model")
        else:
            print(f"Epoch {epoch+1:03d} | Loss: {avg_train_loss:.4f}")

        wandb.log(log_dict)

    wandb.finish()

if __name__ == "__main__":
    ablation_tasks = [
        # --- Temperature Ablations ---
        {"temperature": 0.1, "batch_size": 512},
        {"temperature": 5.0, "batch_size": 512},

        # --- Batch Size Ablations ---
        # {"temperature": 0.5, "batch_size": 256},
        # {"temperature": 0.5, "batch_size": 128},
        # {"temperature": 0.5, "batch_size": 64},
        # {"temperature": 0.5, "batch_size": 32},

        # --- Augmentation Ablation ---
        # {"temperature": 0.5, "batch_size": 512, "aug_mode": "crop_only"},
    ]
    for config in ablation_tasks:
        config_ = CONFIG.copy()
        config_["temperature"] = config["temperature"]
        config_["batch_size"] = config["batch_size"]
        train(config_)
