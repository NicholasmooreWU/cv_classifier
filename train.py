"""
train.py — Training Pipeline
------------------------------
This script loads data, trains the CNN, and saves the best model.

Core PyTorch training loop concepts:
  1. Forward pass  → model makes predictions
  2. Compute loss  → how wrong were the predictions?
  3. Backward pass → compute gradients (which direction to adjust weights)
  4. Optimizer step → actually update the weights

Relevant to PNNL NSD: this kind of pipeline (data → model → evaluation)
is the backbone of operationalized ML systems for sensor/image analysis.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from model import ThreatClassifierCNN, count_parameters

# ── Config ────────────────────────────────────────────────────────────────────
# Keeping hyperparameters at the top makes experiments easy to track (GitOps!)

BATCH_SIZE   = 64     # How many images to process at once
EPOCHS       = 15     # Full passes over the training data
LR           = 1e-3   # Learning rate: step size for weight updates
LR_STEP      = 7      # Reduce LR every N epochs (learning rate scheduling)
LR_GAMMA     = 0.1    # Multiply LR by this factor at each step
DATA_DIR     = "./data"
CHECKPOINT   = "./best_model.pth"

# CIFAR-10 class names — note the national-security-relevant ones!
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_device() -> torch.device:
    """Pick the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")      # Apple Silicon
        print("  Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("  Using CPU (training will be slower, ~5–10 min for 15 epochs)")
    return device


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 and apply transforms.

    Transforms serve two purposes:
      1. Normalization  → zero-mean, unit-variance inputs train faster
      2. Augmentation   → random flips/crops expose the model to more variation,
                          reducing overfitting (critical for small datasets)
    """
    # ImageNet mean/std are commonly reused — they work well even for CIFAR
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # Augmentation: random crop
        transforms.RandomHorizontalFlip(),          # Augmentation: mirror image
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2),       # Augmentation: lighting variation
        transforms.ToTensor(),                      # Convert PIL image → Tensor [0,1]
        transforms.Normalize(mean, std),            # Normalize to ~N(0,1)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # Note: NO augmentation on validation — we want a clean benchmark
    ])

    print(f"\n  Downloading CIFAR-10 to '{DATA_DIR}' (first run only)...")
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=val_transform
    )

    # num_workers speeds up data loading by using parallel CPU threads
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train samples: {len(train_dataset):,}  |  Val samples: {len(val_dataset):,}")
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """
    Run one full training epoch.
    Returns: (average loss, accuracy %)
    """
    model.train()   # Enables dropout and batchnorm training behavior

    total_loss, correct, total = 0.0, 0, 0
    start = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        # Move data to GPU/CPU
        images, labels = images.to(device), labels.to(device)

        # ── Core training loop ──────────────────────────────────────
        optimizer.zero_grad()           # Clear old gradients
        outputs = model(images)         # Forward pass → raw logits
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                 # Backward pass → compute gradients
        optimizer.step()               # Update weights
        # ────────────────────────────────────────────────────────────

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)   # Take class with highest logit
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"    Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    elapsed = time.time() - start
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    print(f"  ✓ Train | Epoch {epoch:02d} | Loss: {avg_loss:.4f} "
          f"| Acc: {accuracy:.2f}% | Time: {elapsed:.1f}s")
    return avg_loss, accuracy


@torch.no_grad()   # Decorator: disable gradient tracking during evaluation (faster, less memory)
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate model on validation set.
    @torch.no_grad() means we don't compute gradients — saves memory and time.
    """
    model.eval()   # Disables dropout, uses running stats for batchnorm

    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train():
    print("=" * 60)
    print("  PNNL-Style Image Classifier — Training Pipeline")
    print("=" * 60)

    device = get_device()
    train_loader, val_loader = build_dataloaders()

    # ── Model, Loss, Optimizer ────────────────────────────────────────
    model = ThreatClassifierCNN(num_classes=10).to(device)
    print(f"\n  Model parameters: {count_parameters(model):,}")

    # CrossEntropyLoss = softmax + negative log-likelihood in one step
    # It measures how far the model's predicted distribution is from the truth
    criterion = nn.CrossEntropyLoss()

    # Adam adapts the learning rate per parameter — great default optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # StepLR reduces LR by LR_GAMMA every LR_STEP epochs
    # This helps the model "fine-tune" in later epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    # ── Training Loop ─────────────────────────────────────────────────
    best_val_acc = 0.0
    history = []

    print(f"\n  Starting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n  [Epoch {epoch}/{EPOCHS}]  LR: {current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"  ✓ Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc,
        })

        # Save checkpoint if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": CLASS_NAMES,
            }, CHECKPOINT)
            print(f"  ★ New best model saved! Val Acc: {val_acc:.2f}%")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Training complete.")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved to: {CHECKPOINT}")
    print("=" * 60)

    return history


if __name__ == "__main__":
    train()
