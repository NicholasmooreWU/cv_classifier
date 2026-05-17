"""
evaluate.py — Inference & Per-Class Analysis
---------------------------------------------
After training, this script loads the saved model and produces:
  - Overall accuracy
  - Per-class accuracy breakdown (useful for spotting biases)
  - Confusion matrix printed to console
  - Single-image inference demo

Relevant to PNNL NSD: in operational settings, per-class performance
matters enormously — a model that's great overall but misses "airplane"
90% of the time is not deployable for border/airspace monitoring.
"""

import sys
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import ThreatClassifierCNN

CHECKPOINT  = "./best_model.pth"
DATA_DIR    = "./data"
BATCH_SIZE  = 128

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_model(checkpoint_path: str, device: torch.device) -> ThreatClassifierCNN:
    """Load a trained model from a checkpoint file."""
    model = ThreatClassifierCNN(num_classes=10).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    epoch   = checkpoint.get("epoch", "?")
    val_acc = checkpoint.get("val_acc", 0.0)
    print(f"  Loaded checkpoint from epoch {epoch} | Val Acc: {val_acc:.2f}%")
    return model


@torch.no_grad()
def run_evaluation(model: ThreatClassifierCNN, device: torch.device):
    """
    Full evaluation on the CIFAR-10 test set with per-class breakdown.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model.eval()

    # Track per-class correct/total for detailed analysis
    class_correct = [0] * 10
    class_total   = [0] * 10
    overall_correct = 0
    overall_total   = 0

    # Confusion matrix: confusion[true][predicted] = count
    confusion = [[0] * 10 for _ in range(10)]

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        for true, pred in zip(labels.cpu(), predicted.cpu()):
            confusion[true.item()][pred.item()] += 1
            class_total[true.item()]   += 1
            class_correct[true.item()] += int(true == pred)
            overall_total   += 1
            overall_correct += int(true == pred)

    # ── Print Results ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Overall Accuracy: {100.0 * overall_correct / overall_total:.2f}%\n")

    print("  Per-Class Accuracy:")
    print("  " + "-" * 40)
    for i, name in enumerate(CLASS_NAMES):
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        bar = "█" * int(acc // 5)   # ASCII bar chart
        print(f"  {name:<12} {acc:5.1f}%  {bar}")
    print("  " + "-" * 40)

    # Show most common confusions (useful for model debugging)
    print("\n  Top Misclassifications:")
    misses = []
    for true_i in range(10):
        for pred_i in range(10):
            if true_i != pred_i and confusion[true_i][pred_i] > 0:
                misses.append((confusion[true_i][pred_i], true_i, pred_i))
    misses.sort(reverse=True)

    for count, true_i, pred_i in misses[:5]:
        print(f"    '{CLASS_NAMES[true_i]}' predicted as '{CLASS_NAMES[pred_i]}': {count}x")

    print("=" * 55)


@torch.no_grad()
def predict_single(model: ThreatClassifierCNN, device: torch.device, image_index: int = 42):
    """
    Run inference on a single image from the test set.
    Demonstrates how you'd use the model in a real pipeline.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=False, transform=transform
    )

    image, true_label = dataset[image_index]
    image = image.unsqueeze(0).to(device)   # Add batch dimension: [H,W,C] → [1,H,W,C]

    model.eval()
    logits = model(image)

    # Convert raw logits to probabilities with softmax
    probs = F.softmax(logits, dim=1).squeeze()

    top5_probs, top5_idx = probs.topk(5)

    print("\n  Single-Image Inference Demo")
    print("  " + "-" * 35)
    print(f"  Ground truth: {CLASS_NAMES[true_label]}")
    print(f"  Model's top-5 predictions:")
    for prob, idx in zip(top5_probs, top5_idx):
        marker = " ← predicted" if idx == top5_idx[0] else ""
        print(f"    {CLASS_NAMES[idx.item()]:<12} {prob.item()*100:5.1f}%{marker}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    model = load_model(CHECKPOINT, device)
    run_evaluation(model, device)

    # Demo: predict one image
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    predict_single(model, device, image_index=idx)
