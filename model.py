"""
model.py — CNN Architecture
----------------------------
This defines our neural network. Think of it as a blueprint
for the model's "brain" — layers stacked together that learn
to recognize patterns in images.

Relevant to PNNL NSD: similar architectures are used for
analyzing radiographic/sensor imagery for threat detection.
"""

import torch
import torch.nn as nn


class ThreatClassifierCNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for image classification.

    Architecture overview:
      - Conv blocks: extract visual features (edges, shapes, textures)
      - Fully connected layers: map those features to class predictions

    nn.Module is PyTorch's base class for all neural networks.
    We inherit from it and define two methods:
      - __init__: build the layers
      - forward: describe how data flows through them
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ── Feature Extractor ─────────────────────────────────────────
        # Each Conv2d layer slides a small filter across the image and
        # learns to detect specific visual patterns.
        # BatchNorm2d stabilizes training. ReLU adds non-linearity.
        # MaxPool2d downsamples (shrinks) the spatial dimensions.

        self.features = nn.Sequential(
            # Block 1: 3 input channels (RGB) → 32 feature maps
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32×32 → 32×32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # 32×32 → 16×16
            nn.Dropout2d(0.25),

            # Block 2: 32 → 64 feature maps, deeper patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16×16 → 16×16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # 16×16 → 8×8
            nn.Dropout2d(0.25),

            # Block 3: 64 → 128 feature maps, high-level abstractions
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # 8×8 → 4×4
            nn.Dropout2d(0.25),
        )

        # ── Classifier ────────────────────────────────────────────────
        # Flatten the 3D feature maps into a 1D vector, then use
        # fully-connected (Linear) layers to produce class scores.
        # 128 feature maps × 4×4 spatial = 2048 values

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),        # Dropout randomly zeros neurons → prevents overfitting
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: define how input tensor x flows through the model.
        PyTorch auto-computes gradients for backpropagation.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x  # Raw logits (not yet probabilities — CrossEntropyLoss handles that)


def count_parameters(model: nn.Module) -> int:
    """Helper: count trainable parameters so you know model size."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
