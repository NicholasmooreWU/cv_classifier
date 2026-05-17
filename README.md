# PNNL-Style Image Classifier

A beginner-friendly PyTorch computer vision project aligned with the skills
in PNNL's National Security Directorate AI/ML role.

## What You'll Learn
- Building and training a **CNN** from scratch in PyTorch
- **Data augmentation** and normalization with `torchvision`
- The core **training loop**: forward pass → loss → backprop → optimizer step
- **Model checkpointing** (saving the best model)
- **Per-class evaluation** — critical for operational ML systems

## Connection to the PNNL JD
| JD Skill | This Project |
|---|---|
| PyTorch | Core framework throughout |
| `torchvision` | Dataset loading + transforms |
| Computer vision / radiographic analysis | CNN image classification |
| Reproducibility & documentation | Comments, config block, checkpointing |
| Research → operational pipeline | train.py → evaluate.py → inference |

## Project Structure
```
pnnl_cv_classifier/
├── model.py        # CNN architecture (ThreatClassifierCNN)
├── train.py        # Full training pipeline with checkpointing
├── evaluate.py     # Per-class accuracy + top-5 inference demo
└── requirements.txt
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (downloads CIFAR-10 automatically)
python train.py

# 3. Evaluate on the test set
python evaluate.py

# 4. Run inference on a specific image (e.g. image #100)
python evaluate.py 100
```

## What is CIFAR-10?
A dataset of 60,000 32×32 color images across 10 classes.
For national-security framing, focus on: **airplane, automobile, ship, truck**.

## Expected Results
| Metric | Target |
|---|---|
| Training time (CPU) | ~5–10 min |
| Training time (GPU) | ~2–3 min |
| Validation accuracy | **~75%** after 15 epochs |

## Key Concepts (Plain English)

**CNN (Convolutional Neural Network)**: scans small patches of an image
to detect edges, shapes, and eventually whole objects.

**Loss**: a number measuring how wrong the model is. Training minimizes it.

**Backpropagation**: calculates which direction to nudge each weight to
reduce the loss. Automatic in PyTorch via `.backward()`.

**Optimizer (Adam)**: applies those nudges to the weights after each batch.

**Epoch**: one full pass over the entire training dataset.

**Overfitting**: model memorizes training data but fails on new images.
We fight it with dropout, batch norm, and data augmentation.

## Extending This Project
Ideas to deepen your skills for PNNL:
- Swap CIFAR-10 for a medical/X-ray dataset (e.g. NIH ChestX-ray14)
- Add a `GradCAM` visualization to see *what* the model is looking at
- Export the model to ONNX for cloud deployment (SageMaker-ready)
- Add a Kafka consumer to run inference on a real-time image stream
