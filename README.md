# Autism Classification via GoogLeNet + Genetic Algorithm Hyperparameter Optimization

A deep learning pipeline for binary classification of **Autism Spectrum (TS)** vs. **Typical Control (TC)** subjects using a pretrained GoogLeNet (Inception v1) backbone with a Genetic Algorithm (GA)-optimized fully-connected head.

---

## Overview

The system addresses two core challenges common in medical imaging research:

- **Class imbalance** — raw data contains 328 TC and 219 TS samples; the pipeline oversamples the minority class via augmentation before training.
- **Hyperparameter search** — instead of manual tuning, a Genetic Algorithm evolves optimal FC head architectures (depth, width, dropout) and training configurations (epochs) automatically.

The best evolved model achieved **94.03% validation accuracy** across 15 generations with a population of 30 individuals.

---

## Architecture

```
Input Image (224×224×3)
        │
        ▼
┌───────────────────┐
│  GoogLeNet        │  Pretrained on ImageNet — weights frozen
│  (Inception v1)   │  Outputs 1024-dimensional feature vector
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  GA-Optimized     │  Evolved FC head — depth, width, dropout
│  FC Head          │  tuned automatically by the GA
└───────────────────┘
        │
        ▼
  Output: TC / TS
```

Only the FC head parameters are trained; the backbone is kept frozen (11.98M non-trainable parameters, 1.17M trainable).

---

## Genetic Algorithm

The GA encodes each candidate architecture as an individual:

```
individual = [[control_gene], [fc_gene_0, fc_gene_1, ...]]
```

| Gene | Fields | Range |
|------|--------|-------|
| `control_gene` | `num_fc_layers`, `num_epochs` | layers: 1–2; epochs: 10–100 (step 10) |
| Each `fc_gene` | `layer_type`, `num_neurons`, `dropout_pct` | type: 1 (no dropout) / 2 (dropout); neurons: paired (1024→512, 512→256, 256→128); dropout: 10–50% |

**Selection & Evolution:**

| Parameter | Value |
|-----------|-------|
| Population size | 30 |
| Generations | 15 |
| Crossover probability | 0.7 (adaptive, decaying) |
| Mutation probability | 0.3 (adaptive, growing) |
| Elitism | Top 10% carried forward |
| Tournament selection size | 3 / 5 |
| Fitness function | Best validation accuracy |

GA runs are resumable via checkpointing (`checkpoints/checkpoint.pkl`). Per-generation results are logged to `Genetic_Result/GA_Results.csv`.

---

## Best Model

The GA converged to the following architecture:

| Parameter | Value |
|-----------|-------|
| FC layers | 2 |
| Epochs | 100 |
| Layer 1 | Linear(1024→818) → ReLU → BatchNorm → Dropout(20%) |
| Layer 2 | Linear(818→397) → ReLU → BatchNorm |
| Output | Linear(397→2) |
| Total params | 13,146,707 |
| Trainable params | 1,166,819 |
| Best validation accuracy | **94.03%** |

---

## Data Pipeline

### 1. Augmentation

Reads raw images from `data/Images/` and oversamples the minority class:

- **TS** — 9–10× augmented copies
- **TC** — 5–8× augmented copies

Augmentations: `HorizontalFlip`, `Rotate`, `Affine` (via Albumentations).

Output: `data/Augmented_Images/` and merged dataset in `data/Merged_Images/`.

### 2. Preprocessing

Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space followed by gamma correction (γ = 1.5) to compensate for the dark, low-contrast nature of the source images.

Output: `data/Preprocessed_Images/` in `TC/` and `TS/` subdirectories (compatible with `torchvision.datasets.ImageFolder`).

### 3. Training

- Split: 70% train / 20% val / 10% test
- Batch size: 64
- Image size: 224×224, normalized with ImageNet mean/std
- Optimizer: Adam with `ReduceLROnPlateau` scheduler
- Early stopping: patience = 5 (restores best weights on trigger)

---

## Directory Structure

```
autismClassification/
├── data/
│   ├── Images/                  # Raw originals (TC/, TS/)
│   ├── Augmented_Images/        # Augmented copies
│   ├── Merged_Images/           # Originals + augmented
│   ├── Preprocessed_Images/     # CLAHE-enhanced (model input)
│   └── Metadata/
│       ├── Metadata_Participants.csv
│       ├── Metadata_TC.csv
│       ├── Metadata_TS.csv
│       ├── Augmented_Metadata/
│       └── Merged_Metadata/
├── Best_Model/
│   ├── best_model.pth
│   ├── best_individual_details.csv
│   ├── best_model_training_log.csv
│   ├── training_history.jpg
│   ├── Classification_Report.jpg
│   └── summary.txt
├── Genetic_Result/
│   └── GA_Results.csv
└── checkpoints/
    └── checkpoint.pkl
```

---

## Setup

Requires Python 3.13 and [`uv`](https://github.com/astral-sh/uv).

```bash
# Install dependencies
uv sync
```

---

## Code Quality

Linting is enforced with [Ruff](https://docs.astral.sh/ruff/):

```bash
uv run ruff check .
uv run ruff format .
```

Active rule sets: `E/W`, `F`, `I` (isort), `UP` (pyupgrade), `B` (bugbear), `C4` (comprehensions), `N` (pep8-naming), `PTH` (pathlib). Line length: 88. Quote style: double.

---

## Results

| Generation | Best Fitness |
|-----------|-------------|
| 1 | 93.31% |
| 3 | 93.67% |
| 5 | 93.85% |
| 14 | **94.03%** |

Training history and classification report plots are saved under `Best_Model/`.

---

## License

This project is intended for academic and research purposes.
