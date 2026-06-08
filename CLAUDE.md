## Project Overview

This is an autism classification research project using GoogLeNet (Inception v1) with Genetic Algorithm (GA) hyperparameter optimization. The model performs binary classification — **TC** (Typical Control) vs **TS** (Autism Spectrum) — on facial/brain scan images using PyTorch.

## Environment & Commands

The project uses `uv` for environment management with Python 3.13.

```bash
# Install dependencies
uv sync

# Launch Jupyter for notebooks
uv run jupyter notebook

# Lint (ruff)
uv run ruff check .
uv run ruff format .
```

## Architecture

All research code lives in three Jupyter notebooks executed in order:

1. **[notebooks/augmentation.ipynb](notebooks/augmentation.ipynb)** — Data augmentation pipeline
   - Reads raw images from `data/Images/` (TC: 328, TS: 219 — imbalanced)
   - Applies Albumentations transforms (HorizontalFlip, Rotate, Affine)
   - Oversamples the minority class (TS: 9–10x, TC: 5–8x augmented copies)
   - Outputs to `data/Augmented_Images/` and merges with originals into `data/Merged_Images/`
   - Also produces merged label CSVs in `data/Metadata/Merged_Metadata/`

2. **[notebooks/Preprocessing.ipynb](notebooks/Preprocessing.ipynb)** — Image preprocessing pipeline
   - Reads from `data/Merged_Images/`
   - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space + gamma correction (γ=1.5) to address the dark/low-contrast nature of the source images
   - Outputs preprocessed images to `data/Preprocessed_Images/` organized as `TC/` and `TS/` subdirectories (compatible with `torchvision.datasets.ImageFolder`)

3. **[notebooks/autism_googlenet.ipynb](notebooks/autism_googlenet.ipynb)** — Model training with GA optimization
   - Reads preprocessed images from `data/Preprocessed_Images/`
   - Splits 70/20/10 (train/val/test), batch size 64, image size 224×224, normalized with ImageNet mean/std
   - Uses pretrained GoogLeNet backbone (frozen) + GA-optimized FC head
   - GA encodes two gene types per individual: **control genes** (number of FC layers: 1–2, epochs: 10–100) and **FC genes** (per-layer: type, neuron count, dropout rate)
   - Fitness = best validation accuracy achieved during training
   - Population: 30, generations: 15, crossover prob: 0.7 (decays), mutation prob: 0.3 (grows)
   - Checkpointing to `checkpoint/checkpoint.pkl` allows resuming interrupted GA runs
   - Best model saved to `Best_Model/`

## Key Classes (in `autism_googlenet.ipynb`)

- **`GoogleNet`** — wraps pretrained GoogLeNet, replaces `fc` with `Identity`, appends GA-configured FC blocks. FC block structure per gene: `Linear → ReLU → BatchNorm1d → (optional Dropout) → ...→ Linear(2)`. Only FC block parameters are trained.
- **`TrainerAndEvaluation`** — handles train/val/test loops, early stopping (patience=5), `ReduceLROnPlateau` scheduler, and training history logging.
- **`EarlyStopping`** — restores best weights on trigger.
- **`run_evolution()`** — the main GA loop with elitism (top 10%), tournament selection (size=3/5), adaptive crossover/mutation rates, and generational CSV logging to `Genetic_Result/GA_Results.csv`.

## Data Directory Structure

```
data/
  Images/          # Raw originals (TC/, TS/)
  Augmented_Images/ # Augmented copies
  Merged_Images/   # Originals + augmented combined (input to Preprocessing)
  Preprocessed_Images/ # CLAHE-enhanced (input to training, ImageFolder format)
  Metadata/
    Metadata_Participants.csv
    Metadata_TC.csv
    Metadata_TS.csv
    Augmented_Metadata/
    Merged_Metadata/
```

## GA Individual Encoding

An individual is a list: `[[control_gene], [fc_gene_layer_0, fc_gene_layer_1, ...]]`

- `control_gene`: `[num_fc_layers (1–2), num_epochs (10–100, step 10)]`
- Each FC gene layer: `[layer_type (1=no dropout, 2=with dropout), num_neurons, dropout_pct (10–50, step 10)]`
- Neuron counts are chosen from fixed pairs: `(1024→512)`, `(512→256)`, `(256→128)`
- FC genes are always sorted descending by neuron count

## Code Style

Enforced by Ruff:

- Line length: 88 characters
- Quote style: double quotes
- Active rule sets: `E/W`, `F`, `I` (isort), `UP` (pyupgrade), `B` (bugbear), `C4` (comprehensions), `N` (pep8-naming), `PTH` (pathlib)

## Branching & Commits

- [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): description`
  - Types: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`
  - Example: `feat(training): add ga-optimized fc head to googlenet`
- Before pushing: `uv run ruff check .` and `uv run pytest` must both pass

## Git Rules
- Always commit using the repo's configured git identity
- Do not override user.name or user.email when committing