"""Dataset loading and splitting utilities for the autism classifier.

Builds train/val/test ``DataLoader`` objects from an ``ImageFolder``-compatible
directory (e.g. ``data/Preprocessed_Images/`` with ``TC/`` and ``TS/``
subdirectories).  The 70/20/10 split is seeded for reproducibility and images
are resized to 224×224 and normalised with ImageNet statistics.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from asd_classifier.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
)


def get_transforms() -> transforms.Compose:
    """Return the standard inference/training transform pipeline (resize → tensor → normalise)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_dataloaders(
    image_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build and return ``(train_loader, val_loader, test_loader)`` from *image_dir*.

    Applies a 70/20/10 stratified random split seeded at 42.  Prints split
    sizes before returning.
    """
    dataset = ImageFolder(str(image_dir), transform=get_transforms())

    dataset_size = len(dataset)
    train_size = int(TRAIN_RATIO * dataset_size)
    val_size = int(VAL_RATIO * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")
    return train_loader, val_loader, test_loader
