"""Data augmentation pipeline for the autism classification dataset.

Reads raw images from ``data/Images/`` (TC: 328, TS: 219 — class-imbalanced),
applies Albumentations transforms (HorizontalFlip, Rotate, Affine), and
oversamples the minority class (TS: 9–10×, TC: 5–8× augmented copies).
Outputs augmented images to ``data/Augmented_Images/`` and merges them with
the originals into ``data/Merged_Images/``.  Corresponding label CSVs are
written to ``data/Metadata/Merged_Metadata/``.

Entry point: ``run_augmentation()``.
"""

import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image  # noqa: F401

from asd_classifier.config import (
    AUGMENTED_DIR,
    AUGMENTED_METADATA_DIR,
    IMAGES_DIR,
    MERGED_DIR,
    MERGED_METADATA_DIR,
    PARTICIPANTS_CSV,
    TC_CSV,
    TS_CSV,
)

_rng = np.random.default_rng(42)

_augmentations = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.Rotate(angle_range=(-45, 45), border_mode=cv2.BORDER_CONSTANT, p=1.0),
    A.Affine(
        translate_percent=(-0.15, 0.15),
        shear=(-10, 10),
        border_mode=cv2.BORDER_CONSTANT,
        p=1.0,
    ),
])


def augmentation(dataset: ImageFolder, image_save_dir: Path, metadata_save_dir: Path) -> None:
    """Generate augmented copies for each sample in *dataset* and save them.

    Clears and recreates *image_save_dir* and *metadata_save_dir* before
    writing.  TS images receive 9–10 copies; TC images receive 5–8 copies.
    A ``label.csv`` is written per class under *metadata_save_dir*.
    """
    for d in [image_save_dir, metadata_save_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    data: dict[str, list] = {"TC": [], "TS": []}

    for idx, (image_path, label) in enumerate(dataset.samples, start=1):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        class_name = "TC" if label == 0 else "TS"
        aug_count = (
            int(_rng.integers(9, 11, endpoint=True))
            if class_name == "TS"
            else int(_rng.integers(5, 8, endpoint=True))
        )

        images_dir = image_save_dir / class_name
        images_dir.mkdir(exist_ok=True)

        for i in range(aug_count):
            augmented = _augmentations(image=image)
            augmented_image = augmented["image"]

            filename = f"{class_name}{idx}_{i}_augmented.png"
            save_path = images_dir / filename
            cv2.imwrite(str(save_path), cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))

            data[class_name].append({"Filename": filename, "Class": class_name})

    for class_name, entries in data.items():
        class_dir = metadata_save_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        csv_path = class_dir / "label.csv"
        pd.DataFrame(entries, columns=["Filename", "Class"]).to_csv(csv_path, index=False)


def copy_files(src_dir: Path, dst_dir: Path) -> None:
    """Recursively copy all files from *src_dir* into *dst_dir* (flat)."""
    for src_file in src_dir.rglob("*"):
        if src_file.is_file():
            dst_file = dst_dir / src_file.name
            shutil.copy2(src_file, dst_file)


def merge_and_copy_files(
    original_images_dir: Path,
    augmented_images_dir: Path,
    destination: Path,
    class_names: list[str],
) -> None:
    """Merge original and augmented images for each class into *destination*.

    Directories are matched by sort order with *class_names*.  *destination*
    is recreated from scratch on each call.
    """
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    orig_folders = sorted(original_images_dir.iterdir())
    aug_folders = sorted(augmented_images_dir.iterdir())

    for orig_folder, aug_folder, cls_name in zip(orig_folders, aug_folders, class_names):
        class_dst = destination / cls_name
        class_dst.mkdir(exist_ok=True)
        copy_files(orig_folder, class_dst)
        copy_files(aug_folder, class_dst)


def label_dataframe(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Extract ``(ParticipantID, Filename)`` columns from *metadata_df*."""
    rows = [
        {"ParticipantID": row["ParticipantID"], "Filename": row["Filename"]}
        for _, row in metadata_df.iterrows()
    ]
    return pd.DataFrame(rows, columns=["ParticipantID", "Filename"])


def save_merged_labels(
    tc_dataframes: list[pd.DataFrame],
    ts_dataframes: list[pd.DataFrame],
    output_dir: Path,
) -> None:
    """Concatenate per-class label DataFrames and write ``TC.csv`` / ``TS.csv``."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for class_name, frames in [("TC", tc_dataframes), ("TS", ts_dataframes)]:
        merged = pd.concat(frames)
        merged.to_csv(output_dir / f"{class_name}.csv", index=False)


def run_augmentation() -> None:
    """Run the full augmentation pipeline using paths from ``config``.

    Generates augmented images, merges them with originals, and writes the
    combined label CSVs.  Prints a short summary on completion.
    """
    dataset = ImageFolder(root=str(IMAGES_DIR))

    augmentation(dataset, AUGMENTED_DIR, AUGMENTED_METADATA_DIR)

    merge_and_copy_files(IMAGES_DIR, AUGMENTED_DIR, MERGED_DIR, ["TC", "TS"])

    participants_df = pd.read_csv(PARTICIPANTS_CSV)
    orig_tc_df = pd.read_csv(TC_CSV)
    orig_ts_df = pd.read_csv(TS_CSV)
    aug_tc_df = pd.read_csv(AUGMENTED_METADATA_DIR / "TC" / "label.csv")
    aug_ts_df = pd.read_csv(AUGMENTED_METADATA_DIR / "TS" / "label.csv")

    drop_cols = ["Gender", "Date of Presentation", "Age", "CARS Score"]

    origin_tc_df = label_dataframe(orig_tc_df)
    origin_tc_df = pd.merge(origin_tc_df, participants_df, on="ParticipantID")
    origin_tc_df = origin_tc_df.drop(columns=["ParticipantID"] + drop_cols)

    origin_ts_df = label_dataframe(orig_ts_df)
    origin_ts_df = pd.merge(origin_ts_df, participants_df, on="ParticipantID")
    origin_ts_df = origin_ts_df.drop(columns=["ParticipantID"] + drop_cols)

    save_merged_labels(
        tc_dataframes=[origin_tc_df, aug_tc_df],
        ts_dataframes=[origin_ts_df, aug_ts_df],
        output_dir=MERGED_METADATA_DIR,
    )

    print("Augmentation complete.")
    print(f"  Augmented images → {AUGMENTED_DIR}")
    print(f"  Merged images    → {MERGED_DIR}")
    print(f"  Merged labels    → {MERGED_METADATA_DIR}")
