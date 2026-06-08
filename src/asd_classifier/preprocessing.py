"""Image preprocessing pipeline for the autism classification dataset.

Reads merged images from ``data/Merged_Images/`` and applies CLAHE (Contrast
Limited Adaptive Histogram Equalization) in LAB colour space followed by
gamma correction (γ=1.5) to address the dark/low-contrast nature of the
source images.  Preprocessed images are written to
``data/Preprocessed_Images/TC/`` and ``data/Preprocessed_Images/TS/`` in an
``ImageFolder``-compatible layout.

Entry point: ``run_preprocessing()``.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from asd_classifier.config import MERGED_DIR, MERGED_METADATA_DIR, PREPROCESSED_DIR


def df_creation(images_dir: Path, labels_dir: Path) -> pd.DataFrame:
    """Build a ``(Filename, Filepath, Class)`` DataFrame by pairing image files with label CSVs.

    Folders in *images_dir* and CSVs in *labels_dir* are matched by sorted order.
    """
    data = []
    for folder, label_file in zip(
        sorted(os.listdir(images_dir)), sorted(os.listdir(labels_dir))
    ):
        images_folder_path = images_dir / folder
        lbl_df = pd.read_csv(labels_dir / label_file)
        for img_file, (_, row) in zip(
            sorted(os.listdir(images_folder_path)), lbl_df.iterrows()
        ):
            img_file_name = Path(img_file).stem
            img_file_path = images_folder_path / img_file
            data.append((img_file_name, str(img_file_path), row["Class"]))
    return pd.DataFrame(data, columns=("Filename", "Filepath", "Class"))


def preprocessing(
    image_path: str,
    gamma: float = 1.5,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Apply CLAHE on the L channel (LAB space) then gamma-correct the image.

    Args:
        image_path: Absolute or relative path to the source image.
        gamma: Gamma correction exponent; >1 brightens, <1 darkens.
        clip_limit: CLAHE clip limit controlling contrast enhancement strength.
        tile_grid_size: CLAHE tile grid dimensions.

    Returns:
        Preprocessed image as a BGR ``np.ndarray``.
    """
    img = cv2.imread(image_path)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l_ch, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_ch)

    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    inv_gamma = 1.0 / gamma
    gamma_table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    img_gamma = cv2.LUT(enhanced_rgb, gamma_table)

    return cv2.cvtColor(img_gamma, cv2.COLOR_RGB2BGR)


def image_saving(
    dataframe: pd.DataFrame,
    processing_func,
    output_dir: Path,
) -> None:
    """Apply *processing_func* to each image in *dataframe* and save results to *output_dir*.

    Recreates *output_dir* with ``TC/`` and ``TS/`` subdirectories.
    Files without a recognised image extension have ``.png`` appended.
    """
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tc_dir = output_dir / "TC"
    ts_dir = output_dir / "TS"
    tc_dir.mkdir()
    ts_dir.mkdir()

    for _, data in dataframe.iterrows():
        filepath = data["Filepath"]
        filename = data["Filename"]
        image_class = data["Class"]

        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            filename += ".png"

        processed = processing_func(filepath)

        out_path = (tc_dir if image_class == "TC" else ts_dir) / filename
        if not cv2.imwrite(str(out_path), processed):
            print(f"Failed to save image at {out_path}")


def run_preprocessing() -> None:
    """Run the full preprocessing pipeline using paths from ``config``.

    Builds the image/label DataFrame, applies CLAHE + gamma correction to
    every image, and writes results to ``data/Preprocessed_Images/``.
    Prints a short summary on completion.
    """
    df = df_creation(MERGED_DIR, MERGED_METADATA_DIR)
    image_saving(df, preprocessing, PREPROCESSED_DIR)

    print("Preprocessing complete.")
    print(f"  Preprocessed images → {PREPROCESSED_DIR}")
