"""Visualization and evaluation utilities for the autism classifier.

Provides functions to display sample images per class, plot class
distributions, render training history curves, format classification
reports as tables, and produce a full confusion-matrix evaluation.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from asd_classifier.config import NUM_CLASSES


def visualizing(dataframe: pd.DataFrame, pts_per_class: int) -> None:
    """Display a grid of up to *pts_per_class* sample images for each class label."""
    df = dataframe.groupby(["Class"], as_index=False).apply(
        lambda x: x.sample(min(len(x), pts_per_class))
    ).reset_index(drop=True)

    for label in df["Class"].unique():
        subset = df[df["Class"] == label]
        subset_size = min(len(subset), pts_per_class)
        cols = min(subset_size, 5)
        rows = (subset_size // cols) + (subset_size % cols > 0)

        fig, axs = plt.subplots(rows, cols, figsize=(20, rows * 4))
        if rows == 1 and cols == 1:
            axs = [axs]
        else:
            axs = axs.flatten() if rows == 1 or cols == 1 else axs.ravel()

        for i, ax in enumerate(axs):
            if i < subset_size:
                img = cv2.cvtColor(cv2.imread(subset.iloc[i]["Filepath"]), cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
            ax.axis("off")

        plt.suptitle(f"Class: {label}", fontsize=24, fontweight="bold", y=0.92)
        plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
        plt.show()


def class_distribution(dataframe: pd.DataFrame, class_column: str) -> None:
    """Plot a bar chart of sample counts per class in *dataframe*."""
    grouped = dataframe.groupby(class_column, as_index=False)["Filename"].count()
    grouped.columns = ["Class", "Counts"]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(grouped["Class"], grouped["Counts"], color=plt.cm.viridis(range(len(grouped))))
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            int(bar.get_height()),
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    plt.title("Class Distribution", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Class", fontsize=14, labelpad=10)
    plt.ylabel("Counts", fontsize=14, labelpad=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_training_history(history: dict, save_path: Path | None = None) -> None:
    """Plot training and validation loss/accuracy curves side by side.

    Optionally saves the figure to *save_path* before displaying.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(15, 8))

    for i, (title, train_key, val_key, tc, vc) in enumerate([
        ("Loss", "train_loss", "val_loss", "b-", "r-"),
        ("Accuracy", "train_acc", "val_acc", "g-", "m-"),
    ], 1):
        plt.subplot(1, 2, i)
        plt.plot(epochs, history[train_key], tc, label=f"Train {title}", marker="o")
        plt.plot(epochs, history[val_key], vc, label=f"Validation {title}", marker="o")
        plt.title(f"Training and Validation {title}")
        plt.xlabel("Epochs")
        plt.ylabel(title)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def display_classification_report_table(
    report: dict,
    num_classes: int = NUM_CLASSES,
    save_path: Path | None = None,
) -> None:
    """Print and render a matplotlib table from a sklearn classification report dict.

    Optionally saves the figure to *save_path*.
    """
    df_report = pd.DataFrame(
        {f"Class {i}": report[str(i)] for i in range(num_classes)}
    ).T[["precision", "recall", "f1-score", "support"]]

    print("\nClassification Report:\n", df_report)

    fig, ax = plt.subplots(figsize=(8, num_classes * 0.5 + 1))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df_report.values,
        colLabels=df_report.columns,
        rowLabels=df_report.index,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df_report.columns))))
    plt.title("Classification Report")
    plt.grid(False)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def evaluation(model, trainer, num_classes: int = NUM_CLASSES) -> tuple[float, float]:
    """Run the test split, display a confusion matrix and classification report table.

    Returns ``(accuracy, loss)`` from the test evaluation.
    """
    eval_dict = trainer.test()

    cm = confusion_matrix(eval_dict["labels"], eval_dict["predictions"], labels=range(num_classes))
    report = classification_report(
        eval_dict["labels"], eval_dict["predictions"],
        labels=range(num_classes), output_dict=True,
    )

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_classes)).plot(
        cmap=plt.cm.Reds
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.grid(False)
    plt.show()

    display_classification_report_table(report, num_classes)

    return eval_dict["accuracy"], eval_dict["loss"]
