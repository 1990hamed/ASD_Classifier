"""CLI entry point for the autism classification pipeline.

Subcommands
-----------
augment
    Generate augmented copies of the raw images and merge them with the
    originals.  Reads from ``data/Images/``, writes to
    ``data/Augmented_Images/`` and ``data/Merged_Images/``.

preprocess
    Apply CLAHE + gamma correction to every merged image.  Reads from
    ``data/Merged_Images/``, writes to ``data/Preprocessed_Images/``.

train
    Run the Genetic Algorithm to search for the optimal FC-head
    configuration, then save the best model to ``Best_Model/``.
    Accepts ``--population``, ``--ngen``, and ``--num-workers``.

evaluate
    Load the best saved model from ``Best_Model/``, plot its training
    history, display the confusion matrix and classification report, and
    print test accuracy/loss.  Accepts ``--num-workers``.

Usage
-----
    uv run python main.py augment
    uv run python main.py preprocess
    uv run python main.py train --population 30 --ngen 15
    uv run python main.py evaluate
"""

import argparse


def cmd_augment(_args: argparse.Namespace) -> None:
    """Delegate to ``run_augmentation()`` from the augmentation module."""
    from asd_classifier.augmentation import run_augmentation
    run_augmentation()


def cmd_preprocess(_args: argparse.Namespace) -> None:
    """Delegate to ``run_preprocessing()`` from the preprocessing module."""
    from asd_classifier.preprocessing import run_preprocessing
    run_preprocessing()


def cmd_train(args: argparse.Namespace) -> None:
    """Seed PyTorch, build data loaders, inject them into the GA module, and run evolution."""
    import torch
    from asd_classifier import ga
    from asd_classifier.config import BEST_MODEL_DIR, CHECKPOINT_PATH, GENETIC_RESULTS_DIR, PREPROCESSED_DIR
    from asd_classifier.dataset import build_dataloaders

    torch.manual_seed(42)

    train_loader, val_loader, test_loader = build_dataloaders(
        PREPROCESSED_DIR, num_workers=args.num_workers
    )
    ga.set_loaders(train_loader, val_loader, test_loader)

    ga.run_evolution(
        population_size=args.population,
        ngen=args.ngen,
        best_model_path=BEST_MODEL_DIR,
        genetic_results_path=GENETIC_RESULTS_DIR,
        checkpoint_path=CHECKPOINT_PATH,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Reconstruct the best model from saved artefacts and evaluate it on the test split."""
    import torch
    from asd_classifier.config import BEST_MODEL_DIR, PREPROCESSED_DIR
    from asd_classifier.dataset import build_dataloaders
    from asd_classifier.model import GoogleNet
    from asd_classifier.trainer import TrainerAndEvaluation
    from asd_classifier.viz import evaluation, plot_training_history

    train_loader, val_loader, test_loader = build_dataloaders(
        PREPROCESSED_DIR, num_workers=args.num_workers
    )

    # Load best individual from saved CSV
    import ast
    import pandas as pd
    details = pd.read_csv(BEST_MODEL_DIR / "best_individual_details.csv")
    control_genes = ast.literal_eval(details["control_genes"].iloc[0])
    fc_genes = ast.literal_eval(details["fc_genes"].iloc[0])
    individual = [control_genes, fc_genes]

    model = GoogleNet(individual)
    # strict=False tolerates the unused GoogLeNet aux-head keys (aux1/aux2)
    # that may be present in checkpoints saved before aux_logits was disabled.
    state_dict = torch.load(BEST_MODEL_DIR / "best_model.pth", weights_only=True)
    missing, _ = model.load_state_dict(state_dict, strict=False)
    fc_missing = [k for k in missing if k.startswith("fc_blocks")]
    if fc_missing:
        raise RuntimeError(
            f"Saved weights are missing FC-head parameters: {fc_missing}. "
            "The saved architecture does not match best_individual_details.csv."
        )

    trainer = TrainerAndEvaluation(model, train_loader, val_loader, test_loader)

    history_df = pd.read_csv(BEST_MODEL_DIR / "best_model_training_log.csv")
    history = {
        "train_loss": history_df["train_loss"].tolist(),
        "train_acc": history_df["train_accuracy"].tolist(),
        "val_loss": history_df["val_loss"].tolist(),
        "val_acc": history_df["val_accuracy"].tolist(),
    }

    plot_training_history(history, save_path=BEST_MODEL_DIR / "training_history.png")
    acc, loss = evaluation(model, trainer, save_dir=BEST_MODEL_DIR)
    print(f"Test accuracy: {acc:.4f}  Test loss: {loss:.4f}")


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand handler."""
    parser = argparse.ArgumentParser(
        description="Autism Classification — GoogLeNet + Genetic Algorithm"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("augment", help="Run data augmentation pipeline")
    sub.add_parser("preprocess", help="Run CLAHE+gamma preprocessing pipeline")

    train_p = sub.add_parser("train", help="Run GA hyperparameter optimisation and training")
    train_p.add_argument("--population", type=int, default=30, help="GA population size")
    train_p.add_argument("--ngen", type=int, default=15, help="Number of GA generations")
    train_p.add_argument("--num-workers", type=int, default=0, dest="num_workers")

    eval_p = sub.add_parser("evaluate", help="Evaluate the best saved model on the test set")
    eval_p.add_argument("--num-workers", type=int, default=0, dest="num_workers")

    args = parser.parse_args()
    {"augment": cmd_augment, "preprocess": cmd_preprocess,
     "train": cmd_train, "evaluate": cmd_evaluate}[args.command](args)


if __name__ == "__main__":
    main()
