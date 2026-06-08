"""Training and evaluation loop for the GoogLeNet autism classifier.

``TrainerAndEvaluation`` wraps a ``GoogleNet`` model with train/val/test
loops, early stopping, optional gradient clipping, and a
``ReduceLROnPlateau`` scheduler.  Training history (loss, accuracy) is
accumulated in ``self.history`` and returned by ``train()``.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from asd_classifier.config import (
    BASE_LEARNING_RATE,
    FC_BLOCKS_LEARNING_RATE,
    LABEL_SMOOTHING,
    WEIGHT_DECAY,
)
from asd_classifier.model import EarlyStopping, GoogleNet


class TrainerAndEvaluation:
    """Manages the train/val/test lifecycle for a ``GoogleNet`` model.

    Args:
        model: The ``GoogleNet`` instance to train.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        test_loader: DataLoader for the test split.
        clip_grad: Max gradient norm for clipping; ``None`` disables clipping.
        label_smoothing: Cross-entropy label smoothing factor.
        use_lr_scheduler: Enable ``ReduceLROnPlateau`` on validation accuracy.
        verbose: Print per-epoch metrics to stdout.
        save_path: Checkpoint path for the best model weights.
        base_learning_rate: Backbone LR (passed through; backbone is frozen).
        fc_blocks_learning_rate: LR for the FC head.
        weight_decay: AdamW weight decay.
    """

    def __init__(
        self,
        model: GoogleNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        clip_grad: float | None = None,
        label_smoothing: float = LABEL_SMOOTHING,
        use_lr_scheduler: bool = True,
        verbose: bool = False,
        save_path: str = "temp_model.pth",
        base_learning_rate: float = BASE_LEARNING_RATE,
        fc_blocks_learning_rate: float = FC_BLOCKS_LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.clip_grad = clip_grad
        self.label_smoothing = label_smoothing
        self.use_lr_scheduler = use_lr_scheduler
        self.verbose = verbose
        self.save_path = save_path
        self.base_learning_rate = base_learning_rate
        self.fc_blocks_learning_rate = fc_blocks_learning_rate
        self.weight_decay = weight_decay

        self.history: dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": [],
        }

        self._initialize_optimizer_and_scheduler()

    def _initialize_optimizer_and_scheduler(self) -> None:
        """Set up loss function, optimizer, accuracy metric, scheduler, and early stopping."""
        self.loss_fxn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = self.model.get_optimizer()
        self.accuracy = Accuracy(task="multiclass", num_classes=2).to(self.device)

        if self.use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=2, factor=0.5
            )
        else:
            self.scheduler = None

        self.early_stopping = EarlyStopping()

    def _compute_loss_and_update_acc(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute cross-entropy loss and update the running accuracy metric."""
        pred = self.model(x)
        y = y.long()
        loss = self.loss_fxn(pred, y)
        self.accuracy.update(pred.softmax(dim=1), y)
        return loss

    def _training_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """One gradient-update step for a single batch."""
        loss = self._compute_loss_and_update_acc(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        return loss

    def _val_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Loss computation under ``torch.no_grad()`` for validation/test batches."""
        with torch.no_grad():
            return self._compute_loss_and_update_acc(x, y)

    def _common_step(self, loader: DataLoader, step) -> tuple[float, float]:
        """Iterate *loader*, apply *step* per batch, return (mean_loss, accuracy)."""
        self.accuracy.reset()
        total_loss = 0.0
        total_samples = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            batch_loss = step(x, y)
            total_loss += batch_loss.item() * x.size(0)
            total_samples += x.size(0)
        return total_loss / total_samples, self.accuracy.compute().item()

    def _validation_step(self, data_loader: DataLoader) -> dict:
        """Evaluate the model on *data_loader* and return accuracy, loss, labels, and predictions."""
        self.model.to(self.device)
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels: list = []
        all_preds: list = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fxn(outputs, labels.long())
                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)

        return {
            "accuracy": correct / total,
            "loss": total_loss / total,
            "labels": all_labels,
            "predictions": all_preds,
        }

    def _log_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        """Append epoch metrics to history, step the LR scheduler, and optionally print."""
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

        if self.scheduler:
            self.scheduler.step(val_acc)

        if self.verbose:
            print(
                f"[Epoch {epoch}] Train: [Loss: {train_loss:.3f}, Acc: {train_acc:.3f}] "
                f"Val: [Loss: {val_loss:.3f}, Acc: {val_acc:.3f}]"
            )

    def train(self) -> dict[str, list]:
        """Run the full training loop and return the accumulated history dict.

        Trains for ``model.num_epochs`` epochs, applies early stopping, saves
        the best checkpoint to ``self.save_path``, then restores those weights
        before returning.
        """
        best_val_acc = -1.0
        saved = False

        for epoch in range(self.model.num_epochs):
            self.model.train()
            self.model.base.eval()
            train_loss, train_acc = self._common_step(self.train_loader, self._training_step)

            self.model.eval()
            val_loss, val_acc = self._common_step(self.val_loader, self._val_step)

            self._log_metrics(epoch + 1, train_loss, train_acc, val_loss, val_acc)

            # Fitness is best val accuracy, so checkpoint on val-acc improvement.
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)
                saved = True

            if self.early_stopping(self.model, val_loss):
                break

        if saved:
            self.model.load_state_dict(torch.load(self.save_path, weights_only=True))
        return self.history

    def validation(self) -> dict:
        """Evaluate on the validation split; returns accuracy, loss, labels, predictions."""
        return self._validation_step(self.val_loader)

    def test(self) -> dict:
        """Evaluate on the test split; returns accuracy, loss, labels, predictions."""
        return self._validation_step(self.test_loader)
