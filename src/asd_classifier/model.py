"""Model definitions for the autism classifier.

Contains ``EarlyStopping`` (val-loss monitor with best-weight restoration) and
``GoogleNet`` (pretrained GoogLeNet backbone with a GA-configured FC head for
binary TC/TS classification).  Only the FC head parameters are trained; the
backbone is frozen.
"""

import copy
import sys
from io import StringIO

import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

from asd_classifier.config import (
    BASE_LEARNING_RATE,
    FC_BLOCKS_LEARNING_RATE,
    WEIGHT_DECAY,
)


class EarlyStopping:
    """Halt training when validation loss stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change in val loss to count as improvement.
        restore_best_weights: If True, load the best-seen weights on trigger.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model: nn.Module, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            return False

        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}."
        else:
            self.counter += 1
            self.status = f"No improvement found, counter increased to {self.counter}."
            if self.counter >= self.patience:
                self.status = "Early stopping triggered."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False


class GoogleNet(nn.Module):
    """Pretrained GoogLeNet with a GA-configured fully-connected head.

    The backbone (all layers up to and including the pool before ``fc``) is
    frozen.  The ``fc`` layer is replaced with ``nn.Identity`` so raw 1024-D
    features flow into ``fc_blocks``, which is built from the GA individual's
    FC genes.  Only ``fc_blocks`` parameters are trained.

    Args:
        individual: GA individual ``[[control_gene], [fc_gene, ...]]``.
        image_size: Spatial dimensions used for the model summary.
        base_learning_rate: LR for backbone (unused — backbone is frozen).
        fc_blocks_learning_rate: LR applied to ``fc_blocks`` by the optimizer.
        weight_decay: AdamW weight decay for ``fc_blocks``.
    """

    def __init__(
        self,
        individual: list,
        image_size: tuple[int, int] = (224, 224),
        base_learning_rate: float = BASE_LEARNING_RATE,
        fc_blocks_learning_rate: float = FC_BLOCKS_LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
    ):
        super().__init__()
        self.individual = individual
        self.image_size = image_size
        self.base_learning_rate = base_learning_rate
        self.fc_blocks_learning_rate = fc_blocks_learning_rate
        self.weight_decay = weight_decay

        control_gene = individual[0]
        if isinstance(control_gene[0], list):
            self.max_fc_layers = control_gene[0][0]
            self.num_epochs = control_gene[0][1]
        else:
            self.max_fc_layers = control_gene[0]
            self.num_epochs = control_gene[1]

        self.fc_genes = individual[1]

        self.base = models.googlenet(pretrained=True, aux_logits=True)
        self.base.fc = nn.Identity()
        for param in self.base.parameters():
            param.requires_grad = False

        fc_blocks: list[nn.Module] = []
        fc_layers = min(self.max_fc_layers, len(self.fc_genes))
        in_features = 1024

        for i in range(fc_layers):
            layer_type, num_neurons, dropout = self.fc_genes[i]
            fc_blocks.append(nn.Linear(in_features, num_neurons))
            fc_blocks.append(nn.ReLU())
            fc_blocks.append(nn.BatchNorm1d(num_neurons))
            if layer_type == 2:
                fc_blocks.append(nn.Dropout(dropout / 100.0))
            in_features = num_neurons

        self.output = nn.Linear(in_features, 2)
        fc_blocks.append(self.output)
        self.fc_blocks = nn.Sequential(*fc_blocks)

        for param in self.fc_blocks.parameters():
            param.requires_grad = True

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features then apply the FC head."""
        features = self.base(x)
        return self.fc_blocks(features)

    def _initialize_weights(self) -> None:
        """Kaiming-normal init for Linear layers; constant init for BatchNorm1d."""
        for m in self.fc_blocks.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Return an AdamW optimizer scoped to ``fc_blocks`` parameters only."""
        return torch.optim.AdamW([
            {
                "params": self.fc_blocks.parameters(),
                "lr": self.fc_blocks_learning_rate,
                "weight_decay": self.weight_decay,
            }
        ])

    def model_summary(self, file_path: str | None = None) -> str:
        """Return a torchinfo summary string; optionally write it to *file_path*."""
        buf = StringIO()
        original_stdout = sys.stdout
        sys.stdout = buf
        summary(self, input_size=(1, 3, self.image_size[0], self.image_size[1]), verbose=0)
        sys.stdout = original_stdout
        output = buf.getvalue()
        if file_path:
            with open(file_path, "w") as f:
                f.write(output)
        return output
