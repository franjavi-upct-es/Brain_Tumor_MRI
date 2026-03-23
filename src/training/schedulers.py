# src/training/schedulers.py — Learning rate schedulers
"""
Learning rate scheduling for the two-phase training protocol.

Implements cosine annealing with linear warmup:
  - Warmup: LR linearly increases from 0 to target over warmup_epochs.
  - Cosine annealing: LR decreases following a cosine curve to min_lr.
"""

from __future__ import annotations

import logging
import math

import torch
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine annealing LR scheduler with linear warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    warmup_epochs : int
        Number of warmup epochs with linearly increasing LR.
    max_epochs : int
        Total number of training epochs.
    min_lr : float
        Minimum learning rate at the end of cosine annealing.
    last_epoch : int
        The index of the last epoch.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 3,
        max_epochs: int = 35,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """
        Compute learning rate for the current epoch.

        Returns
        -------
        list[float]
            Learning rate for each parameter group.
        """
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (self.last_epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                self.max_epochs - self.warmup_epochs, 1
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "cosine_annealing",
    warmup_epochs: int = 3,
    max_epochs: int = 35,
    min_lr: float = 1e-7,
) -> _LRScheduler:
    """
    Factory for LR schedulers.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule
    name : str
        Scheduler name.
    warmup_epochs : int
        Number of warmup epochs.
    max_epochs : int
        Total training epochs.
    min_lr : float
        Minimum LR.

    Returns
    -------
    _LRScheduler
        Learning rate scheduler.
    """
    if name == "cosine_annealing":
        return CosineAnnealingWithWarmup(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            min_lr=min_lr,
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")
