# src/training/callbacks.py — Training callbacks for Lightning
"""
Lightning callbacks implementing the training protocol:
  - Early stopping on val balanced accuracy (patience=10)
  - Model checkpointing by val AUC-ROC (save best)
  - LR monitoring for logging
  - Gradual backbone unfreezing every N epochs

The gradual unfreezing callback transitions from Phase 1 (head-only) to
Phase 2 (full fine-tuning) by progressively unfreezing backbone layers
from deepest to shallowest, every `unfreeze_every_n_epochs` epochs.

References:
    - Howard & Ruder (2018). Universal Language Model Fine-tuning (ULMFiT).
"""

from __future__ import annotations

import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

logger = logging.getLogger(__name__)


def build_callbacks(
    monitor_metric: str = "val/balanced_accuracy",
    checkpoint_metric: str = "val/auc_roc",
    patience: int = 10,
    checkpoint_dir: str = "outputs/models",
    save_top_k: int = 1,
    enable_gradual_unfreeze: bool = True,
    unfreeze_start_epoch: int = 10,
    unfreeze_every_n_epochs: int = 3,
) -> list[Callback]:
    """
    Build the standard set of training callbacks.

    Parameters
    ----------
    monitor_metric : str
        Metric for early stopping.
    checkpoint_metric : str
        Metric for model checkpointing.
    patience : int
        Early stopping patience in epochs.
    checkpoint_dir : str
        Directory for saving model checkpoints.
    save_top_k : int
        Number of best checkpoints to keep.
    enable_gradual_unfreeze : bool
        Whether to enable gradual backbone unfreezing.
    unfreeze_start_epoch : int
        Epoch at which to begin unfreezing (end of Phase 1).
    unfreeze_every_n_epochs : int
        Unfreeze one more layer group every N epochs.

    Returns
    -------
    list[Callback]
        List of Lightning callbacks.
    """
    callbacks = []

    # Early stopping on validation balanced accuracy
    callbacks.append(
        EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            mode="max",
            verbose=True,
        )
    )

    # Save best model by validation AUC
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:02d}-{val/auc_roc:.4f}",
            monitor=checkpoint_metric,
            mode="max",
            save_top_k=save_top_k,
            verbose=True,
        )
    )

    # LR monitoring for W&B logging
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # Gradual unfreezing
    if enable_gradual_unfreeze:
        callbacks.append(
            GradualUnfreezeCallback(
                unfreeze_start_epoch=unfreeze_start_epoch,
                unfreeze_every_n_epochs=unfreeze_every_n_epochs,
            )
        )

    logger.info(
        "Built %d callbacks: EarlyStopping(patient=%d), "
        "ModelCheckpoint(monitor=%s), LRMonitor%s",
        len(callbacks),
        patience,
        checkpoint_metric,
        ", GradualUnfreeze" if enable_gradual_unfreeze else "",
    )
    return callbacks


class GradualUnfreezeCallback(Callback):
    """
    Gradually unfreeze backbone layers during training.

    Implements the Phase 1 → Phase 2 transition:
    1. During Phase 1 (epochs 0 to unfreeze_start_epoch): backbone frozen.
    2. At unfreeze_start_epoch: unfreeze deepest layer group.
    3. Every unfreeze_every_n_epochs: unfreeze the next layer group.
    4. Eventually all layers are unfrozen for full fine-tuning.

    Layers are unfrozen from deepest to shallowest, so the model first
    adapts its deepest (most task-specific) features.

    Parameters
    ----------
    unfreeze_start_epoch : int
        Epoch to begin unfreezing.
    unfreeze_every_n_epochs : int
        Interval between unfreezing successive layer groups.
    """

    def __init__(
        self,
        unfreeze_start_epoch: int = 10,
        unfreeze_every_n_epochs: int = 3,
    ) -> None:
        super().__init__()
        self.unfreeze_start_epoch = unfreeze_start_epoch
        self.unfreeze_every_n_epochs = unfreeze_every_n_epochs
        self._layer_groups: list | None = None
        self._n_unfrozen: int = 0

    def on_train_epochs_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningDataModule,
    ) -> None:
        """
        Check whether to unfreeze the next layer group.

        Parameters
        ----------
        trainer : pl.Trainer
            Lightning trainer instance.
        pl_module : pl.LightningDataModule
            The TumorClassifier module.
        """
        current_epoch = trainer.current_epoch

        if current_epoch < self.unfreeze_start_epoch:
            return

        # Initialize layer groups on first unfreeze
        if self._layer_groups is None:
            if hasattr(pl_module, "get_backbone_layer_groups"):
                self._layer_groups = pl_module.get_backbone_layer_groups()
                logger.info(
                    "GradualUnfreeze: found %d layer groups in backbone",
                    len(self._layer_groups),
                )
            else:
                logger.warning(
                    "Model does not implement get_backbone_layer_groups(). "
                    "Unfreezing entire backbone at once."
                )
                pl_module.unfreeze_backbone()
                return

        # Determine how many groups should be unfrozen by now
        epochs_since_start = current_epoch - self.unfreeze_start_epoch
        target_unfrozen = (
            1 + epochs_since_start // self.unfreeze_every_n_epochs
        )
        target_unfrozen = min(target_unfrozen, len(self._layer_groups))

        # Unfreeze new groups if needed
        while self._n_unfrozen < target_unfrozen:
            group = self._layer_groups[self._n_unfrozen]
            for param in group:
                param.requires_grad = True
            self._n_unfrozen += 1
            logger.info(
                "Epoch %d: Unfroze layer group %d/%d (%d params)",
                current_epoch,
                self._n_unfrozen,
                len(self._layer_groups),
                sum(p.numel() for p in group),
            )
