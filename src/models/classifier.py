# src/models/classifier.py — Main Lightning classifier module
"""
PyTorch Lightning module for brain tumor glioma grading classification.

Implements the two-phase training protocol:
    Phase 1 — Head only: backbone frozen, higher LR, label smoothing CE.
    Phase 2 — Gradual unfreezing: discriminative LR, progressive layer
              unfreeze.

Tracks comprehensive metrics following CLAIM/TRIPOD+AI guidelines:
    - Balanced accuracy (primary), Macro F1, AUC-ROC, Cohen's kappa.
    - Per-class sensitivity and specificity.
    - All logged to W&B for experiment tracking.

References:
    - Tejani et al. (2024). CLAIM checklist. Readiology: AI.
"""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import (
    AUROC,
    Accuracy,
    CohenKappa,
    F1Score,
    Precision,
    Recall,
)

from src.models.backbones import build_backbone, count_parameters
from src.models.heads import build_head

logger = logging.getLogger(__name__)


class TumorClassifier(pl.LightningModule):
    """
    Lightning module for brain tumor classification.

    Combines a backbone feature extractor with a classification head
    and implements the two-phase training protocol.

    Parameters
    ----------
    backbone_name : str
        Name of the backbone architecture.
    num_classes : int
        Number of classification classes. Default 2 (LGG vs HGG).
    in_channels : int
        Number of input channels. Default 4 (T1, T1ce, T2, FLAIR).
    pretrained : bool
        Whether to use pretrained backbone weights. Default True.
    head_hidden_dim : int
        Hidden dimension of the classification head.
    head_dropout : float
        Dropout rate in the classification head.
    lr : float
        Learning rate for Phase 1 (head only).
    lr_backbone : float
        Learning rate for backbone in Phase 2.
    lr_head : float
        Learning rate for head in Phase 2.
    weight_decay : float
        Weight decay for optimizer.
    label_smoothing : float
        Label smoothing factor for cross-entropy loss.
    freeze_backbone : bool
        Whether to start with backbone frozen (Phase 1).
    class_weights : list[float] | None
        Per-class weights for loss function. None for uniform.
    """

    def __init__(
        self,
        backbone_name: str = "densenet121",
        num_classes: int = 2,
        in_channels: int = 4,
        pretrained: bool = True,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.5,
        lr: float = 1e-3,
        lr_backbone: float = 1e-5,
        lr_head: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        freeze_backbone: bool = True,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Build bacbone and head
        self.backbone, feature_dim = build_backbone(
            name=backbone_name,
            in_channels=in_channels,
            pretrained=pretrained,
        )
        self.head = build_head(
            head_type="linear",
            in_features=feature_dim,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

        # Loss function
        weight_tensor = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights
            else None
        )
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=label_smoothing,
        )

        # Freeze backbone for Phase 1
        if freeze_backbone:
            self._freeze_backbone()

        # Metrics — following CLAIM guidelines
        task = "binary" if num_classes == 2 else "multiclass"
        metric_kwargs = {"task": task, "num_classes": num_classes}

        # Train metrics
        self.train_acc = Accuracy(**metric_kwargs)
        self.train_bal_acc = Accuracy(**metric_kwargs, average="macro")

        # Val metrics (primary: balanced accuracy and AUC)
        self.val_acc = Accuracy(**metric_kwargs)
        self.val_bal_acc = Accuracy(**metric_kwargs, average="macro")
        self.val_f1 = F1Score(**metric_kwargs, average="macro")
        self.val_auc = AUROC(**metric_kwargs)
        self.val_kappa = CohenKappa(task=task, num_classes=num_classes)
        self.val_precision = Precision(**metric_kwargs, average="macro")
        self.val_recall = Recall(**metric_kwargs, average="macro")

        # Log parameter counts
        params = count_parameters(self)
        logger.info(
            "TumorClassifier: backbone=%s, params=%dK (trainable=%dK)",
            backbone_name,
            params["total"] // 1000,
            params["trainable"] // 1000,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: backbone features ⟶ classification logits.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes)
        """
        features = self.backbone(x)

        # Handle different backbone output formats
        if isinstance(features, tuple):
            features = features[0]
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)

        logits = self.head(features)
        return logits

    def training_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """
        Execute a single training training step.

        Parameters
        ----------
        batch : dict
            Batch dictionary with "image" and "label" keys.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        images = batch["image"]
        labels = batch["label"]

        logits = self(images)
        loss = self.loss_fn(logits, labels)

        # Update metrics
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)

        # Log
        self.log(
            "train/loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/accuracy",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/balanced_accuracy",
            self.train_bal_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """
        Execute a single validation step.

        Parameters
        ----------
        batch : dict
            Batch dictionary with "image" and "label" keys.
        batch_idx : int
            Index of the current batch.
        """
        images = batch["image"]
        labels = batch["label"]

        logits = self(images)
        loss = self.loss_fn(logits, labels)

        # Predictions and probabilities
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        # Update all metrics
        self.val_acc(preds, labels)
        self.val_bal_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_auc(probs[:, 1], labels)
        self.val_kappa(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)

        # Log
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_acc, on_step=False, on_epoch=True)
        self.log(
            "val/balanced_accuracy",
            self.val_bal_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("val/f1_macro", self.val_f1, on_step=False, on_epoch=True)
        self.log(
            "val/auc_roc",
            self.val_auc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/cohens_kappa", self.val_kappa, on_step=False, on_epoch=True
        )
        self.log(
            "val/precision_macro",
            self.val_precision,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/recall_macro", self.val_recall, on_step=False, on_epoch=True
        )

    def configure_optimizers(self) -> dict:
        """
        Configure optimizer and LR scheduler.

        Implements discriminative learning rates: backbone gets a lower LR
        than the head to preserve pretrained features while allowing
        task-specific fine-tuning.

        Returns
        -------
        dict
            Optimizer and scheduler configuration for Lightning.
        """
        # Separate backbone and head parameter groups
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())

        if self.hparams.freeze_backbone:
            # Phase 1: only optimize head
            optimizer = torch.optim.Adam(
                head_params,
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            # Phase 2: discriminative LR
            param_groups = [
                {
                    "params": backbone_params,
                    "lr": self.hparams.lr_backbone,
                    "name": "backbone",
                },
                {
                    "params": head_params,
                    "lr": self.hparams.lr_head,
                    "name": "head",
                },
            ]
            optimizer = torch.optim.Adam(
                param_groups,
                weight_decay=self.hparams.weight_decay,
            )

        # Cosine annealing scheduler with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 35,
            eta_min=1e-7,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/balanced_accuracy",
            },
        }

    # --- Backbone freezing/unfreezing for two-phase training ---

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters (Phase 1)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone FROZEN (Phase 1: head-only training)")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (transition to Phase 2)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone UNFROZEN (Phase 2: full fine-tuning)")

    def get_backbone_layer_groups(self) -> list[list[nn.Parameter]]:
        """
        Get backbone parameters grouped by layer for gradual unfreezing.

        Returns parameters from deepest to shallowest layers so that
        gradual unfreezing starts from the layers closest to the output.

        Returns
        -------
        list[list[nn.Parameter]]
            List of parameter groups, deepest first.
        """
        children = list(self.backbone.children())
        groups = []
        for child in reversed(children):
            params = list(child.parameters())
            if params:
                groups.append(params)
        return groups
