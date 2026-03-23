# src/training/losses.py — Loss functions for brain tumor classification
"""
Implements loss functions used in the two-phase training protocol:
  - Cross-Entropy with label smoothing (Phase 1)
  - Focal Loss with per-class alpha weights (Phase 2)

Focal Loss (Lin et al., 2017) down-weights well-classified examples,
focusing on hard cases. Particularly important for the imbalanced
UCSF-PDGM grade distribution (55 II, 42 III, 403 IV).

References:
    - Lin et al. (2017). Focal Loss for Dense Object Detection.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher gamma down-weights easy examples more.
        Default 2.0.
    alpha : torch.Tensor | None
        Per-class weights. If None, uniform weighting.
    label_smoothing : float
        Label smoothing factor. Default 0.0.
    reduction : str
        Reduction mode: "mean", "sum", or "none".
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model outputs of shape (B, C).
        targets : torch.Tensor
            Class indices of shape (B,).

        Returns
        -------
        torch.Tensor
            Scalar loss value (if reduction="mean" or "sum")
        """
        num_classes = logits.size(1)

        # Apply label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(
                    logits, self.label_smoothing / (num_classes - 1)
                )
                smooth_targets.scatter_(
                    1, targets.unsqueeze(1), 1.0 - self.label_smoothing
                )
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma

        # Per-class alpha weight
        if self.alpha is not None:
            alpha_weight = self.alpha.unsqueeze(0).expand_as(logits)
            focal_weight = focal_weight * alpha_weight

        # Focal loss = -alpha * (1-p)^gamma * log(p)
        loss = -focal_weight * smooth_targets * log_probs
        loss = loss.sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def compute_class_weights(labels: torch.Tensor | list) -> torch.Tensor:
    """
    Compute inverse frequency class weights for loss balancing.

    Parameters
    ----------
    labels : torch.Tensor or list
        All training labels.

    Returns
    -------
    torch.Tensor
        Normalized class weights (higher weight for rarer classes).
    """
    if isinstance(labels, list):
        labels = torch.tensor(labels)

    classes, counts = torch.unique(labels, return_counts=True)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum() * len(classes)

    logger.info(
        "Class weights: %s (from counts %s)",
        {int(c): f"{w:.3f}" for c, w in zip(classes, weights, strict=False)},
        {int(c): int(n) for c, n in zip(classes, counts, strict=False)},
    )
    return weights


def build_loss(
    name: str,
    num_classes: int = 2,
    label_smoothing: float = 0.0,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """
    Factory for loss functions.

    Parameters
    ----------
    name : str
        Loss name: "cross_entropy" or "focal_loss".
    num_classes : int
        Number of classes.
    label_smoothing : float
        Label smoothing factor.
    gamma : float
        Focal loss focusing parameter.
    class_weights : torch.Tensor, optional
        Per-class weights.

    Returns
    -------
    nn.Module
        Loss function module.
    """
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    elif name == "focal_loss":
        return FocalLoss(
            gamma=gamma,
            alpha=class_weights,
            label_smoothing=label_smoothing,
        )
    else:
        raise ValueError(
            f"Unknown loss: {name}. Options: cross-entropy, focal_loss"
        )
