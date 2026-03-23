# src/models/heads.py — Classification head with dropout
"""
Classification heads that sit on top of backbone feature extractors.

Supports configurable hidden dimension and dropout rate. The head takes
the feature vector from the backbone's global average pooling and produces
class logits.
"""

from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class LinearClassificationHead(nn.Module):
    """
    Two-layer classification head with dropout.

    Architecture: Linear → ReLU → Dropout → Linear → logits

    Parameters
    ----------
    in_features : int
        Number of input features from the backbone.
    num_classes : int
        Number of output classes.
    hidden_dim : int
        Hidden layer dimension. Default 256.
    dropout : float
        Dropout probability. Default 0.5.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        logger.info(
            "ClassificationHead: %d → %d → %d (dropout=%.2f)",
            in_features,
            hidden_dim,
            num_classes,
            dropout,
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Feature vector of shape (B, in_features).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        return self.head(x)


def build_head(
    head_type: str,
    in_features: int,
    num_classes: int,
    hidden_dim: int = 256,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Factory for classification heads.

    Parameters
    ----------
    head_type : str
        Head type. Currently only "linear" is supported.
    in_features : int
        Input feature dimension from backbone.
    num_classes : int
        Number of output classes.
    hidden_dim : int
        Hidden layer size.
    dropout : float
        Dropout rate.

    Returns
    -------
    nn.Module
        Classification head module.
    """
    if head_type == "linear":
        return LinearClassificationHead(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}. Options: 'linear'")
