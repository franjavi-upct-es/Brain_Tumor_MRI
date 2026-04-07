# tests/test_training/test_losses.py
"""Tests for FocalLoss, compute_class_weights, and build_loss."""

from __future__ import annotations

import pytest
import torch

from src.training.losses import FocalLoss, build_loss, compute_class_weights


class TestFocalLoss:
    """Tests for FocalLoss module."""

    def test_forward_basic_shape(self) -> None:
        """Forward pass should return scalar with reduction=mean."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_reduction_sum(self) -> None:
        """reduction=sum should return scalar."""
        loss_fn = FocalLoss(gamma=2.0, reduction="sum")
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_reduction_none(self) -> None:
        """reduction=none should return per-sample losses."""
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, targets)
        assert loss.shape == (4,)

    def test_with_alpha(self) -> None:
        """FocalLoss with per-class alpha weights should run without error."""
        alpha = torch.tensor([0.25, 0.75])
        loss_fn = FocalLoss(gamma=2.0, alpha=alpha)
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_with_label_smoothing(self) -> None:
        """FocalLoss with label smoothing should run without error."""
        loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.1)
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_gradient_flows(self) -> None:
        """Loss should be differentiable."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 2, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_three_classes(self) -> None:
        """Should work with more than 2 classes."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(6, 3)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_no_alpha_attribute(self) -> None:
        """When alpha=None, self.alpha should be None."""
        loss_fn = FocalLoss(gamma=2.0, alpha=None)
        assert loss_fn.alpha is None


class TestComputeClassWeights:
    """Tests for compute_class_weights function."""

    def test_balanced_classes(self) -> None:
        """Equal class counts should give equal weights."""
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        weights = compute_class_weights(labels)
        assert weights.shape == (2,)
        assert abs(weights[0].item() - weights[1].item()) < 1e-5

    def test_imbalanced_classes(self) -> None:
        """Rarer class should get higher weight."""
        labels = torch.tensor([0, 0, 0, 0, 1, 1])  # 4 vs 2
        weights = compute_class_weights(labels)
        assert weights[1] > weights[0]  # Fewer samples → more weight

    def test_list_input(self) -> None:
        """Should accept list input."""
        labels = [0, 0, 1, 1]
        weights = compute_class_weights(labels)
        assert weights.shape == (2,)

    def test_returns_tensor(self) -> None:
        """Should return a torch.Tensor."""
        labels = torch.tensor([0, 1, 0, 1])
        weights = compute_class_weights(labels)
        assert isinstance(weights, torch.Tensor)


class TestBuildLoss:
    """Tests for build_loss factory."""

    def test_cross_entropy(self) -> None:
        """Should build CrossEntropyLoss."""
        import torch.nn as nn
        loss = build_loss("cross_entropy", num_classes=2, label_smoothing=0.1)
        assert isinstance(loss, nn.CrossEntropyLoss)

    def test_focal_loss(self) -> None:
        """Should build FocalLoss."""
        loss = build_loss("focal_loss", num_classes=2, gamma=2.0)
        assert isinstance(loss, FocalLoss)

    def test_focal_loss_with_weights(self) -> None:
        """Should pass class_weights to FocalLoss."""
        weights = torch.tensor([0.5, 0.5])
        loss = build_loss("focal_loss", num_classes=2, class_weights=weights)
        assert isinstance(loss, FocalLoss)

    def test_unknown_loss_raises(self) -> None:
        """Unknown loss name should raise ValueError."""
        with pytest.raises(ValueError):
            build_loss("unknown_loss")
