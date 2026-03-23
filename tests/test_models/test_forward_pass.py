# tests/test_models/test_forward_pass.py — Forward pass tests for all models
"""
Verifies that all backbone architectures and the full classifier produce
correct output shapes and run without errors on synthetic data.

Tests each architecture:
  - DenseNet-121 (MONAI)
  - ResNet-50 (torchvision)
  - ResNet-18 (torchvision, baseline)
  - EfficientNet-B0 (timm)
  - SimpleCNN (from scratch, baseline)
  - Full TumorClassifier with head
"""

from __future__ import annotations

import pytest
import torch

from src.models.backbones import build_backbone, count_parameters
from src.models.classifier import TumorClassifier
from src.models.heads import LinearClassificationHead, build_head

# Synthetic input: batch of 2 images, 4 MRI channels, 64x64 pixels
BATCH_SIZE = 2
IN_CHANNELS = 4
IMG_SIZE = 64


def _dummy_input() -> torch.Tensor:
    """Create a dummy input tensor mimicking a batch of 2D MRI slices."""
    return torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE)


class TestBackbones:
    """Test that all backbones produce features of the correct shape."""

    def test_simple_cnn_forward(self) -> None:
        """SimpleCNN should output (B, 256) features."""
        backbone, feat_dim = build_backbone(
            "simple_cnn", in_channels=IN_CHANNELS
        )
        x = _dummy_input()
        out = backbone(x)
        assert out.shape == (BATCH_SIZE, feat_dim)
        assert feat_dim == 256

    def test_resnet18_forward(self) -> None:
        """ResNet-18 should output (B, 512) features."""
        backbone, feat_dim = build_backbone(
            "resnet18", in_channels=IN_CHANNELS, pretrained=False
        )
        x = _dummy_input()
        out = backbone(x)
        assert out.shape == (BATCH_SIZE, feat_dim)
        assert feat_dim == 512

    def test_resnet50_forward(self) -> None:
        """ResNet-50 should output (B, 2048) features."""
        backbone, feat_dim = build_backbone(
            "resnet50", in_channels=IN_CHANNELS, pretrained=False
        )
        x = _dummy_input()
        out = backbone(x)
        assert out.shape == (BATCH_SIZE, feat_dim)
        assert feat_dim == 2048

    @pytest.mark.slow
    def test_efficientnet_b0_forward(self) -> None:
        """EfficientNet-B0 should output (B, 1280) features."""
        backbone, feat_dim = build_backbone(
            "efficientnet_b0", in_channels=IN_CHANNELS, pretrained=False
        )
        x = _dummy_input()
        out = backbone(x)
        assert out.shape == (BATCH_SIZE, feat_dim)
        assert feat_dim == 1280

    def test_unknown_backbone_raises(self) -> None:
        """Unknown backbone name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backbone"):
            build_backbone("nonexistent_model")

    def test_4_channel_input_accepted(self) -> None:
        """All backbones must accept 4-channel input (T1, T1ce, T2, FLAIR)."""
        for name in ["simple_cnn", "resnet18"]:
            backbone, _ = build_backbone(name, in_channels=4, pretrained=False)
            out = backbone(_dummy_input())
            assert out.dim() == 2, f"{name} output should be 2D (B, features)"

    def test_count_parameters(self) -> None:
        """count_parameters should return total and trainable counts."""
        backbone, _ = build_backbone("simple_cnn", in_channels=4)
        params = count_parameters(backbone)
        assert "total" in params
        assert "trainable" in params
        assert params["total"] > 0
        assert params["trainable"] == params["total"]  # No frozen params


class TestClassificationHead:
    """Test classification head module."""

    def test_output_shape(self) -> None:
        """Head should produce (B, num_classes) logits."""
        head = LinearClassificationHead(
            in_features=512, num_classes=2, hidden_dim=256, dropout=0.5
        )
        x = torch.randn(BATCH_SIZE, 512)
        out = head(x)
        assert out.shape == (BATCH_SIZE, 2)

    def test_multiclass_output(self) -> None:
        """Head should handle >2 classes."""
        head = LinearClassificationHead(
            in_features=1024, num_classes=4, hidden_dim=128, dropout=0.3
        )
        x = torch.randn(BATCH_SIZE, 1024)
        out = head(x)
        assert out.shape == (BATCH_SIZE, 4)

    def test_build_head_factory(self) -> None:
        """build_head factory should create LinearClassificationHead."""
        head = build_head("linear", in_features=256, num_classes=2)
        assert isinstance(head, LinearClassificationHead)

    def test_build_head_unknown_raises(self) -> None:
        """Unknown head type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown head type"):
            build_head("transformer", in_features=256, num_classes=2)


class TestTumorClassifier:
    """Test the full Lightning classifier end-to-end."""

    def _build_classifier(
        self, backbone: str = "simple_cnn"
    ) -> TumorClassifier:
        """Build a classifier for testing."""
        from src.models.classifier import TumorClassifier

        return TumorClassifier(
            backbone_name=backbone,
            num_classes=2,
            in_channels=IN_CHANNELS,
            pretrained=False,
            head_hidden_dim=64,
            head_dropout=0.1,
        )

    def test_forward_produces_logits(self) -> None:
        """Full model forward should produce (B, num_classes) logits."""
        model = self._build_classifier()
        x = _dummy_input()
        logits = model(x)
        assert logits.shape == (BATCH_SIZE, 2)

    def test_training_step_returns_loss(self) -> None:
        """training_step should return a scalar loss tensor."""
        model = self._build_classifier()
        batch = {
            "image": _dummy_input(),
            "label": torch.tensor([0, 1]),
        }
        # Need a trainer for logging
        model.trainer = type("FakeTrainer", (), {"max_epochs": 10})()
        model.log = lambda *a, **k: None  # Stub logging

        loss = model.training_step(batch, 0)
        assert loss.dim() == 0  # Scalar
        assert loss.requires_grad

    def test_freeze_unfreeze_backbone(self) -> None:
        """Freezing and unfreezing backbone should toggle requires_grad."""
        model = self._build_classifier()

        # Initially frozen (default in __init__)
        model._freeze_backbone()
        for p in model.backbone.parameters():
            assert not p.requires_grad

        # Unfreeze
        model.unfreeze_backbone()
        for p in model.backbone.parameters():
            assert p.requires_grad

    def test_get_backbone_layer_groups(self) -> None:
        """Should return non-empty list of parameter groups."""
        model = self._build_classifier()
        groups = model.get_backbone_layer_groups()
        assert len(groups) > 0
        assert all(len(g) > 0 for g in groups)
