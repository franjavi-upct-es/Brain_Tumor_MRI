# tests/test_interpretability/test_gradcam.py — Grad-CAM module tests
"""
Verifies Grad-CAM generation, heatmap binarization, target layer detection,
and end-to-end heatmap generation on the SimpleCNN backbone.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.interpretability.gradcam import (
    GradCAMGenerator,
    binarize_heatmap,
    get_target_layer,
)


class TestBinarizeHeatmap:
    """Tests for heatmap binarization."""

    def test_default_threshold_half_max(self) -> None:
        """Default threshold_ratio=0.5 should binarize at 0.5 × max."""
        heatmap = np.array([[0.0, 0.3, 0.6], [0.8, 1.0, 0.4]])
        binary = binarize_heatmap(heatmap, 0.5)
        expected = np.array([[False, False, True], [True, True, False]])
        np.testing.assert_array_equal(binary, expected)

    def test_returns_boolean_array(self) -> None:
        """Output should be a boolean numpy array."""
        heatmap = np.random.rand(10, 10).astype(np.float32)
        binary = binarize_heatmap(heatmap)
        assert binary.dtype == bool

    def test_custom_threshold(self) -> None:
        """Custom threshold ratio should change binarization."""
        heatmap = np.array([[0.1, 0.5, 0.9]])
        binary_low = binarize_heatmap(heatmap, 0.2)  # threshold = 0.18
        binary_high = binarize_heatmap(heatmap, 0.8)  # threshold = 0.72
        assert binary_low.sum() >= binary_high.sum()


class TestGetTargetLayer:
    """Tests for automatic target layer detection."""

    def test_simple_cnn_target_layer(self) -> None:
        """Should find the last Conv2d in SimpleCNN."""
        from src.models.backbones import build_backbone

        backbone, _ = build_backbone("simple_cnn", in_channels=4)

        # Wrap in a mock model
        class MockModel(nn.Module):
            def __init__(self, bb):
                super().__init__()
                self.backbone = bb

        model = MockModel(backbone)

        layer = get_target_layer(model, "simple_cnn")
        assert isinstance(layer, nn.Conv2d)

    def test_resnet18_target_layer(self) -> None:
        """Should find layer4 for ResNet-18."""
        from src.models.backbones import build_backbone

        backbone, _ = build_backbone(
            "resnet18", in_channels=4, pretrained=False
        )

        class MockModel(nn.Module):
            def __init__(self, bb):
                super().__init__()
                self.backbone = bb

        model = MockModel(backbone)

        layer = get_target_layer(model, "resnet18")
        assert layer is not None

    def test_unknown_backbone_fallback(self) -> None:
        """Should fallback to last Conv2d for unknown architectures."""

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 16, 3)
                self.fc = nn.Linear(16, 2)

        net = SimpleNet()

        class MockModel(nn.Module):
            def __init__(self, bb):
                super().__init__()
                self.backbone = bb

        model = MockModel(net)

        layer = get_target_layer(model, "unknown_arch")
        assert isinstance(layer, nn.Conv2d)


class TestGradCAMGenerator:
    """End-to-end tests for Grad-CAM generation."""

    def _build_simple_model(self):
        """Build a small model for testing Grad-CAM."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(4, 8, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Linear(8, 2)

            def forward(self, x):
                f = self.features(x)
                f = f.flatten(1)
                return self.classifier(f)

        return TinyModel()

    def test_generates_correct_shape(self) -> None:
        """Heatmap should match input spatial dimensions."""
        model = self._build_simple_model()
        target_layer = model.features[0]  # First conv

        gen = GradCAMGenerator(model, target_layer, use_captum=False)
        image = torch.randn(1, 4, 32, 32)
        heatmap = gen.generate(image, target_class=0)

        assert heatmap.shape == (32, 32)

    def test_heatmap_values_normalized(self) -> None:
        """Heatmap values should be in [0, 1]."""
        model = self._build_simple_model()
        target_layer = model.features[0]

        gen = GradCAMGenerator(model, target_layer, use_captum=False)
        image = torch.randn(1, 4, 32, 32)
        heatmap = gen.generate(image)

        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0 + 1e-6

    def test_3d_input_accepted(self) -> None:
        """Should accept (C, H, W) input and add batch dim automatically."""
        model = self._build_simple_model()
        target_layer = model.features[0]

        gen = GradCAMGenerator(model, target_layer, use_captum=False)
        image = torch.randn(4, 32, 32)  # No batch dim
        heatmap = gen.generate(image)

        assert heatmap.shape == (32, 32)

    def test_batch_generation(self) -> None:
        """generate_batch should produce one heatmap per image."""
        model = self._build_simple_model()
        target_layer = model.features[0]

        gen = GradCAMGenerator(model, target_layer, use_captum=False)
        images = torch.randn(3, 4, 32, 32)
        heatmaps = gen.generate_batch(images)

        assert len(heatmaps) == 3
        for hm in heatmaps:
            assert hm.shape == (32, 32)
