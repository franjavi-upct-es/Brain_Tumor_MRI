# src/models/backbones.py — Backbone network factory
"""
Factory for creating backbone feature extractors for brain tumor
classification.

Supported architectures:
    - DenseNet-121 (MONAI): ~8M params, preferred in MONAI and RadImageNet.
      Efficient feature reuse via dense connections.
    - ResNet-50 (torchvision): ~25 params, most widely used standard.
      Facilitates comparison with published literature.
    - EfficientNet-B0 (timm): ~5M params, best efficiency-to-performance ratio.
      Continuity with v1 project for direct comparison.
    - SimpleCNN: 4-5 layer CNN from scratch. Baseline to measure transfer
      learning contribution (Raghu et al. 2019 NeurIPS).
    - ResNet-18 (torchvision): lightweight pretrained baseline.

All pretrained backbones adapt their first convolutional layer to accept
4 input channels (T1, T1ce, T2, FLAIR) instead of the default 3 (RGB).

References:
    - Raghu et al. (2019). Transfusion: Understanding Transfer Learning
      for Medical Imaging. NeurIPS.
    - Mei et al. (2022). RadImageNet. Radiology: AI, 4(5).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_backbone(
    name: str,
    in_channels: int = 4,
    pretrained: bool = True,
    spatial_dims: int = 2,
) -> tuple[nn.Module, int]:
    """
    Build a backbone feature extractor.

    Parameters
    ----------
    name : str
        Backbone name: "densenet121", "resnet50", "resnet18",
        "efficientnet_b0", or "simple_cnn".
    in_channels : int
        Number of input channels. Default 4 for MRI modalities.
    pretrained : bool
        Whether to use ImageNet pretrained weights.
    spatial_dims : int
        Spatial dimensions (2 for 2D slices).

    Returns
    -------
    tuple[nn.Module, int]
        (backbone_module, feature_dim) where feature_dim is the number
        of features output by the backbone's global pooling.
    """
    name = name.lower().strip()

    if name == "densenet121":
        return _build_densenet121(in_channels, pretrained, spatial_dims)
    elif name == "resnet50":
        return _build_resnet50(in_channels, pretrained)
    elif name == "resnet18":
        return _build_resnet18(in_channels, pretrained)
    elif name == "efficientnet_b0":
        return _build_efficientnet_b0(in_channels, pretrained)
    elif name == "simple_cnn":
        return _build_simple_cnn(in_channels)
    else:
        raise ValueError(
            f"Unknown backbone: {name}. "
            "Options: densenet121, resnet50, resnet18, efficientnet_b0, "
            "simple_cnn"
        )


def _build_densenet121(
    in_channels: int,
    pretrained: bool,
    spatial_dims: int,
) -> tuple[nn.Module, int]:
    """
    Build denseNet-121 using MONAI's implementation.

    MONAI's DenseNet supports configurable spatial_dims and in_channels
    natively, making it ideal for medical imaging.

    Returns feature_dim = 1024.
    """
    from monai.networks.nets import DenseNet121

    model = DenseNet121(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=1024,  # Feature dimension before classification head
        pretrained=pretrained,
    )

    # MONAI DenseNet121 returns features from its class_layers
    # We need to remove the final classification layer to get features
    # The model outputs 1024-dim features after global avg pool
    feature_dim = 1024

    logger.info(
        (
            "Built DenseNet-121 (MONAI): in_channels=%d, "
            "pretrained=%s, features=$d"
        ),
        in_channels,
        pretrained,
        feature_dim,
    )
    return model, feature_dim


def _build_resnet50(
    in_channels: int, pretrained: bool
) -> tuple[nn.Module, int]:
    """
    Build ResNet-50 using torchvision with adapted input layer.

    Adapts the first conv layer from 3⟶in_channels by averaging
    pretrained weights across the channel dimension and replicating.
    """
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)

    # Adapt first conv layer for 4-channel input
    if in_channels != 3:
        model.conv1 = _adapt_first_conv(model.conv1, in_channels, pretrained)

    # Remove the final FC layer — we'll attach our own head
    feature_dim = model.fc.in_features  # 2048
    model.fc = nn.Identity()

    logger.info(
        (
            "Built ResNet-50 (torchvision): in_channels=%d, "
            "pretrained=%s, features=%d"
        ),
        in_channels,
        pretrained,
        feature_dim,
    )
    return model, feature_dim


def _build_resnet18(
    in_channels: int, pretrained: bool
) -> tuple[nn.Module, int]:
    """Build ResNet-18 as a lightweight pretrained baseline."""
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    if in_channels != 3:
        model.conv1 = _adapt_first_conv(model.conv1, in_channels, pretrained)

    feature_dim = model.fc.in_features  # 512
    model.fc = nn.Identity()

    logger.info(
        (
            "Built ResNet-18 (torchvision): in_channels=%d, "
            "pretrained=%s, features=%d"
        ),
        in_channels,
        pretrained,
        feature_dim,
    )
    return model, feature_dim


def _build_efficientnet_b0(
    in_channels: int, pretrained: bool
) -> tuple[nn.Module, int]:
    """
    Build EfficientNet-B0 using timm library.

    timm handles in_channels adaptation automatically when specified.
    """
    import timm

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=0,  # Remove classification head, return features
    )

    feature_dim = model.num_features  # 1280

    logger.info(
        (
            "Built EfficientNet-B0 (timm): in_channels=%d, pretrained=%s, "
            "features=%d"
        ),
        in_channels,
        pretrained,
        feature_dim,
    )
    return model, feature_dim


def _build_simple_cnn(in_channels: int) -> tuple[nn.Module, int]:
    """
    Build a simple 4-layer CNN from scratch.

    Raghu et al. (2019, NeurIPS) found that simple CNNs perform comparably
    to large pretrained models on medical imaging, with the main benefit
    of pretraining being convergence speed, not final accuracy.

    Architecture: 4 conv blocks with BatchNorm + ReLU + MaxPool,
    followed by adaptive average pooling.
    """
    feature_dim = 256

    model = nn.Sequential(
        # Block 1: in_channels ⟶ 32
        nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Block 2: 32 ⟶ 64
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Block 3: 64 ⟶ 128
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Block 4: 128 ⟶ 256
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        # Flattern
        nn.Flatten(),
    )

    logger.info(
        "Built SimpleCNN: in_channels=%d, features=%d (no pretraining)",
        in_channels,
        feature_dim,
    )
    return model, feature_dim


def _adapt_first_conv(
    original_conv: nn.Conv2d, new_in_channels: int, pretrained: bool
) -> nn.Conv2d:
    """
    Adapat first conv layer to accept a different number of input channels.

    Strategy: create a new conv with new_in_channels, and if pretrained
    initialize by averaging the original 3-channel weights and replicating.

    Parameters
    ----------
    original_conv : nn.Conv2d
        Original first convolutional layer (3 input channels).
    new_in_channels : int
        Desired number of input channels.
    pretrained : bool
        Whether to transfer averaged weights

    Returns
    -------
    nn.Conv2d
        New convolutional layer with adapted input channels.
    """
    new_conv = nn.Conv2d(
        new_in_channels,
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )

    if pretrained:
        with torch.no_grad():
            # Average across the 3 RGB channels and replicate
            avg_weight = original_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(avg_weight.repeat(1, new_in_channels, 1, 1))
            if original_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)

    logger.debug(
        "Adapted first conv: %d ⟶ %d channels (pretrained=%s)",
        original_conv.in_channels,
        new_in_channels,
        pretrained,
    )
    return new_conv


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Count model parameters (total and trainable).

    Parameters
    ----------
    model : nn.Module
        The model to count parameters for.

    Returns
    -------
    dict[str, int]
        Dictionary with 'total' and 'trainable' parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
