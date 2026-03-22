# src/interpretability/gradcam.py — Grad-CAM via Captum
"""
Generates Grad-CAM attribution maps using the Captum library to visualize
which image regions the model uses for classification decisions.

Grad-CAM (Selvaraju et al., 2017) computes a weighted combination of
feature map activations from a target convolutional layer, using the
gradients of the target class score as weights.

The critical use in this thesis is NOT just visualization but VALIDATION:
Grad-CAM maps are binarized and compared against ground truth tumor
segmentations via IoU to prove the model attends to clinically relevant
regions rather than background artifacts.

Protocol:
    1. Generate Grad-CAM for each correct prediction.
    2. Binarize at threshold = 0.5 × max.
    3. Compare with BraTS segmentation ground truth.
    4. Compute IoU (Intersection over Union).
    5. IoU > 0.5 → model uses clinically relevant features.
    6. IoU < 0.3 → model likely uses shortcuts.

References:
    - Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep
      Networks via Gradient-based Localization. ICCV.
    - Captum: https://captum.ai
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAMGenerator:
    """
    Generates Grad-CAM attribution maps for a classification model.

    Parameters
    ----------
    model : nn.Module
        Trained classification model (TumorClassifier or similar).
    target_layer : nn.Module
        The convolutional layer to compute Grad-CAM from. Typically the
        last conv layer of the backbone before global pooling.
    use_captum : bool
        Whether to use Captum's LayerGradCam. Falls back to manual
        implementation if False or if Captum is unavailable.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        use_captum: bool = True,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.use_captum = use_captum

        self._captum_gc = None
        if use_captum:
            try:
                from captum.attr import LayerGradCam

                self._captum_gc = LayerGradCam(model, target_layer)
                logger.info(
                    "GradCAMGenerator initialized with Captum LayerGradCAM"
                )
            except ImportError:
                logger.warning(
                    "Captum not available; using manual "
                    "Grad-CAM implementation"
                )
                self.use_captum = False

        # For manual implementation: register hooks
        self._activations = None
        self._gradients = None
        if not self.use_captum:
            self._register_hooks()

    def generate(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap for a single image.

        Parameters
        ----------
        image : torch.Tensor
            Input image of shape (1, C, H, W) or (C, H, W)
        target_class : int, optional
            Target class index. If None, uses the predicted class.

        Returns
        -------
        np.ndarray
            Grad-CAM heatmap of shape (H, W), values in [0, 1].
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        self.model.eval()
        image = image.requires_grad_(True)

        if self.use_captum and self._captum_gc is not None:
            return self._generate_captum(image, target_class)
        else:
            return self._generate_manual(image, target_class)

    def generate_batch(
        self,
        images: torch.Tensor,
        target_classes: list[int] | None = None,
    ) -> list[np.ndarray]:
        """
        Generate Grad-CAM heatmaps for a batch of images.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images, shape (B, C, H, W)
        target_classes : list[int], optional
            Target class for each image. If None, uses predicted classes.

        Returns
        -------
        list[np.ndarray]
            List of Grad-CAM heatmaps, each of shape (H, W)
        """
        heatmaps = []
        for i in range(images.shape[0]):
            tc = target_classes[i] if target_classes else None
            heatmap = self.generate(images[i], tc)
            heatmaps.append(heatmap)
        return heatmaps

    def _generate_captum(
        self,
        image: torch.Tensor,
        target_class: int | None,
    ) -> np.ndarray:
        """Generate Grad-CAM using Captum's LayerGradCam."""
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = int(output.argmax(dim=1).item())

        attribution = self._captum_gc.attribute(
            image,
            target=target_class,
            relu_attributions=True,
        )

        # Captum returns (1, 1, H', W') or similar — resize to input
        heatmap = attribution.squeeze().detach().cpu().numpy()
        if heatmap.ndim > 2:
            heatmap = heatmap.mean(axis=0)

        heatmap = self._resize_heatmap(heatmap, image.shape[2], image.shape[3])
        heatmap = self._normalize_heatmap(heatmap)
        return heatmap

    def _generate_manual(
        self,
        image: torch.Tensor,
        target_class: int | None,
    ) -> np.ndarray:
        """Generate Grad-CAM with manual hook-based implementation."""
        self._activations = None
        self._gradients = None

        output = self.model(image)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        # Backward pass for the target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        if self._activations is None or self._gradients is None:
            logger.warning("Hooks did not capture activations/gradients")
            return np.zeros((image.shape[2], image.shape[3]))

        # Compute Grad-CAM: GAP of gradients ⟶ weights, then weighted sum
        weights = self._gradients.mean(
            dim=(2, 3), keepdim=True
        )  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(
            dim=1, keepdim=True
        )  # (1, 1, H', W')
        cam = F.relu(cam)

        heatmap = cam.squeeze().detach().cpu().numpy()
        heatmap = self._resize_heatmap(heatmap, image.shape[2], image.shape[3])
        heatmap = self._normalize_heatmap(heatmap)
        return heatmap

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    @staticmethod
    def _resize_heatmap(
        heatmap: np.ndarray, target_h: int, target_w: int
    ) -> np.ndarray:
        """Resize heatmap to match input image dimensions."""
        if heatmap.shape[0] == target_h and heatmap.shape[1] == target_w:
            return heatmap

        heatmap_tensor = (
            torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
        )
        resized = F.interpolate(
            heatmap_tensor,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze().numpy()

    @staticmethod
    def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
        """Normalize heatmap to [0, 1] range."""
        hmax = heatmap.max()
        if hmax > 0:
            return heatmap / hmax
        return heatmap


def binarize_heatmap(
    heatmap: np.ndarray,
    threshold_ratio: float = 0.5,
) -> np.ndarray:
    """
    Binarize a Grad-CAM heatmap.

    Threshold = threshold_ratio × max(heatmap)

    Parameters
    ----------
    heatmap : np.ndarray
        Grad-CAM heatmap of shape (H, W), values in [0, 1].
    threshold_ratio : float
        Fraction of max value for threshold. Default 0.5.

    Returns
    -------
    np.ndarray
        Binary mask of shape (H, W), dtype bool.
    """
    threshold = threshold_ratio * heatmap.max()
    return heatmap >= threshold


def get_target_layer(model: nn.Module, backbone_name: str) -> nn.Module:
    """
    Identity the appropriate target layer for Grad-CAM.

    The target layer should be the last convolutional layer of the backbone,
    just before global average pooling.

    Parameters
    ----------
    model : nn.Module
        The full model (TumorClassifier).
    backbone_name : str
        Name of the backbone architecture.

    Returns
    -------
    nn.Module
        The target convolutional layer.
    """
    backbone = model.backbone if hasattr(model, "backbone") else model

    if "densenet" in backbone_name:
        # MONAI DenseNet: last feature layer
        if hasattr(backbone, "features"):
            children = list(backbone.features.children())
            return children[-1] if children else backbone
        children = list(backbone.children())
        for child in reversed(children):
            if isinstance(child, (nn.Conv2d, nn.BatchNorm2d)):
                return child
            sub = list(child.children())
            if sub:
                return sub[-1]
    elif "resnet" in backbone_name:
        # torchvision ResNet: layer4 is the last residual block
        if hasattr(backbone, "layer4"):
            return backbone.layer4[-1]

    elif "efficientnet" in backbone_name:
        # timm EfficientNet: last conv block
        if hasattr(backbone, "conv_head"):
            return backbone.conv_head
        if hasattr(backbone, "blocks"):
            return backbone.blocks[-1]

    elif "simple_cnn" in backbone_name:
        # Our SimpleCNN: last Conv2d layer
        for module in reversed(list(backbone.modules())):
            if isinstance(module, nn.Conv2d):
                return module

    # Fallback: find the last Conv2d in the entire model
    last_conv = None
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is not None:
        return last_conv

    raise ValueError(
        f"Could not find target layer for backbone: {backbone_name}"
    )
