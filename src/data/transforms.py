# src/data/transforms.py
# MRI-specific preprocessing and augmentation transforms
"""
Transform pipelines for brain tumor MRI classification using TorchIO.

Augmentations are physically motivated for MRI data, following BraTS challenge
standards. Only anatomically plausible transforms are applied:
- Sagittal flip only (brain is sagittally symmetric but NOT vertically).
- Conservative rotation ±15° (larger angles are anatomically unrealistic).
- MRI-specific artifacts: bias field, ghosting, noise.

What is NOT used and why:
- Vertical flip: brain anatomy is not vertically symmetric.
- Rotations >20°: anatomically irrealistic for brain MRI.
- Color jitter/hue shift: no physical meaning in MRI.
- CutMix/MixUp: not validated in medical imaging; can create unreal pathology.

CRITICAL: Augmentation is ONLY applied during training. Validation and test
sets use NO augmentation — only the normalization already applied in
preprocessing (val_transform = tio.Compose([])).

References:
    - Pérez-García et al. (2021). TorchIO. CMPB, 208, 106236.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torchio as tio

logger = logging.getLogger(__name__)


def build_train_transforms(cfg: dict[str, Any] | None = None) -> tio.Compose:
    """Build the training augmentation pipeline.

    All augmentations are physically motivated for brain MRI and validated
    in BraTS submission literature. Parameters use conservative ranges.

    Parameters
    ----------
    cfg : dict, optional
        Configuration overrides. If None, uses default parameters

    Returns
    -------
    tio.Compose
        Composed TorchIO transform pipeline for training data.
    """
    cfg = cfg or {}

    transforms = []

    # --- Spatial transforms (conservative ranges) ---
    if cfg.get("enable_affine", True):
        transforms.append(
            tio.RandomAffine(
                scales=cfg.get("affine_scales", (0.9, 1.1)),  # ±10%
                degrees=cfg.get("affine_degrees", 15),  # ±15°
                translation=cfg.get("affine_translation", 10),  # ±10mm
                p=cfg.get("affine_p", 0.5),
            )
        )

    # Sagittal flip only — brain is sagittally symmetric
    if cfg.get("enable_flip", True):
        transforms.append(
            tio.RandomFlip(
                axes=(0,),  # Sagittal axis only
                p=cfg.get("flip_p", 0.5),
            )
        )

    # --- MRI-specific artifact simulation ---
    # Bias field inhomogeneity (B0 field non-uniformity)
    if cfg.get("enable_bias_field", True):
        transforms.append(
            tio.RandomBiasField(
                coefficients=cfg.get("bias_coefficients", 0.3),
                p=cfg.get("bias_p", 0.3),
            )
        )

    # Gaussian noise (thermal noise in MRI acquisition)
    if cfg.get("enable_noise", True):
        transforms.append(
            tio.RandomNoise(
                std=cfg.get("noise_std", (0, 0.05)),
                p=cfg.get("noise_p", 0.3),
            )
        )

    # Ghosting artifacts (periodic motion during acquisition)
    if cfg.get("enable_ghosting", True):
        transforms.append(
            tio.RandomGhosting(
                intensity=cfg.get("ghosting_intensity", (0.5, 1.0)),
                p=cfg.get("ghosting_p", 0.15),
            )
        )

    # --- Intensity transforms ---
    if cfg.get("enable_gamma", True):
        transforms.append(
            tio.RandomGamma(
                log_gamma=cfg.get("gamma_range", (-0.2, 0.2)),
                p=cfg.get("gamma_p", 0.3),
            )
        )

    composed = tio.Compose(transforms)
    logger.info(
        "Built training transforms: %d augmentations enabled",
        len(transforms),
    )
    return composed


def build_val_transforms() -> tio.Compose:
    """Build the validation/test transform pipeline.

    CRITICAL: No augmentation is applied during validation or testing.
    Z-score normalization is already applied during preprocessing.

    Returns
    -------
    tio.Compose
        Empty compose (identity transform) for validation/test data.
    """
    logger.info("Built validation transforms: NO augmentation (identity)")
    return tio.Compose([])


def build_preprocessing_transforms(
    z_score: bool = True,
    foreground_only: bool = True,
) -> tio.Compose:
    """Build the offline preprocessing transform pipeline.

    Applied once during data preparation (not during training loop).
    Includes z-score normalization computed per-volume on foreground voxels.

    Parameters
    ----------
    z_score : bool
        Whether to apply z-score normalization. Default True.
    foreground_only : bool
        If True, compute mean/std only on voxels > 0. Default True.

    Returns
    -------
    tio.Compose
        Preprocessing transform pipeline.
    """
    transforms = []

    if z_score:
        transforms.append(
            tio.ZNormalization(
                masking_method=_foreground_mask if foreground_only else None,
            )
        )

    return tio.Compose(transforms)


def extract_2d_slice(
    volume: np.ndarray | torch.Tensor,
    segmentation: np.ndarray | torch.Tensor | None = None,
    axis: int = 2,
    method: str = "max_tumor_area",
    context_slices: int = 0,
) -> tuple[np.ndarray | torch.Tensor, int]:
    """Extract a representative 2D slice from a 3D volume.

    For the 2D approach, selects the axial slice with maximum tumor area
    from the segmentation mask. For 2.5D, also returns adjacent slices.

    Parameters
    ----------
    volume : np.ndarray or torch.Tensor
        3D or 4D volume (C, H, W, D) or (H, W, D).
    segmentation : np.ndarray or torch.Tensor, optional
        Segmentation mask for tumor-guided slice selection.
    axis : int
        Axis along which to extract slices. Default 2 (axial plane).
    method : str
        Slice selection method. Options: "max_tumor_area", "center".
    context_slices : int
        Number of adjacent slices on each side for 2.5D. 0 for pure 2D.

    Returns
    -------
    tuple
        (extracted_slice, slice_index) where extracted_slice has shape
        (C, H, W) for 2D or (C * (2*context+1), H, W) for 2.5D.
    """

    if method == "max_tumor_area" and segmentation is not None:
        slice_idx = _find_max_tumor_slice(segmentation, axis=axis)
    elif method == "center":
        n_slices = (
            volume.shape[axis] if volume.ndim == 3 else volume.shape[axis + 1]
        )
        slice_idx = n_slices // 2
    else:
        # Fallback to center if no segmentation available
        n_slices = (
            volume.shape[axis] if volume.ndim == 3 else volume.shape[axis + 1]
        )
        slice_idx = n_slices // 2
        logger.warning(
            "No segmentation available for max_tumor_area; "
            "using center slice %d",
            slice_idx,
        )

    # Extract slice(s) along the specified axis
    if context_slices > 0:
        slices = _extract_context_slices(
            volume, slice_idx, axis, context_slices
        )
    else:
        slices = _extract_single_slice(volume, slice_idx, axis)

    return slices, slice_idx


def _foreground_mask(tensor: torch.Tensor) -> torch.Tensor:
    """Create a foreground mask for z-score normalization.

    Parameters
    ----------
    tensor : torch.Tensor
        Input volume tensor.

    Returns
    -------
    torch.Tensor
        Boolean mask where voxels > 0.
    """
    return tensor > 0


def _find_max_tumor_slice(
    segmentation: np.ndarray | torch.Tensor,
    axis: int = 2,
) -> int:
    """Find the slice index with maximum tumor area.

    Parameters
    ----------
    segmentation : np.ndarray or torch.Tensor
        Segmentation mask (any non-zero value indicates tumor).
    axis : int
        Axis along which to search for the maximum tumor slice.

    Returns
    -------
    int
        Index of the slice with maximum tumor area.
    """
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.numpy()

    # Handle channel dimension: squeeze if present
    if segmentation.ndim == 4:
        segmentation = segmentation[0]

    # Sum non-zero voxels along all axes except the target axis
    axes_to_sum = tuple(i for i in range(segmentation.ndim) if i != axis)
    tumor_area_per_slice = np.sum(segmentation > 0, axis=axes_to_sum)

    max_idx = int(np.argmax(tumor_area_per_slice))

    if tumor_area_per_slice[max_idx] == 0:
        logger.warning(
            "No tumor found in segmentation; defaulting to center slice."
        )
        max_idx = segmentation.shape[axis] // 2

    return max_idx


def _extract_single_slice(
    volume: np.ndarray | torch.Tensor,
    slice_idx: int,
    axis: int,
) -> np.ndarray | torch.Tensor:
    """Extract a single 2D slice from a 3D/4D volume.

    Parameters
    ----------
    volume : np.ndarray or torch.Tensor
        Input volume with shape (C, H, W, D) or (H, W, D).
    slice_idx : int
        Index of the slice to extract.
    axis : int
        Axis along which to slice.

    Returns
    -------
    np.ndarray or torch.Tensor
        2D slice with shape (C, H, W) or (H, W).
    """
    if volume.ndim == 4:
        # Shape: (C, H, W, D) -> slice along axis+1 to account for channel dim
        return _index_along_axis(volume, slice_idx, axis + 1)
    else:
        # Shape: (H, W, D) -> slice directly
        return _index_along_axis(volume, slice_idx, axis)


def _extract_context_slices(
    volume: np.ndarray | torch.Tensor,
    center_idx: int,
    axis: int,
    context: int,
) -> np.ndarray | torch.Tensor:
    """Extract adjacent slices for 2.5D approach.

    Concatenates (2*context + 1) slices along the channel dimension.
    If slices go out of bounds, pads with zeros.

    Parameters
    ----------
    volume : np.ndarray or torch.Tensor
        Input volume with shape (C, H, W, D).
    center_idx : int
        Index of the center slice.
    axis : int
        Axis along which to slice.
    context : int
        Number of adjacent slices on each side.

    Returns
    -------
    np.ndarray or torch.Tensor
        2.5D slice with shape (C * (2*context+1), H, W).
    """
    has_channels = volume.ndim == 4
    effective_axis = axis + 1 if has_channels else axis
    n_slices_total = volume.shape[effective_axis]

    all_slices = []
    for offset in range(-context, context + 1):
        idx = center_idx + offset
        if 0 <= idx < n_slices_total:
            s = _index_along_axis(volume, idx, effective_axis)
        else:
            # Zero-pad out-of-bounds slices
            s = _index_along_axis(volume, center_idx, effective_axis)
            if isinstance(s, torch.Tensor):
                s = torch.zeros_like(s)
            else:
                s = np.zeros_like(s)
        all_slices.append(s)

    # Concatenate along channel dimension
    if isinstance(volume, torch.Tensor):
        return (
            torch.cat(all_slices, dim=0)
            if has_channels
            else torch.stack(all_slices)
        )
    else:
        return (
            np.concatenate(all_slices, axis=0)
            if has_channels
            else np.stack(all_slices)
        )


def _index_along_axis(
    arr: np.ndarray | torch.Tensor,
    idx: int,
    axis: int,
) -> np.ndarray | torch.Tensor:
    """Index a single slice along a specific axis.

    Parameters
    ----------
    arr : np.ndarray or torch.Tensor
        Input array.
    idx : int
        Index to select.
    axis : int
        Axis along which to index.

    Returns
    -------
    np.ndarray or torch.Tensor
        Array with one fewer dimension.
    """
    slicing = [slice(None)] * arr.ndim
    slicing[axis] = idx
    return arr[tuple(slicing)]
