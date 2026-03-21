# src/data/preprocessing.py
# Preprocessing pipelines for BraTS and UCSF-PDGM
"""
Implements validated preprocessing pipelines.

BraTS 2023:
    Already co-registered to SRI24, 1mm³ isotropic, skull-stripped.
    Additional preprocessing: quality control ⟶ z-score normalization ⟶
    slice selection ⟶ save as NPZ.

UCSF-PDGM:
    Requires full preprocessing: select modalities ⟶ N4ITK bias field
    correction ⟶ verify skull stripping ⟶ resample to 1mm³ ⟶ z-score
    normalization ⟶ slice selection ⟶ save as NPZ.

CRITICAL: Preprocessing must be identical across all classes (Section on
validated preprocessing in the methodological guide). Data-dependent steps
(normalization fitting) must happen AFTER the train/test split.

References:
    - Tustison et al. (2010). N4ITK: Improved N3 Bias Correction. IEEE TMI.
    - Isensee et al. (2021). nnU-Net. Nature Methods (z-score default).
    - Foltyn-Dumitru et al. (2023). N4 + normalization improves glioma
      molecular subtype prediction AUC from 0.84 to 0.87. European Radiology.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingStats:
    """
    Statistics collected during preprocessing for reproducibility.

    Attributes
    ----------
    patient_id : str
        Patient identifier.
    original_shape : tuple[int, ...]
        Shape of the volume before preprocessing.
    final_shape : tuple[int, ...]
        Shape of the volume after preprocessing.
    selected_slice : int
        Index of the selected 2D slice.
    tumor_area_on_slice : int
        Number of tumor voxels on the selected slice.
    n4_applied : bool
        Whether N4 bias field correction was applied.
    resampled : bool
        Whether resampling was applied.
    fg_mean : float
        Foreground mean before z-score normalization.
    fg_std : float
        Foreground std before z-score normalization.
    """

    patient_id: str = ""
    original_shape: tuple[int, ...] = ()
    final_shape: tuple[int, ...] = ()
    selected_slice: int = -1
    tumor_area_on_slice: int = 0
    n4_applied: bool = False
    resampled: bool = False
    fg_mean: float = 0.0
    fg_std: float = 0.0


def apply_n4_bias_correction(
    image_sitk: Any,
    shrink_factor: int = 4,
    iterations: list[int] | None = None,
    convergence_threshold: float = 0.001,
) -> Any:
    """Apply N4ITK bias field correction using SimpleITK.

    N4ITK (Tustison et al., 2010) models the MRI bias field as a smooth
    multiplicative field using B-spline approximation. This is the gold
    standard for MRI bias correction.

    Parameters: shrinkFactor=4, iterations=[50,50,50,50].

    Foltyn-Dumitru et al. (2023) showed N4 + normalization improved glioma
    molecular subtype prediction AUC from 0.84 to 0.87 on 615 patients.

    Parameters
    ----------
    image_sitk : SimpleITK.Image
        Input MRI volume as a SimpleITK image.
    shrink_factor : int
        Factor by which to shrink the image for faster computation.
    iterations : list[int]
        Number of iterations at each resolution level.
    convergence_threshold : float
        Convergence threshold for the algorithm.

    Returns
    -------
    SimpleITK.Image
        Bias-corrected image.
    """
    import SimpleITK as sitk

    if iterations is None:
        iterations = [50, 50, 50, 50]

    # Cast to float32 for N4
    image_float = sitk.Cast(image_sitk, sitk.sitkFloat32)

    # Create mask of non-zero voxels (foreground)
    mask = sitk.OtsuThreshold(image_float, 0, 1, 200)

    # Shrink for speed
    shrunk_image = sitk.Shrink(
        image_float, [shrink_factor] * image_float.GetDimension()
    )
    shrunk_mask = sitk.Shrink(mask, [shrink_factor] * mask.GetDimension())

    # Configure and run N4
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(iterations)
    corrector.SetConvergenceThreshold(convergence_threshold)

    corrected = corrector.Execute(shrunk_image, shrunk_mask)

    # Get the bias field at full resolution
    log_bias_field = corrector.GetLogBiasFieldAsImage(image_float)
    bias_field = sitk.Exp(log_bias_field)

    # Apply correction at full resolution
    corrected_full = image_float / bias_field

    logger.debug(
        "N4 correction applied (iterations=%s, shrink=%d)",
        iterations,
        shrink_factor,
    )
    return corrected_full


def resample_volume(
    image_sitk: Any,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolation: str = "bspline3",
) -> Any:
    """
    Resample a volume to target isotropic spacing using SimpleITK.

    Parameters: target 1mm³ isotropic, B-spline 3rd order for images,
    nearest-neighbor for labels.

    Parameters
    ----------
    image_sitk : SimpleITK.Image
        Input volume.
    target_spacing : tuple[float, float, float]
        Target voxel spacing in mm.
    interpolation : str
        Interpolation method: "bspline3" for images, "nearest" for labels.

    Returns
    -------
    SimpleITK.Image
        Resampled volume.
    """
    import SimpleITK as sitk

    original_spacing = image_sitk.GetSpacing()
    original_size = image_sitk.GetSize()

    # Compute new size based on target spacing
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(
            original_size, original_spacing, target_spacing, strict=False
        )
    ]

    # Select interpolator
    interp_map = {
        "bspline3": sitk.sitkBSpline,
        "nearest": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
    }
    interpolator = interp_map.get(interpolation, sitk.sitkBSpline)

    resampled = sitk.Resample(
        image_sitk,
        new_size,
        sitk.Transform(),
        interpolator,
        image_sitk.GetOrigin(),
        target_spacing,
        image_sitk.GetDirection(),
        0.0,  # Default pixel value
        image_sitk.GetPixelID(),
    )

    logger.debug(
        "Resampled: spacing %s ⟶ %s, size %s ⟶ %s",
        original_spacing,
        target_spacing,
        original_size,
        new_size,
    )
    return resampled


def z_score_normalize(
    volume: np.ndarray,
    foreground_only: bool = True,
) -> tuple[np.ndarray, float, float]:
    """
    Apply z-score normalization per volume.

    Computes mean and std on foreground voxels (> 0) only, following the
    nnU-Net default for MRI (Isensee et al., 2021, Nature Methods).

    Parameters
    ----------
    volume : np.ndarray
        3D MRI volume.
    foreground_only : bool
        If True, compute statistics only on voxels > 0.

    Returns
    -------
    tuple[np.ndarray, float, float]
        (normalized_volume, mean, std)
    """
    if foreground_only:
        mask = volume > 0
    else:
        mask = np.ones_like(volume, dtype=bool)

    if mask.sum() == 0:
        return volume, 0.0, 1.0

    fg_mean = float(volume[mask].mean())
    fg_std = float(volume[mask].std())

    if fg_std < 1e-8:
        return volume, fg_mean, fg_std

    normalized = np.zeros_like(volume)
    normalized[mask] = (volume[mask] - fg_mean) / fg_std
    return normalized, fg_mean, fg_std


def select_representative_slice(
    segmentation: np.ndarray | None,
    volume_depth: int,
    method: str = "max_tumor_area",
    axis: int = 2,
) -> int:
    """
    Select the representative 2D slice for classification.

    Parameters
    ----------
    segmentation : np.ndarray or None
        3D segmentation mask (H, W, D). None triggers center fallback.
    volume_depth : int
        Total number of slices along the target axis.
    method : str
        Selection method: "max_tumor_area" or "center"
    axis : int
        Axis along which to extract the slice (2 = axial).

    Returns
    -------
    int
        Index of the selected slice.
    """
    if method == "center" or segmentation is None:
        return volume_depth // 2

    if method == "max_tumor_area":
        tumor_mask = segmentation > 0
        axes_to_sum = tuple(i for i in range(tumor_mask.ndim) if i != axis)
        area_per_slice = tumor_mask.sum(axis=axes_to_sum)

        if area_per_slice.max() == 0:
            logger.warning("No tumor in segmentation; using center slice.")
            return volume_depth // 2

        return int(np.argmax(area_per_slice))

    raise ValueError(f"Unknown slice selection method: {method}")


def preprocess_brats_patient(
    patient_dir: Path,
    output_dir: Path,
    modalities: list[str] | None = None,
    expected_shape: tuple[int, int, int] = (240, 240, 155),
    slice_method: str = "max_tumor_area",
    context_slices: int = 0,
) -> PreprocessingStats | None:
    """
    Preprocess a single BraTS patient.

    BraTS is already co-registered, isotropic, skull-stripped.
    Preprocessing: verify ⟶ z-score normalize ⟶ select slice ⟶ save NPZ.

    Parameters
    ----------
    patient_dir : Path
        Path to the BraTS patient directory.
    output_dir : Path
        Path to save preprocessed output.
    modalities : list[str]
        Modality keys to load.
    expected_shape : tuple
        Expected volume dimensions for QC.
    slice_method : str
        Slice selection method.
    context_slices : int
        Adjacent slices for 2.5D (0 for pure 2D).

    Returns
    -------
    PreprocessingStats or None
        Preprocessing statistics, or None if patient failed QC.
    """
    from src.data.brats_dataset import MODALITY_SUFFIXES, SEG_SUFFIX

    modalities = modalities or ["t1", "t1ce", "t2", "flair"]
    patient_id = patient_dir.name
    stats = PreprocessingStats(patient_id=patient_id)

    # Load and verify all modalities
    volumes = {}
    for mod in modalities:
        suffix = MODALITY_SUFFIXES[mod]
        filepath = patient_dir / f"{patient_id}{suffix}"
        if not filepath.exists():
            logger.warning(
                "Missing %s for %s. Skipping patient.", mod, patient_id
            )
            return None

        nii = nib.load(str(filepath))
        vol = nii.get_fdata().astype(np.float32)
        stats.original_shape = vol.shape

        # QC: check dimensions
        if expected_shape and vol.shape != expected_shape:
            logger.warning(
                "Shape mismatch for %s: got %s, expected %s. Skipping.",
                patient_id,
                vol.shape,
                expected_shape,
            )
            return None

        # QC: check for empty volumes
        if np.all(vol == 0):
            logger.warning(
                "Empty volumes %s for %s. Skipping.", mod, patient_id
            )
            return None

        # Z-score normalize
        vol_norm, fg_mean, fg_std = z_score_normalize(
            vol, foreground_only=True
        )
        stats.fg_mean = fg_mean
        stats.fg_std = fg_std
        volumes[mod] = vol_norm

    # Load segmentation
    seg_path = patient_dir / f"{patient_id}{SEG_SUFFIX}"
    if not seg_path.exists():
        logger.warning("Missing segmentation for %s. Skipping.", patient_id)
        return None

    seg_nii = nib.load(str(seg_path))
    seg_vol = seg_nii.get_fdata().astype(np.int32)

    # Select representative slice
    depth = vol.shape[2]
    slice_idx = select_representative_slice(
        seg_vol, depth, method=slice_method
    )
    stats.selected_slice = slice_idx

    # Extract 2D slices and stack modalities as channels
    image_slices = []
    half = context_slices
    if half == 0:
        indices = [slice_idx]
    else:
        indices = list(
            range(max(0, slice_idx - half), min(depth, slice_idx + half + 1))
        )
        # Pad if near edges
        while len(indices) < 2 * half + 1:
            if indices[0] > 0:
                indices.insert(0, indices[0] - 1)
            elif indices[-1] < depth - 1:
                indices.append(indices[-1] + 1)
            else:
                break

    for si in indices:
        for mod in modalities:
            image_slices.append(volumes[mod][:, :, si])

    image = np.stack(image_slices, axis=0)  # (C, H, W)
    seg_slice = seg_vol[:, :, slice_idx]
    stats.tumor_area_on_slice = int(np.sum(seg_slice > 0))
    stats.final_shape = image.shape

    # Save as NPZ
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{patient_dir}.npz"
    np.savez_compressed(
        out_path,
        image=image,
        segmentation=seg_slice,
        patient_id=patient_id,
        slice_index=slice_idx,
    )

    logger.debug(
        "Preprocessed %s ⟶ %s (slice %d)", patient_id, out_path, slice_idx
    )
    return stats


def preprocess_ucsf_patient(
    patient_dir: Path,
    output_dir: Path,
    modalities: list[str] | None = None,
    apply_n4: bool = True,
    n4_params: dict[str, Any] | None = None,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    do_resample: bool = True,
    slice_method: str = "max_tumor_area",
    context_slices: int = 0,
) -> PreprocessingStats | None:
    """
    Preprocess a single UCSF-PDGM patient.

    Full pipeline:
        load ⟶ N4 bias correction ⟶ resample z-score ⟶ slice ⟶ save.

    Parameters
    ----------
    patient_dir : Path
        Path to the patient directory with NIfTI files.
    output_dir : Path
        Directory to save preprocessed output.
    modalities : list[str]
        Modality keys to preocess.
    apply_n4 : bool
        Whether to apply N4 bias field correction.
    n4_params : dict
        N4 parameters (shrink_factor, iterations, converge_threshold).
    target_spacing : tuple
        Target voxel spacing for resampling.
    do_resample : bool
        Whether to resample to target spacing.
    slice_method : str
        Slice selection method.
    context_slices : int
        Adjacent slices for 2.5D.

    Returns
    -------
    PreprocessingStats or None
        Preprocessing statistics, or None if patient failed.
    """
    import SimpleITK as sitk

    modalities = modalities or ["t1", "t1ce", "t2", "flair"]
    n4_params = n4_params or {}
    patient_id = patient_dir.name
    stats = PreprocessingStats(patient_id=patient_id)

    volumes = {}
    for mod in modalities:
        # UCSF-PDGM naming convention
        vol_path = patient_dir / f"{patient_id}_{mod}.nii.gz"
        if not vol_path.exists():
            logger.warning(
                "Missing %s for %s. Skipping patient.", mod, patient_id
            )
            return None

        # Load with SimpleITK for N4 and resampling
        image_sitk = sitk.ReadImage(str(vol_path), sitk.sitkFloat32)
        stats.original_shape = image_sitk.GetSize()

        # Step 1: N4 bias field correction (Tustison et al., 2010)
        if apply_n4:
            try:
                image_sitk = apply_n4_bias_correction(
                    image_sitk,
                    shrink_factor=n4_params.get("shrink_factor", 4),
                    iterations=n4_params.get("iterations", [50, 50, 50, 50]),
                    convergence_threshold=n4_params.get(
                        "convergence_threshold", 0.001
                    ),
                )
                stats.n4_applied = True
            except Exception as e:
                logger.warning(
                    "N4 failed for %s/%s: %s. Continuing without.",
                    patient_id,
                    mod,
                    e,
                )

        # Step 2: Resample to target spacing
        if do_resample:
            image_sitk = resample_volume(
                image_sitk, target_spacing, interpolation="bspline3"
            )
            stats.resampled = True

        # Convert to numpy for z-score normalization
        vol = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
        # SimpleITK returns (D, H, W), transpose to (H, W, D) for consistency
        vol = vol.transpose(1, 2, 0)

        # Step 3: Z-score normalization (same protocol as BraTS)
        vol_norm, fg_mean, fg_std = z_score_normalize(
            vol, foreground_only=True
        )
        stats.fg_mean = fg_mean
        stats.fg_std = fg_std
        volumes[mod] = vol_norm

    # Load segmentation if available
    seg_vol = None
    seg_path = patient_dir / f"{patient_id}_seg.nii.gz"
    if seg_path.exists():
        seg_sitk = sitk.ReadImage(str(seg_path), sitk.sitkInt32)
        if do_resample:
            seg_sitk = resample_volume(
                seg_sitk, target_spacing, interpolation="nearest"
            )
        seg_vol = (
            sitk.GetArrayFromImage(seg_sitk)
            .astype(np.int32)
            .transpose(1, 2, 0)
        )

    # Select slice
    first_vol = list(volumes.values())[0]
    depth = first_vol.shape[2]
    slice_idx = select_representative_slice(
        seg_vol, depth, method=slice_method
    )
    stats.selected_slice = slice_idx

    # Extract and stack slices
    image_slices = []
    half = context_slices
    if half == 0:
        indices = [slice_idx]
    else:
        indices = list(
            range(max(0, slice_idx - half), min(depth, slice_idx + half + 1))
        )
        while len(indices) < 2 * half + 1:
            if indices[0] > 0:
                indices.insert(0, indices[0] - 1)
            elif indices[-1] < depth - 1:
                indices.append(indices[-1] + 1)
            else:
                break

    for si in indices:
        for mod in modalities:
            image_slices.append(volumes[mod][:, :, si])

    image = np.stack(image_slices, axis=0)
    seg_slice = (
        seg_vol[:, :, slice_idx]
        if seg_vol is not None
        else np.zeros(image.shape[1:], dtype=np.int32)
    )
    stats.tumor_area_on_slice = int(np.sum(seg_slice > 0))
    stats.final_shape = image.shape

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{patient_id}.npz"
    np.savez_compressed(
        out_path,
        image=image,
        segmentation=seg_slice,
        patient_id=patient_id,
        slice_index=slice_idx,
    )

    logger.debug(
        "Preprocessed %s ⟶ %s (slice %d)", patient_id, out_path, slice_idx
    )
    return stats
