# src/data/brats_dataset.py — BraTS 2023 Adult Glioma dataset loader
"""
MONAI-based dataset loader for BraTS 2023 Adult Glioma data.

BraTS data is organized per patient with 4 MRI modalities (T1, T1ce, T2, FLAIR)
and expert segmentation labels in NIfTI format. Data is already co-registered
to SRI24 atlas at 1mm³ isotropic resolution and skull-stripped.

The loader:
1. Discovers all patient directories in the BraTS root folder.
2. Verifies that all 4 modalities + segmentation exist per patient.
3. Derives classification labels (HGG vs LGG) from the dataset metadata.
4. Extracts the representative 2D slice (max tumor area) for 2D classification.
5. Stacks modalities as channels: [T1, T1ce, T2, FLAIR] ⟶ (4, H, W).

Patient-level organization prevents data leakage by design.

References:
    - BraTS 2023: Synapse syn51156910
    - Menze et al. (2015), IEEE TMI, 34(10), 1993-2024
"""

import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# BraTS 2023 file naming conventions
MODALITY_SUFFIXES = {
    "t1": "-t1n.nii.gz",
    "t1ce": "-t1c.nii.gz",
    "t2": "-t2w.nii.gz",
    "flair": "-t2f.nii.gz",
}
SEG_SUFFIX = "-seg.nii.gz"

# BraTS segmentation label values
SEG_LABELS = {
    "background": 0,
    "necrotic_core": 1,  # NCR
    "edema": 2,  # ED
    "enhancing_tumor": 3,  # ET
}


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS 2023 Adult Glioma classification.

    Each sample contains a 2D multi-modal MRI slice (4 channels) and the
    corresponding glioma grade label (0=LGG, 1=HGG).

    Args:
        root_dir: Path to the BraTS dataset root (containing patient folders).
        grade_labels: Dictionary mapping patient_id to grade (0=LGG, 1=HGG).
        modalities: List of modality keys to load. Default: all four.
        slice_method: Method for selecting the representative axial slice.
            Options: "max_tumor_area" (default), "center".
        context_slices: Number of adjacent slices for 2.5D. 0 for pure 2D.
        transform: Optional callable transform applied to the sample dict.
        z_score_normalize: Whether to apply per-volume z-score normalization
            on foreground voxels.
    """

    def __init__(
        self,
        root_dir: str,
        grade_labels: dict[str, int],
        modalities: list[str] | None = None,
        slice_method: str = "max_tumor_area",
        context_slices: int = 0,
        transform: Any | None = None,
        z_score_normalize: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.grade_labels = grade_labels
        self.modalities = modalities or list(MODALITY_SUFFIXES.keys())
        self.slice_method = slice_method
        self.context_slices = context_slices
        self.transform = transform
        self.z_score_normalize = z_score_normalize

        # Discover and validate patient directories
        self.patient_ids, self.patient_dirs = self._discover_patients()
        logger.info(
            "BraTSDataset initialized: %d patients, modalities=%s, "
            "slice_method=%s, context_slices=%d",
            len(self.patient_ids),
            self.modalities,
            slice_method,
            context_slices,
        )

    def _discover_patients(self) -> tuple[list[str], list[Path]]:
        """
        Discover valid patient directories that have all required files.

        Returns:
            Tuple of (patient_ids, patient_dirs) for valid patients only.
        """
        patient_ids = []
        patient_dirs = []

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"BraTS root directory not found: {self.root_dir}"
            )

        for patient_dir in sorted(self.root_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            patient_id = patient_dir.name

            # Skip patients without grade labels
            if patient_id not in self.grade_labels:
                logger.debug(
                    "Skipping %s: no grade label available".patient_id
                )
                continue

            # Verify all modality files exist
            if self._verify_patient_files(patient_dir, patient_id):
                patient_ids.append(patient_id)
                patient_dirs.append(patient_dir)

        if not patient_ids:
            raise RuntimeError(
                f"No valid patients found in {self.root_dir}. "
                "Check that patient directories contain all required "
                "NIfTI files."
            )

        logger.info(
            "Discovered %d valid patients out of available directories.",
            len(patient_dirs),
        )

        return patient_ids, patient_dirs

    def _verify_patient_files(
        self, patient_dir: Path, patient_id: str
    ) -> bool:
        """
        Verify that all required modality and segmentation files exist.

        Args:
            patient_dir: Path to the patient_directory.
            patient_id: Patient identifier string.

        Returns:
            True if all required files are present, False otherwise.
        """
        for modality in self.modalities:
            suffix = MODALITY_SUFFIXES[modality]
            modality_file = patient_dir / f"{patient_id}{suffix}"
            if not modality_file.exists():
                logger.warning(
                    "Missing %s for patient %s: %s",
                    modality,
                    patient_id,
                    modality_file,
                )
                return False

        seg_file = patient_dir / f"{patient_id}{SEG_SUFFIX}"
        if not seg_file.exists():
            logger.warning("Missing segmentation for patient %s", patient_id)
            return False

        return True

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Load and process a single patient sample.

        Returns:
            Dictonary with keys:
                - "image": Tensor of shape (C, H, W) were C = num_modalities
                    (or C = num_modalities * num_context_slices for 2.5D)
                - "label": Integer grade label (0=LGG, 1=HGG).
                - "patient_id": String patient identifier.
                - "slice_idx": Index of the selected axial slice.
                - "segmentation": Tensor of shape (H, W) with tumor
                    segmentation for the selected slice (used for Grad-CAM IoU
                    validation).
        """
        patient_id = self.patient_ids[idx]
        patient_dir = self.patient_dirs[idx]
        label = self.grade_labels[patient_id]

        # Load all modalities volumes
        volumes = self._load_modality_volumes(patient_dir, patient_id)

        # Load segmentation for slice selection and interpretability validation
        seg_volume = self._load_segmentation(patient_dir, patient_id)

        # Select representative slice(s)
        slice_idx = self._select_slice(seg_volume)

        # Extract 2D slices (or 2.5D multi-slice)
        image = self._extract_slice(volumes, slice_idx)
        seg_slice = seg_volume[:, :, slice_idx]

        # Build sample dictionary
        sample = {
            "image": torch.from_numpy(image).float(),
            "label": label,
            "patient_id": patient_id,
            "slice_idx": slice_idx,
            "segmentation": torch.from_numpy(seg_slice).long(),
        }

        # Apply transform if provided
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _load_modality_volumes(
        self, patient_dir: Path, patient_id: str
    ) -> dict[str, np.ndarray]:
        """
        Load NIfTI volumes for all requested modalities.

        Applies z-score normalization on foreground voxels if enabled
        (mu and sigma computed on voxels > 0).

        Args:
            patient_dir: Path to the patient directory.
            patient_id: Patient identifier.

        Returns:
            Dictionary mapping modality name to 3D numpy array.
        """
        volumes = {}
        for modality in self.modalities:
            suffix = MODALITY_SUFFIXES[modality]
            filepath = patient_dir / f"{patient_id}{suffix}"
            nii = nib.load(str(filepath))
            volume = nii.get_fdata().astype(np.float32)

            if self.z_score_normalize:
                volume = self._apply_z_score(volume)

            volumes[modality] = volume

        return volumes

    def _load_segmentation(
        self,
        patient_dir: Path,
        patient_id: str,
    ) -> np.ndarray:
        """
        Load the segmentation volume.

        Args:
            patient_dir: Path to the patient directory.
            patient_id: Patient identifier.

        Returns:
            3D numpy array with segmentation labels.
        """
        seg_path = patient_dir / f"{patient_dir}{SEG_SUFFIX}"
        nii = nib.load(str(seg_path))
        return nii.get_fdata().astype(np.int32)

    @staticmethod
    def _apply_z_score(volume: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization on foreground voxels only.

        Foreground is defined as voxels with intensity > 0, following the
        nnU-Net default for MRI data (Isensee et al., 2021).

        Args:
            volume: 3D numpy array.

        Returns:
            Normalized 3D numpy array.
        """
        foreground_mask = volume > 0
        if foreground_mask.sum() == 0:
            return volume

        fg_mean = volume[foreground_mask].mean()
        fg_std = volume[foreground_mask].std()

        if fg_std < 1e-8:
            return volume

        normalized = np.zeros_like(volume)
        normalized[foreground_mask] = (
            volume[foreground_mask] - fg_mean
        ) / fg_std
        return normalized

    def _select_slice(self, seg_volume: np.ndarray) -> int:
        """
        Select the representative axial slice for 2D classification.

        Args:
            seg_volume: 3D segmentation array (H, W, D).

        Returns:
            Index of the selected axial slice.
        """
        if self.slice_method == "max_tumor_area":
            # Select axual slice with maximum tumor area.
            tumor_mask = seg_volume > 0  # Any tumor label
            tumor_area_per_slice = tumor_mask.sum(axis=(0, 1))

            if tumor_area_per_slice.max() == 0:
                # Fallback to center slice if not tumor visible
                logger.warning(
                    "No tumor found in segmentation. Using center slice."
                )
                return seg_volume.shape[2] // 2

            return int(np.argmax(tumor_area_per_slice))

        elif self.slice_method == "center":
            return seg_volume.shape[2] // 2

        else:
            raise ValueError(
                f"Unknown slice selection method: {self.slice_method}"
            )

    def _extract_slices(
        self,
        volumes: dict[str, np.ndarray],
        center_idx: int,
    ) -> np.ndarray:
        """
        Extract 2D slice(s) from all modalitites and stack as channels.

        For pure 2D (context_slices=0): output shape is (num_modalities, H, W).
        For 2.5D (context_slices>0): output shape is
            (num_modalities * (2*context_slices+1), H, W)

        Args:
            volumes: Dictionary of modality modules.
            center_idx: Index of the center axial slice.

        Returns:
            Numpy array of shape (C, H, W).
        """
        slices = []
        depth = list(volumes.values())[0].shape[2]

        # Compute range of slices to extract
        if self.context_slices == 0:
            slice_indices = [center_idx]
        else:
            half = self.context_slices
            slice_indices = list(
                range(
                    max(0, center_idx - half),
                    min(depth, center_idx + half + 1),
                )
            )
            # Pad if near edges
            while len(slice_indices) < 2 * half + 1:
                if slice_indices[0] > 0:
                    slice_indices.insert(0, slice_indices[0] - 1)
                elif slice_indices[-1] < depth - 1:
                    slice_indices.append(slice_indices[-1] + 1)
                else:
                    break

        # Stack modalities for each slice index
        for si in slice_indices:
            for modality in self.modalities:
                slices.append(volumes[modality][:, :, si])

        return np.stack(slices, axis=0)

    def get_patient_ids(self) -> list[str]:
        """
        Return list of all patient identifiers in this dataset.

        Used by the splitter and leakage tests to verify patient-level
        separation.
        """
        return list(self.patient_ids)

    def get_labels(self) -> np.ndarray:
        """
        Return array of all labels for stratification.

        Returns:
            Numpy array of integer labels, one per patient.
        """
        return np.array([self.grade_labels[pid] for pid in self.patient_ids])
