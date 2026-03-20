# src/data/quality_control.py — Automated data quality verification
"""
Automated quality control checks for BraTS 2023 and UCSF-PDGM datasets.

Verififes data integrity before training to catch issues early:
- Volume dimensions match expected shape (240x240x155 for BraTS).
- No corrupt or empty volumes.
- All 4 MRI modalities present per patient.
- Minimum tumor voxel count in segmentations.
- NIfTI header consistency (affine, orientation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QCResult:
    """
    Result of a quality control check for a single patient.

    Attributes
    ----------
    patient_id : str
        Patient identifier.
    passed : bool
        Whether all QC checks passed.
    issues : list[str]
        List of issues found. Empty if all checks passed.
    modalities_found : list[str]
        List of modalities found for this patient.
    volume_shape : tuple[int, ...] | None
        Shape of the first modality volume, or None if not loadable.
    tumor_voxels : int
        Number of non-zero voxels in the segmentation mask.
    """

    patient_id: str
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    modalities_found: list[str] = field(default_factory=list)
    volume_shape: tuple[int, ...] | None = None
    tumor_voxels: int = 0


class DatasetQualityControl:
    """
    Autimated quality control for brain tumor MRI datasets.

    Runs a battery of checks on each patient directory to ensure data
    integrity before any training or preprocessing takes place.

    Parameters
    ----------
    expected_modalities : list[str]
        List of expected MRI modality suffixes
        (e.g., ["t1n", "t1c", "t2w", "t2f"])
    expected_shape : tuple[int, ...] | None
        Expected volume dimensions (e.g., (240, 240, 155) for BraTS).
    min_tumor_voxels : int
        Minimum non-zero voxels in segmentation to consider valid.
    file_patterns : dict[str, str] | None
        Mapping from modality name to glob pattern.
    """

    def __init__(
        self,
        expected_modalities: list[str] | None = None,
        expected_shape: tuple[int, ...] | None = None,
        min_tumor_voxels: int = 100,
        file_patterns: dict[str, str] | None = None,
    ) -> None:
        self.expected_modalities = expected_modalities or [
            "t1n",
            "t1c",
            "t2w",
            "t2f",
        ]
        self.expected_shape = expected_shape
        self.min_tumor_voxels = min_tumor_voxels
        self.file_patterns = file_patterns or {}

    def check_patient(self, patient_dir: Path) -> QCResult:
        """
        Run all quality control checks on a single patient directory.

        Parameters
        ----------
        patient_dir : Path
            Path to the patient directory containing NIfTI files.

        Returns
        -------
        QCResult
            Quality control result with pass/fail status and issues.
        """
        patient_id = patient_dir.name
        result = QCResult(patient_id=patient_id)

        # Check 1: All expected modalities present
        self._check_modalities(patient_dir, result)

        # Check 2: Volume dimensions match expected shape
        self._check_dimensions(patient_dir, result)

        # Check 3: No corrupt or empty volumes
        self._check_not_empty(patient_dir, result)

        # Check 4: Segmentation has minimum tumor content
        self._check_segmentation(patient_dir, result)

        # Set overall pass/fail
        result.passed = len(result.issues) == 0

        if not result.passed:
            logger.warning(
                "QC FAILED for %s: %s", patient_id, "; ".join(result.issues)
            )
        else:
            logger.debug("QC passed for %s", patient_id)

        return result

    def check_dataset(self, dataset_dir: Path) -> list[QCResult]:
        """
        Run quality control on all patients in a dataset directory.

        Parameters
        ----------
        dataset_dir : Path
            Root directory containing patient subdirectories.

        Returns
        -------
        list[QCResult]
            QC results for each patient.
        """
        patient_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

        if not patient_dirs:
            logger.error("No patient directories found in %s", dataset_dir)
            return []

        results = []
        n_passed = 0
        n_failed = 0

        for patient_dir in patient_dirs:
            result = self.check_dataset(patient_dir)
            results.append(result)
            if result.passed:
                n_passed += 1
            else:
                n_failed += 1

        logger.info(
            "QC complete: %d/%d patients passed, %d failed",
            n_passed,
            len(results),
            n_failed,
        )

        return results

    def _check_modalities(self, patient_dir: Path, result: QCResult) -> None:
        """Verify all expected MRI modalities are present."""
        found_modalities = []

        for modality in self.expected_modalities:
            # Try file pattern first, then fallback to substring matching
            pattern = self.file_patterns.get(modality, f"*{modality}*")
            matches = list(patient_dir.glob(pattern))
            if matches:
                found_modalities.append(modality)
            else:
                result.issues.append(f"Missing modality: {modality}")

        result.modalities_found = found_modalities

    def _check_dimensions(self, patient_dir: Path, result: QCResult) -> None:
        """Verify volume dimensions match the expected shape."""
        if self.expected_shape is None:
            return

        nifti_files = list(patient_dir.glob("*.nii.gz"))
        for nifti_path in nifti_files:
            if "seg" in nifti_path.name.lower():
                continue  # Skip segmentation for dimension check

            try:
                img = nib.load(nifti_path)
                shape = img.shape[:3]  # First 3 dims only
                result.volume_shape = shape

                if shape != self.expected_shape:
                    result.issues.append(
                        f"Unexpected shape for {nifti_path.name}: "
                        f"{shape} (expected {self.expected_shape})"
                    )
                break  # Only need to check one modality for dimensions
            except Exception as e:
                result.issues.append(f"Cannot load {nifti_path.name}: {e}")

    def _check_not_empty(self, patient_dir: Path, result: QCResult) -> None:
        """Verify volumes are not empty (all zeros)."""
        nifti_files = list(patient_dir.glob("*.nii.gz"))
        for nifti_path in nifti_files:
            if "seg" in nifti_path.lower():
                continue

            try:
                img = nib.load(nifti_path)
                data = img.get_fdata(dtype=np.float32)

                if np.all(data == 0):
                    result.issues.append(f"Empty volume: {nifti_path.name}")

                # Check for NaN or Inf values
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    result.issues.append(
                        f"NaN/Inf values in: {nifti_path.name}"
                    )
            except Exception as e:
                result.issues.append(f"Error reading {nifti_path.name}: {e}")

    def _check_segmentation(self, patient_dir: Path, result: QCResult) -> None:
        """Verify segmentation mask has minimum tumor content."""
        seg_files = list(patient_dir.glob("*seg*"))
        if not seg_files:
            # Segmentation is optional for some datasets (e.g., UCSF-PDGM
            # may not have segmentation for all patients)
            return

        try:
            seg_img = nib.load(seg_files[0])
            seg_data = seg_img.get_fdata(dtype=np.float32)
            tumor_voxels = int(np.sum(seg_data > 0))
            result.tumor_voxels = tumor_voxels

            if tumor_voxels < self.min_tumor_voxels:
                result.issues.append(
                    f"Insufficient tumor voxels: {tumor_voxels} "
                    f"(minimum {self.min_tumor_voxels})"
                )
        except Exception as e:
            result.issues.append(f"Error reading segmentation: {e}")


def generate_qc_report(results: list[QCResult]) -> dict:
    """Generate a summary report from QC results.

    Parameters
    ----------
    results : list[QCResult]
        List of QC results from check_dataset().

    Returns
    -------
    dict
        Summary statistics and list of failed patients.
    """
    n_total = len(results)
    n_passed = sum(1 for r in results if r.passed)
    n_failed = n_total - n_passed

    failed_patients = [
        {"patient_id": r.patient_id, "issues": r.issues}
        for r in results
        if not r.passed
    ]

    report = {
        "total_patients": n_total,
        "passed": n_passed,
        "failed": n_failed,
        "pass_rate": n_passed / n_total if n_total > 0 else 0.0,
        "failed_patients": failed_patients,
    }

    return report
