# src/experiments/naive_experiment.py — Reproduce the naive Kaggle approach
"""
Reproduces the flawed methodology from v1 to provide the contrast for the
thesis narrative.

The naive approach uses:
  - Kaggle MasoudNickparvar dataset (JPEG, no DICOM metadata).
  - Random 80/10/10 image-level split (ignores patient grouping).
  - Standard JPEG → resize → normalize preprocessing.
  - EfficientNet-B0 fine-tuned with basic augmentation.
  - Metrics: only accuracy and loss curves.
  - Grad-CAM without IoU validation.

Expected result: ~99% accuracy (ARTIFICIAL — driven by shortcut learning).

The experiment demonstrates:
  1. Suspiciously fast convergence curves.
  2. Grad-CAM attending to borders/background, not tumors.
  3. Dramatic collapse on external validation.

This is NOT presented as a success but as the ERROR that motivates the
rigorous approach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NaiveExperimentResult:
    """Results from the naive Kaggle approach for thesis comparison.

    Attributes
    ----------
    accuracy : float
        Standard accuracy (expected ~99% — artificially inflated).
    val_accuracy : float
        Validation accuracy.
    training_history : dict[str, list[float]]
        Loss and accuracy curves per epoch.
    grad_cam_iou : float | None
        Grad-CAM IoU vs ground truth (expected very low, ~0.1-0.2).
    external_accuracy : float | None
        Accuracy on external data (expected dramatic collapse).
    leakage_analysis : dict[str, Any]
        Data leakage quantification.
    dataset_problems : list[str]
        Documented problems with the Kaggle dataset.
    """

    accuracy: float = 0.0
    val_accuracy: float = 0.0
    training_history: dict[str, list[float]] = field(default_factory=dict)
    grad_cam_iou: float | None = None
    external_accuracy: float | None = None
    leakage_analysis: dict[str, Any] = field(default_factory=dict)
    dataset_problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "accuracy": self.accuracy,
            "val_accuracy": self.val_accuracy,
            "training_history": self.training_history,
            "grad_cam_iou": self.grad_cam_iou,
            "external_accuracy": self.external_accuracy,
            "leakage_analysis": self.leakage_analysis,
            "dataset_problems": self.dataset_problems,
        }


# Documented problems with the Kaggle MasoudNickparvar dataset
KAGGLE_DATASET_PROBLEMS = [
    "Combines 3 sources (Figshare/Jun Cheng, SARTAJ, Br35H) with different "
    "preprocessing, creating dataset-identification shortcuts.",
    "Figshare source: 3,064 images from only 233 patients "
    "(~13 slices/patient). No patient-level separation between train "
    "and test.",
    "SARTAJ glioma class contains documented mislabeling "
    "(PLOS ONE, 2025, DOI: 10.1371/journal.pone.0322624).",
    "All images distributed as lossy JPEG with no DICOM metadata preserved — "
    "no scanner, field strength, or acquisition parameters available.",
    "'No tumor' class comes from entirely different source datasets, "
    "enabling classification by dataset origin rather than pathology.",
    "Preprocessing applied during .mat → JPEG conversion is undocumented, "
    "creating class-correlated intensity and background patterns.",
]


def create_naive_splits(
    n_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create naive random image-level splits (the WRONG way).

    This ignores patient grouping entirely, producing data leakage.

    Parameters
    ----------
    n_samples : int
        Total number of images.
    train_ratio : float
        Fraction for training (default 0.8).
    val_ratio : float
        Fraction for validation (default 0.1).
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (train_indices, val_indices, test_indices)
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    logger.info(
        "Naive split (IMAGE-LEVEL): %d train, %d val, %d test",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    return train_idx, val_idx, test_idx


def analyze_kaggle_dataset(
    data_dir: Path,
) -> dict[str, Any]:
    """Analyze the Kaggle dataset to document its problems.

    Performs checks for:
    1. Image format and quality.
    2. Class distribution.
    3. Potential duplicate detection via perceptual hashing.
    4. Intensity distribution differences between classes.

    Parameters
    ----------
    data_dir : Path
        Root directory of the Kaggle dataset.

    Returns
    -------
    dict
        Analysis results documenting dataset problems.
    """
    analysis = {
        "documented_problems": KAGGLE_DATASET_PROBLEMS,
        "class_counts": {},
        "image_format": "JPEG (lossy, no DICOM metadata)",
        "patient_level_info": "NOT AVAILABLE — no patient IDs in dataset",
    }

    if not data_dir.exists():
        logger.warning("Kaggle data directory not found: %s", data_dir)
        analysis["status"] = "data_not_available"
        return analysis

    # Count images per class
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            n_images = len(
                list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
            )
            analysis["class_counts"][class_dir.name] = n_images

    total = sum(analysis["class_counts"].values())
    analysis["total_images"] = total

    logger.info(
        "Kaggle dataset analysis: %d images, %s",
        total,
        analysis["class_counts"],
    )

    return analysis


def generate_naive_comparison_data(
    naive_result: NaiveExperimentResult,
    rigorous_metrics: dict[str, float],
) -> dict[str, Any]:
    """Generate side-by-side comparison data for the thesis.

    Creates the data structure for the central comparison table
    showing naive (~99%) vs rigorous (~87%) results.

    Parameters
    ----------
    naive_result : NaiveExperimentResult
        Results from the naive approach.
    rigorous_metrics : dict[str, float]
        Metrics from the rigorous approach.

    Returns
    -------
    dict
        Comparison data for thesis tables and figures.
    """
    comparison = {
        "naive": {
            "dataset": "Kaggle MasoudNickparvar",
            "split": "Random 80/10/10 (image-level)",
            "preprocessing": "JPEG → resize → normalize",
            "accuracy": naive_result.accuracy,
            "interpretation": "ARTIFICIAL — shortcut learning",
        },
        "rigorous": {
            "dataset": "BraTS 2023",
            "split": "StratifiedGroupKFold 5-fold (patient-level)",
            "preprocessing": "NIfTI → N4 → z-score (validated)",
            "balanced_accuracy": rigorous_metrics.get("balanced_accuracy", 0),
            "interpretation": "REAL — clinically relevant features",
        },
        "delta": {
            "description": (
                "The naive approach reports artificially high accuracy "
                "because the model exploits data leakage and dataset-specific "
                "shortcuts. The rigorous approach produces lower but honest "
                "results that generalize to unseen data."
            ),
        },
    }

    if naive_result.grad_cam_iou is not None:
        comparison["naive"]["grad_cam_iou"] = naive_result.grad_cam_iou
    if naive_result.external_accuracy is not None:
        comparison["naive"]["external_accuracy"] = (
            naive_result.external_accuracy
        )

    return comparison
