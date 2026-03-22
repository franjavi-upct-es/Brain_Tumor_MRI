# src/experiments/split_comparison.py
# Image-level vs patient-level split comparison
"""
The most important ablation study in the thesis: quantifying how much
image-level splitting inflates accuracy compared to patient-level splitting.

Yagis et al. (2021, Scientific Reports) demonstrated:
  - OASIS dataset: 30% accuracy inflation from slice-level CV.
  - PPMI dataset: 48% accuracy inflation.
  - Randomly labeled data: ~96% accuracy with slice-level splitting,
    proving the model learned patient identity, not disease features.

This experiment reproduces the finding on BraTS data by training the SAME
model with the SAME data under two splitting strategies:

  Image-level split: random 80/10/10 by image (LEAKY — images from the
    same patient can appear in both train and test).
  Patient-level split: StratifiedGroupKFold by patient (CORRECT — all
    images from one patient in the same fold).

The expected result is a 7-30 percentage point gap, directly demonstrating
why methodological rigor matters more than accuracy numbers.

References:
    - Yagis et al. (2021). Effect of data leakage in brain MRI classification
      using 2D CNNs. Scientific Reports, 11, 22544.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class SplitComparisonResult:
    """Result of the image-level vs patient-level split comparison.

    Attributes
    ----------
    image_level_metrics : dict[str, float]
        Metrics from image-level (leaky) splitting.
    patient_level_metrics : dict[str, float]
        Metrics from patient-level (correct) splitting.
    delta : dict[str, float]
        Difference (patient - image) for each metric. Expected to be negative
        because patient-level splitting is harder (no leakage to exploit).
    image_level_per_fold : list[dict[str, float]]
        Per-fold metrics for image-level splitting.
    patient_level_per_fold : list[dict[str, float]]
        Per-fold metrics for patient-level splitting.
    n_patients : int
        Total number of unique patients in the dataset.
    n_samples : int
        Total number of samples (slices).
    leakage_detected : bool
        Whether data leakage was confirmed in the image-level splits.
    """

    image_level_metrics: dict[str, float] = field(default_factory=dict)
    patient_level_metrics: dict[str, float] = field(default_factory=dict)
    delta: dict[str, float] = field(default_factory=dict)
    image_level_per_fold: list[dict[str, float]] = field(default_factory=list)
    patient_level_per_fold: list[dict[str, float]] = field(
        default_factory=list
    )
    n_patients: int = 0
    n_samples: int = 0
    leakage_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "image_level": {
                "mean_metrics": self.image_level_metrics,
                "per_fold": self.image_level_per_fold,
            },
            "patient_level": {
                "mean_metrics": self.patient_level_metrics,
                "per_fold": self.patient_level_per_fold,
            },
            "delta": self.delta,
            "dataset_info": {
                "n_patients": self.n_patients,
                "n_samples": self.n_samples,
            },
            "leakage_detected": self.leakage_detected,
        }


def create_image_level_splits(
    labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create image-level (LEAKY) stratified k-fold splits.

    This is the INCORRECT approach used in most Kaggle notebooks. It ignores
    patient grouping entirely, allowing slices from the same patient to
    appear in both training and validation sets.

    Parameters
    ----------
    labels : np.ndarray
        Class labels for each sample.
    n_splits : int
        Number of folds.
    random_state : int
        Random seed.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) for each fold.
    """
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    return list(skf.split(np.arange(len(labels)), labels))


def create_patient_level_splits(
    labels: np.ndarray,
    patient_ids: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create patient-level (CORRECT) stratified group k-fold splits.

    This is the correct approach. All samples from one patient stay together.

    Parameters
    ----------
    labels : np.ndarray
        Class labels for each sample.
    patient_ids : np.ndarray
        Patient ID for each sample.
    n_splits : int
        Number of folds.
    random_state : int
        Random seed.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) for each fold.
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    return list(sgkf.split(np.arange(len(labels)), labels, groups=patient_ids))


def detect_leakage_in_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    patient_ids: np.ndarray,
) -> dict[str, Any]:
    """Detect and quantify data leakage in a set of splits.

    Checks whether any patient appears in both training and validation
    sets for any fold, and quantifies the extent of leakage.

    Parameters
    ----------
    splits : list[tuple]
        List of (train_indices, val_indices).
    patient_ids : np.ndarray
        Patient ID for each sample.

    Returns
    -------
    dict
        Leakage analysis with counts and patient lists.
    """
    total_leaked_patients = set()
    per_fold_leakage = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_patients = set(patient_ids[train_idx])
        val_patients = set(patient_ids[val_idx])
        overlap = train_patients & val_patients

        leaked_samples_train = sum(
            1 for i in train_idx if patient_ids[i] in overlap
        )
        leaked_samples_val = sum(
            1 for i in val_idx if patient_ids[i] in overlap
        )

        per_fold_leakage.append(
            {
                "fold": fold_idx,
                "n_leaked_patients": len(overlap),
                "leaked_patient_ids": sorted(str(p) for p in overlap),
                "leaked_samples_in_train": leaked_samples_train,
                "leaked_samples_in_val": leaked_samples_val,
            }
        )

        total_leaked_patients.update(overlap)

    unique_patients = set(patient_ids)
    leakage_rate = len(total_leaked_patients) / max(len(unique_patients), 1)

    result = {
        "has_leakage": len(total_leaked_patients) > 0,
        "n_leaked_patients_total": len(total_leaked_patients),
        "n_unique_patients": len(unique_patients),
        "leakage_rate": float(leakage_rate),
        "per_fold": per_fold_leakage,
    }

    if result["has_leakage"]:
        logger.warning(
            "DATA LEAKAGE DETECTED: %d/%d patients (%.1f%%) appear in "
            "both train and val across folds",
            len(total_leaked_patients),
            len(unique_patients),
            leakage_rate * 100,
        )
    else:
        logger.info(
            "No leakage detected — all splits are patient-level clean."
        )

    return result


def run_split_comparison(
    labels: np.ndarray,
    patient_ids: np.ndarray,
    evaluate_fn: Any,
    n_splits: int = 5,
    random_state: int = 42,
) -> SplitComparisonResult:
    """Run the complete image-level vs patient-level comparison.

    Parameters
    ----------
    labels : np.ndarray
        Class labels for each sample.
    patient_ids : np.ndarray
        Patient ID for each sample.
    evaluate_fn : callable
        Function that takes (train_indices, val_indices) and returns
        a dict of metric_name → value.
    n_splits : int
        Number of cross-validation folds.
    random_state : int
        Random seed.

    Returns
    -------
    SplitComparisonResult
        Complete comparison results.
    """
    result = SplitComparisonResult(
        n_patients=len(set(patient_ids)),
        n_samples=len(labels),
    )

    # --- Image-level (leaky) splits ---
    logger.info("Running image-level (LEAKY) %d-fold CV...", n_splits)
    image_splits = create_image_level_splits(labels, n_splits, random_state)

    # Detect and quantify leakage
    leakage_info = detect_leakage_in_splits(image_splits, patient_ids)
    result.leakage_detected = leakage_info["has_leakage"]

    image_fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(image_splits):
        metrics = evaluate_fn(train_idx, val_idx)
        image_fold_metrics.append(metrics)
        logger.info(
            "  Image-level fold %d: %s",
            fold_idx,
            {k: f"{v:.4f}" for k, v in metrics.items()},
        )

    result.image_level_per_fold = image_fold_metrics

    # --- Patient-level (correct) splits ---
    logger.info("Running patient-level (CORRECT) %d-fold CV...", n_splits)
    patient_splits = create_patient_level_splits(
        labels, patient_ids, n_splits, random_state
    )

    patient_fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(patient_splits):
        metrics = evaluate_fn(train_idx, val_idx)
        patient_fold_metrics.append(metrics)
        logger.info(
            "  Patient-level fold %d: %s",
            fold_idx,
            {k: f"{v:.4f}" for k, v in metrics.items()},
        )

    result.patient_level_per_fold = patient_fold_metrics

    # --- Aggregate and compute deltas ---
    metric_keys = set()
    for m in image_fold_metrics + patient_fold_metrics:
        metric_keys.update(m.keys())

    for key in sorted(metric_keys):
        img_values = [m[key] for m in image_fold_metrics if key in m]
        pat_values = [m[key] for m in patient_fold_metrics if key in m]

        if img_values and pat_values:
            img_mean = float(np.mean(img_values))
            pat_mean = float(np.mean(pat_values))

            result.image_level_metrics[key] = img_mean
            result.patient_level_metrics[key] = pat_mean
            result.delta[key] = float(pat_mean - img_mean)

    # --- Log summary ---
    logger.info("\n" + "=" * 60)
    logger.info("SPLIT COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info(
        "Leakage detected in image-level splits: %s", result.leakage_detected
    )
    for key in sorted(result.delta.keys()):
        logger.info(
            "  %s: image=%.4f, patient=%.4f, delta=%+.4f",
            key,
            result.image_level_metrics[key],
            result.patient_level_metrics[key],
            result.delta[key],
        )

    return result
