# src/evaluation/bootstrap.py — Bootstrap confidence intervals
"""
Patient-level bootstrap for computing 95% confidence intervals on all metrics.

CRITICAL: Bootstrap must be at the PATIENT level, not the sample level.
Multiple slices from the same patient are correlated, so resampling at the
slice level would underestimate confidence interval width.

Procedure:
    1. Group predictions by patient_id.
    2. Resample patients with replacement (1,000 iterations).
    3. Compute metrics on each bootstrap sample.
    4. Report mean and 2.5th/97.5th percentiles as 95% CI.

References:
    - Efron & Tibshirani (1993). An Introduction to the Bootstrap.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.evaluation.metrics import compute_full_report

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """
    Result of bootstrap confidence interval estimation.

    Attributes
    ----------
    metric_name : str
        Name of the metric.
    point_estimate : float
        Metric computed on the full dataset.
    mean : float
        Mean across bootstrap iterations.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    std : float
        Standard deviation across bootstrap interations.
    n_iterations : int
        Number of bootstrap iterations performed.
    confidence_level : float
        Confidence level (e.g., 0.95, for 95% CI).
    """

    metric_name: str = ""
    point_estimate: float = 0.0
    mean: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    std: float = 0.0
    n_iterations: int = 0
    confidence_level: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "metric": self.metric_name,
            "point_estimate": round(self.point_estimate, 4),
            "mean": round(self.mean, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "srd": round(self.std, 4),
            "n_iterations": self.n_iterations,
            "confidence_level": self.confidence_level,
        }

    def format_str(self) -> str:
        """Format as 'point_estimate (CI_lower - CI_upper)'."""
        return (
            f"{self.point_estimate:.4f} "
            f"({self.ci_lower:.4f} — {self.ci_upper:.4f})"
        )


def patient_level_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
    y_prob: np.ndarray | None = None,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, BootstrapResult]:
    """
    Compute bootstrap CIs for all primary metrics at the patient level.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted labels, shape (n_samples,).
    patient_ids : np.ndarray
        Patient ID for each sample. Used for patient-level resampling.
    y_prob : np.ndarray, optional
        Predicted probabilities, shape (n_samples, n_classes).
    n_iterations : int
        Number of bootstrap iterations. Default 1000
    confidence_level : float
        Confidence level for the interval. Default 0.95
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, BootstrapResult]
        Mapping from metric name to bootstrap result
    """
    rng = np.random.RandomState(seed)
    alpha = 1.0 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    # Get unique patients
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    # Build patient-to-index mapping
    patient_to_indices = {}
    for i, pid in enumerate(patient_ids):
        if pid not in patient_to_indices:
            patient_to_indices[pid] = []
        patient_to_indices[pid].append(i)

    # Compute point estimates on full data
    full_report = compute_full_report(y_true, y_pred, y_prob)
    point_estimates = {
        "balanced_accuracy": full_report.balanced_accuracy,
        "f1_macro": full_report.f1_macro,
        "cohens_kappa": full_report.cohens_kappa,
        "accuracy": full_report.accuracy,
    }
    if y_prob is not None:
        point_estimates["auc_roc_macro"] = full_report.auc_roc_macro

    # Bootstrap iterations
    bootstrap_values: dict[str, list[float]] = {k: [] for k in point_estimates}

    for _ in range(n_iterations):
        # Resample patients with replacement
        resampled_patients = rng.choice(
            unique_patients, size=n_patients, replace=True
        )

        # Gather indices for resampled patients
        indices = []
        for pid in resampled_patients:
            indices.extend(patient_to_indices[pid])

        boot_true = y_true[indices]
        boot_pred = y_pred[indices]
        boot_prob = y_prob[indices] if y_prob is not None else None

        # Compute metrics on bootstrap sample
        try:
            boot_report = compute_full_report(boot_true, boot_pred, boot_prob)
            bootstrap_values["balanced_accuracy"].append(
                boot_report.balanced_accuracy
            )
            bootstrap_values["f1_macro"].append(boot_report.f1_macro)
            bootstrap_values["cohens_kappa"].append(boot_report.cohens_kappa)
            bootstrap_values["accuracy"].append(boot_report.accuracy)
            if y_prob is not None:
                bootstrap_values["auc_roc_macro"].append(
                    boot_report.auc_roc_macro
                )
        except Exception:
            # Skip failed iterations (can happen with degenerated samples)
            continue

    # Compute CIs
    results = {}
    for metric_name, values in bootstrap_values.items():
        if not values:
            continue

        values_arr = np.array(values)
        results[metric_name] = BootstrapResult(
            metric_name=metric_name,
            point_estimate=point_estimates[metric_name],
            mean=float(np.mean(values_arr)),
            ci_lower=float(np.percentile(values_arr, lower_percentile)),
            ci_upper=float(np.percentile(values_arr, upper_percentile)),
            std=float(np.std(values_arr)),
            n_iterations=len(values),
            confidence_level=confidence_level,
        )

    logger.info(
        "Bootstrap complete (%d iterations, %d patients):",
        n_iterations,
        n_patients,
    )
    for name, res in results.items():
        logger.info("   %s: %s", name, res.format_str())

    return results


def bootstrap_single_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    metric_name: str = "custom",
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """
    Bootstrap CI for a single custom metric function.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    patient_ids : np.ndarray
        Patient IDs for patient-level resampling.
    metric_fn : callable
        Function that takes (y_true, y_pred) and returns a float.
    metric_name : str
        Name for the result.
    n_iterations : int
        Number of bootstrap iterations.
    confidence_level : flaot
        Confidence level.
    seed : int
        Random seed.

    Returns
    -------
    BootstrapResult
        Bootstrap confidence interval result.
    """
    rng = np.random.RandomState(seed)
    alpha = 1.0 - confidence_level

    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    patient_to_indices = {}
    for i, pid in enumerate(patient_ids):
        if pid not in patient_to_indices:
            patient_to_indices[pid] = []
        patient_to_indices[pid].append(i)

    point_estimate = float(metric_fn(y_true, y_pred))
    values = []

    for _ in range(n_iterations):
        resampled = rng.choice(unique_patients, size=n_patients, replace=True)
        indices = []
        for pid in resampled:
            indices.extend(patient_to_indices[pid])

        try:
            val = float(metric_fn(y_true[indices], y_pred[indices]))
            values.append(val)
        except Exception:
            continue

    values_arr = np.array(values)
    return BootstrapResult(
        metric_name=metric_name,
        point_estimate=point_estimate,
        mean=float(np.mean(values_arr)),
        ci_lower=float(np.percentile(values_arr, (alpha / 2) * 100)),
        ci_upper=float(np.percentile(values_arr, (1 - alpha / 2) * 100)),
        std=float(np.std(values_arr)),
        n_iterations=len(values),
        confidence_level=confidence_level,
    )
