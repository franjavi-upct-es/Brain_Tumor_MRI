# src/evaluation/calibration.py — Model calibration metrics
"""
Calibration metrics assess whether predicted probabilities match observed
frequencies. A model predicting 70% probability should be correct ~70% of
the time. Poorly calibrated models can be dangerously overconfident.

Metrics:
    - Expected Calibration Error (ECE): weighted average of bin-wise
      |accuracy - confidence| differences.
    - Brier Score: mean squared error of probability estimates.
    - Reliability diagram data: for plotting calibration curves.

References:
    - Naeini et al. (2015). Obtaining well calibrated probabilities using
      Bayesian binning into quantiles. AAAI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """
    Calibration evaluation results.

    Attributes
    ----------
    ece : float
        Expected Calibration Error.
    brier_score : float
        Brier score (mean squared probability error).
    bin_accuracies : list[float]
        Accuracy in each calibration bin.
    bin_confidences : list[float]
        Mean predicted confidence in each bin.
    bin_counts : list[int]
        Number of samples in each bin.
    n_bins : int
        Number of calibration bins used.
    """

    ece: float = 0.0
    brier_score: float = 0.0
    bin_accuracies: list[float] = field(default_factory=list)
    bin_confidences: list[float] = field(default_factory=list)
    bin_counts: list[float] = field(default_factory=list)
    n_bins: int = 10

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "ece": round(self.ece, 4),
            "bier_score": round(self.brier_score, 4),
            "reliability_diagram": {
                "bin_accuracies": [round(a, 4) for a in self.bin_accuracies],
                "bin_confidences": [round(c, 4) for c in self.bin_confidences],
                "bin_counts": self.bin_counts,
                "n_bins": self.n_bins,
            },
        }


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """
    Compute all calibration metrics

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1), shape (n_samples,).
    y_prob : np.ndarray
        Predicted probabilities for the positive class, shape (n_samples,)
        or (n_samples, n_classes).
    n_bins : int
        Number of bins for ECE and reliability diagram. Default 10.

    Returns
    -------
    CalibrationResult
        Complete calibration assessment.
    """
    # Handle multi-class probabilities
    if y_prob.ndim == 2:
        # Use max probability as confidence, predicted class for correctness
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
    else:
        # Binary case: y_prob if P(class=1)
        confidences = np.where(y_prob >= 0.5, y_prob, 1 - y_prob)
        predictions = (y_prob >= 0.5).astype(int)
        accuracies = (predictions == y_true).astype(float)

    result = CalibrationResult(n_bins=n_bins)

    # --- Expected Calibration Error (ECE) ---
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Find samples in this bin
        if i == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        bin_count = int(in_bin.sum())
        result.bin_counts.append(bin_count)

        if bin_count > 0:
            bin_acc = float(accuracies[in_bin].mean())
            bin_conf = float(confidences[in_bin].mean())
            ece += (bin_count / n_total) * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = (lower + upper) / 2

        result.bin_accuracies.append(bin_acc)
        result.bin_confidences.append(bin_conf)

    result.ece = float(ece)

    # --- Brier Score ---
    if y_prob.ndim == 2:
        # Multi-class Brier score
        n_classes = y_prob.shape[1]
        one_hot = np.eye(n_classes)[y_true]
        result.brier_score = float(
            np.mean(np.sum((y_prob - one_hot) ** 2, axis=1))
        )
    else:
        # Binary Brier score
        result.brier_score = float(np.mean((y_prob - y_true) ** 2))

    logger.info(
        "Calibration: ECE=%.4f, Brier=%.4f, (n_bins=%d)",
        result.ece,
        result.brier_score,
        n_bins,
    )

    return result
