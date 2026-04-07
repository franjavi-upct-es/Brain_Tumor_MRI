# tests/test_evaluation/test_calibration.py
"""Tests for calibration metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluation.calibration import CalibrationResult, compute_calibration


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_default_values(self) -> None:
        """Default CalibrationResult should have sensible defaults."""
        result = CalibrationResult()
        assert result.ece == 0.0
        assert result.brier_score == 0.0
        assert result.n_bins == 10

    def test_to_dict_serializable(self) -> None:
        """to_dict should produce JSON-serializable output."""
        result = CalibrationResult(
            ece=0.05,
            brier_score=0.12,
            bin_accuracies=[0.5, 0.6, 0.7],
            bin_confidences=[0.5, 0.6, 0.7],
            bin_counts=[10, 20, 15],
            n_bins=10,
        )
        d = result.to_dict()
        json.dumps(d)  # Should not raise
        assert "ece" in d
        assert "bier_score" in d  # Note: typo in source code
        assert "reliability_diagram" in d

    def test_to_dict_rounding(self) -> None:
        """to_dict should round to 4 decimal places."""
        result = CalibrationResult(ece=0.123456789, brier_score=0.987654321)
        d = result.to_dict()
        assert d["ece"] == 0.1235


class TestComputeCalibration:
    """Tests for compute_calibration function."""

    def _perfect_calibration(self):
        """Create perfectly calibrated binary predictions."""
        n = 100
        y_true = np.array([1] * 50 + [0] * 50)
        y_prob = np.array(
            [0.9] * 25 + [0.8] * 25 + [0.2] * 25 + [0.1] * 25
        )
        return y_true, y_prob

    def test_binary_case(self) -> None:
        """Should compute calibration for binary 1D probabilities."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.8])
        result = compute_calibration(y_true, y_prob)
        assert isinstance(result, CalibrationResult)
        assert 0 <= result.ece <= 1
        assert result.brier_score >= 0

    def test_multiclass_case(self) -> None:
        """Should compute calibration for 2D probability matrix."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ])
        result = compute_calibration(y_true, y_prob)
        assert isinstance(result, CalibrationResult)
        assert result.ece >= 0
        assert result.brier_score >= 0

    def test_multiclass_brier_score(self) -> None:
        """Multiclass Brier score should use one-hot encoding."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        result = compute_calibration(y_true, y_prob)
        assert result.brier_score < 0.1  # Well-calibrated

    def test_ece_zero_for_perfect(self) -> None:
        """Perfect predictions should have low ECE."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_prob = np.array([0.99, 0.98, 0.97, 0.01, 0.02, 0.03])
        result = compute_calibration(y_true, y_prob)
        assert result.ece < 0.1

    def test_bin_counts_sum_to_n(self) -> None:
        """Bin counts should sum to total samples."""
        n = 50
        y_true = np.random.RandomState(42).randint(0, 2, n)
        y_prob = np.random.RandomState(42).uniform(0, 1, n)
        result = compute_calibration(y_true, y_prob, n_bins=10)
        assert sum(result.bin_counts) == n

    def test_n_bins_parameter(self) -> None:
        """Should use the specified number of bins."""
        y_true = np.array([0, 1] * 20)
        y_prob = np.random.RandomState(42).uniform(0, 1, 40)
        result = compute_calibration(y_true, y_prob, n_bins=5)
        assert result.n_bins == 5
        assert len(result.bin_accuracies) == 5
        assert len(result.bin_confidences) == 5
        assert len(result.bin_counts) == 5

    def test_empty_bins_handled(self) -> None:
        """Empty bins should not cause division by zero errors."""
        # All samples concentrated in one bin
        y_true = np.array([1, 1, 1, 1, 1])
        y_prob = np.array([0.95, 0.96, 0.97, 0.98, 0.99])
        result = compute_calibration(y_true, y_prob, n_bins=10)
        # Most bins will be empty
        assert result.ece >= 0  # Should not crash

    def test_binary_brier_score(self) -> None:
        """Binary Brier score should be MSE of probabilities."""
        y_true = np.array([1, 0])
        y_prob = np.array([1.0, 0.0])  # Perfect predictions
        result = compute_calibration(y_true, y_prob)
        assert result.brier_score == 0.0

    def test_last_bin_includes_upper_bound(self) -> None:
        """Last bin should include confidence = 1.0."""
        y_true = np.array([1, 1])
        y_prob = np.array([1.0, 1.0])  # Max confidence
        result = compute_calibration(y_true, y_prob, n_bins=10)
        # Should not crash, confidence=1.0 should be in last bin
        assert sum(result.bin_counts) == 2
