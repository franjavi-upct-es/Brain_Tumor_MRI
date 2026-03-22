# tests/test_evaluation/test_metrics.py — Tests for evaluation metrics
"""
Verifies that all CLAIM-compliant metrics are computed correctly:
- Balanced accuracy, F1, AUC-ROC, Cohen's kappa
- Per-class sensitivity, specificity, precision
- Confusion matrix (absolute and normalized)
- Classification report structure
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    compute_full_report,
    compute_roc_curve_data,
)


class TestMetricsComputation:
    """Tests for metric correctness."""

    def _perfect_predictions(self) -> tuple:
        """Create perfect predictions for sanity checks."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.95, 0.05],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.05, 0.95],
                [0.15, 0.85],
                [0.1, 0.9],
            ]
        )
        return y_true, y_pred, y_prob

    def _imperfect_predictions(self) -> tuple:
        """Create realistic imperfect predictions."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
        y_prob = np.array(
            [
                [0.8, 0.2],
                [0.7, 0.3],
                [0.6, 0.4],
                [0.4, 0.6],
                [0.3, 0.7],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.1, 0.9],
                [0.6, 0.4],
                [0.7, 0.3],
            ]
        )
        return y_true, y_pred, y_prob

    def test_perfect_predictions_give_perfect_scores(self) -> None:
        """Perfect predictions should yield metrics = 1.0."""
        y_true, y_pred, y_prob = self._perfect_predictions()
        report = compute_full_report(y_true, y_pred, y_prob)

        assert report.balanced_accuracy == 1.0
        assert report.f1_macro == 1.0
        assert report.accuracy == 1.0
        assert report.cohens_kappa == 1.0
        assert report.auc_roc_macro == 1.0

    def test_balanced_accuracy_handles_imbalance(self) -> None:
        """Balanced accuracy should differ from accuracy on imbalanced data."""
        # Heavily imbalanced: 8 class-0, 2 class-1
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Predict all 0
        report = compute_full_report(y_true, y_pred)

        assert report.accuracy == 0.8  # 8/10 correct
        assert report.balanced_accuracy == 0.5  # (1.0 + 0.0) / 2

    def test_per_class_sensitivity(self) -> None:
        """Per-class sensitivity should equal per-class recall."""
        y_true, y_pred, y_prob = self._imperfect_predictions()
        report = compute_full_report(y_true, y_pred, y_prob)

        # Class 0: 3 TP out of 5 positives → sensitivity = 0.6
        assert abs(report.per_class_sensitivity[0] - 0.6) < 1e-6
        # Class 1: 3 TP out of 5 positives → sensitivity = 0.6
        assert abs(report.per_class_sensitivity[1] - 0.6) < 1e-6

    def test_per_class_specificity(self) -> None:
        """Specificity should be TN / (TN + FP)."""
        y_true, y_pred, y_prob = self._imperfect_predictions()
        report = compute_full_report(y_true, y_pred, y_prob)

        # Class 0: TN=3, FP=2 → specificity = 3/5 = 0.6
        assert abs(report.per_class_specificity[0] - 0.6) < 1e-6

    def test_confusion_matrix_shape(self) -> None:
        """Confusion matrix should be (n_classes, n_classes)."""
        y_true, y_pred, y_prob = self._imperfect_predictions()
        report = compute_full_report(y_true, y_pred, y_prob)

        assert report.confusion_matrix_absolute.shape == (2, 2)
        assert report.confusion_matrix_normalized.shape == (2, 2)

    def test_confusion_matrix_rows_sum_correctly(self) -> None:
        """Absolute CM rows should sum to class count; normalized to ~100%."""
        y_true, y_pred, _ = self._imperfect_predictions()
        report = compute_full_report(y_true, y_pred)

        cm = report.confusion_matrix_absolute
        assert cm[0].sum() == 5  # 5 actual class-0 samples
        assert cm[1].sum() == 5  # 5 actual class-1 samples

        cm_norm = report.confusion_matrix_normalized
        for row in cm_norm:
            assert abs(row.sum() - 100.0) < 0.2  # Rounding tolerance

    def test_auc_roc_requires_probabilities(self) -> None:
        """AUC-ROC should be 0 when no probabilities are provided."""
        y_true, y_pred, _ = self._imperfect_predictions()
        report = compute_full_report(y_true, y_pred, y_prob=None)
        assert report.auc_roc_macro == 0.0

    def test_n_samples_tracked(self) -> None:
        """Report should track total samples and per-class counts."""
        y_true, y_pred, y_prob = self._imperfect_predictions()
        report = compute_full_report(y_true, y_pred, y_prob)
        assert report.n_samples == 10
        assert report.n_per_class[0] == 5
        assert report.n_per_class[1] == 5


class TestReportSerialization:
    """Tests for ClassificationReport JSON export."""

    def test_to_dict_structure(self) -> None:
        """to_dict() should contain all required sections."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.6, 0.4]])
        report = compute_full_report(y_true, y_pred, y_prob)

        d = report.to_dict()
        assert "primary_metrics" in d
        assert "secondary_metrics" in d
        assert "per_class" in d
        assert "confusion_matrix" in d
        assert "dataset_info" in d

    def test_primary_metrics_keys(self) -> None:
        """Primary metrics should include all CLAIM-required metrics."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        report = compute_full_report(y_true, y_pred)
        d = report.to_dict()

        required_keys = {
            "balanced_accuracy",
            "f1_macro",
            "auc_roc_macro",
            "cohens_kappa",
        }
        assert required_keys <= set(d["primary_metrics"].keys())


class TestROCCurveData:
    """Tests for ROC curve data generation."""

    def test_roc_curve_structure(self) -> None:
        """ROC curve data should contain fpr, tpr, thresholds, and auc."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array(
            [
                [0.9, 0.1],
                [0.7, 0.3],
                [0.6, 0.4],
                [0.3, 0.7],
                [0.2, 0.8],
                [0.1, 0.9],
            ]
        )
        data = compute_roc_curve_data(y_true, y_prob)

        assert "fpr" in data
        assert "tpr" in data
        assert "auc" in data
        assert 0 <= data["auc"] <= 1

    def test_perfect_roc_auc(self) -> None:
        """Perfect separation should give AUC = 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.1, 0.9],
                [0.0, 1.0],
                [0.05, 0.95],
            ]
        )
        data = compute_roc_curve_data(y_true, y_prob)
        assert data["auc"] == 1.0
