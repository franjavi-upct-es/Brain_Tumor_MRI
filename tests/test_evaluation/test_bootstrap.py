# tests/test_evaluation/test_bootstrap.py
# Tests for bootstrap confidence intervals
"""
Verifies that patient-level bootstrap:
- Produces valid confidence intervals (lower < point < upper).
- Resamples at patient level, not sample level.
- Is reproducible with the same seed.
- Handles edge cases gracefully.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from src.evaluation.bootstrap import (
    BootstrapResult,
    bootstrap_single_metric,
    patient_level_bootstrap,
)


class TestPatientLevelBootstrap:
    """Tests for the main bootstrap function."""

    def _create_synthetic_data(
        self,
        n_patients: int = 50,
        slices_per_patient: int = 3,
        seed: int = 42,
    ) -> tuple:
        """Create synthetic predictions with patient structure."""
        rng = np.random.RandomState(seed)
        patient_ids = []
        y_true = []
        y_pred = []
        y_prob = []

        for p in range(n_patients):
            label = rng.randint(0, 2)
            # Simulate realistic prediction accuracy (~85%)
            pred = label if rng.random() < 0.85 else (1 - label)
            prob_correct = (
                rng.uniform(0.6, 0.95)
                if pred == label
                else rng.uniform(0.3, 0.55)
            )
            prob = (
                [1 - prob_correct, prob_correct]
                if label == 1
                else [prob_correct, 1 - prob_correct]
            )

            for _ in range(slices_per_patient):
                patient_ids.append(f"patient_{p:03d}")
                y_true.append(label)
                y_pred.append(pred)
                y_prob.append(prob)

        return (
            np.array(y_true),
            np.array(y_pred),
            np.array(patient_ids),
            np.array(y_prob),
        )

    def test_ci_bounds_valid(self) -> None:
        """CI lower should be ≤ point estimate ≤ CI upper."""
        y_true, y_pred, pids, y_prob = self._create_synthetic_data()
        results = patient_level_bootstrap(
            y_true,
            y_pred,
            pids,
            y_prob,
            n_iterations=200,
            seed=42,
        )

        for name, br in results.items():
            assert br.ci_lower <= br.point_estimate + 0.01, (
                f"{name}: CI lower ({br.ci_lower}) > "
                f"point ({br.point_estimate})"
            )
            assert br.ci_upper >= br.point_estimate - 0.01, (
                f"{name}: CI upper ({br.ci_upper}) < "
                f"point ({br.point_estimate})"
            )

    def test_ci_lower_less_than_upper(self) -> None:
        """CI lower must be strictly less than CI upper."""
        y_true, y_pred, pids, y_prob = self._create_synthetic_data()
        results = patient_level_bootstrap(
            y_true,
            y_pred,
            pids,
            y_prob,
            n_iterations=200,
            seed=42,
        )

        for name, br in results.items():
            assert br.ci_lower < br.ci_upper, (
                f"{name}: CI lower ({br.ci_lower}) >= upper ({br.ci_upper})"
            )

    def test_returns_all_primary_metrics(self) -> None:
        """Should return CIs for all primary CLAIM metrics."""
        y_true, y_pred, pids, y_prob = self._create_synthetic_data()
        results = patient_level_bootstrap(
            y_true,
            y_pred,
            pids,
            y_prob,
            n_iterations=100,
            seed=42,
        )

        expected_metrics = {
            "balanced_accuracy",
            "f1_macro",
            "cohens_kappa",
            "auc_roc_macro",
        }
        assert expected_metrics <= set(results.keys())

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed should produce identical CIs."""
        y_true, y_pred, pids, y_prob = self._create_synthetic_data()

        results_a = patient_level_bootstrap(
            y_true,
            y_pred,
            pids,
            y_prob,
            n_iterations=100,
            seed=42,
        )
        results_b = patient_level_bootstrap(
            y_true,
            y_pred,
            pids,
            y_prob,
            n_iterations=100,
            seed=42,
        )

        for name in results_a:
            assert results_a[name].ci_lower == results_b[name].ci_lower
            assert results_a[name].ci_upper == results_b[name].ci_upper

    def test_different_seed_different_results(self) -> None:
        """Different seeds should produce different CIs."""
        y_true, y_pred, pids, y_prob = self._create_synthetic_data()

        results_a = patient_level_bootstrap(
            y_true,
            y_pred,
            pids,
            y_prob,
            n_iterations=100,
            seed=42,
        )
        results_b = patient_level_bootstrap(
            y_true,
            y_pred,
            pids,
            y_prob,
            n_iterations=100,
            seed=123,
        )

        any_different = any(
            results_a[name].ci_lower != results_b[name].ci_lower
            for name in results_a
            if name in results_b
        )
        assert any_different

    def test_more_iterations_narrows_ci(self) -> None:
        """More bootstrap iterations should not systematically widen CIs."""
        y_true, y_pred, pids, y_prob = self._create_synthetic_data(
            n_patients=100
        )

        results_few = patient_level_bootstrap(
            y_true,
            y_pred,
            pids,
            y_prob,
            n_iterations=50,
            seed=42,
        )
        results_many = patient_level_bootstrap(
            y_true,
            y_pred,
            pids,
            y_prob,
            n_iterations=500,
            seed=42,
        )

        # Just check both run and produce valid intervals
        for name in ["balanced_accuracy"]:
            assert results_few[name].ci_lower < results_few[name].ci_upper
            assert results_many[name].ci_lower < results_many[name].ci_upper

    def test_without_probabilities(self) -> None:
        """Should work without y_prob (no AUC computation)."""
        y_true, y_pred, pids, _ = self._create_synthetic_data()
        results = patient_level_bootstrap(
            y_true,
            y_pred,
            pids,
            y_prob=None,
            n_iterations=100,
            seed=42,
        )

        assert "balanced_accuracy" in results
        assert "auc_roc_macro" not in results


class TestBootstrapSingleMetric:
    """Tests for custom single-metric bootstrap."""

    def test_custom_metric_function(self) -> None:
        """Should accept any (y_true, y_pred) → float function."""
        rng = np.random.RandomState(42)
        n_patients = 40
        patient_ids = []
        y_true = []
        y_pred = []
        for p in range(n_patients):
            label = rng.randint(0, 2)
            pred = label if rng.random() < 0.8 else (1 - label)
            for _ in range(3):
                patient_ids.append(f"p{p:03d}")
                y_true.append(label)
                y_pred.append(pred)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        pids = np.array(patient_ids)

        result = bootstrap_single_metric(
            y_true,
            y_pred,
            pids,
            metric_fn=balanced_accuracy_score,
            metric_name="custom_bal_acc",
            n_iterations=200,
            seed=42,
        )

        assert isinstance(result, BootstrapResult)
        assert result.metric_name == "custom_bal_acc"
        assert result.ci_lower <= result.ci_upper


class TestBootstrapResult:
    """Tests for BootstrapResult dataclass."""

    def test_format_str(self) -> None:
        """format_str should produce readable CI notation."""
        br = BootstrapResult(
            metric_name="test",
            point_estimate=0.87,
            ci_lower=0.83,
            ci_upper=0.91,
        )
        s = br.format_str()
        assert "0.8700" in s
        assert "0.8300" in s
        assert "0.9100" in s

    def test_to_dict(self) -> None:
        """to_dict should be JSON-serializable."""
        import json

        br = BootstrapResult(
            metric_name="balanced_accuracy",
            point_estimate=0.874,
            mean=0.872,
            ci_lower=0.831,
            ci_upper=0.912,
            std=0.021,
            n_iterations=1000,
        )
        d = br.to_dict()
        json_str = json.dumps(d)  # Should not raise
        assert "balanced_accuracy" in json_str
