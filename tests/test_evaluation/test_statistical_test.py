# tests/test_evaluation/test_statistical_tests.py — Tests for statistical tests
"""
Verifies McNemar's test, DeLong's test, and Wilcoxon signed-rank test.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.evaluation.statistical_tests import (
    TestResult,
    delong_test,
    mcnemar_test,
    wilcoxon_signed_rank_test,
)


class TestMcNemarTest:
    """Tests for McNemar's test."""

    def test_identical_models_not_significant(self) -> None:
        """Two models with identical errors should not be significant."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 1])
        result = mcnemar_test(y_true, y_pred, y_pred)
        assert result.p_value == 1.0
        assert not result.significant

    def test_very_different_models_significant(self) -> None:
        """Highly discordant predictions should be significant."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        y_pred_a = y_true.copy()  # Perfect model
        y_pred_b = rng.randint(0, 2, 200)  # Random model
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert result.p_value < 0.05
        assert result.significant

    def test_returns_test_result(self) -> None:
        """Should return a TestResult dataclass."""
        y_true = np.array([0, 1, 0, 1])
        y_pred_a = np.array([0, 1, 0, 0])
        y_pred_b = np.array([0, 0, 1, 1])
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert isinstance(result, TestResult)
        assert result.test_name == "McNemar"

    def test_to_dict_serializable(self) -> None:
        """to_dict should produce valid JSON."""
        import json

        y_true = np.array([0, 1, 0, 1])
        y_pred_a = np.array([0, 1, 0, 0])
        y_pred_b = np.array([0, 0, 1, 1])
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        d = result.to_dict()
        json.dumps(d)  # Should not raise


class TestDeLongTest:
    """Tests for DeLong's AUC comparison test."""

    def test_identical_aucs_not_significant(self) -> None:
        """Same probabilities should give p ≈ 1."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.2, 0.3, 0.4, 0.7, 0.8, 0.9])
        result = delong_test(y_true, y_prob, y_prob)
        assert result.p_value > 0.5

    def test_different_aucs(self) -> None:
        """Models with clearly different AUCs should be distinguishable."""
        rng = np.random.RandomState(42)
        y_true = np.concatenate([np.zeros(100), np.ones(100)])
        # Good model
        y_prob_a = np.concatenate(
            [rng.uniform(0.0, 0.4, 100), rng.uniform(0.6, 1.0, 100)]
        )
        # Weak model
        y_prob_b = np.concatenate(
            [rng.uniform(0.2, 0.6, 100), rng.uniform(0.4, 0.8, 100)]
        )
        result = delong_test(y_true, y_prob_a, y_prob_b)
        assert result.effect_size > 0  # AUC difference should be positive

    def test_returns_effect_size(self) -> None:
        """Effect size should be the absolute AUC difference."""
        y_true = np.array([0, 0, 1, 1])
        y_prob_a = np.array([0.1, 0.2, 0.8, 0.9])
        y_prob_b = np.array([0.3, 0.4, 0.6, 0.7])
        result = delong_test(y_true, y_prob_a, y_prob_b)
        assert result.effect_size >= 0


class TestWilcoxonTest:
    """Tests for Wilcoxon signed-rank test."""

    def test_identical_scores_not_significant(self) -> None:
        """Same scores across folds should not be significant."""
        scores = [0.85, 0.87, 0.84, 0.86, 0.88]
        result = wilcoxon_signed_rank_test(scores, scores)
        assert result.p_value == 1.0
        assert not result.significant

    def test_consistently_better_model(self) -> None:
        """Model consistently better across folds should trend significant."""
        scores_a = [0.90, 0.91, 0.89, 0.92, 0.88, 0.93, 0.90, 0.91]
        scores_b = [0.80, 0.79, 0.78, 0.81, 0.77, 0.82, 0.80, 0.79]
        result = wilcoxon_signed_rank_test(scores_a, scores_b)
        # With 8 pairs and large difference, should be significant
        assert result.p_value < 0.05

    def test_mismatched_lengths_raise(self) -> None:
        """Different array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            wilcoxon_signed_rank_test([0.8, 0.9], [0.7, 0.8, 0.9])

    def test_effect_size_is_mean_diff(self) -> None:
        """Effect size should be the mean difference."""
        scores_a = [0.90, 0.92]
        scores_b = [0.80, 0.82]
        result = wilcoxon_signed_rank_test(scores_a, scores_b)
        assert abs(result.effect_size - 0.10) < 1e-6
