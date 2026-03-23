# src/evaluation/statisical_tests.py — Statistical significance tests
"""
Statistical tests for comparing classification models, as required by
CLAIM and TRITPOD+AI guidelines.

Tests implemented:
    - McNemar's test: compares error rates between two classifiers on the
      same test set (e.g., naive vs rigorous approach).
    - DeLong's test: compares AUCs between two models.
    - Wilcoxon signed-rank test: compares paired performance across CV folds.

All tests report p-values and effect sizes. A significance threshold of
alpha=0.05 is standard, but the actual p-value should always be reported.

References:
    - McNemar (1947). Note on the sampling error of the difference between
      correlated proportions or percentages. Psychometrika.
    - DeLong et al. (1988). Comparing the areas under two or more correlated
      ROC curves. Biometrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """
    Result of a statistical significance test.

    Attributes
    ----------
    test_name : str
        Name of the statistical test
    statistic : float
        Test statistic value.
    p_value : float
        P-value of the test.
    significant : bool
        Whether p < alpha.
    alpha : float
        Significance threshold.
    effect_size : float | None
        Effect size measure (if applicable).
    description : str
        Human-readable description of the result.
    """

    test_name: str = ""
    statistic: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    alpha: float = 0.05
    effect_size: float | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = {
            "test": self.test_name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
            "alpha": self.alpha,
            "description": self.description,
        }
        if self.effect_size is not None:
            d["effect_size"] = round(self.effect_size, 4)
        return d


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    alpha: float = 0.05,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> StatisticalTestResult:
    """
    McNemar's test for comparing two classifiers' error rates.

    Tests whether the two models make significantly different types of
    errors on the same test data. Uses the exact binomial version for
    small contingency tables and chi-squared approximation for larger ones.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred_a : np.ndarray
        Predictions from model A.
    y_pred_b : np.ndarray
        Predictions from model B.
    alpha : float
        Significance threshold.
    model_a_name : str
        Name of model A for reporting.
    model_b_name : str
        Name of model B for reporting.

    Returns
    -------
    StatisticalTestResult
        Statistical test result.
    """
    # Build the 2x2 contingency table
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    # b: A correct, B wrong; c: A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    # McNemar statistic with continuity correction
    n_discordant = b + c
    if n_discordant == 0:
        return StatisticalTestResult(
            test_name="McNemar",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            description=(
                f"{model_a_name} and {model_b_name} make identical errors."
            ),
        )

    if n_discordant < 25:
        # Extract binomial test for small samples
        p_value = float(stats.binomtest(b, n_discordant, 0.5).pvalue)
        statistic = float(b)
    else:
        # Chi-squared approximation with continuity correction
        statistic = float((abs(b - c) - 1) ** 2 / (b + c))
        p_value = float(1 - stats.chi2.cdf(statistic, df=1))

    significant = p_value < alpha

    description = (
        f"{model_a_name} vs {model_b_name}: "
        f"discordant pairs b={b}, c={c}, "
        f"p={p_value:.6f} "
        f"({'significant' if significant else 'not significant'})"
    )

    logger.info("McNemar's test: %s", description)

    return StatisticalTestResult(
        test_name="McNemar",
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=abs(b - c) / max(n_discordant, 1),
        description=description,
    )


def delong_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    alpha: float = 0.05,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> StatisticalTestResult:
    """
    DeLong's test for comparing two ROC-AUCs.

    Tests whether to AUC-ROC of model A is significantly different from
    model B on the same data. Uses the nonparametric approach of DeLong
    et al. (1988).

    Simplified implementation using the Mann-Whitney U statistic approach.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_prob_a : np.ndarray
        Predicted probabilities from model A for the positive class.
    y_prob_b : np.ndarray
        Predicted probabilities from model B for the positive class.
    alpha : float
        Significance threshold.

    Returns
    -------
    StatisticalTestResult
        Statistical test result with AUC comparison.
    """
    from sklearn.metrics import roc_auc_score

    auc_a = roc_auc_score(y_true, y_prob_a)
    auc_b = roc_auc_score(y_true, y_prob_b)
    auc_diff = abs(auc_a - auc_b)

    # Compute DeLong variance using the structural components
    # (Simplified: using bootstrap variance estimation)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    # Placement values for model A
    v_a_pos = np.array(
        [np.mean(y_prob_a[neg_idx] < y_prob_a[i]) for i in pos_idx]
    )
    v_a_neg = np.array(
        [np.mean(y_prob_a[pos_idx] > y_prob_a[i]) for i in pos_idx]
    )

    # Placement values for model B
    v_b_pos = np.array(
        [np.mean(y_prob_b[neg_idx] < y_prob_b[i]) for i in pos_idx]
    )
    v_b_neg = np.array(
        [np.mean(y_prob_b[pos_idx] > y_prob_b[i]) for i in pos_idx]
    )

    # Variance components
    s10_a = np.var(v_a_pos, ddof=1) if n_pos > 1 else 0
    s01_a = np.var(v_a_neg, ddof=1) if n_neg > 1 else 0
    s10_b = np.var(v_b_pos, ddof=1) if n_pos > 1 else 0
    s01_b = np.var(v_b_neg, ddof=1) if n_neg > 1 else 0

    # Covariance between AUCs
    s10_ab = np.cov(v_a_pos, v_b_pos, ddof=1)[0, 1] if n_pos > 1 else 0
    s01_ab = np.cov(v_a_neg, v_b_neg, ddof=1)[0, 1] if n_neg > 1 else 0

    # Variance of the difference
    var_diff = (s10_a + s10_b - 2 * s10_ab) / max(n_pos, 1) + (
        s01_a + s01_b - 2 * s01_ab
    ) / max(n_neg, 1)

    if var_diff <= 0:
        p_value = 1.0
        z_stat = 0.0
    else:
        se_diff = np.sqrt(var_diff)
        z_stat = float(auc_diff / se_diff)
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    significant = p_value < alpha

    description = (
        f"AUC {model_a_name}={auc_a:.4f} vs {model_b_name}={auc_a:.4f}, "
        f"diff={auc_diff:.4f}, z={z_stat:.3f}, p={p_value:.6f} "
        f"({'significant' if significant else 'not significant'})"
    )
    logger.info("DeLong's test: %s", description)

    return StatisticalTestResult(
        test_name="DeLong",
        statistic=z_stat,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=float(auc_diff),
        description=description,
    )


def wilcoxon_signed_rank_test(
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
    alpha: float = 0.05,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    metric_name: str = "metric",
) -> StatisticalTestResult:
    """
    Wilcoxon signed-rank test for paired fold comparisons.

    Compares performance between two models across CV folds. Each entry
    is the metric value for one fold, producing paired observations.

    Parameters
    ----------
    scores_a : array-like
        Per-fold scores for model A (e.g., 5 values for 5-fold CV).
    scores_b : array-like
        Per-fold scores for model B.
    alpha : float
        Significance threshold, by default 0.05
    metric_name : str
        Name of the metric being compared

    Returns
    -------
    StatisticalTestResult
        Statistical test result.
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")

    if len(scores_a) < 5:
        logger.warning(
            "Wilcoxon test with <5 pairs has very low power. "
            "Consider using more folds or a different test."
        )

    differences = scores_a - scores_b

    # Check if all differences are zero
    if np.all(differences == 0):
        return StatisticalTestResult(
            test_name="Wilcoxon signed-rank",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            description=f"No difference in {metric_name} across folds.",
        )

    try:
        stat, p_value = stats.wilcoxon(
            scores_a, scores_b, alternative="two-sided"
        )
    except ValueError:
        # Can fail if all differences have the same sign
        stat, p_value = 0.0, 1.0

    significant = p_value < alpha
    mean_diff = float(np.mean(differences))

    description = (
        f"{metric_name}: {model_a_name} mean={np.mean(scores_a):.4f} vs "
        f"{model_b_name} mean={np.mean(scores_b):.4f}, "
        f"mean_diff={mean_diff:+.4f}, p={p_value:.6f} "
        f"({'significant' if significant else 'not significant'})"
    )
    logger.info("Wilcoxon signed-rank test: %s", description)

    return StatisticalTestResult(
        test_name="Wilcoxon signed-rank",
        statistic=float(stat),
        p_value=float(p_value),
        alpha=alpha,
        effect_size=mean_diff,
        description=description,
    )
