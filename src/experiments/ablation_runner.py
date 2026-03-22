# src/experiments/ablation_runner.py — Systematic ablation study framework
"""
Framework for running ablation studies where one variable is changed at a time
while keeping everything else constant.

Each ablation answers a specific methodological question:
  - Preprocessing steps: Does each step (N4, z-score, resampling) contribute?
  - Augmentation type: Do MRI-specific augmentations outperform generic ones?
  - Transfer learning: Does ImageNet pretraining help vs from scratch?
  - Architecture: Which backbone performs best with proper evaluation?
  - 2D vs 2.5D: Does volumetric context from adjacent slices help?

Results are structured for direct inclusion in thesis tables, with
statistical significance tests between baseline and each ablation variant.

References:
    - configs/experiment/ablation_preproc.yaml
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AblationVariant:
    """
    Definition of a single ablation variant.

    Attributes
    ----------
    name : str
        Identifier for this variant (e.g., "no_n4", "no_zscore").
    description : str
        What is changed in this variant.
    question : str
        The methodological question this ablation answers.
    config_overrides : dict
        Configuration overrides to apply for this variant.
    """

    name: str
    description: str
    question: str = ""
    config_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationResult:
    """
    Result of a single ablation variant evaluation.

    Attributes
    ----------
    variant_name : str
        Name of the ablation variant.
    metrics : dict[str, float]
        Mean metrics across CV folds.
    per_fold_metrics : list[dict[str, float]]
        Per-fold metrics.
    delta_from_baseline : dict[str, float]
        Difference from baseline for each metric.
    training_time_seconds : float
        Total training time.
    """

    variant_name: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    per_fold_metrics: list[dict[str, float]] = field(default_factory=list)
    delta_from_baseline: dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "variant": self.variant_name,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "per_fold": self.per_fold_metrics,
            "delta_from_baseline": {
                k: round(v, 4) for k, v in self.delta_from_baseline.items()
            },
            "training_time_seconds": round(self.training_time_seconds, 1),
        }


@dataclass
class AblationStudy:
    """
    Complete ablation study with baseline and variants.

    Attributes
    ----------
    study_name : str
        Name of the ablation study.
    question : str
        The overarching research question.
    baseline : AblationResult
        Baseline (full pipeline) result.
    variants : list[AblationResult]
        Results for each ablation variant.
    statistical_tests : list[dict]
        Significance tests between baseline and variants.
    """

    study_name: str = ""
    question: str = ""
    baseline: AblationResult = field(default_factory=AblationResult)
    variants: list[AblationResult] = field(default_factory=list)
    statistical_tests: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "study_name": self.study_name,
            "question": self.question,
            "baseline": self.baseline.to_dict(),
            "variants": [v.to_dict() for v in self.variants],
            "statistical_tests": self.statistical_tests,
        }

    def generate_latex_table(
        self, metric_key: str = "balanced_accuracy"
    ) -> str:
        """
        Generate a LaTeX ablation table for the thesis.

        Parameters
        ----------
        metric_key : str
            Primary metric to display.

        Returns
        -------
        str
            LaTeX table string.
        """
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Ablation study: {self.study_name}}}",
            f"\\label{{tab:ablation_"
            f"{self.study_name.lower().replace(' ', '_')}}}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            f"Variant & {metric_key.replace('_', ' ').title()} & "
            "$\\Delta$ & Significant \\\\",
            "\\midrule",
        ]

        # Baseline row
        base_val = self.baseline.metrics.get(metric_key, 0)
        lines.append(
            f"Baseline (full pipeline) & {base_val:.3f} & --- & --- \\\\"
        )

        # Variant rows
        for variant in self.variants:
            val = variant.metrics.get(metric_key, 0)
            delta = variant.delta_from_baseline.get(metric_key, 0)

            # Find significance from tests
            sig = "---"
            for test in self.statistical_tests:
                if test.get("variant") == variant.variant_name:
                    sig = (
                        "$p < 0.05$"
                        if test.get("significant", False)
                        else "n.s."
                    )
                    break

            lines.append(
                f"{variant.variant_name} & {val:.3f} & "
                f"{delta:+.3f} & {sig} \\\\"
            )

        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ]
        )

        return "\n".join(lines)


def run_ablation_study(
    study_name: str,
    question: str,
    baseline_config: dict[str, Any],
    variants: list[AblationVariant],
    train_and_evaluate_fn: Callable[[dict[str, Any]], AblationResult],
    run_significance_tests: bool = True,
) -> AblationStudy:
    """
    Run a complete ablation study with baseline and variants.

    Parameters
    ----------
    study_name : str
        Name for the study.
    question : str
        Research question being addressed.
    baseline_config : dict
        Configuration for the baseline run.
    variants : list[AblationVariant]
        List of ablation variants to evaluate.
    train_and_evaluate_fn : callable
        Function that takes a config dict and returns an AblationResult.
    run_significance_tests : bool
        Whether to run Wilcoxon tests between baseline and variants.

    Returns
    -------
    AblationStudy
        Complete ablation study results.
    """
    study = AblationStudy(study_name=study_name, question=question)

    # Run baseline
    logger.info("=" * 60)
    logger.info("ABLATION STUDY: %s", study_name)
    logger.info("Question: %s", question)
    logger.info("=" * 60)

    logger.info("\n--- Running baseline ---")
    t_start = time.time()
    study.baseline = train_and_evaluate_fn(baseline_config)
    study.baseline.variant_name = "baseline"
    study.baseline.training_time_seconds = time.time() - t_start
    logger.info(
        "Baseline metrics: %s",
        {k: f"{v:.4f}" for k, v in study.baseline.metrics.items()},
    )

    # Run each variant
    for variant_def in variants:
        logger.info("\n--- Running variant: %s ---", variant_def.name)
        logger.info("  %s", variant_def.description)

        # Merge baseline config with variant overrides
        variant_config = {**baseline_config, **variant_def.config_overrides}

        t_start = time.time()
        variant_result = train_and_evaluate_fn(variant_config)
        variant_result.variant_name = variant_def.name
        variant_result.training_time_seconds = time.time() - t_start

        # Compute delta from baseline
        for key in study.baseline.metrics:
            if key in variant_result.metrics:
                variant_result.delta_from_baseline[key] = (
                    variant_result.metrics[key] - study.baseline.metrics[key]
                )

        study.variants.append(variant_result)
        logger.info(
            "  %s metrics: %s (delta: %s)",
            variant_def.name,
            {k: f"{v:.4f}" for k, v in variant_result.metrics.items()},
            {
                k: f"{v:+.4f}"
                for k, v in variant_result.delta_from_baseline.items()
            },
        )

    # Statistical significance tests
    if run_significance_tests:
        study.statistical_tests = _run_significance_tests(study)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION STUDY COMPLETE: %s", study_name)
    logger.info("=" * 60)
    for variant in study.variants:
        for key, delta in variant.delta_from_baseline.items():
            logger.info("  %s → %s: %+.4f", variant.variant_name, key, delta)

    return study


def _run_significance_tests(study: AblationStudy) -> list[dict]:
    """
    Run Wilcoxon tests between baseline and each variant.

    Parameters
    ----------
    study : AblationStudy
        Study with baseline and variant per-fold metrics.

    Returns
    -------
    list[dict]
        Test results for each variant.
    """
    from src.evaluation.statistical_tests import wilcoxon_signed_rank_test

    test_results = []
    metric_key = "balanced_accuracy"

    baseline_folds = [
        m.get(metric_key, 0) for m in study.baseline.per_fold_metrics
    ]

    for variant in study.variants:
        variant_folds = [
            m.get(metric_key, 0) for m in variant.per_fold_metrics
        ]

        if (
            len(baseline_folds) != len(variant_folds)
            or len(baseline_folds) < 3
        ):
            test_results.append(
                {
                    "variant": variant.variant_name,
                    "test": "wilcoxon",
                    "note": "Insufficient folds for testing",
                }
            )
            continue

        try:
            result = wilcoxon_signed_rank_test(
                baseline_folds,
                variant_folds,
                model_a_name="Baseline",
                model_b_name=variant.variant_name,
                metric_name=metric_key,
            )
            test_results.append(
                {
                    "variant": variant.variant_name,
                    **result.to_dict(),
                }
            )
        except Exception as e:
            logger.warning(
                "Significance test failed for %s: %s", variant.variant_name, e
            )
            test_results.append(
                {
                    "variant": variant.variant_name,
                    "error": str(e),
                }
            )

    return test_results


def define_preprocessing_ablations() -> list[AblationVariant]:
    """
    Define the preprocessing ablation variants.

    Returns
    -------
    list[AblationVariant]
        List of preprocessing ablation variants.
    """
    return [
        AblationVariant(
            name="no_n4",
            description="Skip N4 bias field correction",
            question="Does N4 bias field correction improve generalization?",
            config_overrides={"apply_n4": False},
        ),
        AblationVariant(
            name="no_zscore",
            description="Skip z-score normalization",
            question=(
                "Does z-score normalization affect cross-site performance?"
            ),
            config_overrides={"z_score_normalize": False},
        ),
        AblationVariant(
            name="no_augmentation",
            description="Disable all training augmentation",
            question="Do MRI-specific augmentations improve robustness?",
            config_overrides={"enable_augmentation": False},
        ),
        AblationVariant(
            name="generic_augmentation",
            description="Use generic (non-MRI) augmentations only",
            question="Do MRI-specific augmentations outperform generic ones?",
            config_overrides={"augmentation_type": "generic"},
        ),
        AblationVariant(
            name="no_pretrained",
            description="Train from scratch (no ImageNet pretraining)",
            question="Does ImageNet pretraining help? (Raghu et al., 2019)",
            config_overrides={"pretrained": False},
        ),
    ]


def define_architecture_ablations() -> list[AblationVariant]:
    """
    Define the architecture comparison variants.

    Returns
    -------
    list[AblationVariant]
        List of architecture variants.
    """
    return [
        AblationVariant(
            name="resnet50",
            description="ResNet-50 (torchvision, ~25M params)",
            question="Does the larger ResNet-50 outperform DenseNet-121?",
            config_overrides={"backbone": "resnet50"},
        ),
        AblationVariant(
            name="efficientnet_b0",
            description="EfficientNet-B0 (timm, ~5M params)",
            question="Does the efficient EfficientNet match larger models?",
            config_overrides={"backbone": "efficientnet_b0"},
        ),
        AblationVariant(
            name="resnet18",
            description="ResNet-18 pretrained baseline (~11M params)",
            question="How does a lightweight pretrained model perform?",
            config_overrides={"backbone": "resnet18"},
        ),
        AblationVariant(
            name="simple_cnn",
            description="Simple 4-layer CNN from scratch (~0.5M params)",
            question=(
                "How much does transfer learning contribute? "
                "(Raghu et al., 2019)"
            ),
            config_overrides={"backbone": "simple_cnn", "pretrained": False},
        ),
    ]


def define_slice_ablations() -> list[AblationVariant]:
    """
    Define 2D vs 2.5D comparison variants.

    Returns
    -------
    list[AblationVariant]
        Slice strategy variants.
    """
    return [
        AblationVariant(
            name="2.5d_3slices",
            description="2.5D with 3 adjacent slices (center ± 1)",
            question="Does volumetric context from adjacent slices help?",
            config_overrides={"context_slices": 1, "in_channels": 12},
        ),
        AblationVariant(
            name="2.5d_5slices",
            description="2.5D with 5 adjacent slices (center ± 2)",
            question=(
                "Does more volumetric context further improve performance?"
            ),
            config_overrides={"context_slices": 2, "in_channels": 20},
        ),
        AblationVariant(
            name="center_slice",
            description="Use center slice instead of max tumor area",
            question="Does tumor-guided slice selection matter?",
            config_overrides={"slice_method": "center"},
        ),
    ]
