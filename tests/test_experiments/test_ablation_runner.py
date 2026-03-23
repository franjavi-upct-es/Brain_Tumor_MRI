# tests/test_experiments/test_ablation_runner.py — Tests for ablation framework
"""
Verifies the ablation study framework:
- Baseline runs first, then each variant.
- Deltas are computed correctly.
- LaTeX table generation works.
- Predefined ablation definitions are valid.
"""

from __future__ import annotations

import json

import numpy as np

from src.experiments.ablation_runner import (
    AblationResult,
    AblationStudy,
    AblationVariant,
    define_architecture_ablations,
    define_preprocessing_ablations,
    define_slice_ablations,
    run_ablation_study,
)


def _mock_train_fn(config: dict) -> AblationResult:
    """Mock training function that returns config-dependent metrics."""
    # Simulate different performance based on config
    base = 0.87
    if config.get("pretrained") is False:
        base -= 0.05
    if config.get("apply_n4") is False:
        base -= 0.02
    if config.get("z_score_normalize") is False:
        base -= 0.03
    if config.get("enable_augmentation") is False:
        base -= 0.04

    # Add fake per-fold results
    rng = np.random.RandomState(42)
    folds = [
        {"balanced_accuracy": base + rng.normal(0, 0.01)} for _ in range(5)
    ]

    return AblationResult(
        metrics={"balanced_accuracy": base, "f1_macro": base - 0.01},
        per_fold_metrics=folds,
    )


class TestAblationStudyRunner:
    """Tests for the ablation study orchestrator."""

    def test_runs_baseline_and_variants(self) -> None:
        """Should run baseline plus all variants."""
        variants = [
            AblationVariant(
                name="v1",
                description="Test v1",
                config_overrides={"pretrained": False},
            ),
            AblationVariant(
                name="v2",
                description="Test v2",
                config_overrides={"apply_n4": False},
            ),
        ]

        study = run_ablation_study(
            study_name="Test",
            question="Testing",
            baseline_config={"pretrained": True, "apply_n4": True},
            variants=variants,
            train_and_evaluate_fn=_mock_train_fn,
            run_significance_tests=False,
        )

        assert study.baseline.variant_name == "baseline"
        assert len(study.variants) == 2
        assert study.variants[0].variant_name == "v1"
        assert study.variants[1].variant_name == "v2"

    def test_deltas_computed_correctly(self) -> None:
        """Delta should be variant_metric - baseline_metric."""
        variants = [
            AblationVariant(
                name="no_pretrained",
                description="No pretrain",
                config_overrides={"pretrained": False},
            ),
        ]

        study = run_ablation_study(
            study_name="Test",
            question="Testing",
            baseline_config={"pretrained": True},
            variants=variants,
            train_and_evaluate_fn=_mock_train_fn,
            run_significance_tests=False,
        )

        baseline_ba = study.baseline.metrics["balanced_accuracy"]
        variant_ba = study.variants[0].metrics["balanced_accuracy"]
        delta = study.variants[0].delta_from_baseline["balanced_accuracy"]

        assert abs(delta - (variant_ba - baseline_ba)) < 1e-6

    def test_latex_table_generation(self) -> None:
        """LaTeX table should be valid and contain key elements."""
        study = AblationStudy(
            study_name="Preprocessing",
            baseline=AblationResult(
                variant_name="baseline",
                metrics={"balanced_accuracy": 0.87},
            ),
            variants=[
                AblationResult(
                    variant_name="no_n4",
                    metrics={"balanced_accuracy": 0.85},
                    delta_from_baseline={"balanced_accuracy": -0.02},
                ),
            ],
        )

        latex = study.generate_latex_table()
        assert "\\begin{table}" in latex
        assert "Baseline" in latex
        assert "no_n4" in latex
        assert "0.870" in latex
        assert "-0.020" in latex

    def test_to_dict_serializable(self) -> None:
        """Study result should be JSON-serializable."""
        study = AblationStudy(
            study_name="Test",
            baseline=AblationResult(
                variant_name="baseline", metrics={"acc": 0.87}
            ),
            variants=[
                AblationResult(
                    variant_name="v1",
                    metrics={"acc": 0.85},
                    delta_from_baseline={"acc": -0.02},
                )
            ],
        )
        d = study.to_dict()
        json.dumps(d)  # Should not raise


class TestPredefinedAblations:
    """Tests for predefined ablation variant definitions."""

    def test_preprocessing_ablations_defined(self) -> None:
        """Should return at least 4 preprocessing variants."""
        variants = define_preprocessing_ablations()
        assert len(variants) >= 4
        names = {v.name for v in variants}
        assert "no_n4" in names
        assert "no_zscore" in names
        assert "no_pretrained" in names

    def test_architecture_ablations_defined(self) -> None:
        """Should return variants for all architectures."""
        variants = define_architecture_ablations()
        assert len(variants) >= 3
        names = {v.name for v in variants}
        assert "resnet50" in names
        assert "efficientnet_b0" in names
        assert "simple_cnn" in names

    def test_slice_ablations_defined(self) -> None:
        """Should return 2D vs 2.5D variants."""
        variants = define_slice_ablations()
        assert len(variants) >= 2
        names = {v.name for v in variants}
        assert "2.5d_3slices" in names
        assert "center_slice" in names

    def test_all_variants_have_names_and_descriptions(self) -> None:
        """Every variant must have a name and description."""
        all_variants = (
            define_preprocessing_ablations()
            + define_architecture_ablations()
            + define_slice_ablations()
        )
        for v in all_variants:
            assert v.name, f"Variant missing name: {v}"
            assert v.description, f"Variant {v.name} missing description"
            assert v.config_overrides, (
                f"Variant {v.name} has no config overrides"
            )
