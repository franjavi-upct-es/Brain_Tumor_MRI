# tests/test_evaluation/test_report_generator.py
"""Tests for publication-ready report generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from src.evaluation.bootstrap import BootstrapResult
from src.evaluation.metrics import ClassificationReport, compute_full_report
from src.evaluation.report_generator import (
    format_metric_with_ci,
    generate_comparison_table,
    generate_confusion_matrix_latex,
    generate_main_results_table,
    save_full_report,
)


def _make_bootstrap_result(
    metric: str = "balanced_accuracy",
    point: float = 0.87,
    lower: float = 0.83,
    upper: float = 0.91,
) -> BootstrapResult:
    return BootstrapResult(
        metric_name=metric,
        point_estimate=point,
        mean=point,
        ci_lower=lower,
        ci_upper=upper,
        std=0.02,
        n_iterations=1000,
    )


class TestGenerateMainResultsTable:
    """Tests for generate_main_results_table."""

    def test_returns_latex_string(self) -> None:
        """Should return a LaTeX table string."""
        results = {
            "DenseNet-121": {
                "balanced_accuracy": _make_bootstrap_result("balanced_accuracy"),
                "f1_macro": _make_bootstrap_result("f1_macro", 0.85, 0.81, 0.89),
                "auc_roc_macro": _make_bootstrap_result("auc_roc_macro", 0.92, 0.88, 0.96),
                "cohens_kappa": _make_bootstrap_result("cohens_kappa", 0.74, 0.70, 0.78),
            }
        }
        table = generate_main_results_table(results)
        assert isinstance(table, str)
        assert "\\begin{table}" in table
        assert "DenseNet-121" in table

    def test_missing_metric_shows_dashes(self) -> None:
        """Missing metrics should display as '---'."""
        results = {
            "Model A": {
                "balanced_accuracy": _make_bootstrap_result(),
                # f1_macro is missing
            }
        }
        table = generate_main_results_table(results)
        assert "---" in table

    def test_multiple_models(self) -> None:
        """Should include all models in table."""
        results = {
            "ModelA": {"balanced_accuracy": _make_bootstrap_result()},
            "ModelB": {"balanced_accuracy": _make_bootstrap_result(point=0.80)},
        }
        table = generate_main_results_table(results)
        assert "ModelA" in table
        assert "ModelB" in table

    def test_latex_structure(self) -> None:
        """Should have proper LaTeX table structure."""
        results = {
            "Test": {"balanced_accuracy": _make_bootstrap_result()}
        }
        table = generate_main_results_table(results)
        assert "\\toprule" in table
        assert "\\midrule" in table
        assert "\\bottomrule" in table
        assert "\\end{table}" in table


class TestGenerateConfusionMatrixLatex:
    """Tests for generate_confusion_matrix_latex."""

    def test_returns_latex_string(self) -> None:
        """Should return a LaTeX confusion matrix."""
        cm_abs = np.array([[85, 15], [10, 90]])
        cm_norm = np.array([[85.0, 15.0], [10.0, 90.0]])
        class_names = ["LGG", "HGG"]
        table = generate_confusion_matrix_latex(cm_abs, cm_norm, class_names)
        assert "\\begin{table}" in table
        assert "LGG" in table
        assert "HGG" in table

    def test_includes_counts_and_percentages(self) -> None:
        """Should show both counts and percentages."""
        cm_abs = np.array([[85, 15], [10, 90]])
        cm_norm = np.array([[85.0, 15.0], [10.0, 90.0]])
        table = generate_confusion_matrix_latex(cm_abs, cm_norm, ["A", "B"])
        assert "85" in table
        assert "%" in table

    def test_custom_model_name(self) -> None:
        """Should include model name in caption."""
        cm_abs = np.array([[10, 2], [3, 15]])
        cm_norm = np.array([[83.3, 16.7], [16.7, 83.3]])
        table = generate_confusion_matrix_latex(
            cm_abs, cm_norm, ["LGG", "HGG"], model_name="DenseNet-121"
        )
        assert "DenseNet-121" in table


class TestGenerateComparisonTable:
    """Tests for generate_comparison_table."""

    def test_returns_latex_string(self) -> None:
        """Should return a LaTeX comparison table."""
        naive = {"balanced_accuracy": _make_bootstrap_result(point=0.99)}
        rigorous = {"balanced_accuracy": _make_bootstrap_result(point=0.87)}
        table = generate_comparison_table(naive, rigorous)
        assert "\\begin{table}" in table
        assert "Na" in table  # Part of "Naïve"

    def test_computes_delta(self) -> None:
        """Should compute and display delta between approaches."""
        naive = {"balanced_accuracy": _make_bootstrap_result(point=0.99)}
        rigorous = {"balanced_accuracy": _make_bootstrap_result(point=0.87)}
        table = generate_comparison_table(naive, rigorous)
        # Delta = 0.87 - 0.99 = -0.120
        assert "-0.120" in table

    def test_missing_metric_shows_dashes(self) -> None:
        """Missing metrics should show '---'."""
        naive = {}
        rigorous = {}
        table = generate_comparison_table(naive, rigorous)
        assert "---" in table

    def test_with_test_results(self) -> None:
        """Should handle optional test_results parameter."""
        naive = {"balanced_accuracy": _make_bootstrap_result(point=0.99)}
        rigorous = {"balanced_accuracy": _make_bootstrap_result(point=0.87)}
        test_results = [{"test": "mcnemar", "p_value": 0.001}]
        table = generate_comparison_table(naive, rigorous, test_results)
        assert isinstance(table, str)


class TestSaveFullReport:
    """Tests for save_full_report."""

    def test_saves_json_file(self) -> None:
        """Should save a JSON report file."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        report = compute_full_report(y_true, y_pred)

        bootstrap_results = {
            "balanced_accuracy": _make_bootstrap_result()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_full_report(
                report, bootstrap_results, output_dir,
                model_name="densenet", dataset_name="brats"
            )
            expected_file = output_dir / "densenet_brats_report.json"
            assert expected_file.exists()

    def test_json_is_valid(self) -> None:
        """Saved JSON should be valid and parseable."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        report = compute_full_report(y_true, y_pred)
        bootstrap_results = {"balanced_accuracy": _make_bootstrap_result()}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_full_report(report, bootstrap_results, output_dir)
            path = output_dir / "model_dataset_report.json"
            with open(path) as f:
                data = json.load(f)
            assert "model" in data
            assert "metrics" in data
            assert "bootstrap_ci" in data

    def test_creates_output_dir(self) -> None:
        """Should create output directory if it doesn't exist."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        report = compute_full_report(y_true, y_pred)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "dir"
            save_full_report(report, {}, output_dir)
            assert output_dir.exists()


class TestFormatMetricWithCI:
    """Tests for format_metric_with_ci."""

    def test_basic_format(self) -> None:
        """Should format as 'value (lower — upper)'."""
        result = _make_bootstrap_result(point=0.870, lower=0.830, upper=0.910)
        formatted = format_metric_with_ci(result)
        assert "0.870" in formatted
        assert "0.830" in formatted
        assert "0.910" in formatted

    def test_custom_decimals(self) -> None:
        """Should use specified number of decimal places."""
        result = _make_bootstrap_result(point=0.87, lower=0.83, upper=0.91)
        formatted = format_metric_with_ci(result, decimals=2)
        assert "0.87" in formatted
        assert "0.83" in formatted

    def test_returns_string(self) -> None:
        """Should return a string."""
        result = _make_bootstrap_result()
        formatted = format_metric_with_ci(result)
        assert isinstance(formatted, str)
