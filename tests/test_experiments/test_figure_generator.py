# tests/test_experiments/test_figure_generator.py — Tests for figure generation
"""
Verifies that all thesis figure generators produce valid PNG files:
- Split comparison bar chart (Chapter 4).
- Ablation study horizontal bars (Chapter 6).
- ROC curves (Chapter 6).
- Confusion matrix heatmap (Chapter 6).
- Calibration reliability diagram (Chapter 6).
- Cross-dataset comparison (Chapter 6).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.experiments.figure_generator import (
    plot_ablation_study,
    plot_calibration_diagram,
    plot_confusion_matrix,
    plot_cross_dataset_comparison,
    plot_roc_curves,
    plot_split_comparison,
)


class TestFigureGeneration:
    """Tests that each figure generator creates a valid PNG file."""

    def test_split_comparison_figure(self) -> None:
        """Split comparison should produce a PNG file."""
        data = {
            "image_level": {
                "mean_metrics": {"balanced_accuracy": 0.95},
                "per_fold": [
                    {"balanced_accuracy": 0.94},
                    {"balanced_accuracy": 0.96},
                ],
            },
            "patient_level": {
                "mean_metrics": {"balanced_accuracy": 0.87},
                "per_fold": [
                    {"balanced_accuracy": 0.86},
                    {"balanced_accuracy": 0.88},
                ],
            },
            "delta": {"balanced_accuracy": -0.08},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_split.png"
            plot_split_comparison(data, path)
            assert path.exists()
            assert path.stat().st_size > 1000  # Non-trivial file size

    def test_ablation_study_figure(self) -> None:
        """Ablation figure should produce a PNG file."""
        data = {
            "study_name": "Preprocessing",
            "baseline": {"metrics": {"balanced_accuracy": 0.87}},
            "variants": [
                {"variant": "no_n4", "metrics": {"balanced_accuracy": 0.85}},
                {
                    "variant": "no_zscore",
                    "metrics": {"balanced_accuracy": 0.84},
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_ablation.png"
            plot_ablation_study(data, path)
            assert path.exists()
            assert path.stat().st_size > 1000

    def test_roc_curves_figure(self) -> None:
        """ROC curves figure should produce a PNG file."""
        roc_data = {
            "DenseNet-121": {
                "fpr": [0.0, 0.1, 0.3, 1.0],
                "tpr": [0.0, 0.7, 0.9, 1.0],
                "auc": 0.92,
            },
            "ResNet-50": {
                "fpr": [0.0, 0.15, 0.4, 1.0],
                "tpr": [0.0, 0.65, 0.85, 1.0],
                "auc": 0.88,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_roc.png"
            plot_roc_curves(roc_data, path)
            assert path.exists()
            assert path.stat().st_size > 1000

    def test_confusion_matrix_figure(self) -> None:
        """Confusion matrix figure should produce a PNG file."""
        cm = np.array([[85, 15], [10, 90]])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_cm.png"
            plot_confusion_matrix(cm, ["LGG", "HGG"], path)
            assert path.exists()
            assert path.stat().st_size > 1000

    def test_calibration_diagram_figure(self) -> None:
        """Calibration diagram should produce a PNG file."""
        cal_data = {
            "ece": 0.035,
            "brier_score": 0.12,
            "reliability_diagram": {
                "bin_accuracies": [
                    0.1,
                    0.3,
                    0.5,
                    0.6,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
                "bin_confidences": [
                    0.05,
                    0.15,
                    0.25,
                    0.35,
                    0.45,
                    0.55,
                    0.65,
                    0.75,
                    0.85,
                    0.95,
                ],
                "bin_counts": [10, 15, 20, 25, 30, 35, 40, 30, 20, 10],
                "n_bins": 10,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_cal.png"
            plot_calibration_diagram(cal_data, path)
            assert path.exists()
            assert path.stat().st_size > 1000

    def test_cross_dataset_figure(self) -> None:
        """Cross-dataset comparison should produce a PNG file."""
        internal = {
            "balanced_accuracy": 0.87,
            "f1_macro": 0.85,
            "auc_roc_macro": 0.92,
        }
        external = {
            "balanced_accuracy": 0.82,
            "f1_macro": 0.80,
            "auc_roc_macro": 0.88,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_cross.png"
            plot_cross_dataset_comparison(internal, external, path)
            assert path.exists()
            assert path.stat().st_size > 1000

    def test_figures_create_parent_directories(self) -> None:
        """All figure generators should create parent dirs if needed."""
        data = {
            "image_level": {
                "mean_metrics": {"balanced_accuracy": 0.95},
                "per_fold": [],
            },
            "patient_level": {
                "mean_metrics": {"balanced_accuracy": 0.87},
                "per_fold": [],
            },
            "delta": {"balanced_accuracy": -0.08},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "figure.png"
            plot_split_comparison(data, path)
            assert path.exists()
