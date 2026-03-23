# src/experiments/figure_generator.py
# Publication-ready thesis figure generation
"""
Generates all figures for the thesis at 300 DPI, following publication
standards. Figures are organized by thesis chapter.

Chapter 3 figures (naive approach):
  - Suspicious training curves (fast convergence).
  - Grad-CAM on borders/background.

Chapter 4 figures (problem discovery):
  - Image-level vs patient-level accuracy comparison bar chart.
  - Intensity distribution differences between dataset sources.

Chapter 5 figures (rigorous methodology):
  - Preprocessing pipeline flowchart.
  - Augmentation examples.

Chapter 6 figures (results):
  - ROC curves per model.
  - Confusion matrices.
  - Ablation study bar charts.
  - Grad-CAM comparison (naive vs rigorous).
  - Calibration reliability diagrams.
  - Cross-dataset performance comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Publication-quality settings
FIGURE_DPI = 300
FONT_SIZE = 11
TITLE_SIZE = 13
COLORS = {
    "primary": "#1976D2",
    "secondary": "#FF9800",
    "success": "#4CAF50",
    "danger": "#F44336",
    "neutral": "#757575",
    "naive": "#F44336",
    "rigorous": "#4CAF50",
}

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": FIGURE_DPI,
    }
)


def plot_split_comparison(
    comparison_result: dict[str, Any],
    output_path: Path,
    metric_key: str = "balanced_accuracy",
) -> None:
    """Plot image-level vs patient-level split comparison (Chapter 4).

    This is one of the most important figures in the thesis, showing
    the accuracy inflation from data leakage.

    Parameters
    ----------
    comparison_result : dict
        Output from SplitComparisonResult.to_dict().
    output_path : Path
        Where to save the figure.
    metric_key : str
        Metric to display.
    """
    img_val = comparison_result["image_level"]["mean_metrics"].get(
        metric_key, 0
    )
    pat_val = comparison_result["patient_level"]["mean_metrics"].get(
        metric_key, 0
    )
    delta = comparison_result["delta"].get(metric_key, 0)

    # Per-fold values for error bars
    img_folds = [
        f.get(metric_key, 0)
        for f in comparison_result["image_level"]["per_fold"]
    ]
    pat_folds = [
        f.get(metric_key, 0)
        for f in comparison_result["patient_level"]["per_fold"]
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    x = [0, 1]
    values = [img_val, pat_val]
    stds = [
        np.std(img_folds) if img_folds else 0,
        np.std(pat_folds) if pat_folds else 0,
    ]
    colors = [COLORS["danger"], COLORS["success"]]
    labels = ["Image-level split\n(LEAKY)", "Patient-level split\n(CORRECT)"]

    bars = ax.bar(
        x,
        values,
        yerr=stds,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=8,
        width=0.5,
    )

    # Annotate values
    for bar, val, std in zip(bars, values, stds, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.01,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Delta annotation
    ax.annotate(
        f"$\\Delta$ = {delta:+.1%}",
        xy=(0.5, max(values) + 0.03),
        fontsize=12,
        ha="center",
        color=COLORS["danger"],
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title(
        "Impact of Data Leakage: Image-Level vs Patient-Level Splitting",
        fontweight="bold",
    )
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Split comparison figure saved to %s", output_path)


def plot_ablation_study(
    study_result: dict[str, Any],
    output_path: Path,
    metric_key: str = "balanced_accuracy",
) -> None:
    """Plot ablation study results as a horizontal bar chart (Chapter 6).

    Parameters
    ----------
    study_result : dict
        Output from AblationStudy.to_dict().
    output_path : Path
        Where to save the figure.
    metric_key : str
        Metric to display.
    """
    baseline_val = study_result["baseline"]["metrics"].get(metric_key, 0)
    names = ["Baseline"]
    values = [baseline_val]
    colors = [COLORS["primary"]]

    for variant in study_result["variants"]:
        names.append(variant["variant"])
        val = variant["metrics"].get(metric_key, 0)
        values.append(val)
        colors.append(
            COLORS["success"] if val >= baseline_val else COLORS["danger"]
        )

    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(names))))

    y_pos = range(len(names))
    bars = ax.barh(
        y_pos,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.3,
        height=0.5,
    )

    # Annotate with values and deltas
    for i, (bar, val) in enumerate(zip(bars, values, strict=False)):
        delta = val - baseline_val if i > 0 else 0
        label = f"{val:.3f}" if i == 0 else f"{val:.3f} ({delta:+.3f})"
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel(metric_key.replace("_", " ").title())
    ax.set_title(
        f"Ablation Study: {study_result.get('study_name', '')}",
        fontweight="bold",
    )
    ax.axvline(
        baseline_val,
        color=COLORS["neutral"],
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Ablation figure saved to %s", output_path)


def plot_roc_curves(
    roc_data: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot ROC curves for multiple models on the same axes (Chapter 6).

    Parameters
    ----------
    roc_data : dict
        Mapping from model_name to ROC curve data dict with fpr, tpr, auc.
    output_path : Path
        Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    color_cycle = ["#1976D2", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, (model_name, data) in enumerate(roc_data.items()):
        color = color_cycle[i % len(color_cycle)]
        ax.plot(
            data["fpr"],
            data["tpr"],
            color=color,
            linewidth=2,
            label=f"{model_name} (AUC = {data['auc']:.3f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=0.8,
        alpha=0.5,
        label="Random (AUC = 0.500)",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves figure saved to %s", output_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Plot confusion matrix as a heatmap (Chapter 6).

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (absolute counts).
    class_names : list[str]
        Display names for each class.
    output_path : Path
        Where to save the figure.
    title : str
        Figure title.
    normalize : bool
        Whether to show row-normalized percentages.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm / row_sums * 100
    else:
        cm_display = cm.astype(float)

    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, label="%" if normalize else "Count")

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_display[i, j]
            text = f"{count}\n({pct:.1f}%)" if normalize else f"{count}"
            color = (
                "white" if cm_display[i, j] > cm_display.max() / 2 else "black"
            )
            ax.text(
                j, i, text, ha="center", va="center", color=color, fontsize=10
            )

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels([f"Pred. {c}" for c in class_names])
    ax.set_yticklabels([f"True {c}" for c in class_names])
    ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix figure saved to %s", output_path)


def plot_calibration_diagram(
    calibration_data: dict[str, Any],
    output_path: Path,
    model_name: str = "Model",
) -> None:
    """Plot reliability diagram for model calibration (Chapter 6).

    Parameters
    ----------
    calibration_data : dict
        Output from CalibrationResult.to_dict().
    output_path : Path
        Where to save the figure.
    model_name : str
        Model name for the title.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    rd = calibration_data["reliability_diagram"]
    bin_acc = rd["bin_accuracies"]
    bin_conf = rd["bin_confidences"]
    bin_counts = rd["bin_counts"]

    # Reliability diagram
    ax1.bar(
        bin_conf,
        bin_acc,
        width=0.08,
        alpha=0.7,
        color=COLORS["primary"],
        edgecolor="black",
        linewidth=0.3,
        label="Model",
    )
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(f"Reliability Diagram — {model_name}", fontweight="bold")
    ax1.legend()
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_aspect("equal")

    # Confidence histogram
    ax2.bar(
        bin_conf,
        bin_counts,
        width=0.08,
        color=COLORS["secondary"],
        edgecolor="black",
        linewidth=0.3,
    )
    ax2.set_xlabel("Mean Predicted Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Confidence Distribution", fontweight="bold")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Add ECE/Brier annotations
    ece = calibration_data.get("ece", 0)
    brier = calibration_data.get("brier_score", 0)
    fig.text(
        0.5,
        -0.02,
        f"ECE = {ece:.4f} | Brier Score = {brier:.4f}",
        ha="center",
        fontsize=FONT_SIZE,
        style="italic",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration diagram saved to %s", output_path)


def plot_cross_dataset_comparison(
    internal_metrics: dict[str, float],
    external_metrics: dict[str, float],
    output_path: Path,
    internal_name: str = "BraTS 2023",
    external_name: str = "UCSF-PDGM",
) -> None:
    """Plot internal vs external validation performance (Chapter 6).

    Shows the expected 3-8 percentage point drop on external data,
    demonstrating honest generalization measurement.

    Parameters
    ----------
    internal_metrics : dict
        Metrics from the training dataset (BraTS).
    external_metrics : dict
        Metrics from external validation (UCSF-PDGM).
    output_path : Path
        Where to save the figure.
    """
    metrics_to_plot = ["balanced_accuracy", "f1_macro", "auc_roc_macro"]
    labels = ["Bal. Accuracy", "Macro F1", "AUC-ROC"]

    internal_vals = [internal_metrics.get(m, 0) for m in metrics_to_plot]
    external_vals = [external_metrics.get(m, 0) for m in metrics_to_plot]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(labels))
    width = 0.3

    ax.bar(
        x - width / 2,
        internal_vals,
        width,
        color=COLORS["primary"],
        edgecolor="black",
        linewidth=0.3,
        label=internal_name,
    )
    ax.bar(
        x + width / 2,
        external_vals,
        width,
        color=COLORS["secondary"],
        edgecolor="black",
        linewidth=0.3,
        label=external_name,
    )

    # Annotate with values and deltas
    for i, (iv, ev) in enumerate(
        zip(internal_vals, external_vals, strict=False)
    ):
        delta = ev - iv
        ax.text(
            i - width / 2,
            iv + 0.01,
            f"{iv:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            i + width / 2,
            ev + 0.01,
            f"{ev:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        mid_y = (iv + ev) / 2
        ax.annotate(
            f"$\\Delta$={delta:+.3f}",
            xy=(i + width, mid_y),
            fontsize=8,
            color=COLORS["danger"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title(
        "Cross-Dataset Generalization: Internal vs External Validation",
        fontweight="bold",
    )
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Cross-dataset comparison figure saved to %s", output_path)
