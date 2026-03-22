# src/interpretability/visualization.py
# Publication-ready interpretability figures
"""
Generates publication-quality interpretability figures at 300 DPI.

Figures types:
    1. Grad-CAM overlay on MRI slices (with segmentation contour).
    2. Side-by-side naive vs rigorous Grad-CAM comparison.
    3. IoU distribution histograms per class.
    4. Grad-CAM + segmentation + IoU grid for multiple patients.
    5. Integrated Gradients attribution maps.

All figures follow a consistency style matching the figure_generator module.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

FIGURE_DPI = 300


def plot_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    segmentation: np.ndarray | None = None,
    output_path: Path | None = None,
    title: str = "Grad-CAM Overlay",
    iou: float | None = None,
    modality_idx: int = 0,
) -> None:
    """
    Plot Grad-CAM heatmap overlaid on a MRI slice.

    Optionally shows the ground truth segmentation contour for comparison.

    Parameters
    ----------
    image : np.ndarray
        Multi-channel image of shape (C, H, W).
    heatmap : np.ndarray
        Grad-CAM heatmap of shape (H, W), values in [0, 1].
    segmentation : np.ndarray, optional
        Ground truth segmentation of shape (H, W).
    output_path : Path, optional
        Where to save the figure. If None, displays interactively.
    title : str
        Figure title
    iou : float, optional
        IoU value to display as annotation
    modality_idx : int
        Which channel to display as the background image.
    """
    n_cols = 3 if segmentation is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # Get the background MRI slice
    bg = image[modality_idx] if image.ndim == 3 else image

    # Panel 1: Original MRI
    axes[0].imshow(bg.T, cmap="gray", origin="lower")
    axes[0].set_title("MRI (T1ce)", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Grad-CAM overlay
    axes[1].imshow(bg.T, cmap="gray", origin="lower")
    heatmap_masked = np.ma.masked_where(heatmap.T < 0.1, heatmap.T)
    axes[1].imshow(
        heatmap_masked, cmap="jet", alpha=0.5, origin="lower", vmin=0, vmax=1
    )
    subtitle = "Grad-CAM"
    if iou is not None:
        subtitle += f" (IoU = {iou:.3f})"
    axes[1].set_title(subtitle, fontsize=11)
    axes[1].axis("off")

    # Panel 3: Segmentation comparison (if available)
    if segmentation is not None and n_cols == 3:
        axes[2].imshow(bg.T, cmap="gray", origin="lower")
        # Show heatmap overlay
        axes[2].imshow(
            heatmap_masked,
            cmap="jet",
            alpha=0.3,
            origin="lower",
            vmin=0,
            vmax=1,
        )
        # Show segmentation contour
        seg_mask = (segmentation > 0).astype(float)
        axes[2].contour(
            seg_mask.T,
            levels=[0.5],
            colors="lime",
            linewidths=2,
            origin="lower",
        )
        axes[2].set_title("Grad-CAM + Tumor Contour", fontsize=11)
        axes[2].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_naive_vs_rigorous_gradcam(
    image_naive: np.ndarray,
    heatmap_naive: np.ndarray,
    image_rigorous: np.ndarray,
    heatmap_rigorous: np.ndarray,
    seg_rigorous: np.ndarray | None = None,
    iou_naive: float | None = None,
    iou_rigorous: float | None = None,
    output_path: Path | None = None,
) -> None:
    """
    Side-by-side Grad-CAM comparison: naive vs rigorous model

    This is a KEY thesis figure showing that the naive model looks
    at artifacts while the rigorous model looks at the tumor.

    Parameters
    ----------
    image_naive : np.ndarray
        MRI slice from the naive approach (C, H, W).
    heatmap_naive : np.ndarray
        Grad-CAM from the naive model.
    image_rigorous : np.ndarray
        MRI slice from the rigorous approach (C, H, W).
    heatmap_rigorous : np.ndarray
        Grad-CAM from the rigorous model.
    seg_rigorous : np.ndarray, optional
        Ground truth segmentation (only available for BraTS).
    iou_naive : float, optional
        IoU for the naive model.
    iou_rigorous : float, optional
        IoU for the rigorous model.
    output_path : Path, optional
        Where to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Naive model
    bg_naive = image_naive[0] if image_naive.ndim == 3 else image_naive
    axes[0].imshow(bg_naive.T, cmap="gray", origin="lower")
    hm_naive = np.ma.masked_where(heatmap_naive.T < 0.1, heatmap_naive.T)
    axes[0].imshow(
        hm_naive, cmap="jet", alpha=0.5, origin="lower", vmin=0, vmax=1
    )
    title_naive = "Naive Model (Kaggle)"
    if iou_naive is not None:
        title_naive += f"\nIoU = {iou_naive:.3f}"
    axes[0].set_title(
        title_naive, fontsize=12, color="#F44336", fontweight="bold"
    )
    axes[0].axis("off")

    # Rigorous model
    bg_rig = image_rigorous[0] if image_rigorous.ndim == 3 else image_rigorous
    axes[1].imshow(bg_rig.T, cmap="gray", origin="lower")
    hm_rig = np.ma.masked_where(heatmap_rigorous.T < 0.1, heatmap_rigorous.T)
    axes[1].imshow(
        hm_rig, cmap="jet", alpha=0.5, origin="lower", vmin=0, vmax=1
    )
    if seg_rigorous is not None:
        seg_mask = (seg_rigorous > 0).astype(float)
        axes[1].contour(
            seg_mask.T,
            levels=[0.5],
            colors="lime",
            linewidths=2,
            origin="lower",
        )
    title_rig = "Rigorous Model (BraTS)"
    if iou_rigorous is not None:
        title_rig += f"\nIoU = {iou_rigorous:.3f}"
    axes[1].set_title(
        title_rig, fontsize=12, color="#4CAF50", fontweight="bold"
    )
    axes[1].axis("off")

    fig.suptitle(
        "Where Does the Model Look? — Naive vs Rigorous Grad-CAM",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Naive vs rigorous Grad-CAM figure saved to %s", output_path)


def plot_iou_distribution(
    iou_values: list[float],
    labels: list[int] | None = None,
    class_names: dict[int, str] | None = None,
    output_path: Path | None = None,
    title: str = "Grad-CAM IoU Distribution",
) -> None:
    """
    Plot histogram of IoU values with interpretation thresholds.

    Parameters
    ----------
    iou_values : list[float]
        IoU values for all validated samples.
    labels : list[int], optional
        Class label for each sample (for per-class histograms).
    class_names : dict[int, str], optional
        Class display names.
    output_path : Path, optional
        Where to save the figure.
    title : str
        Figure title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if labels is not None and class_names:
        classes = sorted(set(labels))
        for cls in classes:
            cls_ious = [
                iou
                for iou, lab in zip(iou_values, labels, strict=False)
                if lab == cls
            ]
            ax.hist(
                cls_ious,
                bins=20,
                alpha=0.6,
                label=(
                    f"{class_names.get(cls, f'Class {cls}')} "
                    f"(n={len(cls_ious)})"
                ),
            )
        ax.legend()
    else:
        ax.hist(
            iou_values,
            bins=30,
            color="#1976D2",
            edgecolor="black",
            linewidth=0.3,
            alpha=0.8,
        )

    # Add interpretation thresholds
    ax.axvline(
        0.3,
        color="#F44336",
        linestyle="--",
        linewidth=1.5,
        label="Shortcut threshold (0.3)",
    )
    ax.axvline(
        0.5,
        color="#4CAF50",
        linestyle="--",
        linewidth=1.5,
        label="Clinical threshold (0.5)",
    )

    mean_iou = float(np.mean(iou_values))
    ax.axvline(
        mean_iou,
        color="#FF9800",
        linestyle="-",
        linewidth=2,
        label=f"Mean IoU = {mean_iou:.3f}",
    )

    ax.set_xlabel("IoU (Grad-CAM vs Ground Truth)")
    ax.set_ylabel("Number of Samples")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_gradcam_grid(
    samples: list[dict],
    output_path: Path | None = None,
    n_cols: int = 4,
    title: str = "Grad-CAM Validation Grid",
) -> None:
    """
    Plot a grid of Grad-CAM overlays for multiple patients.

    Parameters
    ----------
    samples : list[dict]
        List of sample dicts with keys: "image", "heatmap", "segmentation",
        "patient_id", "iou", "true_class", "predicted_class".
    output_path : Path, optional
        Where to save the figure.
    n_cols : int
        Number of columns in the grid.
    title : str
        Figure title.
    """
    n_samples = len(samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, sample in enumerate(samples):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        image = sample["image"]
        bg = image[0] if image.ndim == 3 else image

        ax.imshow(bg.T, cmap="gray", origin="lower")

        heatmap = sample["heatmap"]
        hm_masked = np.ma.masked_where(heatmap.T < 0.1, heatmap.T)
        ax.imshow(
            hm_masked, cmap="jet", alpha=0.4, origin="lower", vmin=0, vmax=1
        )

        seg = sample.get("segmentation")
        if seg is not None:
            seg_mask = (seg > 0).astype(float)
            if seg_mask.sum() > 0:
                ax.contour(
                    seg_mask.T,
                    levels=[0.5],
                    colors="lime",
                    linewidths=1.5,
                    origin="lower",
                )

        iou = sample.get("iou", 0)
        pid = sample.get("patient_id", "")
        color = (
            "#4CAF50" if iou > 0.5 else "#F44336" if iou < 0.3 else "#FF9800"
        )
        ax.set_title(f"{pid}\nIoU={iou:.3f}", fontsize=9, color=color)
        ax.axis("off")

    # Hide empty axes
    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grad-CAM grid saved to %s", output_path)
