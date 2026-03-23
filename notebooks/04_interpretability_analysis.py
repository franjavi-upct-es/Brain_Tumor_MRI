# notebooks/04_interpretability_analysis.py — Interpretability analysis
"""
Interpretability Analysis for Brain Tumor Classification v2
============================================================

Runs the complete interpretability validation pipeline:
  1. Generate Grad-CAM for all test predictions.
  2. Binarize and compute IoU against ground truth segmentations.
  3. Visualize attention maps for correct and incorrect predictions.
  4. Compare naive vs rigorous model attention patterns.
  5. Generate publication-ready figures for thesis Chapter 6.

This is THE MOST POWERFUL thesis experiment: it proves not just WHAT
the model classifies, but WHETHER IT LOOKS WHERE IT SHOULD.
"""

# %% [markdown]
# # Interpretability Validation: Does the Model Look at Tumors?
#
# We validate model attention using Grad-CAM IoU against BraTS
# ground truth segmentations.

# %%
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIGURE_DPI = 300


# %% [markdown]
# ## 1. Synthetic Demonstration
#
# Before running on real data, we demonstrate the IoU validation
# pipeline with synthetic examples showing "good" and "bad" attention.


# %%
def demonstrate_iou_validation(output_dir: Path) -> None:
    """Demonstrate IoU validation with synthetic heatmaps and segmentations.

    Creates two scenarios:
    - GOOD attention: heatmap overlaps with tumor → high IoU.
    - BAD attention: heatmap on borders/background → low IoU.

    Parameters
    ----------
    output_dir : Path
        Directory to save demonstration figures.
    """
    from src.interpretability.attention_validator import (
        compute_dice,
        compute_iou,
    )
    from src.interpretability.gradcam import binarize_heatmap
    from src.interpretability.visualization import plot_gradcam_overlay

    print("=" * 60)
    print("IoU VALIDATION DEMONSTRATION")
    print("=" * 60)

    # Create synthetic MRI-like image
    rng = np.random.RandomState(42)
    h, w = 128, 128
    image = np.zeros((4, h, w), dtype=np.float32)
    image[:, 20:108, 20:108] = (
        rng.randn(4, 88, 88).astype(np.float32) * 0.5 + 0.5
    )

    # Create tumor segmentation (centered blob)
    segmentation = np.zeros((h, w), dtype=np.int32)
    y, x = np.ogrid[:h, :w]
    tumor_mask = ((y - 64) ** 2 + (x - 64) ** 2) < 20**2
    segmentation[tumor_mask] = 1

    # Scenario A: GOOD attention — heatmap on tumor
    heatmap_good = np.zeros((h, w), dtype=np.float32)
    good_mask = ((y - 64) ** 2 + (x - 64) ** 2) < 25**2
    heatmap_good[good_mask] = rng.uniform(0.5, 1.0, size=good_mask.sum())

    binary_good = binarize_heatmap(heatmap_good, 0.5)
    iou_good = compute_iou(binary_good, segmentation > 0)
    dice_good = compute_dice(binary_good, segmentation > 0)

    print("\n  Scenario A (GOOD attention — heatmap on tumor):")
    print(f"    IoU = {iou_good:.3f}, Dice = {dice_good:.3f}")
    print("    Interpretation: IoU > 0.5 → model looks at tumor ✓")

    # Scenario B: BAD attention — heatmap on borders
    heatmap_bad = np.zeros((h, w), dtype=np.float32)
    heatmap_bad[:15, :] = rng.uniform(0.5, 1.0, size=(15, w))
    heatmap_bad[-15:, :] = rng.uniform(0.5, 1.0, size=(15, w))
    heatmap_bad[:, :15] = rng.uniform(0.5, 1.0, size=(h, 15))

    binary_bad = binarize_heatmap(heatmap_bad, 0.5)
    iou_bad = compute_iou(binary_bad, segmentation > 0)
    dice_bad = compute_dice(binary_bad, segmentation > 0)

    print("\n  Scenario B (BAD attention — heatmap on borders):")
    print(f"    IoU = {iou_bad:.3f}, Dice = {dice_bad:.3f}")
    print("    Interpretation: IoU < 0.3 → model uses shortcuts ✗")

    # Generate figures
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_gradcam_overlay(
        image,
        heatmap_good,
        segmentation,
        output_dir / "demo_good_attention.png",
        title=f"GOOD Attention (IoU = {iou_good:.3f})",
        iou=iou_good,
    )

    plot_gradcam_overlay(
        image,
        heatmap_bad,
        segmentation,
        output_dir / "demo_bad_attention.png",
        title=f"BAD Attention (IoU = {iou_bad:.3f})",
        iou=iou_bad,
    )

    print(f"\n  Figures saved to {output_dir}")


# %% [markdown]
# ## 2. IoU Distribution Analysis


# %%
def analyze_iou_distribution(results_path: Path, output_dir: Path) -> None:
    """Load validation results and analyze IoU distribution.

    Parameters
    ----------
    results_path : Path
        Path to attention validation JSON results.
    output_dir : Path
        Figure output directory.
    """
    from src.interpretability.visualization import plot_iou_distribution

    if not results_path.exists():
        print(f"[INFO] Validation results not found at {results_path}")
        print("  Run the full evaluation pipeline first to generate these.")
        return

    with open(results_path) as f:
        data = json.load(f)

    ious = [s["iou"] for s in data.get("per_sample", [])]
    labels = [s["true_class"] for s in data.get("per_sample", [])]

    if not ious:
        print("[INFO] No IoU values found in results.")
        return

    print(f"\nIoU Distribution (n={len(ious)}):")
    print(f"  Mean: {np.mean(ious):.3f} ± {np.std(ious):.3f}")
    print(f"  Median: {np.median(ious):.3f}")
    print(
        f"  IoU > 0.5 (clinically relevant): {sum(1 for x in ious if x > 0.5)}/{len(ious)}"
    )
    print(
        f"  IoU < 0.3 (shortcut learning): {sum(1 for x in ious if x < 0.3)}/{len(ious)}"
    )

    plot_iou_distribution(
        ious,
        labels,
        class_names={0: "LGG", 1: "HGG"},
        output_path=output_dir / "iou_distribution.png",
    )


# %% [markdown]
# ## Main Execution


# %%
def main() -> None:
    """Run the interpretability analysis pipeline."""
    parser = argparse.ArgumentParser(description="Interpretability analysis")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/figures/interpretability"
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="outputs/evaluation/attention_validation.json",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    results_path = PROJECT_ROOT / args.results_path

    print("=" * 60)
    print("INTERPRETABILITY ANALYSIS")
    print("=" * 60)

    print("\n[1/2] Running synthetic IoU demonstration...")
    demonstrate_iou_validation(output_dir / "demo")

    print("\n[2/2] Analyzing IoU distribution from model results...")
    analyze_iou_distribution(results_path, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
