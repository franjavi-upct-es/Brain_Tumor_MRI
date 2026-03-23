# scripts/generate_figures.py — Generate all thesis figures
"""
Master script for generating all publication-ready thesis figures at 300 DPI.

Reads evaluation results from outputs/ and generates figures organized by
thesis chapter. Can generate all figures at once or specific subsets.

Usage:
    python scripts/generate_figures.py --all
    python scripts/generate_figures.py --roc --confusion-matrix
    python scripts/generate_figures.py --gradcam --iou-validation
    python scripts/generate_figures.py --split-comparison

Output directory: outputs/figures/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.figure_generator import (  # noqa: E402
    plot_ablation_study,
    plot_calibration_diagram,
    plot_confusion_matrix,
    plot_cross_dataset_comparison,
    plot_roc_curves,
    plot_split_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_all_figures(output_dir: Path, eval_dir: Path) -> None:
    """Generate all available figures from evaluation results.

    Parameters
    ----------
    output_dir : Path
        Directory to save figures.
    eval_dir : Path
        Directory containing evaluation JSON results.
    """
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GENERATING THESIS FIGURES")
    logger.info("=" * 60)

    generated = 0

    # Split comparison (Chapter 4)
    split_path = eval_dir / "split_comparison.json"
    if split_path.exists():
        logger.info("\n[1] Split comparison figure...")
        with open(split_path) as f:
            data = json.load(f)
        plot_split_comparison(data, fig_dir / "ch4_split_comparison.png")
        generated += 1

    # Ablation study (Chapter 6)
    ablation_path = eval_dir / "ablation_preprocessing.json"
    if ablation_path.exists():
        logger.info("\n[2] Ablation study figure...")
        with open(ablation_path) as f:
            data = json.load(f)
        plot_ablation_study(data, fig_dir / "ch6_ablation_preprocessing.png")
        generated += 1

    # ROC curves (Chapter 6)
    roc_files = list(eval_dir.glob("roc_*.json"))
    if roc_files:
        logger.info("\n[3] ROC curves figure...")
        roc_data = {}
        for roc_file in roc_files:
            with open(roc_file) as f:
                roc_data[roc_file.stem] = json.load(f)
        plot_roc_curves(roc_data, fig_dir / "ch6_roc_curves.png")
        generated += 1

    # Cross-dataset comparison (Chapter 6)
    internal_path = eval_dir / "final_metrics.json"
    external_path = eval_dir / "external_metrics.json"
    if internal_path.exists() and external_path.exists():
        logger.info("\n[4] Cross-dataset comparison figure...")
        with open(internal_path) as f:
            internal = json.load(f)
        with open(external_path) as f:
            external = json.load(f)

        internal_agg = internal.get("aggregated", {})
        external_agg = external.get("aggregated", {})

        int_metrics = {k: v["mean"] for k, v in internal_agg.items()}
        ext_metrics = {k: v["mean"] for k, v in external_agg.items()}

        if int_metrics and ext_metrics:
            plot_cross_dataset_comparison(
                int_metrics, ext_metrics,
                fig_dir / "ch6_cross_dataset.png",
            )
            generated += 1

    # Confusion matrices (Chapter 6)
    for report_file in eval_dir.glob("*_report.json"):
        logger.info("\n[5] Confusion matrix for %s...", report_file.stem)
        with open(report_file) as f:
            report = json.load(f)

        cm_data = report.get("metrics", {}).get("confusion_matrix", {})
        cm_abs = cm_data.get("absolute", [])
        if cm_abs:
            cm_arr = np.array(cm_abs)
            plot_confusion_matrix(
                cm_arr, ["LGG", "HGG"],
                fig_dir / f"ch6_cm_{report_file.stem}.png",
                title=f"Confusion Matrix — {report.get('model', '')}",
            )
            generated += 1

    # Calibration diagrams (Chapter 6)
    for report_file in eval_dir.glob("*_report.json"):
        with open(report_file) as f:
            report = json.load(f)
        cal_data = report.get("calibration")
        if cal_data:
            logger.info("\n[6] Calibration diagram for %s...", report_file.stem)
            plot_calibration_diagram(
                cal_data,
                fig_dir / f"ch6_calibration_{report_file.stem}.png",
                model_name=report.get("model", "Model"),
            )
            generated += 1

    logger.info("\n" + "=" * 60)
    logger.info("Figure generation complete: %d figures generated", generated)
    logger.info("Output directory: %s", fig_dir)

    if generated == 0:
        logger.info(
            "\nNo evaluation results found. Run experiments first:\n"
            "  python scripts/run_cross_validation.py\n"
            "  python scripts/compare_splits.py\n"
            "  python scripts/ablation_preprocessing.py"
        )


def main() -> None:
    """Entry point for figure generation."""
    parser = argparse.ArgumentParser(description="Generate thesis figures")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--split-comparison", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--roc", action="store_true")
    parser.add_argument("--confusion-matrix", action="store_true")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--cross-dataset", action="store_true")
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--iou-validation", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    eval_dir = output_dir / "evaluation"

    # Default to --all if no specific flag
    if not any([args.split_comparison, args.ablation, args.roc,
                args.confusion_matrix, args.calibration,
                args.cross_dataset, args.gradcam, args.iou_validation]):
        args.all = True

    if args.all:
        generate_all_figures(output_dir, eval_dir)
    else:
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        if args.split_comparison:
            path = eval_dir / "split_comparison.json"
            if path.exists():
                with open(path) as f:
                    plot_split_comparison(json.load(f), fig_dir / "split_comparison.png")

        if args.ablation:
            path = eval_dir / "ablation_preprocessing.json"
            if path.exists():
                with open(path) as f:
                    plot_ablation_study(json.load(f), fig_dir / "ablation_preprocessing.png")

        logger.info("Selected figures generated.")


if __name__ == "__main__":
    main()
