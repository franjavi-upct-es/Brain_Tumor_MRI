# scripts/evaluate.py — Comprehensive evaluation with bootstrap CI
"""
Evaluation entry point that computes all CLAIM-compliant metrics with
patient-level bootstrap confidence intervals.

Modes:
    --all-folds: Evaluate all 5 CV folds, aggregate mean ± std.
    --dataset ucsf_pdgm: Evaluate on external UCSF-PDGM validation.
    --compare: Run statistical tests comparing two models.

Output:
    - JSON metrics file (for DVC tracking).
    - LaTeX tables (for thesis).
    - Console summary.

Usage:
    python scripts/evaluate.py --all-folds --bootstrap 1000
    python scripts/evaluate.py --dataset ucsf_pdgm --bootstrap 1000
    python scripts/evaluate.py --compare naive rigorous --test mcnemar
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

from src.evaluation.bootstrap import patient_level_bootstrap  # noqa: E402
from src.evaluation.calibration import compute_calibration  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    compute_full_report,
    compute_roc_curve_data,
)
from src.evaluation.report_generator import (  # noqa: E402
    format_metric_with_ci,
    save_full_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_predictions(predictions_path: Path) -> dict:
    """Load saved model predictions for evaluation.

    Expected format (JSON):
        {
            "y_true": [0, 1, 1, 0, ...],
            "y_pred": [0, 1, 0, 0, ...],
            "y_prob": [[0.8, 0.2], [0.3, 0.7], ...],
            "patient_ids": ["p001", "p002", ...],
            "model": "densenet121",
            "fold": 0
        }

    Parameters
    ----------
    predictions_path : Path
        Path to the JSON predictions file.

    Returns
    -------
    dict
        Loaded predictions dictionary.
    """
    with open(predictions_path) as f:
        data = json.load(f)

    return {
        "y_true": np.array(data["y_true"]),
        "y_pred": np.array(data["y_pred"]),
        "y_prob": np.array(data["y_prob"]) if "y_prob" in data else None,
        "patient_ids": np.array(data["patient_ids"]),
        "model": data.get("model", "unknown"),
        "fold": data.get("fold", -1),
    }


def evaluate_single_fold(
    predictions_path: Path,
    n_bootstrap: int = 1000,
    output_dir: Path | None = None,
) -> dict:
    """Evaluate predictions from a single fold with bootstrap CI.

    Parameters
    ----------
    predictions_path : Path
        Path to predictions JSON.
    n_bootstrap : int
        Number of bootstrap iterations.
    output_dir : Path, optional
        Directory to save reports.

    Returns
    -------
    dict
        Evaluation results including metrics and bootstrap CIs.
    """
    preds = load_predictions(predictions_path)

    # Full metrics report
    report = compute_full_report(
        preds["y_true"], preds["y_pred"], preds["y_prob"]
    )

    # Bootstrap CIs
    bootstrap_results = patient_level_bootstrap(
        y_true=preds["y_true"],
        y_pred=preds["y_pred"],
        patient_ids=preds["patient_ids"],
        y_prob=preds["y_prob"],
        n_iterations=n_bootstrap,
    )

    # Calibration
    calibration = None
    if preds["y_prob"] is not None:
        calibration = compute_calibration(
            preds["y_true"], preds["y_prob"]
        )

    # ROC curve data
    roc_data = None
    if preds["y_prob"] is not None:
        roc_data = compute_roc_curve_data(
            preds["y_true"], preds["y_prob"]
        )

    # Print summary
    logger.info("\n--- Fold %d Evaluation Results ---", preds["fold"])
    for name, br in bootstrap_results.items():
        logger.info("  %s: %s", name, format_metric_with_ci(br))
    if calibration:
        logger.info("  ECE: %.4f, Brier: %.4f", calibration.ece, calibration.brier_score)

    # Save report
    if output_dir:
        save_full_report(
            report=report,
            bootstrap_results=bootstrap_results,
            output_dir=output_dir,
            model_name=preds["model"],
            dataset_name=f"fold_{preds['fold']}",
        )

        # Save ROC curve data for DVC plots
        if roc_data:
            roc_path = output_dir / f"roc_fold_{preds['fold']}.json"
            with open(roc_path, "w") as f:
                json.dump(roc_data, f, indent=2)

    return {
        "report": report.to_dict(),
        "bootstrap": {k: v.to_dict() for k, v in bootstrap_results.items()},
        "calibration": calibration.to_dict() if calibration else None,
    }


def evaluate_all_folds(
    output_dir: Path,
    n_bootstrap: int = 1000,
) -> dict:
    """Evaluate all 5 CV folds and aggregate results.

    Parameters
    ----------
    output_dir : Path
        Base output directory containing fold predictions.
    n_bootstrap : int
        Number of bootstrap iterations per fold.

    Returns
    -------
    dict
        Aggregated results with per-fold and mean ± std.
    """
    logger.info("=" * 60)
    logger.info("EVALUATING ALL FOLDS")
    logger.info("=" * 60)

    eval_output = output_dir / "evaluation"
    eval_output.mkdir(parents=True, exist_ok=True)

    fold_results = []
    for fold in range(5):
        pred_path = output_dir / "metrics" / f"fold_{fold}" / "predictions.json"
        if not pred_path.exists():
            logger.warning("Predictions not found for fold %d: %s", fold, pred_path)
            continue

        result = evaluate_single_fold(pred_path, n_bootstrap, eval_output)
        fold_results.append(result)

    if not fold_results:
        logger.error(
            "No fold predictions found. Train first:\n"
            "  python scripts/run_cross_validation.py"
        )
        return {}

    # Aggregate across folds
    aggregated = _aggregate_fold_results(fold_results)

    # Save final metrics
    final_path = eval_output / "final_metrics.json"
    with open(final_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    logger.info("\n--- Aggregated Results (mean ± std across folds) ---")
    if "aggregated" in aggregated:
        for metric, stats in aggregated["aggregated"].items():
            logger.info("  %s: %.4f ± %.4f", metric, stats["mean"], stats["std"])

    return aggregated


def _aggregate_fold_results(fold_results: list[dict]) -> dict:
    """Aggregate evaluation results across CV folds.

    Parameters
    ----------
    fold_results : list[dict]
        Per-fold evaluation results.

    Returns
    -------
    dict
        Aggregated statistics.
    """
    # Collect primary metrics across folds
    metric_keys = ["balanced_accuracy", "f1_macro", "auc_roc_macro", "cohens_kappa"]
    aggregated = {}

    for key in metric_keys:
        values = []
        for result in fold_results:
            if "bootstrap" in result and key in result["bootstrap"]:
                values.append(result["bootstrap"][key]["point_estimate"])
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "per_fold": values,
            }

    return {
        "n_folds": len(fold_results),
        "per_fold": fold_results,
        "aggregated": aggregated,
    }


def main() -> None:
    """Entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate brain tumor classifier")
    parser.add_argument(
        "--all-folds", action="store_true",
        help="Evaluate all 5 CV folds and aggregate",
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Evaluate a specific fold",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["brats2023", "ucsf_pdgm"],
        help="Evaluate on a specific dataset",
    )
    parser.add_argument(
        "--predictions", type=str, default=None,
        help="Path to predictions JSON file",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=1000,
        help="Number of bootstrap iterations (default: 1000)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir

    if args.all_folds:
        evaluate_all_folds(output_dir, args.bootstrap)
    elif args.predictions:
        evaluate_single_fold(
            Path(args.predictions), args.bootstrap,
            output_dir / "evaluation",
        )
    elif args.fold is not None:
        pred_path = output_dir / "metrics" / f"fold_{args.fold}" / "predictions.json"
        evaluate_single_fold(pred_path, args.bootstrap, output_dir / "evaluation")
    elif args.dataset == "ucsf_pdgm":
        pred_path = output_dir / "evaluation" / "ucsf_pdgm_predictions.json"
        if pred_path.exists():
            evaluate_single_fold(pred_path, args.bootstrap, output_dir / "evaluation")
        else:
            logger.error("UCSF-PDGM predictions not found: %s", pred_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
