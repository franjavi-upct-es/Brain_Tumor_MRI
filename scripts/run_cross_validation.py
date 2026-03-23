# scripts/run_cross_validation.py — Run complete 5-fold cross-validation
"""
Orchestrates the complete 5-fold StratifiedGroupKFold cross-validation.

For each fold:
  1. Trains with the two-phase protocol (head-only → gradual unfreeze).
  2. Saves per-fold metrics to outputs/metrics/fold_{i}/metrics.json.

After all folds:
  3. Aggregates results (mean ± std across folds).
  4. Saves summary to outputs/evaluation/cv_summary.json.

Usage:
    python scripts/run_cross_validation.py
    python scripts/run_cross_validation.py --backbone resnet50 --n-folds 5
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

from scripts.train import train_fold  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_cross_validation(
    n_folds: int = 5,
    backbone: str = "densenet121",
    **train_kwargs,
) -> dict:
    """Run complete k-fold cross-validation.

    Parameters
    ----------
    n_folds : int
        Number of CV folds.
    backbone : str
        Backbone architecture.
    **train_kwargs
        Additional arguments passed to train_fold().

    Returns
    -------
    dict
        Summary with per-fold and aggregated metrics.
    """
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION: %d folds, backbone=%s", n_folds, backbone)
    logger.info("=" * 60)

    all_fold_results = []

    for fold in range(n_folds):
        logger.info("\n{'='*40}")
        logger.info("Starting fold %d/%d", fold + 1, n_folds)
        logger.info("{'='*40}")

        fold_results = train_fold(
            fold=fold,
            backbone=backbone,
            **train_kwargs,
        )
        all_fold_results.append(fold_results)

    # Aggregate metrics across folds
    summary = aggregate_cv_results(all_fold_results)
    summary["n_folds"] = n_folds
    summary["backbone"] = backbone

    # Save summary
    output_dir = PROJECT_ROOT / train_kwargs.get("output_dir", "outputs")
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "cv_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("CROSS-VALIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Summary saved to %s", summary_path)

    # Print aggregated results
    if "aggregated" in summary:
        for metric, stats in summary["aggregated"].items():
            logger.info(
                "  %s: %.4f ± %.4f",
                metric, stats["mean"], stats["std"],
            )

    return summary


def aggregate_cv_results(fold_results: list[dict]) -> dict:
    """Aggregate metrics across CV folds (mean ± std).

    Parameters
    ----------
    fold_results : list[dict]
        List of per-fold result dictionaries.

    Returns
    -------
    dict
        Summary with per-fold results and aggregated statistics.
    """
    # Collect all metric keys that are numeric
    metric_keys = set()
    for result in fold_results:
        for key, value in result.items():
            if isinstance(value, (int, float)) and key.startswith("val/"):
                metric_keys.add(key)

    # Compute mean and std for each metric
    aggregated = {}
    for key in sorted(metric_keys):
        values = [r[key] for r in fold_results if key in r]
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "per_fold": values,
            }

    return {
        "per_fold": fold_results,
        "aggregated": aggregated,
    }


def main() -> None:
    """Entry point for cross-validation."""
    parser = argparse.ArgumentParser(description="Run 5-fold cross-validation")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--backbone", type=str, default="densenet121")
    parser.add_argument("--phase1-epochs", type=int, default=10)
    parser.add_argument("--phase2-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default="data/processed/brats2023")
    parser.add_argument("--splits-path", type=str, default="data/splits/brats_5fold.json")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-mode", type=str, default="disabled")
    args = parser.parse_args()

    kwargs = vars(args)
    n_folds = kwargs.pop("n_folds")
    backbone = kwargs.pop("backbone")
    run_cross_validation(n_folds=n_folds, backbone=backbone, **kwargs)


if __name__ == "__main__":
    main()
