# scripts/ablation_preprocessing.py — Preprocessing ablation study
"""
Runs the preprocessing ablation study: systematically
removes one preprocessing step at a time to measure its contribution.

Ablations:
    1. no_n4: Skip N4 bias field correction.
    2. no_zscore: Skip z-score normalization.
    3. no_augmentation: Disable all training augmentation.
    4. generic_augmentation: Replace MRI-specific with generic augmentations.
    5. no_pretrained: Train from scratch (no ImageNet weights).

Each ablation keeps everything else identical to the baseline.

Usage:
    python scripts/ablation_preprocessing.py
    python scripts/ablation_preprocessing.py --backbone densenet121

Output:
    - outputs/evaluation/ablation_preprocessing.json
    - outputs/figures/ablation_preprocessing.png

Reference: configs/experiment/ablation_preproc.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.ablation_runner import (  # noqa: E402
    AblationResult,
    define_preprocessing_ablations,
    run_ablation_study,
)
from src.experiments.figure_generator import plot_ablation_study  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the preprocessing ablation study."""
    parser = argparse.ArgumentParser(
        description="Preprocessing ablation study"
    )
    parser.add_argument("--backbone", type=str, default="densenet121")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--data-dir", type=str, default="data/processed/brats2023"
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    eval_dir = output_dir / "evaluation"
    fig_dir = output_dir / "figures"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PREPROCESSING ABLATION STUDY")
    logger.info("=" * 60)

    # Define baseline configuration
    baseline_config = {
        "backbone": args.backbone,
        "n_folds": args.n_folds,
        "data_dir": args.data_dir,
        "seed": args.seed,
        "apply_n4": True,
        "z_score_normalize": True,
        "enable_augmentation": True,
        "augmentation_type": "mri_specific",
        "pretrained": True,
    }

    # Define ablation variants
    variants = define_preprocessing_ablations()

    # Training function placeholder — in production, this runs the
    # full train.py pipeline and returns fold metrics.
    def train_and_evaluate(config):
        """Placeholder training function.

        Replace with actual training pipeline for production runs.
        Returns simulated results for pipeline testing.
        """
        logger.info(
            "  [Placeholder] Would train with config: %s",
            {k: v for k, v in config.items() if k not in ("data_dir",)},
        )

        # Return a placeholder result
        return AblationResult(
            metrics={
                "balanced_accuracy": 0.0,
                "f1_macro": 0.0,
                "auc_roc_macro": 0.0,
            },
            per_fold_metrics=[],
        )

    # Run ablation study
    study = run_ablation_study(
        study_name="Preprocessing",
        question="Does each preprocessing step contribute to performance?",
        baseline_config=baseline_config,
        variants=variants,
        train_and_evaluate_fn=train_and_evaluate,
        run_significance_tests=False,
    )

    # Save results
    result_path = eval_dir / "ablation_preprocessing.json"
    with open(result_path, "w") as f:
        json.dump(study.to_dict(), f, indent=2)
    logger.info("Results saved to %s", result_path)

    # Generate figure
    plot_ablation_study(
        study.to_dict(), fig_dir / "ablation_preprocessing.png"
    )

    # Generate LaTeX table
    latex_table = study.generate_latex_table()
    latex_path = eval_dir / "ablation_preprocessing.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)
    logger.info("LaTeX table saved to %s", latex_path)

    logger.info("\nAblation study complete.")


if __name__ == "__main__":
    main()
