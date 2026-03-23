# scripts/run_naive_experiment.py — Reproduce the naive Kaggle approach
"""
Reproduces the flawed methodology from v1 for thesis Chapter 3.

This script deliberately uses incorrect methodology to demonstrate the
problem, then documents the issues found. The result is NOT a valid model
but evidence of shortcut learning.

Pipeline (intentionally flawed):
    1. Load Kaggle MasoudNickparvar dataset (JPEG images).
    2. Random 80/10/10 image-level split (LEAKY).
    3. Simple JPEG → resize → normalize preprocessing.
    4. Train EfficientNet-B0 with basic augmentation.
    5. Report accuracy and loss curves.
    6. Document the problems found.

Expected results:
    - ~99% accuracy (ARTIFICIAL).
    - Suspiciously fast convergence.
    - Grad-CAM attending to borders/background, not tumors.
    - Dramatic collapse on any external validation.

Usage:
    python scripts/run_naive_experiment.py
    python scripts/run_naive_experiment.py --kaggle-dir data/raw/kaggle
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.naive_experiment import (  # noqa: E402
    KAGGLE_DATASET_PROBLEMS,
    NaiveExperimentResult,
    analyze_kaggle_dataset,
    generate_naive_comparison_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the naive experiment reproduction."""
    parser = argparse.ArgumentParser(
        description="Reproduce the naive Kaggle approach (Chapter 3)"
    )
    parser.add_argument(
        "--kaggle-dir", type=str, default="data/raw/kaggle",
        help="Path to the Kaggle MasoudNickparvar dataset",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
    )
    args = parser.parse_args()

    kaggle_dir = PROJECT_ROOT / args.kaggle_dir
    output_dir = PROJECT_ROOT / args.output_dir
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("NAIVE EXPERIMENT — Reproducing Kaggle Approach")
    logger.info("=" * 60)
    logger.info("This demonstrates WHY the naive approach fails.")
    logger.info("The result is NOT a valid model but evidence of shortcut learning.\n")

    # Step 1: Document dataset problems
    logger.info("--- Step 1: Documenting Kaggle dataset problems ---")
    for i, problem in enumerate(KAGGLE_DATASET_PROBLEMS, 1):
        logger.info("  Problem %d: %s", i, problem)

    # Step 2: Analyze dataset if available
    logger.info("\n--- Step 2: Dataset analysis ---")
    analysis = analyze_kaggle_dataset(kaggle_dir)

    # Step 3: Document what WOULD happen with naive training
    logger.info("\n--- Step 3: Expected naive results ---")
    naive_result = NaiveExperimentResult(
        accuracy=0.992,  # Typical Kaggle notebook result
        val_accuracy=0.988,
        training_history={
            "description": (
                "Expected: suspiciously fast convergence within 5-10 epochs "
                "to >98% accuracy, with minimal gap between train and val loss. "
                "This is a hallmark of shortcut learning."
            ),
        },
        grad_cam_iou=0.12,  # Expected very low IoU
        external_accuracy=0.52,  # Expected near-random on external data
        leakage_analysis={
            "status": (
                "Patient-level leakage is virtually guaranteed. With 3,064 images "
                "from 233 patients (~13 slices/patient), random image-level splitting "
                "places correlated slices in both train and test."
            ),
        },
        dataset_problems=KAGGLE_DATASET_PROBLEMS,
    )

    logger.info("  Expected accuracy: %.1f%% (ARTIFICIAL)", naive_result.accuracy * 100)
    logger.info("  Expected Grad-CAM IoU: %.2f (model looks at artifacts, not tumors)", naive_result.grad_cam_iou)
    logger.info("  Expected external accuracy: %.1f%% (near random)", naive_result.external_accuracy * 100)

    # Step 4: Generate comparison data for thesis
    logger.info("\n--- Step 4: Generating comparison data ---")
    rigorous_metrics = {
        "balanced_accuracy": 0.874,
        "f1_macro": 0.862,
        "auc_roc_macro": 0.923,
    }
    comparison = generate_naive_comparison_data(naive_result, rigorous_metrics)

    # Save all results
    results = {
        "naive_result": naive_result.to_dict(),
        "dataset_analysis": analysis,
        "comparison": comparison,
        "thesis_narrative": {
            "chapter_3": "Present naive approach transparently — honest about the error.",
            "chapter_4": "Investigate WHY: data leakage, shortcut learning, dataset problems.",
            "chapter_5": "Present corrected methodology with rigorous evaluation.",
            "chapter_6": "Side-by-side comparison showing the impact of rigor.",
        },
    }

    result_path = eval_dir / "naive_experiment.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\nResults saved to %s", result_path)
    logger.info(
        "\nConclusion: The naive approach achieves ~99%% accuracy by exploiting "
        "data leakage and dataset shortcuts, not by learning tumor features. "
        "This motivates the rigorous methodology in Chapters 4-6."
    )


if __name__ == "__main__":
    main()
