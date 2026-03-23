# scripts/compare_splits.py — Image-level vs patient-level split comparison
"""
Runs the KEY ablation demonstrating data leakage impact.

Trains the SAME model on the SAME data with two different splitting strategies
and compares the resulting metrics. The expected outcome is a 7-30 pp gap.

Yagis et al. (2021): slice-level CV inflated accuracy by 30% (OASIS) to 48% (PPMI).

Usage:
    python scripts/compare_splits.py
    python scripts/compare_splits.py --backbone densenet121 --n-folds 5

Output:
    - outputs/evaluation/split_comparison.json
    - outputs/figures/split_comparison.png
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
    plot_split_comparison,
)
from src.experiments.split_comparison import (  # noqa: E402
    create_image_level_splits,
    detect_leakage_in_splits,
    run_split_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run image-level vs patient-level split comparison."""
    parser = argparse.ArgumentParser(
        description="Compare image-level vs patient-level splitting"
    )
    parser.add_argument("--data-dir", type=str, default="data/processed/brats2023")
    parser.add_argument("--backbone", type=str, default="densenet121")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    eval_dir = output_dir / "evaluation"
    fig_dir = output_dir / "figures"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data manifest
    data_dir = PROJECT_ROOT / args.data_dir
    manifest_path = data_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error(
            "Manifest not found at %s. Run preprocessing first:\n"
            "  python scripts/preprocess.py --dataset brats2023",
            manifest_path,
        )
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    patient_ids = np.array([s["patient_id"] for s in manifest["samples"]])
    labels = np.array([s["label"] for s in manifest["samples"]])

    logger.info("=" * 60)
    logger.info("SPLIT COMPARISON EXPERIMENT")
    logger.info("=" * 60)
    logger.info("Dataset: %d samples from %d patients", len(labels), len(set(patient_ids)))
    logger.info("Backbone: %s", args.backbone)
    logger.info("Folds: %d", args.n_folds)

    # Step 1: Quantify leakage in image-level splits
    logger.info("\n--- Step 1: Quantifying data leakage ---")
    image_splits = create_image_level_splits(labels, args.n_folds, args.seed)
    leakage = detect_leakage_in_splits(image_splits, patient_ids)

    leakage_path = eval_dir / "leakage_analysis.json"
    with open(leakage_path, "w") as f:
        json.dump(leakage, f, indent=2)
    logger.info("Leakage analysis saved to %s", leakage_path)

    # Step 2: Define the evaluate function
    # In production, this trains a model and evaluates — here we provide
    # a placeholder that can be replaced with actual training
    def evaluate_fold(train_idx, val_idx):
        """Evaluate a single fold using SVM baseline for quick comparison.

        For the full comparison with deep learning models, replace this
        with the training pipeline from scripts/train.py.
        """
        from src.models.baseline_svm import (
            evaluate_baseline,
            train_svm_baseline,
        )

        # Load preprocessed samples
        train_samples = []
        val_samples = []
        for i in train_idx:
            s = manifest["samples"][i]
            npz = np.load(data_dir / s["file"])
            train_samples.append({
                "image": npz["image"],
                "label": s["label"],
                "patient_id": s["patient_id"],
            })
        for i in val_idx:
            s = manifest["samples"][i]
            npz = np.load(data_dir / s["file"])
            val_samples.append({
                "image": npz["image"],
                "label": s["label"],
                "patient_id": s["patient_id"],
            })

        from src.models.baseline_svm import extract_features_from_dataset
        X_train, y_train, _ = extract_features_from_dataset(train_samples)
        X_val, y_val, _ = extract_features_from_dataset(val_samples)

        svm = train_svm_baseline(X_train, y_train)
        metrics = evaluate_baseline(svm, X_val, y_val, "SVM")
        return metrics

    # Step 3: Run the comparison
    logger.info("\n--- Step 2: Running split comparison ---")
    result = run_split_comparison(
        labels=labels,
        patient_ids=patient_ids,
        evaluate_fn=evaluate_fold,
        n_splits=args.n_folds,
        random_state=args.seed,
    )

    # Save results
    result_path = eval_dir / "split_comparison.json"
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Results saved to %s", result_path)

    # Generate figure
    plot_split_comparison(
        result.to_dict(),
        fig_dir / "split_comparison.png",
    )

    logger.info("\nExperiment complete.")


if __name__ == "__main__":
    main()
