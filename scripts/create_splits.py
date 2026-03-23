# scripts/create_splits.py — Create patient-level cross-validation splits
"""
Creates StratifiedGroupKFold splits from preprocessed BraTS data and saves
the split definitions to JSON for reproducibility and DVC tracking.

This is a critical step: patient-level splitting prevents data leakage.
Yagis et al. (2021) showed image-level splitting inflates accuracy by 30-48%.

The output JSON contains train/val indices and patient IDs for each fold,
which the DataModule reads during setup().

Usage:
    python scripts/create_splits.py --seed 42
    python scripts/create_splits.py --seed 42 --n-folds 5
    python scripts/create_splits.py --seed 42 --output data/splits/brats_5fold.json
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

from src.data.splitter import PatientLevelSplitter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load patient IDs and labels from a preprocessed dataset manifest.

    Parameters
    ----------
    manifest_path : Path
        Path to manifest.json produced by preprocess.py.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (patient_ids, labels) arrays — one entry per sample.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    patient_ids = []
    labels = []
    for sample in manifest["samples"]:
        patient_ids.append(sample["patient_id"])
        labels.append(sample["label"])

    return np.array(patient_ids), np.array(labels)


def main() -> None:
    """Create patient-level stratified k-fold splits."""
    parser = argparse.ArgumentParser(
        description="Create patient-level cross-validation splits"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to preprocessed data directory (default: data/processed/brats2023)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for splits JSON (default: data/splits/brats_5fold.json)",
    )
    args = parser.parse_args()

    # Resolve paths
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "processed" / "brats2023"
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "splits" / "brats_5fold.json"

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(
            "Manifest not found at %s. Run preprocessing first:\n"
            "  python scripts/preprocess.py --dataset brats2023",
            manifest_path,
        )
        sys.exit(1)

    # Load data
    logger.info("Loading manifest from %s", manifest_path)
    patient_ids, labels = load_manifest(manifest_path)

    unique_patients = len(set(patient_ids))
    n_samples = len(patient_ids)
    n_hgg = int(np.sum(labels == 1))
    n_lgg = int(np.sum(labels == 0))

    logger.info(
        "Dataset: %d samples from %d patients (%d HGG, %d LGG)",
        n_samples, unique_patients, n_hgg, n_lgg,
    )

    # Create splits
    logger.info(
        "Creating %d-fold StratifiedGroupKFold splits (seed=%d)...",
        args.n_folds, args.seed,
    )
    splitter = PatientLevelSplitter(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=args.seed,
    )
    splits = splitter.create_splits(patient_ids, labels)

    # Log fold statistics
    logger.info("\n--- Fold statistics ---")
    for fold in splits:
        train_labels = labels[fold["train_indices"]]
        val_labels = labels[fold["val_indices"]]
        logger.info(
            "Fold %d: Train=%d patients (%d HGG, %d LGG), "
            "Val=%d patients (%d HGG, %d LGG)",
            fold["fold"],
            fold["n_train_patients"],
            int(np.sum(train_labels == 1)), int(np.sum(train_labels == 0)),
            fold["n_val_patients"],
            int(np.sum(val_labels == 1)), int(np.sum(val_labels == 0)),
        )

    # Save splits
    splitter.save_splits(output_path)
    logger.info("\nSplits saved to %s", output_path)
    logger.info(
        "Verify with: PYTHONPATH=. pytest tests/test_data/test_no_leakage.py -v"
    )


if __name__ == "__main__":
    main()
