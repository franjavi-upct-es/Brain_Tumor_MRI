# scripts/preprocess.py — Preprocessing pipeline for BraTS 2023 and UCSF-PDGM
"""
Runs the validated preprocessing pipeline on raw NIfTI data and produces
preprocessed 2D slices saved as NPZ files for efficient training.

BraTS 2023:
    QC → z-score normalization → slice selection → save NPZ.

UCSF-PDGM:
    Select modalities → N4ITK bias correction → resample to 1mm³ →
    z-score normalization → slice selection → save NPZ.

Also generates a manifest.json listing all preprocessed samples with
metadata (patient_id, label, file path) for the DataModule to consume.

Usage:
    python scripts/preprocess.py --config configs/data/brats2023.yaml
    python scripts/preprocess.py --config configs/data/ucsf_pdgm.yaml
    python scripts/preprocess.py --dataset brats2023
    python scripts/preprocess.py --dataset ucsf_pdgm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.label_derivation import (  # noqa: E402
    derive_brats_labels,
    load_ucsf_labels,
    save_labels,
)
from src.data.preprocessing import (  # noqa: E402
    preprocess_brats_patient,
    preprocess_ucsf_patient,
)
from src.data.quality_control import (  # noqa: E402
    DatasetQualityControl,
    generate_qc_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def preprocess_brats(
    raw_dir: Path,
    output_dir: Path,
    metadata_path: Path | None = None,
    slice_method: str = "max_tumor_area",
    context_slices: int = 0,
) -> None:
    """Run the complete BraTS 2023 preprocessing pipeline.

    Parameters
    ----------
    raw_dir : Path
        Path to raw BraTS data (patient subdirectories).
    output_dir : Path
        Directory for preprocessed NPZ files.
    metadata_path : Path, optional
        Path to BraTS metadata CSV for grade labels.
    slice_method : str
        Slice selection method.
    context_slices : int
        Adjacent slices for 2.5D.
    """
    logger.info("=" * 60)
    logger.info("BraTS 2023 PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info("Input:  %s", raw_dir)
    logger.info("Output: %s", output_dir)

    if not raw_dir.exists():
        logger.error("Raw data directory not found: %s", raw_dir)
        sys.exit(1)

    # Check if data needs extraction first
    patient_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("BraTS-")]
    zip_dir = raw_dir / "Data" / "BraTS-GLI"
    if not patient_dirs and zip_dir.exists():
        zip_files = list(zip_dir.glob("*TrainingData.zip"))
        if zip_files:
            logger.error(
                "Found %d zip file(s) in %s but no extracted patient directories. "
                "Run 'make extract-brats' first to extract the data.",
                len(zip_files),
                zip_dir,
            )
            sys.exit(1)

    # Step 1: Derive grade labels
    logger.info("\n--- Step 1: Deriving grade labels ---")
    labels = derive_brats_labels(raw_dir, metadata_path)
    if not labels:
        logger.error("No grade labels could be derived. Aborting.")
        sys.exit(1)

    # Save labels for reproducibility
    labels_path = output_dir / "grade_labels.json"
    save_labels(labels, labels_path)

    # Step 2: Quality control
    logger.info("\n--- Step 2: Running quality control ---")
    qc = DatasetQualityControl(
        expected_modalities=["t1n", "t1c", "t2w", "t2f"],
        expected_shape=(240, 240, 155),
        min_tumor_voxels=100,
    )
    qc_results = qc.check_dataset(raw_dir)
    qc_report = generate_qc_report(qc_results)

    qc_report_path = output_dir / "qc_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(qc_report_path, "w") as f:
        json.dump(qc_report, f, indent=2)
    logger.info(
        "QC: %d/%d passed (%.1f%%)",
        qc_report["passed"], qc_report["total_patients"],
        qc_report["pass_rate"] * 100,
    )

    # Step 3: Preprocess each patient
    logger.info("\n--- Step 3: Preprocessing patients ---")
    passed_ids = {r.patient_id for r in qc_results if r.passed}
    patient_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])

    manifest_samples = []
    all_stats = []
    t_start = time.time()

    for i, patient_dir in enumerate(patient_dirs):
        pid = patient_dir.name

        # Skip QC failures
        if pid not in passed_ids:
            continue

        # Skip patients without labels
        if pid not in labels:
            continue

        stats = preprocess_brats_patient(
            patient_dir=patient_dir,
            output_dir=output_dir,
            slice_method=slice_method,
            context_slices=context_slices,
        )

        if stats is not None:
            all_stats.append(stats)
            manifest_samples.append({
                "file": f"{pid}.npz",
                "patient_id": pid,
                "label": labels[pid],
                "slice_index": stats.selected_slice,
                "tumor_area": stats.tumor_area_on_slice,
            })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            logger.info(
                "Progress: %d/%d patients (%.1f sec)",
                i + 1, len(patient_dirs), elapsed,
            )

    # Step 4: Save manifest
    manifest = {
        "dataset": "brats2023",
        "preprocessing": {
            "z_score": True,
            "foreground_only": True,
            "slice_method": slice_method,
            "context_slices": context_slices,
        },
        "n_samples": len(manifest_samples),
        "n_hgg": sum(1 for s in manifest_samples if s["label"] == 1),
        "n_lgg": sum(1 for s in manifest_samples if s["label"] == 0),
        "samples": manifest_samples,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Save preprocessing stats
    stats_path = output_dir / "preprocessing_stats.json"
    with open(stats_path, "w") as f:
        stats_dicts = [
            {
                "patient_id": s.patient_id,
                "original_shape": list(s.original_shape),
                "final_shape": list(s.final_shape),
                "selected_slice": s.selected_slice,
                "tumor_area_on_slice": s.tumor_area_on_slice,
                "fg_mean": s.fg_mean,
                "fg_std": s.fg_std,
            }
            for s in all_stats
        ]
        json.dump(stats_dicts, f, indent=2)

    elapsed = time.time() - t_start
    logger.info("\n--- BraTS preprocessing complete ---")
    logger.info("Patients processed: %d", len(manifest_samples))
    logger.info("HGG: %d, LGG: %d", manifest["n_hgg"], manifest["n_lgg"])
    logger.info("Time: %.1f seconds", elapsed)
    logger.info("Manifest: %s", manifest_path)


def preprocess_ucsf(
    raw_dir: Path,
    output_dir: Path,
    metadata_path: Path | None = None,
    apply_n4: bool = True,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    slice_method: str = "max_tumor_area",
    context_slices: int = 0,
) -> None:
    """Run the complete UCSF-PDGM preprocessing pipeline.

    Parameters
    ----------
    raw_dir : Path
        Path to raw UCSF-PDGM data.
    output_dir : Path
        Directory for preprocessed NPZ files.
    metadata_path : Path, optional
        Path to clinical metadata CSV.
    apply_n4 : bool
        Whether to apply N4 bias correction.
    target_spacing : tuple
        Target voxel spacing.
    slice_method : str
        Slice selection method.
    context_slices : int
        Adjacent slices for 2.5D.
    """
    logger.info("=" * 60)
    logger.info("UCSF-PDGM PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info("Input:  %s", raw_dir)
    logger.info("Output: %s", output_dir)

    if not raw_dir.exists():
        logger.error("Raw data directory not found: %s", raw_dir)
        sys.exit(1)

    # Step 1: Load metadata
    logger.info("\n--- Step 1: Loading clinical metadata ---")
    if metadata_path is None:
        candidates = [
            raw_dir / "clinical_metadata.csv",
            raw_dir / "metadata.csv",
            raw_dir.parent / "ucsf_pdgm_metadata.csv",
        ]
        for c in candidates:
            if c.exists():
                metadata_path = c
                break

    if metadata_path is None or not metadata_path.exists():
        logger.error(
            "UCSF-PDGM metadata file not found. Provide --metadata-path. "
            "Download from: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/"
        )
        sys.exit(1)

    ucsf_metadata = load_ucsf_labels(metadata_path)
    labels = {pid: m["binary_label"] for pid, m in ucsf_metadata.items()}

    # Save labels
    labels_path = output_dir / "grade_labels.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_labels(labels, labels_path)

    # Save full metadata as JSON for dataset loader
    meta_json_path = output_dir / "patient_metadata.json"
    with open(meta_json_path, "w") as f:
        json.dump(ucsf_metadata, f, indent=2)

    # Step 2: Preprocess each patient
    logger.info("\n--- Step 2: Preprocessing patients ---")
    patient_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])

    manifest_samples = []
    all_stats = []
    t_start = time.time()

    for i, patient_dir in enumerate(patient_dirs):
        pid = patient_dir.name
        if pid not in labels:
            continue

        stats = preprocess_ucsf_patient(
            patient_dir=patient_dir,
            output_dir=output_dir,
            apply_n4=apply_n4,
            target_spacing=target_spacing,
            slice_method=slice_method,
            context_slices=context_slices,
        )

        if stats is not None:
            all_stats.append(stats)
            manifest_samples.append({
                "file": f"{pid}.npz",
                "patient_id": pid,
                "label": labels[pid],
                "slice_index": stats.selected_slice,
                "tumor_area": stats.tumor_area_on_slice,
            })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            logger.info("Progress: %d/%d patients (%.1f sec)", i + 1, len(patient_dirs), elapsed)

    # Step 3: Save manifest
    manifest = {
        "dataset": "ucsf_pdgm",
        "preprocessing": {
            "n4_bias_correction": apply_n4,
            "target_spacing": list(target_spacing),
            "z_score": True,
            "foreground_only": True,
            "slice_method": slice_method,
            "context_slices": context_slices,
        },
        "n_samples": len(manifest_samples),
        "n_hgg": sum(1 for s in manifest_samples if s["label"] == 1),
        "n_lgg": sum(1 for s in manifest_samples if s["label"] == 0),
        "samples": manifest_samples,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t_start
    logger.info("\n--- UCSF-PDGM preprocessing complete ---")
    logger.info("Patients processed: %d", len(manifest_samples))
    logger.info("HGG: %d, LGG: %d", manifest["n_hgg"], manifest["n_lgg"])
    logger.info("Time: %.1f seconds", elapsed)


def main() -> None:
    """Entry point for preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess brain tumor MRI datasets"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["brats2023", "ucsf_pdgm"],
        help="Dataset to preprocess (alternative to --config)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to Hydra YAML config file",
    )
    parser.add_argument(
        "--raw-dir", type=str, default=None,
        help="Override raw data directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--metadata-path", type=str, default=None,
        help="Path to metadata CSV (for labels)",
    )
    parser.add_argument(
        "--slice-method", type=str, default="max_tumor_area",
        choices=["max_tumor_area", "center"],
    )
    parser.add_argument(
        "--context-slices", type=int, default=0,
        help="Adjacent slices for 2.5D (0 for pure 2D)",
    )
    parser.add_argument(
        "--no-n4", action="store_true",
        help="Skip N4 bias field correction (UCSF-PDGM only)",
    )
    args = parser.parse_args()

    data_raw = PROJECT_ROOT / "data" / "raw"
    data_processed = PROJECT_ROOT / "data" / "processed"
    metadata = Path(args.metadata_path) if args.metadata_path else None

    if args.dataset == "brats2023" or (args.config and "brats" in args.config):
        raw = Path(args.raw_dir) if args.raw_dir else data_raw / "brats2023"
        out = Path(args.output_dir) if args.output_dir else data_processed / "brats2023"
        preprocess_brats(
            raw_dir=raw,
            output_dir=out,
            metadata_path=metadata,
            slice_method=args.slice_method,
            context_slices=args.context_slices,
        )

    elif args.dataset == "ucsf_pdgm" or (args.config and "ucsf" in args.config):
        raw = Path(args.raw_dir) if args.raw_dir else data_raw / "ucsf_pdgm"
        out = Path(args.output_dir) if args.output_dir else data_processed / "ucsf_pdgm"
        preprocess_ucsf(
            raw_dir=raw,
            output_dir=out,
            metadata_path=metadata,
            apply_n4=not args.no_n4,
            slice_method=args.slice_method,
            context_slices=args.context_slices,
        )

    else:
        parser.error("Specify --dataset or --config")


if __name__ == "__main__":
    main()
