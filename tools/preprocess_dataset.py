"""
tools/preprocess_dataset.py - Unified Preprocessing Wrapper
-----------------------------------------------------------
Intelligently routes preprocessing based on configuration.
Supports both legacy and medical-grade methods.
"""

import os
import sys
import argparse
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
import json

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_legacy(input_dir: str, output_dir: str):
    """
    Legacy preprocessing method (simple skull stripping).
    Kept for backward compatibility and speed.
    """
    from src.imgproc import crop_brain_contour

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in input_path.rglob(
        "*") if p.suffix.lower() in valid_exts]

    print(f"[INFO] Legacy preprocessing: {len(files)} images")

    success = 0
    for src_file in tqdm(files, desc="Legacy preprocessing"):
        rel_path = src_file.relative_to(input_path)
        dest_file = output_path / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            cropped = crop_brain_contour(str(src_file))
            if cropped is not None:
                cv2.imwrite(str(dest_file), cropped)
                success += 1
        except Exception as e:
            print(f"\n[WARNING] Failed on {src_file}: {e}")

    print(f"[OK] Legacy preprocessing completed: {
          success}/{len(files)} images")
    return success


def preprocess_medical(input_dir: str, output_dir: str, config: dict):
    """
    Medical-grade preprocessing pipeline.

    Args:
        input_dir: Source directory
        output_dir: Destination directory
        config: Preprocessing configuration from config.yaml
    """
    from tools.mri_specific_preprocessing import MRIPreprocessor

    medical_config = config.get("preprocessing", {}).get("medical", {})

    # Initialize preprocessor with config
    preprocessor = MRIPreprocessor(
        target_size=(config["data"]["image_size"],
                     config["data"]["image_size"]),
        apply_n4_bias_correction=medical_config.get(
            "apply_n4_bias_correction", True),
        apply_rician_denoising=medical_config.get(
            "apply_rician_denoising", True),
        skull_strip_method=medical_config.get("skull_strip_method", "bet"),
        intensity_normalization=medical_config.get(
            "intensity_normalization", "nyul"),
        enhance_method=medical_config.get("enhance_method", "clahe"),
    )

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in input_path.rglob(
        "*") if p.suffix.lower() in valid_exts]

    print(f"[INFO] Medical-grade preprocessing: {len(files)} images")
    print(f"[CONFIG] Bias correction: {
          medical_config.get('apply_n4_bias_correction')}")
    print(f"[CONFIG] Skull strip: {medical_config.get('skull_strip_method')}")
    print(f"[CONFIG] Normalization: {
          medical_config.get('intensity_normalization')}")
    print(f"[CONFIG] Enhancement: {medical_config.get('enhance_method')}")

    stats = {
        "total": len(files),
        "processed": 0,
        "failed": 0,
        "quality_filtered": 0,
        "metadata": [],
    }

    save_metadata = medical_config.get("save_metadata", False)
    quality_filter = medical_config.get("quality_filter", False)

    for src_file in tqdm(files, desc="Medical preprocessing"):
        try:
            img = cv2.imread(str(src_file))
            if img is None:
                stats["failed"] += 1
                continue

            # Process with medical pipeline
            processed, metadata = preprocessor.process(
                img, return_metadata=True)

            # Quality filtering (optional)
            if quality_filter:
                quality_metrics = metadata.get("quality_metrics", {})
                snr = quality_metrics.get("snr_estimate", 0)
                contrast = quality_metrics.get("contrast", 0)

                # Simple quality thresholds
                if snr < 2.0 or contrast < 50:
                    stats["quality_filtered"] += 1
                    continue

            # Save processed image
            rel_path = src_file.relative_to(input_path)
            dest_file = output_path / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(dest_file), processed)
            stats["processed"] += 1

            # Save metadata if enabled
            if save_metadata:
                meta_file = dest_file.with_suffix(".json")
                metadata_serializable = {
                    "source_file": str(src_file),
                    "stages": metadata.get("stages", {}),
                    "quality_metrics": {
                        k: float(v)
                        for k, v in metadata.get("quality_metrics", {}).items()
                    },
                }
                with open(meta_file, "w") as f:
                    json.dump(metadata_serializable, f, indent=2)

        except Exception as e:
            print(f"\n[ERROR] Failed on {src_file}: {e}")
            stats["failed"] += 1

    # Save summary
    summary_file = output_path / "preprocessing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("MEDICAL PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files:        {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed:             {stats['failed']}")
    print(f"Quality filtered:   {stats['quality_filtered']}")
    print(f"Success rate:       {stats['processed'] / stats['total']:.1%}")
    print(f"\nSummary saved to: {summary_file}")

    return stats["processed"]


def main():
    parser = argparse.ArgumentParser(
        description="Unified preprocessing wrapper - supports legacy and medical-grade methods"
    )
    parser.add_argument(
        "--input_dir", required=True, help="Source directory (e.g., data/train)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination directory (e.g., data/train_medical)",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "legacy", "medical"],
        default="auto",
        help="Preprocessing mode (auto reads from config)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force reprocessing even if output exists"
    )

    args = parser.parse_args()

    # Check if output already exists
    if Path(args.output_dir).exists() and not args.force:
        file_count = sum(1 for _ in Path(args.output_dir).rglob("*.jpg")) + sum(
            1 for _ in Path(args.output_dir).rglob("*.png")
        )

        if file_count > 0:
            print(f"[INFO] Output directory already contains {
                  file_count} images.")
            print(f"[INFO] Skipping preprocessing. Use --force to reprocess.")
            return

    # Load config
    config = load_config(args.config)

    # Determine preprocessing mode
    if args.mode == "auto":
        mode = config.get("preprocessing", {}).get("mode", "medical")
    else:
        mode = args.mode

    print(f"\n{'=' * 60}")
    print(f"PREPROCESSING MODE: {mode.upper()}")
    print(f"{'=' * 60}\n")

    # Execute preprocessing
    if mode == "legacy":
        preprocess_legacy(args.input_dir, args.output_dir)

    elif mode == "medical":
        preprocess_medical(args.input_dir, args.output_dir, config)

    else:
        print(f"[ERROR] Unknown mode: {mode}")
        sys.exit(1)

    print("\n[OK] Preprocessing completed successfully!")


if __name__ == "__main__":
    main()
