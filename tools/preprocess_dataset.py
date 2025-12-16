"""
tools/preprocess_dataset.py - Unified Preprocessing Wrapper (Multiprocessing Optimized)
-------------------------------------------------------------------------------------
Intelligently routes preprocessing based on configuration.
Uses ProcessPoolExecutor to utilize all CPU cores effectively.
"""

import sys
import argparse
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
import json
import concurrent.futures
import multiprocessing


def load_config(config_path: str = "../configs/config.yaml") -> dict:
    """Load configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _process_single_legacy(args):
    """Helper function for processing a single image in legacy mode."""
    src_file, input_path, output_path = args
    from src.imgproc import crop_brain_contour

    try:
        rel_path = src_file.relative_to(input_path)
        dest_file = output_path / rel_path

        # Check if already exists to skip efficiently if needed
        # if dest_file.exists(): return True

        dest_file.parent.mkdir(parents=True, exist_ok=True)

        cropped = crop_brain_contour(str(src_file))
        if cropped is not None:
            cv2.imwrite(str(dest_file), cropped)
            return True
        return False
    except Exception as e:
        # Return error as string to print later (avoid print collisions in threads)
        return str(e)


def preprocess_legacy(input_dir: str, output_dir: str):
    """
    Legacy preprocessing method with Multiprocessing.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in input_path.rglob("*") if p.suffix.lower() in valid_exts]

    print(f"[INFO] Legacy preprocessing: {len(files)} images")

    # Determine CPUs
    num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave 1 core free
    print(f"[INFO] Using {num_workers} parallel workers.")

    success = 0
    failed_log = []

    # Prepare arguments for map
    # We pass paths as strings or Path objects.
    # Note: Objects passed to workers must be picklable.
    tasks = [(f, input_path, output_path) for f in files]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(_process_single_legacy, tasks),
                total=len(tasks),
                desc="Legacy (Parallel)",
            )
        )

    for res in results:
        if res is True:
            success += 1
        elif res is False:
            pass  # Just didn't detect contour/crop
        else:
            # It's an error string
            failed_log.append(res)

    if failed_log:
        print(f"\n[WARNING] {len(failed_log)} errors occurred. First few:")
        for err in failed_log[:5]:
            print(f" - {err}")

    print(f"[OK] Legacy preprocessing completed: {success}/{len(files)} images")
    return success


def _process_single_medical(args):
    """Helper function for medical preprocessing worker."""
    src_file, input_path, output_path, config_params = args

    # Re-import inside process to avoid pickling issues with cv2 objects if any
    from tools.mri_specific_preprocessing import MRIPreprocessor
    import cv2
    import json

    # Initialize preprocessor inside the worker process
    # This creates a new instance per process, which is safe
    preprocessor = MRIPreprocessor(**config_params["init_args"])

    save_metadata = config_params["save_metadata"]
    quality_filter = config_params["quality_filter"]
    q_thresholds = config_params.get("quality_filter_thresholds", {})
    min_snr = q_thresholds.get("min_snr", 2.0)
    min_contrast = q_thresholds.get("min_contrast", 50.0)

    try:
        img = cv2.imread(str(src_file))
        if img is None:
            return "failed_read"

        # Process
        processed, metadata = preprocessor.process(img, return_metadata=True)

        # Quality filtering
        if quality_filter:
            q_metrics = metadata.get("quality_metrics", {})
            snr = q_metrics.get("snr_estimate", 0)
            contrast = q_metrics.get("contrast", 0)
            if snr < min_snr or contrast < min_contrast:
                return "quality_filtered"

        # Save
        rel_path = src_file.relative_to(input_path)
        dest_file = output_path / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(dest_file), processed)

        # Metadata
        if save_metadata:
            meta_file = dest_file.with_suffix(".json")
            meta_serializable = {
                "source_file": str(src_file),
                "stages": metadata.get("stages", {}),
                "quality_metrics": {
                    k: float(v) for k, v in metadata.get("quality_metrics", {}).items()
                },
            }
            with open(meta_file, "w") as f:
                json.dump(meta_serializable, f, indent=2)

        return "success"

    except Exception as e:
        return f"Error: {str(e)}"


def preprocess_medical(input_dir: str, output_dir: str, config: dict):
    """
    Medical-grade preprocessing pipeline (Multiprocessed).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in input_path.rglob("*") if p.suffix.lower() in valid_exts]

    medical_config = config.get("preprocessing", {}).get("medical", {})

    print(f"[INFO] Medical-grade preprocessing: {len(files)} images")
    print(
        f"[CONFIG] Bias correction: {
            medical_config.get('apply_n4_bias_correction', True)
        }"
    )
    print(f"[CONFIG] Skull strip: {medical_config.get('skull_strip_method', 'bet')}")
    print(
        f"[CONFIG] Normalization: {
            medical_config.get('intensity_normalization', 'nyul')
        }"
    )

    # Prepare configuration dictionary to pass to workers
    # We don't pass the class instance, just the args to init it
    config_params = {
        "init_args": {
            "target_size": (config["data"]["image_size"], config["data"]["image_size"]),
            "apply_n4_bias_correction": medical_config.get(
                "apply_n4_bias_correction", True
            ),
            "apply_rician_denoising": medical_config.get(
                "apply_rician_denoising", True
            ),
            "skull_strip_method": medical_config.get("skull_strip_method", "bet"),
            "intensity_normalization": medical_config.get(
                "intensity_normalization", "nyul"
            ),
            "enhance_method": medical_config.get("enhance_method", "clahe"),
        },
        "save_metadata": medical_config.get("save_metadata", False),
        "quality_filter": medical_config.get("quality_filter", False),
        "quality_filter_thresholds": medical_config.get(
            "quality_filter_thresholds", {}
        ),
    }

    stats = {"total": len(files), "processed": 0, "failed": 0, "quality_filtered": 0}

    # Leave 2 cores free for system
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"[INFO] Using {num_workers} parallel workers for heavy processing.")

    tasks = [(f, input_path, output_path, config_params) for f in files]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(_process_single_medical, tasks),
                total=len(tasks),
                desc="Medical (Parallel)",
            )
        )

    for res in results:
        if res == "success":
            stats["processed"] += 1
        elif res == "quality_filtered":
            stats["quality_filtered"] += 1
        elif res == "failed_read":
            stats["failed"] += 1
        elif res.startswith("Error"):
            stats["failed"] += 1
            # print(res) # Uncomment to see individual errors

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
    print(f"Success rate:       {stats['processed'] / max(1, stats['total']):.1%}")
    print(f"\nSummary saved to: {summary_file}")

    return stats["processed"]


def main():
    parser = argparse.ArgumentParser(
        description="Unified preprocessing wrapper - supports legacy and medical-grade methods"
    )
    parser.add_argument("--input_dir", required=True, help="Source directory")
    parser.add_argument("--output_dir", required=True, help="Destination directory")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config"
    )
    parser.add_argument("--mode", choices=["auto", "legacy", "medical"], default="auto")
    parser.add_argument("--force", action="store_true", help="Force reprocessing")

    args = parser.parse_args()

    # Check if output already exists
    if Path(args.output_dir).exists() and not args.force:
        file_count = sum(1 for _ in Path(args.output_dir).rglob("*.jpg")) + sum(
            1 for _ in Path(args.output_dir).rglob("*.png")
        )

        if file_count > 0:
            print(f"[INFO] Output directory already contains {file_count} images.")
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
    print(f"PREPROCESSING MODE: {mode.upper()} (PARALLEL)")
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
    multiprocessing.set_start_method("spawn", force=True)
    main()
