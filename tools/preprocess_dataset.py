import os
import sys
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm

# Add root to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.imgproc import crop_brain_contour


def main():
    parser = argparse.ArgumentParser(
        description="Applies contour cropping (skull stripping) to an entire dataset."
    )
    parser.add_argument(
        "--input_dir", required=True, help="Source directory (e.g., data/train)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination directory (e.g., data/train_cropped)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    # Supportes image extensions
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    files = [p for p in input_path.rglob("*") if p.suffix.lower() in valid_exts]
    print(f"[INFO] Found {len(files)} images in {input_path}")
    print(f"[INFO] Processing and saving to {output_path}...")

    for src_file in tqdm(files):
        # Calculate relative path to maintain folder structure (e.g., glioma/img01.jpg)
        rel_path = src_file.relative_to(input_path)
        dest_file = output_path / rel_path

        # Create destination folder if it doesn't exist
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        # Process
        try:
            cropped = crop_brain_contour(str(src_file))
            if cropped is not None:
                cv2.imwrite(str(dest_file), cropped)
            else:
                # If fails, skip or copy original
                pass
        except Exception as e:
            print(f"[ERROR] Failed on {src_file}: {e}")

    print("[OK] Preprocessing completed.")


if __name__ == "__main__":
    main()
