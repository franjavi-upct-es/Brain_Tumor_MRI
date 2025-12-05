"""
tools/download_data.py - Master Data Script
-------------------------------------------
Downloads, normalizes, merges, and structures ALL datasets for the project.

Managed Datasets:
1. MasoudNickparvar (Base): Multiclass classification.
2. Pradeep2665 (Extra): Multiclass classification (merged with Base).
3. Navoneel (External): Binary validation (Tumor vs No Tumor).
4. MateuszBuda (LGG): Segmentation (Masks).

Resulting Structure:
data/
  ├── train/              (Masoud + Pradeep merged)
  ├── val/                (Masoud + Pradeep merged)
  ├── test/               (Masoud + Pradeep merged)
  ├── external_navoneel/  (Original binary dataset)
  └── external_lgg/       (Segmentation dataset)
"""

import os
import shutil
import argparse
import kagglehub
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

# --- Configuration ---
DATASETS = {
    "masoud": "masoudnickparvar/brain-tumor-mri-dataset",
    "pradeep": "pradeep2665/brain-mri",
    "navoneel": "navoneel/brain-mri-images-for-brain-tumor-detection",
    "lgg": "mateuszbuda/lgg-mri-segmentation",
}

DEF_CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]

# --- Utilities ---


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_image(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def normalize_class_name(name: str) -> str:
    """Maps varied folder names to the 4 canonical classes."""
    n = name.lower().strip().replace(" ", "_").replace("-", "_")

    # Pradeep uses 'glioma_tumor', Masoud uses 'glioma', etc.
    if "no_tumor" in n or "notumor" in n:
        return "no_tumor"
    if "glioma" in n:
        return "glioma"
    if "meningioma" in n:
        return "meningioma"
    if "pituitary" in n:
        return "pituitary"

    return None  # Unkown or irrelevant folder


def safe_copy(src: Path, dst: Path):
    """Copies file, creating parent directories if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def merge_dataset(source_root: Path, target_base: Path, dataset_prefix: str):
    """
    Traverses a downloaded dataset, identifies split (train/test/val) and class,
    and copies files to the merged target with a prefix to avoid collisions.
    """
    print(f"--> Processing {dataset_prefix} from: {source_root}")

    # Mapping source folder names to standard target splits
    split_map = {
        "training": "train",
        "train": "train",
        "testing": "test",
        "test": "test",
        "validation": "val",
        "val": "val",
    }

    count = 0
    # Recursive walk
    for root, dirs, files in os.walk(source_root):
        root_path = Path(root)

        # Try to guess the split (train/val/test) based on path parts
        parts = root_path.parts
        target_split = None

        # Look for keywords in the path (e.g., .../Training/glioma)
        for part in parts:
            if part.lower() in split_map:
                target_split = split_map[part.lower()]
                break

        # If not explicit split found, skip (or assume train, but safety first)
        if not target_split:
            continue

        # Try to guess the class (glioma/meningioma...)
        class_name = normalize_class_name(root_path.name)

        # If current folder is not a valid class, skip
        if class_name not in DEF_CLASSES:
            continue

        # Copy images
        for f in files:
            if is_image(f):
                src_file = root_path / f
                # Reanme: pradeep_img001.jpg to avoid overwritting masoud_img001.jpg
                new_name = f"{dataset_prefix}_{f}"
                dst_file = target_base / target_split / class_name / new_name

                safe_copy(src_file, dst_file)
                count += 1

    print(
        f"    + {count} images merged into data/{target_split if count > 0 else '...'}"
    )


# --- Specific Functions ---


def process_navoneel(target_root: Path):
    """Handles the external binary dataset (yes/no)"""
    print("[INFO] Processing Navoneel (External/Binary)...")
    path = kagglehub.dataset_download(DATASETS["navoneel"])
    source = Path(path)

    # Sometimes nested
    if (source / "brain_tumor_dataset").exists():
        source = source / "brain_tumor_dataset"

    dest = target_root / "external_navoneel"
    if dest.exists():
        shutil.rmtree(dest)

    shutil.copytree(source, dest, dirs_exist_ok=True)
    print(f"    + Saved to {dest}")


def process_logg(target_root: Path):
    """Handles the segmentation dataset"""
    print("[INFO] Processing LGG Segmentation (Mateusz)...")
    path = kagglehub.dataset_download(DATASETS["lgg"])
    dest = target_root / "external_lgg"

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(path, dest, dirs_exist_ok=True)
    print(f"    + Saved to {dest}")


def check_and_fix_splits(data_root: Path):
    """
    If 'val' is empty after merging, create a stratified split from 'train'.
    """
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    # Count images in val
    val_count = sum(1 for _ in val_dir.rglob("*") if _.is_file())

    if val_count < 100:  # Arbitrary threshold
        print(f"[WARN] Validation split is too small or empty ({val_count} imgs).")
        print("    Creating stratified 15% split from data/train...")

        # Collect train files
        files = []
        labels = []
        classes = [d.name for d in train_dir.iterdir() if d.is_dir()]

        for cls in classes:
            for img in (train_dir / cls).glob("*"):
                if is_image(img.name):
                    files.append(img)
                    labels.append(cls)

        if not files:
            print("[ERROR] No images in train to split.")
            return

        # Split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        for train_idx, val_idx in sss.split(files, labels):
            for i in val_idx:
                src = files[i]
                cls = labels[i]
                dst = val_dir / cls / src.name

                # Move files from train to val
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

                val_count += 1

        print(f"    [OK] Split generated. Val now has {val_count} images.")
    else:
        print(f"[INFO] Validation ready with {val_count} images.")


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Download and prepare all datasets.")
    parser.add_argument("--project_root", default=".", help="Project root directory")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    data_dir = ensure_dir(root / "data")

    print("==============================================")
    print("   BRAIN TUMOR MRI - UNIFIED DATA MANAGER")
    print("==============================================")

    # 1. Initial cleanup
    print("[INFO] Cleaning previous train/val/test folders for clean build...")
    for split in ["train", "val", "test"]:
        p = data_dir / split
        if p.exists():
            shutil.rmtree(p)

    # 2. Download and Merge Classification Datasets
    print("[INFO] Phase 1: Classification Datasets (Merge)")

    # Masoud (Base)
    path_masoud = kagglehub.dataset_download(DATASETS["masoud"])
    merge_dataset(Path(path_masoud), data_dir, dataset_prefix="masoud")

    # Pradeep (Extra)
    path_pradeep = kagglehub.dataset_download(DATASETS["pradeep"])
    merge_dataset(Path(path_pradeep), data_dir, dataset_prefix="pradeep")

    # 3. Verify Splits
    check_and_fix_splits(data_dir)

    # 4. External/Specific Datasets
    print("[INFO] Phase 2: Specialized Datasets")
    process_navoneel(data_dir)
    process_logg(data_dir)

    # Final Summary
    print("==============================================")
    print("   DATA SUMMARY")
    print("==============================================")
    for split in ["train", "val", "test"]:
        n = sum(1 for _ in (data_dir / split).rglob("*") if is_image(_.name))
        print(f"    - {split.upper()}: {n} images")

    n_nav = sum(
        1 for _ in (data_dir / "external_navoneel").rglob("*") if is_image(_.name)
    )
    print(f"    - EXTERNAL (Navoneel): {n_nav} images")

    n_lgg = sum(1 for _ in (data_dir / "external_lgg").rglob("*") if is_image(_.name))
    print(f"    - SEGMENTATION (LGG): {n_lgg} images")


if __name__ == "__main__":
    main()
