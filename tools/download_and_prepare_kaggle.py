"""
Download + prepare the Kaggle Brain Tumor MRI dataset fot this project

What it does
------------
1) Downloads `masoudnickparvar/brain-tumor-mri-dataset` with kagglehub
2) Finds the source split (usually `Training/` and `Testing/`
3) Normalizes class folder names to lowercase: glioma, meningiona, pituitary, no_tumor
3) Writes the project layout:
    data/
        train/<class>/*
        val/<class>/*
        test/<class>/*

If there is no `val/` split in the source, we create one via stratified split
from train (default 10%).

Usage
-----
$ python tools/download_and_prepare_kaggle.py --project-root . \
    --val-size 0.1 --use-symlinks

Notes
-----
- Symlinks avoid duplicating data on disk; use --copy if you prefer hard copies.
- If you already have data/, the script will be careful and only populate missing pieces.
"""

from __future__ import annotations
import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# Light imports; sklearn only used for stratified split
from sklearn.model_selection import StratifiedShuffleSplit

# We rely on kagglehub (no need to set API credencials explicitly for public datasets)
import kagglehub

# -------------------------- config & CLI --------------------------------
DEF_CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".", help="Path to repository root (where data/ lives).")
    p.add_argument("--dataset", default="masoudnickparvar/brain-tumor-mri-dataset",
                   help="Kaggle dataset ref for kagglehub.")
    p.add_argument("--val-size", type=float, default=0.10, help="Fraction of TRAIN to use for validation (if missing).")
    p.add_argument("--use-symlinks", action="store_true", help="Use symlinks instead of copying files")
    p.add_argument("--copy", dest="use_symlinks", action="store_false", help="Force copy instead of symlinks.")
    return p.parse_args()

# -------------------------- fs helpers ----------------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def is_image(name: str) -> bool:
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))

def norm_class_name(name: str) -> str:
    """
    Normalize variantes like 'no_tumor', 'notumor', 'no tumor', 'Pituitary', etc.
    We map several common variantions to our canonical names.
    """
    n = name.strip().lower()
    n = re.sub(r"[\s\-]+", "_", n) # spaces/dashes -> underscore
    # common aliases
    if n in {"no_tumor", "no-tumor", "notumor", "no__tumor"}:
        return "no_tumor"
    if "pituit" in n:
        return "pituitary"
    if "meningi" in n:
        return "meningioma"
    if "glioma" in n:
        return "glioma"
    # final fallback: keep cleaned version
    return n

def safe_link_or_copy(src: Path, dst: Path, use_symlinks: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_symlinks:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            # Windows without admin often can't create symlinks; fallback to copy
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)

def collect_split(root: Path, split_name_candidates: List[str]) -> Path | None:
    """
    Find a split directory (e.g., "Training" or "Train" under root, case-insensitively.
    Return the first match or None if not found.
    """
    for cand in split_name_candidates:
        for child in root.iterdir():
            if child.is_dir() and child.name.lower() == cand.lower():
                return child
    return None

def scan_classes(root: Path) -> Dict[str, List[Path]]:
    """
    Scan class subfolders under `root` and return mapping: canonical_class_name -> [image files].
    """
    classes: Dict[str, List[Path]] = {}
    if not root or not root.exists():
        return classes
    for sub in sorted(x for x in root.iterdir() if x.is_dir()):
        cname = norm_class_name(sub.name)
        files = [p for p in sub.rglob("*") if p.is_file() and is_image(p.name)]
        if not files:
            continue
        classes.setdefault(cname, []).extend(files)
    return classes

# -------------------------- core pipeline ----------------------------------
def download_kaggle_dataset(dataset_ref: str) -> Path:
    """
    Uses kagglehub to download; return the local path to the dataset folder.
    kagglehub returns a versioned directory path.
    """
    local_path_str = kagglehub.dataset_download(dataset_ref)
    return Path(local_path_str)

def prepare_object_layout(project_root: Path) -> Tuple[Path, Path, Path]:
    data_dir = ensure_dir(project_root / "data")
    train_dir = ensure_dir(data_dir / "train")
    val_dir = ensure_dir(data_dir / "val")
    test_dir = ensure_dir(data_dir / "test")
    return train_dir, val_dir, test_dir

def write_split(src_split_dir: Path, dst_split_dir: Path, use_symlinks: bool, classes_filter: List[str] | None = None):
    """
    Mirror a split (`Training`/`Testing`) into our layout (train/ or test/),
    normalizing class folder names and linking/copying images.
    """
    class_map = scan_classes(src_split_dir)
    for cname, files in class_map.items():
        if classes_filter and cname not in classes_filter:
            continue
        for f in files:
            rel_name = f.name
            dst = dst_split_dir / cname / rel_name
            safe_link_or_copy(f, dst, use_symlinks)

def stratified_val_from_train(train_root: Path, val_root: Path, val_frac: float, use_symlinks: bool):
    """
    Create a stratified validation split by *moving/copying* a fraction of training files
    into val/. We keep train/ untouched and *populate* val/ from train/.
    """
    X: List[Path] = []
    y: List[int] = []
    classes = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
    index_for = {c: i for i, c in enumerate(classes)}

    for c in classes:
        for img in (train_root / c).glob("*"):
            if img.is_file() and is_image(img.name):
                X.append(img)
                y.append(index_for[c])

    if not X:
        print("[WARN] No training images found to make a validation split.")
        return

    X = [Path(p) for p in X]
    y = list(y)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
    for _, val_idx in splitter.split(X, y):
        for i in val_idx:
            src = X[i]
            cname = classes[y[i]]
            dst = val_root / cname / src.name
            # If we symlink, we still keep the file in train/ and *link* into val/ to avoid duplication.
            # If we copy, we copy the file (train remains as is).
            safe_link_or_copy(src, dst, use_symlinks)
    print(f"[INFO] Created validation split ~{len(val_idx)} samples across {len(classes)} classes.")

# -------------------------- main ----------------------------------

def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    print(f"[INFO] Project root: {project_root}")

    # 1) Download
    dl_dir = download_kaggle_dataset(args.dataset)
    print(f"[INFO] Downloaded to: {dl_dir}")

    # The Kaggle dataset root sometimes has an extra top-level version folder
    # We try to find 'Training'/'Testing' inside dl_dir or one level below.
    search_root = dl_dir
    # Heuristic: if dl_dir contains a single directory with train/test inside, descend into it.
    children_dirs = [d for d in search_root.iterdir() if d.is_dir()]
    if len(children_dirs) == 1 and (children_dirs[0] / "Training").exists() or (children_dirs[0] / "Testing").exists():
        search_root = children_dirs[0]

    # 2) Detect source splits (case-insensitive)
    # Some versions use "Training" / "Testing", others "train" / "test".
    training_dir = collect_split(search_root, ["Training", "Train", "training", "train"])
    testing_dir = collect_split(search_root, ["Testing", "Test", "testing", "test"])

    if not training_dir and not testing_dir:
        # Fallback: maybe everything is directly under a single folder with classes
        print(f"[WARN] Could not find Training/Testing folders. Will treat root as 'train' and create val/test later.")
        training_dir = search_root

    # 3) Prepare project layout
    train_dir, val_dir, test_dir = prepare_object_layout(project_root)

    # 4) Write train/test splits from source (if found)
    if training_dir:
        print(f"[INFO] Preparing TRAIN from: {training_dir}")
        write_split(training_dir, train_dir, args.use_symlinks, classes_filter=DEF_CLASSES)
    if testing_dir:
        print(f"[INFO] Preparing TEST from: {testing_dir}")
        write_split(testing_dir, test_dir, args.use_symlinks, classes_filter=DEF_CLASSES)

    # 5) If there is no dedicated validation split in the source, make one
    #    (some Kaggle versions don't ship a val/ folder).
    #    We *never* remove from train/ - when using symlinks, val/ points to the same files.
    if not any(val_dir.glob("*/*")): # check if val is empty
        print(f"[INFO] Creating stratified validation split from TRAIN.")
        stratified_val_from_train(training_dir, val_dir, args.val_size, args.use_symlinks)

    # 6) Final report
    def count_image(root: Path) -> int:
        return sum(1 for p in root.rglob("*") if p.is_file() and is_image(p.name))

    print("\n[SUMMARY]")
    print(f"Train:  {count_image(training_dir)} images at {training_dir}")
    print(f"Val:    {count_image(val_dir)} images at {val_dir}")
    print(f"Test:   {count_image(test_dir)} images at {test_dir}")
    print("\n[OK] Dataset is ready for the project layout.")

if __name__ == "__main__":
    main()