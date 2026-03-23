# notebooks/01_data_exploration.py — Data exploration and visualization
"""
Data Exploration for Brain Tumor Classification v2
===================================================

This script explores the BraTS 2023 and UCSF-PDGM datasets to understand
their structure, class distributions, and imaging characteristics before
training any models.

Sections:
  1. Dataset structure and organization
  2. Class distribution analysis (HGG vs LGG)
  3. Volume dimensions and modality verification
  4. Intensity distribution analysis across modalities
  5. Tumor region statistics from segmentations
  6. Representative slice visualization
  7. Cross-dataset comparison (BraTS vs UCSF-PDGM)

Run as script:
    python notebooks/01_data_exploration.py --data-dir data/raw/brats2023

Convert to notebook:
    jupytext --to notebook notebooks/01_data_exploration.py

Reference: Phase 2 (Weeks 3-4) of brain_tumor_v2_redesign.md
"""

# %% [markdown]
# # Brain Tumor MRI Classification v2 — Data Exploration
#
# Before training any models, we explore the BraTS 2023 and UCSF-PDGM datasets
# to verify data integrity, understand class distributions, and characterize
# the imaging data.

# %%
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for script mode
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# %% [markdown]
# ## 1. Dataset Structure
#
# BraTS 2023 organizes data per patient, preventing data leakage by design.
# Each patient directory contains 4 MRI modalities + segmentation in NIfTI format.


# %%
def explore_dataset_structure(data_dir: Path) -> dict:
    """Explore the top-level structure of a dataset directory.

    Parameters
    ----------
    data_dir : Path
        Root directory of the dataset.

    Returns
    -------
    dict
        Summary statistics about the dataset structure.
    """
    if not data_dir.exists():
        print(f"[WARNING] Data directory not found: {data_dir}")
        print(
            "  → This script requires downloaded data to produce visualizations."
        )
        print("  → Run: python scripts/download_data.py --dataset brats2023")
        return {"error": "directory_not_found"}

    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    n_patients = len(patient_dirs)

    print(f"Dataset root: {data_dir}")
    print(f"Total patient directories: {n_patients}")

    if n_patients == 0:
        return {"n_patients": 0}

    # Examine first patient as reference
    sample = patient_dirs[0]
    files = sorted(sample.glob("*.nii.gz"))
    print(f"\nSample patient: {sample.name}")
    print(f"  Files ({len(files)}):")
    for f in files:
        print(f"    {f.name}")

    # Count files per patient
    files_per_patient = []
    for pd in patient_dirs:
        n_files = len(list(pd.glob("*.nii.gz")))
        files_per_patient.append(n_files)

    print(
        f"\nFiles per patient: min={min(files_per_patient)}, "
        f"max={max(files_per_patient)}, "
        f"mode={Counter(files_per_patient).most_common(1)[0][0]}"
    )

    return {
        "n_patients": n_patients,
        "sample_patient": sample.name,
        "files_per_patient": files_per_patient,
    }


# %% [markdown]
# ## 2. Class Distribution Analysis


# %%
def analyze_class_distribution(
    labels: dict[str, int],
    dataset_name: str = "BraTS 2023",
    output_dir: Path | None = None,
) -> None:
    """Analyze and visualize the class distribution.

    Parameters
    ----------
    labels : dict[str, int]
        Mapping from patient_id to binary label (0=LGG, 1=HGG).
    dataset_name : str
        Name for plot titles.
    output_dir : Path, optional
        Directory to save figures.
    """
    label_values = list(labels.values())
    n_total = len(label_values)
    n_hgg = sum(1 for v in label_values if v == 1)
    n_lgg = n_total - n_hgg

    print(f"\n{'=' * 50}")
    print(f"Class Distribution — {dataset_name}")
    print(f"{'=' * 50}")
    print(f"Total patients: {n_total}")
    print(f"  HGG (Grade IV):    {n_hgg:4d} ({100 * n_hgg / n_total:.1f}%)")
    print(f"  LGG (Grade II-III): {n_lgg:4d} ({100 * n_lgg / n_total:.1f}%)")
    print(
        f"  Imbalance ratio:   {max(n_hgg, n_lgg) / max(min(n_hgg, n_lgg), 1):.1f}:1"
    )

    # Create bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    classes = ["LGG\n(Grade II-III)", "HGG\n(Grade IV)"]
    counts = [n_lgg, n_hgg]
    colors = ["#4CAF50", "#F44336"]

    bars = ax.bar(
        classes, counts, color=colors, edgecolor="black", linewidth=0.5
    )
    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{count}\n({100 * count / n_total:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Number of Patients", fontsize=12)
    ax.set_title(f"Glioma Grade Distribution — {dataset_name}", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir
            / f"class_distribution_{dataset_name.lower().replace(' ', '_')}.png",
            dpi=300,
        )
        print(f"  Saved figure to {output_dir}")
    plt.close(fig)


# %% [markdown]
# ## 3. Volume Dimensions and Modality Verification


# %%
def verify_volumes(data_dir: Path, max_patients: int = 50) -> dict:
    """Verify volume dimensions and modality completeness.

    Parameters
    ----------
    data_dir : Path
        Root dataset directory.
    max_patients : int
        Maximum patients to check (for speed on large datasets).

    Returns
    -------
    dict
        Verification summary.
    """
    try:
        import nibabel as nib
    except ImportError:
        print("[WARNING] nibabel not installed. Skipping volume verification.")
        return {}

    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[
        :max_patients
    ]
    shapes = []
    spacings = []
    orientations = []

    for pd in patient_dirs:
        nifti_files = sorted(pd.glob("*.nii.gz"))
        for nf in nifti_files:
            if "seg" in nf.name.lower():
                continue
            try:
                img = nib.load(str(nf))
                shapes.append(img.shape[:3])
                spacings.append(tuple(np.round(img.header.get_zooms()[:3], 2)))
                orientations.append(nib.aff2axcodes(img.affine))
                break  # Only check first modality per patient
            except Exception as e:
                print(f"  Error loading {nf.name}: {e}")

    if not shapes:
        return {}

    shape_counts = Counter(shapes)
    spacing_counts = Counter(spacings)

    print(f"\n{'=' * 50}")
    print("Volume Dimensions Verification")
    print(f"{'=' * 50}")
    print(f"Checked: {len(shapes)} patients")
    print(f"Unique shapes: {dict(shape_counts)}")
    print(f"Unique spacings: {dict(spacing_counts)}")
    print(f"Orientations: {Counter(orientations).most_common(3)}")

    return {
        "shapes": dict(shape_counts),
        "spacings": dict(spacing_counts),
    }


# %% [markdown]
# ## 4. Intensity Distribution Analysis


# %%
def analyze_intensity_distributions(
    data_dir: Path,
    max_patients: int = 30,
    output_dir: Path | None = None,
) -> None:
    """Analyze intensity distributions across modalities.

    Parameters
    ----------
    data_dir : Path
        Root dataset directory.
    max_patients : int
        Maximum patients to sample.
    output_dir : Path, optional
        Directory to save figures.
    """
    try:
        import nibabel as nib
    except ImportError:
        print("[WARNING] nibabel not installed. Skipping intensity analysis.")
        return

    from src.data.brats_dataset import MODALITY_SUFFIXES

    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[
        :max_patients
    ]

    modality_intensities = {mod: [] for mod in MODALITY_SUFFIXES}

    for pd in patient_dirs:
        pid = pd.name
        for mod, suffix in MODALITY_SUFFIXES.items():
            fpath = pd / f"{pid}{suffix}"
            if not fpath.exists():
                continue
            try:
                vol = nib.load(str(fpath)).get_fdata().astype(np.float32)
                foreground = vol[vol > 0]
                if len(foreground) > 0:
                    # Sample for efficiency
                    sample = np.random.choice(
                        foreground,
                        size=min(10000, len(foreground)),
                        replace=False,
                    )
                    modality_intensities[mod].append(sample)
            except Exception:
                pass

    # Plot intensity distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    mod_names = {
        "t1": "T1",
        "t1ce": "T1ce (Gadolinium)",
        "t2": "T2",
        "flair": "FLAIR",
    }

    for ax, (mod, mod_name) in zip(axes.flat, mod_names.items(), strict=True):
        if modality_intensities[mod]:
            all_values = np.concatenate(modality_intensities[mod])
            ax.hist(
                all_values, bins=100, density=True, alpha=0.7, color="#1976D2"
            )
            ax.set_title(mod_name, fontsize=12)
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Density")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Raw Intensity Distributions per MRI Modality", fontsize=14, y=1.01
    )
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / "intensity_distributions.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("  Intensity distribution analysis complete.")


# %% [markdown]
# ## 5. Tumor Region Statistics


# %%
def analyze_tumor_statistics(
    data_dir: Path,
    max_patients: int = 100,
    output_dir: Path | None = None,
) -> None:
    """Analyze tumor size and location statistics from segmentations.

    Parameters
    ----------
    data_dir : Path
        Root dataset directory with segmentation files.
    max_patients : int
        Maximum patients to analyze.
    output_dir : Path, optional
        Directory to save figures.
    """
    try:
        import nibabel as nib
    except ImportError:
        print("[WARNING] nibabel not installed. Skipping tumor statistics.")
        return

    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[
        :max_patients
    ]

    tumor_volumes = []  # Total tumor voxels per patient
    enhancing_volumes = []  # Enhancing tumor (label 3) voxels
    max_slice_areas = []  # Max tumor area on any single slice

    for pd in patient_dirs:
        seg_files = list(pd.glob("*seg*"))
        if not seg_files:
            continue

        try:
            seg = nib.load(str(seg_files[0])).get_fdata().astype(np.int32)
            total_tumor = int(np.sum(seg > 0))
            enhancing = int(np.sum(seg == 3))
            area_per_slice = np.sum(seg > 0, axis=(0, 1))
            max_area = int(area_per_slice.max())

            tumor_volumes.append(total_tumor)
            enhancing_volumes.append(enhancing)
            max_slice_areas.append(max_area)
        except Exception:
            pass

    if not tumor_volumes:
        return

    print(f"\n{'=' * 50}")
    print("Tumor Region Statistics")
    print(f"{'=' * 50}")
    print(f"Patients analyzed: {len(tumor_volumes)}")
    print("Total tumor volume (voxels):")
    print(f"  Mean: {np.mean(tumor_volumes):.0f}")
    print(f"  Median: {np.median(tumor_volumes):.0f}")
    print(f"  Range: [{min(tumor_volumes)}, {max(tumor_volumes)}]")
    print("Enhancing tumor volume (voxels):")
    print(f"  Mean: {np.mean(enhancing_volumes):.0f}")
    print(
        f"  Patients with enhancement: {sum(1 for v in enhancing_volumes if v > 0)}/{len(enhancing_volumes)}"
    )
    print("Max tumor area on single slice:")
    print(f"  Mean: {np.mean(max_slice_areas):.0f}")
    print(f"  Range: [{min(max_slice_areas)}, {max(max_slice_areas)}]")

    # Plot tumor volume distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(
        tumor_volumes,
        bins=50,
        color="#FF9800",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[0].set_title("Total Tumor Volume")
    axes[0].set_xlabel("Voxels")
    axes[0].set_ylabel("Patients")

    axes[1].hist(
        enhancing_volumes,
        bins=50,
        color="#E91E63",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[1].set_title("Enhancing Tumor Volume")
    axes[1].set_xlabel("Voxels")

    axes[2].hist(
        max_slice_areas,
        bins=50,
        color="#9C27B0",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[2].set_title("Max Tumor Area (Single Slice)")
    axes[2].set_xlabel("Pixels")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Tumor Region Statistics from BraTS Segmentations", fontsize=13
    )
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / "tumor_statistics.png", dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


# %% [markdown]
# ## 6. Representative Slice Visualization


# %%
def visualize_representative_slices(
    data_dir: Path,
    n_patients: int = 4,
    output_dir: Path | None = None,
) -> None:
    """Visualize the selected representative slices with all modalities.

    Parameters
    ----------
    data_dir : Path
        Root dataset directory.
    n_patients : int
        Number of patients to display.
    output_dir : Path, optional
        Directory to save figures.
    """
    try:
        import nibabel as nib
    except ImportError:
        print("[WARNING] nibabel not installed. Skipping slice visualization.")
        return

    from src.data.brats_dataset import MODALITY_SUFFIXES, SEG_SUFFIX

    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[
        :n_patients
    ]
    modality_names = {"t1": "T1", "t1ce": "T1ce", "t2": "T2", "flair": "FLAIR"}

    fig, axes = plt.subplots(
        n_patients,
        5,
        figsize=(20, 4 * n_patients),
        gridspec_kw={"wspace": 0.05, "hspace": 0.2},
    )
    if n_patients == 1:
        axes = axes[np.newaxis, :]

    for row, pd in enumerate(patient_dirs):
        pid = pd.name

        # Load segmentation to find max tumor slice
        seg_path = pd / f"{pid}{SEG_SUFFIX}"
        if not seg_path.exists():
            continue
        seg = nib.load(str(seg_path)).get_fdata().astype(np.int32)
        area_per_slice = np.sum(seg > 0, axis=(0, 1))
        best_slice = int(np.argmax(area_per_slice))

        # Plot each modality
        for col, (mod, mod_name) in enumerate(modality_names.items()):
            suffix = MODALITY_SUFFIXES[mod]
            fpath = pd / f"{pid}{suffix}"
            if not fpath.exists():
                continue
            vol = nib.load(str(fpath)).get_fdata().astype(np.float32)
            axes[row, col].imshow(
                vol[:, :, best_slice].T, cmap="gray", origin="lower"
            )
            if row == 0:
                axes[row, col].set_title(
                    mod_name, fontsize=12, fontweight="bold"
                )
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(
                    pid, fontsize=9, rotation=0, labelpad=80
                )

        # Plot segmentation overlay
        seg_slice = seg[:, :, best_slice].T
        vol_t1ce = nib.load(
            str(pd / f"{pid}{MODALITY_SUFFIXES['t1ce']}")
        ).get_fdata()
        axes[row, 4].imshow(
            vol_t1ce[:, :, best_slice].T, cmap="gray", origin="lower"
        )
        seg_masked = np.ma.masked_where(seg_slice == 0, seg_slice)
        axes[row, 4].imshow(seg_masked, cmap="Set1", alpha=0.5, origin="lower")
        if row == 0:
            axes[row, 4].set_title(
                "Segmentation", fontsize=12, fontweight="bold"
            )
        axes[row, 4].axis("off")

    fig.suptitle(
        "Representative Slices (Max Tumor Area) with 4 MRI Modalities + Segmentation",
        fontsize=14,
        y=1.01,
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / "representative_slices.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("  Representative slice visualization complete.")


# %% [markdown]
# ## 7. Preprocessed Data Verification


# %%
def verify_preprocessed_data(processed_dir: Path) -> None:
    """Verify the preprocessed NPZ files and manifest.

    Parameters
    ----------
    processed_dir : Path
        Directory with preprocessed NPZ files and manifest.json.
    """
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[INFO] No preprocessed data found at {processed_dir}")
        print("  → Run: python scripts/preprocess.py --dataset brats2023")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    n_samples = manifest["n_samples"]
    n_hgg = manifest["n_hgg"]
    n_lgg = manifest["n_lgg"]

    print(f"\n{'=' * 50}")
    print("Preprocessed Data Verification")
    print(f"{'=' * 50}")
    print(f"Dataset: {manifest['dataset']}")
    print(f"Samples: {n_samples} ({n_hgg} HGG, {n_lgg} LGG)")
    print(f"Preprocessing: {json.dumps(manifest['preprocessing'], indent=2)}")

    # Verify a few NPZ files
    n_check = min(5, n_samples)
    print(f"\nVerifying {n_check} sample files...")
    for sample in manifest["samples"][:n_check]:
        npz_path = processed_dir / sample["file"]
        if npz_path.exists():
            data = np.load(npz_path)
            print(
                f"  {sample['file']}: image={data['image'].shape}, "
                f"label={sample['label']}, slice={sample['slice_index']}"
            )
        else:
            print(f"  [MISSING] {sample['file']}")


# %% [markdown]
# ## Main Execution


# %%
def main() -> None:
    """Run the complete data exploration pipeline."""
    parser = argparse.ArgumentParser(
        description="Explore brain tumor MRI datasets"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to raw BraTS data directory",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Path to preprocessed data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save exploration figures",
    )
    args = parser.parse_args()

    raw_dir = (
        Path(args.data_dir)
        if args.data_dir
        else PROJECT_ROOT / "data" / "raw" / "brats2023"
    )
    processed_dir = (
        Path(args.processed_dir)
        if args.processed_dir
        else PROJECT_ROOT / "data" / "processed" / "brats2023"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "figures" / "exploration"
    )

    print("=" * 60)
    print("BRAIN TUMOR MRI CLASSIFICATION v2 — DATA EXPLORATION")
    print("=" * 60)

    # 1. Dataset structure
    print("\n[1/7] Exploring dataset structure...")
    info = explore_dataset_structure(raw_dir)

    if info.get("error") == "directory_not_found":
        print("\n[INFO] Raw data not available. Checking preprocessed data...")
        verify_preprocessed_data(processed_dir)
        print("\nTo run full exploration, download data first:")
        print("  python scripts/download_data.py --dataset brats2023")
        return

    # 2. Class distribution (requires label derivation)
    print("\n[2/7] Analyzing class distribution...")
    try:
        from src.data.label_derivation import derive_brats_labels

        labels = derive_brats_labels(raw_dir)
        if labels:
            analyze_class_distribution(labels, "BraTS 2023", output_dir)
    except Exception as e:
        print(f"  Could not derive labels: {e}")

    # 3. Volume verification
    print("\n[3/7] Verifying volumes...")
    verify_volumes(raw_dir)

    # 4. Intensity distributions
    print("\n[4/7] Analyzing intensity distributions...")
    analyze_intensity_distributions(raw_dir, output_dir=output_dir)

    # 5. Tumor statistics
    print("\n[5/7] Analyzing tumor statistics...")
    analyze_tumor_statistics(raw_dir, output_dir=output_dir)

    # 6. Representative slices
    print("\n[6/7] Visualizing representative slices...")
    visualize_representative_slices(raw_dir, output_dir=output_dir)

    # 7. Preprocessed data verification
    print("\n[7/7] Verifying preprocessed data...")
    verify_preprocessed_data(processed_dir)

    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    if output_dir:
        print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

# %%
