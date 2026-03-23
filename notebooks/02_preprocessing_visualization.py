# notebooks/02_preprocessing_visualization.py — Visualize preprocessing effects
"""
Preprocessing Visualization for Brain Tumor Classification v2
==============================================================

Visualizes the effect of each preprocessing step to verify correctness
and provide figures for the thesis (Chapter 5: Methodology).

Sections:
  1. Before/after z-score normalization
  2. Before/after N4 bias field correction (UCSF-PDGM)
  3. Slice selection verification (max tumor area vs center)
  4. Augmentation examples (TorchIO MRI-specific transforms)
  5. Cross-dataset intensity comparison (BraTS vs UCSF-PDGM)

All figures are saved at 300 DPI for publication-ready quality.
"""

# %% [markdown]
# # Preprocessing Pipeline Visualization
#
# Each preprocessing step has validation evidence. This notebook
# verifies correctness by visualizing before/after effects.

# %%
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIGURE_DPI = 300  # Publication quality


# %% [markdown]
# ## 1. Z-Score Normalization Effect


# %%
def visualize_zscore_effect(
    data_dir: Path,
    output_dir: Path,
    n_patients: int = 3,
) -> None:
    """Show intensity histograms before and after z-score normalization.

    Parameters
    ----------
    data_dir : Path
        BraTS raw data directory.
    output_dir : Path
        Figure output directory.
    n_patients : int
        Number of patients to visualize.
    """
    try:
        import nibabel as nib
    except ImportError:
        print("[WARNING] nibabel not installed.")
        return

    from src.data.brats_dataset import MODALITY_SUFFIXES
    from src.data.preprocessing import z_score_normalize

    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[
        :n_patients
    ]

    fig, axes = plt.subplots(n_patients, 2, figsize=(12, 3 * n_patients))
    if n_patients == 1:
        axes = axes[np.newaxis, :]

    for row, pd in enumerate(patient_dirs):
        pid = pd.name
        fpath = pd / f"{pid}{MODALITY_SUFFIXES['flair']}"
        if not fpath.exists():
            continue

        vol = nib.load(str(fpath)).get_fdata().astype(np.float32)
        foreground = vol[vol > 0]

        vol_norm, fg_mean, fg_std = z_score_normalize(
            vol, foreground_only=True
        )
        fg_norm = vol_norm[vol > 0]

        # Raw histogram
        axes[row, 0].hist(
            foreground, bins=100, density=True, alpha=0.7, color="#1976D2"
        )
        axes[row, 0].axvline(
            foreground.mean(),
            color="red",
            linestyle="--",
            label=f"μ={foreground.mean():.0f}",
        )
        axes[row, 0].set_title(f"{pid} — Raw FLAIR", fontsize=10)
        axes[row, 0].legend(fontsize=8)

        # Normalized histogram
        axes[row, 1].hist(
            fg_norm, bins=100, density=True, alpha=0.7, color="#4CAF50"
        )
        axes[row, 1].axvline(
            fg_norm.mean(),
            color="red",
            linestyle="--",
            label=f"μ={fg_norm.mean():.2f}",
        )
        axes[row, 1].set_title(f"{pid} — Z-Score Normalized", fontsize=10)
        axes[row, 1].legend(fontsize=8)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Z-Score Normalization: Foreground-Only (nnU-Net Default)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "zscore_normalization_effect.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  Z-score visualization saved.")


# %% [markdown]
# ## 2. Slice Selection Comparison


# %%
def visualize_slice_selection(
    data_dir: Path,
    output_dir: Path,
    n_patients: int = 3,
) -> None:
    """Compare max-tumor-area slice vs center slice.

    This justifies why we select by tumor area instead of center.

    Parameters
    ----------
    data_dir : Path
        BraTS raw data directory.
    output_dir : Path
        Figure output directory.
    n_patients : int
        Number of patients to display.
    """
    try:
        import nibabel as nib
    except ImportError:
        print("[WARNING] nibabel not installed.")
        return

    from src.data.brats_dataset import MODALITY_SUFFIXES, SEG_SUFFIX

    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[
        :n_patients
    ]

    fig, axes = plt.subplots(n_patients, 3, figsize=(12, 4 * n_patients))
    if n_patients == 1:
        axes = axes[np.newaxis, :]

    for row, pd in enumerate(patient_dirs):
        pid = pd.name
        t1ce_path = pd / f"{pid}{MODALITY_SUFFIXES['t1ce']}"
        seg_path = pd / f"{pid}{SEG_SUFFIX}"
        if not t1ce_path.exists() or not seg_path.exists():
            continue

        vol = nib.load(str(t1ce_path)).get_fdata().astype(np.float32)
        seg = nib.load(str(seg_path)).get_fdata().astype(np.int32)

        # Find max tumor slice
        area_per_slice = np.sum(seg > 0, axis=(0, 1))
        max_slice = int(np.argmax(area_per_slice))
        center_slice = vol.shape[2] // 2

        # Max tumor slice
        axes[row, 0].imshow(
            vol[:, :, max_slice].T, cmap="gray", origin="lower"
        )
        seg_overlay = np.ma.masked_where(
            seg[:, :, max_slice].T == 0, seg[:, :, max_slice].T
        )
        axes[row, 0].imshow(seg_overlay, cmap="hot", alpha=0.4, origin="lower")
        axes[row, 0].set_title(f"Max Tumor (slice {max_slice})", fontsize=10)
        axes[row, 0].axis("off")

        # Center slice
        axes[row, 1].imshow(
            vol[:, :, center_slice].T, cmap="gray", origin="lower"
        )
        seg_center = np.ma.masked_where(
            seg[:, :, center_slice].T == 0, seg[:, :, center_slice].T
        )
        axes[row, 1].imshow(seg_center, cmap="hot", alpha=0.4, origin="lower")
        axes[row, 1].set_title(f"Center (slice {center_slice})", fontsize=10)
        axes[row, 1].axis("off")

        # Tumor area profile
        axes[row, 2].plot(area_per_slice, color="#1976D2", linewidth=1.5)
        axes[row, 2].axvline(
            max_slice, color="red", linestyle="--", label=f"Max: {max_slice}"
        )
        axes[row, 2].axvline(
            center_slice,
            color="green",
            linestyle=":",
            label=f"Center: {center_slice}",
        )
        axes[row, 2].set_xlabel("Slice Index")
        axes[row, 2].set_ylabel("Tumor Area (px)")
        axes[row, 2].set_title(f"{pid}", fontsize=10)
        axes[row, 2].legend(fontsize=8)
        axes[row, 2].spines["top"].set_visible(False)
        axes[row, 2].spines["right"].set_visible(False)

    fig.suptitle(
        "Slice Selection: Max Tumor Area vs Center — Justification for Selection Strategy",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "slice_selection_comparison.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  Slice selection visualization saved.")


# %% [markdown]
# ## 3. TorchIO Augmentation Examples


# %%
def visualize_augmentations(
    data_dir: Path,
    output_dir: Path,
) -> None:
    """Show examples of MRI-specific augmentations from TorchIO.

    Visualizes: original, affine, bias field, noise, ghosting, gamma.
    This demonstrates physically motivated augmentations.

    Parameters
    ----------
    data_dir : Path
        BraTS raw data directory.
    output_dir : Path
        Figure output directory.
    """
    try:
        import nibabel as nib
        import torchio as tio
    except ImportError:
        print("[WARNING] nibabel or torchio not installed.")
        return

    from src.data.brats_dataset import MODALITY_SUFFIXES, SEG_SUFFIX

    # Find first patient with all files
    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not patient_dirs:
        return

    pd = patient_dirs[0]
    pid = pd.name
    flair_path = pd / f"{pid}{MODALITY_SUFFIXES['flair']}"
    seg_path = pd / f"{pid}{SEG_SUFFIX}"

    if not flair_path.exists():
        return

    vol = nib.load(str(flair_path)).get_fdata().astype(np.float32)
    seg = (
        nib.load(str(seg_path)).get_fdata().astype(np.int32)
        if seg_path.exists()
        else None
    )

    # Select max tumor slice
    if seg is not None:
        area = np.sum(seg > 0, axis=(0, 1))
        sl = int(np.argmax(area))
    else:
        sl = vol.shape[2] // 2

    # Create TorchIO subject with a single slice (add dummy depth)
    slice_2d = vol[:, :, sl]
    tensor_4d = (
        __import__("torch")
        .from_numpy(slice_2d[np.newaxis, :, :, np.newaxis])
        .float()
    )

    # Define individual augmentations
    augmentations = {
        "Original": None,
        "Random Affine\n(±15°, ±10%)": tio.RandomAffine(
            scales=(0.9, 1.1), degrees=15, p=1.0
        ),
        "Bias Field\n(B0 inhomogeneity)": tio.RandomBiasField(
            coefficients=0.5, p=1.0
        ),
        "Gaussian Noise\n(thermal)": tio.RandomNoise(std=(0.03, 0.08), p=1.0),
        "Ghosting\n(motion artifact)": tio.RandomGhosting(
            intensity=(0.5, 1.0), p=1.0
        ),
        "Gamma\n(intensity)": tio.RandomGamma(log_gamma=(-0.3, 0.3), p=1.0),
    }

    fig, axes = plt.subplots(
        1, len(augmentations), figsize=(4 * len(augmentations), 4)
    )

    for ax, (name, transform) in zip(axes, augmentations.items(), strict=True):
        if transform is None:
            img = tensor_4d[0, :, :, 0].numpy()
        else:
            subject = tio.Subject(image=tio.ScalarImage(tensor=tensor_4d))
            result = transform(subject)
            img = result["image"].data[0, :, :, 0].numpy()

        ax.imshow(img.T, cmap="gray", origin="lower")
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        "TorchIO MRI-Specific Augmentations (Physically Motivated)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "augmentation_examples.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  Augmentation visualization saved.")


# %% [markdown]
# ## Main Execution


# %%
def main() -> None:
    """Run all preprocessing visualizations."""
    parser = argparse.ArgumentParser(
        description="Visualize preprocessing effects"
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    raw_dir = (
        Path(args.data_dir)
        if args.data_dir
        else PROJECT_ROOT / "data" / "raw" / "brats2023"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "figures" / "preprocessing"
    )

    print("=" * 60)
    print("PREPROCESSING VISUALIZATION")
    print("=" * 60)

    if not raw_dir.exists():
        print(f"\n[INFO] Raw data not found at {raw_dir}")
        print(
            "  → Download data first: python scripts/download_data.py --dataset brats2023"
        )
        print(
            "  → These visualizations will be generated once data is available."
        )
        return

    print("\n[1/3] Visualizing z-score normalization...")
    visualize_zscore_effect(raw_dir, output_dir)

    print("\n[2/3] Visualizing slice selection...")
    visualize_slice_selection(raw_dir, output_dir)

    print("\n[3/3] Visualizing augmentations...")
    visualize_augmentations(raw_dir, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
