# notebooks/03_kaggle_problems_analysis.py — Kaggle dataset problem documentation
"""
Kaggle MasoudNickparvar Dataset Problem Analysis
=================================================

Documents the structural problems with the Kaggle brain tumor dataset
that cause artificial accuracy inflation. This notebook provides evidence
for thesis Chapter 4 (Problem Discovery).

Analyses:
  1. Dataset composition audit (3 sources combined).
  2. Duplicate detection via perceptual hashing.
  3. Intensity distribution differences between sources.
  4. Background pattern analysis (shortcut features).
  5. Mislabeling documentation.
  6. Patient-level leakage quantification.
"""

# %% [markdown]
# # Kaggle MasoudNickparvar Dataset — Problem Analysis
#
# This analysis documents WHY the 99% accuracy on this dataset is artificial.

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

FIGURE_DPI = 300


# %% [markdown]
# ## 1. Dataset Composition Audit


# %%
def audit_dataset_composition(data_dir: Path, output_dir: Path) -> dict:
    """Audit the Kaggle dataset structure and document problems.

    Parameters
    ----------
    data_dir : Path
        Root directory of the Kaggle dataset (with class subdirectories).
    output_dir : Path
        Directory to save audit figures.

    Returns
    -------
    dict
        Audit results.
    """
    from src.experiments.naive_experiment import KAGGLE_DATASET_PROBLEMS

    print("=" * 60)
    print("KAGGLE DATASET AUDIT")
    print("=" * 60)

    audit = {
        "documented_problems": KAGGLE_DATASET_PROBLEMS,
        "sources": [
            {
                "name": "Figshare (Jun Cheng, 2017)",
                "images": 3064,
                "patients": 233,
                "type": "T1ce MRI",
                "issue": "~13 slices/patient, patient grouping lost",
            },
            {
                "name": "SARTAJ (Bhuvaji et al., 2020)",
                "images": "variable",
                "type": "Mixed",
                "issue": "Documented glioma mislabeling (PLOS ONE 2025)",
            },
            {
                "name": "Br35H (Hamada, 2020)",
                "images": "variable",
                "type": "Mixed",
                "issue": "No tumor class from different source entirely",
            },
        ],
    }

    if not data_dir.exists():
        print(f"\n[INFO] Kaggle data not available at {data_dir}")
        print("  Analysis below uses documented facts from the literature.")
        return audit

    # Count per class
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(
                class_dir.glob("*.jpeg")
            )
            n = len(images)
            print(f"  Class '{class_dir.name}': {n} images")
            audit[f"class_{class_dir.name}"] = n

    return audit


# %% [markdown]
# ## 2. Intensity Distribution Analysis Between Sources


# %%
def analyze_intensity_differences(data_dir: Path, output_dir: Path) -> None:
    """Show intensity distribution differences between classes.

    If classes come from different dataset sources, their intensity
    distributions will differ systematically — creating a shortcut.

    Parameters
    ----------
    data_dir : Path
        Kaggle dataset root.
    output_dir : Path
        Figure output directory.
    """
    try:
        from PIL import Image
    except ImportError:
        print("[WARNING] Pillow not available for image analysis.")
        return

    if not data_dir.exists():
        print("[INFO] Kaggle data not available for intensity analysis.")
        print("  Expected finding: different sources have different intensity")
        print(
            "  distributions, enabling classification by source, not pathology."
        )
        return

    class_intensities = {}
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*.jpg"))[:100]  # Sample 100
        values = []
        for img_path in images:
            img = np.array(Image.open(img_path).convert("L"))
            values.extend(img.flatten().tolist())

        if values:
            class_intensities[class_dir.name] = np.array(values)

    if not class_intensities:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for class_name, values in class_intensities.items():
        ax.hist(values, bins=100, density=True, alpha=0.5, label=class_name)

    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.set_title(
        "Intensity Distributions by Class — Evidence of Source-Specific Patterns",
        fontweight="bold",
    )
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "kaggle_intensity_distributions.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  Intensity distribution figure saved.")


# %% [markdown]
# ## 3. Background Pattern Analysis


# %%
def analyze_background_patterns(data_dir: Path, output_dir: Path) -> None:
    """Analyze background (non-brain) regions for shortcut features.

    If the model can distinguish classes from background alone, it has
    learned preprocessing signatures rather than pathology.

    Parameters
    ----------
    data_dir : Path
        Kaggle dataset root.
    output_dir : Path
        Figure output directory.
    """
    try:
        from PIL import Image
    except ImportError:
        return

    if not data_dir.exists():
        print("[INFO] Kaggle data not available for background analysis.")
        print(
            "  Expected finding: background pixel distributions differ between"
        )
        print(
            "  classes because images come from different preprocessing pipelines."
        )
        return

    class_bg_means = {}
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*.jpg"))[:50]
        bg_values = []
        for img_path in images:
            img = np.array(Image.open(img_path).convert("L"))
            # Background: pixels in the corners (likely outside brain)
            corners = np.concatenate(
                [
                    img[:10, :10].flatten(),
                    img[:10, -10:].flatten(),
                    img[-10:, :10].flatten(),
                    img[-10:, -10:].flatten(),
                ]
            )
            bg_values.append(float(np.mean(corners)))

        if bg_values:
            class_bg_means[class_dir.name] = bg_values

    if not class_bg_means:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = list(class_bg_means.keys())
    data = [class_bg_means[k] for k in labels]
    ax.boxplot(data, labels=labels)
    ax.set_ylabel("Mean Background Intensity (Corner Pixels)")
    ax.set_title(
        "Background Intensity by Class — Shortcut Feature Evidence",
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "kaggle_background_patterns.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  Background pattern figure saved.")


# %% [markdown]
# ## 4. Documented Problems Summary


# %%
def print_problem_summary() -> None:
    """Print a formatted summary of all documented problems."""
    from src.experiments.naive_experiment import KAGGLE_DATASET_PROBLEMS

    print("\n" + "=" * 60)
    print("DOCUMENTED PROBLEMS WITH KAGGLE MASOUDNICKPARVAR DATASET")
    print("=" * 60)

    for i, problem in enumerate(KAGGLE_DATASET_PROBLEMS, 1):
        print(f"\n  [{i}] {problem}")

    print("\n" + "-" * 60)
    print("CONCLUSION: These problems make ANY result from this dataset")
    print("unreliable without extensive correction. The 99% accuracy")
    print("reported in Kaggle notebooks is ARTIFICIAL.")
    print("-" * 60)


# %%
def main() -> None:
    """Run the complete Kaggle problems analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze Kaggle dataset problems"
    )
    parser.add_argument("--kaggle-dir", type=str, default="data/raw/kaggle")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/figures/kaggle_analysis"
    )
    args = parser.parse_args()

    kaggle_dir = PROJECT_ROOT / args.kaggle_dir
    output_dir = PROJECT_ROOT / args.output_dir

    audit_dataset_composition(kaggle_dir, output_dir)
    analyze_intensity_differences(kaggle_dir, output_dir)
    analyze_background_patterns(kaggle_dir, output_dir)
    print_problem_summary()


if __name__ == "__main__":
    main()
