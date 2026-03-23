# Data Guide

How to obtain, preprocess, and validate the datasets used in this project.

---

## Dataset Overview

### BraTS 2023 Adult Glioma (Training)

- **Source:** Synapse platform, ID: syn51156910
- **URL:** https://www.synapse.org/brats2024
- **Patients:** ~1,470 adult glioma cases
- **Modalities:** T1, T1ce (gadolinium), T2, T2-FLAIR (NIfTI format)
- **Annotations:** Expert neuroradiologist segmentations (NCR, ED, ET)
- **Organization:** Per patient — data leakage impossible by design
- **Pre-applied:** Co-registered to SRI24 atlas, 1mm³ isotropic, skull-stripped
- **Institutions:** 19+ centers (real scanner heterogeneity)
- **Citations:** ~5,000+ (gold standard benchmark)

### UCSF-PDGM (External Validation)

- **Source:** The Cancer Imaging Archive (TCIA)
- **URL:** https://www.cancerimagingarchive.net/collection/ucsf-pdgm/
- **Patients:** 495 with histopathologically confirmed diffuse gliomas
- **Grade distribution:** 55 Grade II, 42 Grade III, 403 Grade IV
- **Protocol:** Standardized 3T (GE Discovery 750), 8-channel coil
- **Modalities:** 11 sequences per patient (we use T1, T1ce, T2, FLAIR)
- **Genetic markers:** IDH (all patients), MGMT (Grade III-IV), 1p/19q
- **Survival data:** Overall survival + resection extent
- **CRITICAL:** Used ONLY for external validation. Never for training.

### Kaggle MasoudNickparvar (Naive Comparison Only)

- **Problems:** Combines 3 sources, documented mislabeling, no patient IDs, JPEG with no DICOM metadata, preprocessing shortcuts across classes.
- **Role:** Demonstrates why the naive approach fails. NOT used for any valid model.
- **See:** `notebooks/03_kaggle_problems_analysis.py` for full problem documentation.

---

## Data Acquisition

### BraTS 2023

1. Create a free account at https://www.synapse.org
2. Navigate to https://www.synapse.org/brats2024
3. Accept the data use terms
4. Generate a personal access token (Profile → Settings → Personal Access Tokens)
5. Download:

```bash
export SYNAPSE_AUTH_TOKEN="your-token-here"
python scripts/download_data.py --dataset brats2023
```

**Disk space:** ~50+ GB for the training data with labels.

### UCSF-PDGM

1. Visit https://www.cancerimagingarchive.net/collection/ucsf-pdgm/
2. Install the NBIA Data Retriever from TCIA
3. Download the collection (select T1, T1ce, T2, FLAIR sequences)
4. Place data in `data/raw/ucsf_pdgm/`

```bash
python scripts/download_data.py --dataset ucsf_pdgm
# Follow the printed manual instructions
```

**Disk space:** ~150 GB for full 11-sequence data.

---

## Preprocessing Pipeline

### BraTS (Section 6.1 — Minimal, Already Preprocessed)

BraTS arrives co-registered, isotropic, and skull-stripped. Our preprocessing:

```
Raw NIfTI (.nii.gz)
    │
    ├── Quality control
    │   ├── Check dimensions = (240, 240, 155)
    │   ├── Verify 4 modalities + segmentation per patient
    │   ├── Detect empty/corrupt volumes
    │   └── Verify minimum tumor voxels (≥100)
    │
    ├── Z-score normalization (per volume)
    │   └── μ and σ computed on foreground voxels only (voxels > 0)
    │
    ├── Slice selection
    │   └── Axial slice with maximum tumor area (from segmentation)
    │
    └── Save as NPZ (image: C×H×W, segmentation: H×W, metadata)
```

**Command:**
```bash
python scripts/preprocess.py --dataset brats2023
```

**Output:** `data/processed/brats2023/` containing NPZ files + `manifest.json` + `qc_report.json`

### UCSF-PDGM (Section 6.2 — Full Preprocessing)

```
Raw NIfTI (.nii.gz)
    │
    ├── Select 4 modalities (T1, T1ce, T2, FLAIR)
    │   └── Discard advanced sequences (DTI, ASL) for BraTS comparability
    │
    ├── N4ITK Bias Field Correction (SimpleITK)
    │   ├── shrinkFactor = 4
    │   ├── iterations = [50, 50, 50, 50]
    │   └── Reference: Tustison et al. (2010)
    │
    ├── Resample to 1mm³ isotropic
    │   ├── Images: 3rd-order B-spline interpolation
    │   └── Labels: nearest-neighbor interpolation
    │
    ├── Z-score normalization (identical protocol to BraTS)
    │
    ├── Slice selection (same criteria as BraTS)
    │
    └── Save as NPZ
```

**Command:**
```bash
python scripts/preprocess.py --dataset ucsf_pdgm --metadata-path data/raw/ucsf_pdgm/clinical_metadata.csv
```

---

## Label Derivation

### BraTS Grade Labels

Grade labels (HGG vs LGG) are derived in priority order:

1. **Metadata CSV** (authoritative): if `name_mapping.csv` exists, reads Grade/Type column.
2. **Segmentation heuristic** (fallback): counts enhancing tumor voxels (BraTS label 3). Threshold ≥ 100 enhancing voxels → HGG.

The heuristic is documented as a limitation if used.

### UCSF-PDGM Grade Labels

Read directly from the clinical metadata CSV:
- Grade II, III → LGG (label 0)
- Grade IV → HGG (label 1)

IDH status and MGMT methylation are preserved for optional secondary tasks.

---

## Patient-Level Splitting

**This is non-negotiable.** All splits use `StratifiedGroupKFold`:

```bash
python scripts/create_splits.py --seed 42
```

Output: `data/splits/brats_5fold.json` containing:
- Train/val indices for each of 5 folds
- Patient IDs per fold (for leakage verification)
- Class distribution per fold

**Verification:**
```bash
make test-leakage
# Runs 12 tests confirming zero patient overlap
```

---

## Quality Control

The QC pipeline (`src/data/quality_control.py`) checks:

| Check | Threshold | Action on Failure |
|-------|-----------|-------------------|
| Volume dimensions | (240, 240, 155) for BraTS | Skip patient, log warning |
| All 4 modalities present | Must have T1, T1ce, T2, FLAIR | Skip patient |
| Non-empty volumes | At least one non-zero voxel | Skip patient |
| NaN/Inf values | Zero tolerance | Skip patient |
| Minimum tumor voxels | ≥ 100 in segmentation | Skip patient |

QC report saved to `data/processed/{dataset}/qc_report.json`.

---

## Manifest Format

After preprocessing, `manifest.json` lists all valid samples:

```json
{
  "dataset": "brats2023",
  "preprocessing": {
    "z_score": true,
    "foreground_only": true,
    "slice_method": "max_tumor_area",
    "context_slices": 0
  },
  "n_samples": 1350,
  "n_hgg": 945,
  "n_lgg": 405,
  "samples": [
    {
      "file": "BraTS-GLI-00001.npz",
      "patient_id": "BraTS-GLI-00001",
      "label": 1,
      "slice_index": 77,
      "tumor_area": 1250
    }
  ]
}
```
