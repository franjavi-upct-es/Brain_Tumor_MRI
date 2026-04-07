# src/data/label_derivation.py
# Derive classification labels from BraTS and UCSF-PDGM
"""
Derives glioma grade labels for classification from dataset metadata.

BraTS 2023:
    HGG/LGG labels can be derived from the official name mapping CSV or
    from tumor segmentation characteristics. BraTS organizes patients with
    a naming convention that often encodes grade information, but the
    definitive labels come from the accompanying metadata files.

UCSF-PDGM:
    Grade labels (II, III, IV) come from the clinical CSV provided by TCIA.
    For binary classification: Grade II-III ⟶ LGG (0), Grade IV ⟶ HGG (1).

References:
    - WHO 2021 CNS tumor classification
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# BraTS segmentation-based heuristic for HGG or LGG
# HGG (Grade IV / Glioblastoma) typically has enhancing tumor (label 3)
# LGG (Grade II-III) typically lacks significant enhancement.
ENHANCING_TUMOR_LABEL = 3
ENHANCING_TUMOR_THRESHOLD = 100  # Minimum enhancing voxels to call HGG


def derive_brats_labels_from_metadata(
    metadata_path: Path,
) -> dict[str, int]:
    """
    Load HGG/LGG labels from BraTS official metadata CSV.

    BraTS 2023 provides a name mapping file that associates each patient ID
    with the originating dataset and grade information.

    Parameters
    ----------
    metadata_path : Path
        Path to the BraTS metadata CSV file (name_mapping.csv or similar).
        Expected columns: BraTS2023ID, Grade (or Type with HGG/LGG).

    Returns
    -------
    dict[str, int]
        Mapping from patient_id to binary label (0=LGG, 1=HGG).
    """
    labels = {}
    path = Path(metadata_path)

    if not path.exists():
        raise FileNotFoundError(f"BraTS metadata file not found: {path}")

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row.get("BraTS2023ID", row.get("ID", "")).strip()
            if not patient_id:
                continue

            # Try different column names used across BraTS versions
            grade_raw = (
                (
                    row.get("Grade", "")
                    or row.get("Type", "")
                    or row.get("grade", "")
                )
                .strip()
                .upper()
            )

            if grade_raw in ("HGG", "HIGH", "GBM", "IV", "4"):
                labels[patient_id] = 1  # HGG
            elif grade_raw in ("LGG", "LOW", "II", "III", "2", "3"):
                labels[patient_id] = 0  # LGG
            else:
                logging.debug(
                    "Unknown grade '%s' for %s. Will attempt "
                    "segmentation-based derivation.",
                    grade_raw,
                    patient_id,
                )

    logger.info(
        "Loaded %d grade labels from metadata: %d HGG, %d LGG",
        len(labels),
        sum(1 for v in labels.values() if v == 1),
        sum(1 for v in labels.values() if v == 0),
    )
    return labels


def derive_brats_labels_from_segmentation(
    data_dir: Path,
    seg_suffix: str = "-seg.nii.gz",
    enhancing_threshold: int = ENHANCING_TUMOR_THRESHOLD,
) -> dict[str, int]:
    """
    Derive HGG/LGG labels from segmentation masks (heuristic fallback).

    High-grade gliomas (HGG, Grade IV) typically show enhancing tumor
    (BraTS label 3). Low-grade gliomas (LGG, Grade II-III) typically
    have minimal or no enhancement.

    This is a HEURISTIC and should only be used when official metadata
    labels are unavailable. The threshold and accuracy should be reported
    in the thesis as a limitation.

    Parameters
    ----------
    data_dir : Path
        Root directory with patient subdirectories.
    seg_suffix : str
        Segmentation file suffix.
    enhancing_threshold : int
        Minimum enhancing tumor voxels to classify as HGG.

    Returns
    -------
    dict[str, int]
        Mapping from patient_id to binary label.
    """
    import nibabel as nib

    labels = {}

    for patient_dir in sorted(data_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        seg_path = patient_dir / f"{patient_id}{seg_suffix}"

        if not seg_path.exists():
            continue

        seg = nib.load(str(seg_path)).get_fdata().astype(np.int32)

        # Count enhancing tumor voxels (BraTS label 3)
        n_enhancing = int(np.sum(seg == ENHANCING_TUMOR_LABEL))

        if n_enhancing >= enhancing_threshold:
            labels[patient_id] = 1  # HGG (significant enhancement)
        else:
            labels[patient_id] = 0  # LGG (minimal/no enhancement)

    logger.info(
        "Derived %d labels from segmentation (threshold=%d): %d HGG, %d LGG",
        len(labels),
        enhancing_threshold,
        sum(1 for v in labels.values() if v == 1),
        sum(1 for v in labels.values() if v == 0),
    )
    return labels


def derive_brats_labels_from_mapping_xlsx(
    xlsx_path: Path,
) -> dict[str, int]:
    """
    Derive HGG/LGG labels from BraTS2023-2017 mapping Excel file.

    Uses the "Cohort Name" column to assign labels:
        TCGA-GBM, CPTAC-GBM, IvyGAP, ACRIN-FMISO-Brain → HGG (1)
        TCGA-LGG → LGG (0)
        Private Collection, priorToBraTS2017 → unknown (skipped)

    Parameters
    ----------
    xlsx_path : Path
        Path to BraTS2023_2017_GLI_Mapping.xlsx.

    Returns
    -------
    dict[str, int]
        Mapping from patient_id to binary label for patients with known cohorts.
    """
    import pandas as pd

    df = pd.read_excel(xlsx_path)

    cohort_col = "Cohort Name (if publicly available)"
    id_col = "BraTS2023"

    if cohort_col not in df.columns or id_col not in df.columns:
        logger.warning(
            "Expected columns not found in %s. Columns: %s",
            xlsx_path, list(df.columns),
        )
        return {}

    # Map cohort names to grade labels
    hgg_cohorts = {"TCGA-GBM", "CPTAC-GBM", "IvyGAP", "ACRIN-FMISO-Brain (ACRIN 6684)"}
    lgg_cohorts = {"TCGA-LGG"}

    labels = {}
    for _, row in df.iterrows():
        pid = str(row[id_col]).strip()
        cohort = str(row[cohort_col]).strip()

        if cohort in hgg_cohorts:
            labels[pid] = 1
        elif cohort in lgg_cohorts:
            labels[pid] = 0
        # else: unknown cohort, skip — will fall back to segmentation heuristic

    logger.info(
        "Loaded %d labels from mapping xlsx (cohort-based): %d HGG, %d LGG, "
        "%d unknown (skipped)",
        len(labels),
        sum(1 for v in labels.values() if v == 1),
        sum(1 for v in labels.values() if v == 0),
        len(df) - len(labels),
    )
    return labels


def derive_brats_labels(
    data_dir: Path,
    metadata_path: Path | None = None,
) -> dict[str, int]:
    """
    Derive BraTS grade labels using metadata first, segmentation fallback.

    Strategy:
    1. If Excel mapping file exists, extract labels from cohort names.
    2. If metadata CSV exists, load labels from there.
    3. For patients missing from both, use segmentation heuristic.
    4. Log statistics and coverage.

    Parameters
    ----------
    data_dir : Path
        Root BraTS data directory.
    metadata_path : Path, optional
        Path to BraTS metadata CSV. If None, searches common locations.

    Returns
    -------
    dict[str, int]
        Complete label mapping for all patients.
    """
    labels = {}

    # Try Excel mapping file first (most reliable for BraTS 2023)
    xlsx_candidates = [
        data_dir / "Data" / "BraTS-GLI" / "BraTS2023_2017_GLI_Mapping.xlsx",
        data_dir / "BraTS2023_2017_GLI_Mapping.xlsx",
    ]
    for xlsx_path in xlsx_candidates:
        if xlsx_path.exists():
            labels = derive_brats_labels_from_mapping_xlsx(xlsx_path)
            break

    # Try metadata CSV as secondary source
    if metadata_path is None:
        # Search common metadata file locations
        candidates = [
            data_dir / "name_mapping.csv",
            data_dir / "metadata.csv",
            data_dir / "labels.csv",
            data_dir / "clinical_metadata.csv",
            data_dir.parent / "name_mapping.csv",
            data_dir.parent / "clinical_metadata.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                metadata_path = candidate
                break

    if metadata_path is not None and metadata_path.exists():
        csv_labels = derive_brats_labels_from_metadata(metadata_path)
        for pid, label in csv_labels.items():
            if pid not in labels:
                labels[pid] = label

    if not labels:
        logger.info(
            "No metadata CSV or mapping xlsx found. "
            "Using segmentation-based derivation."
        )

    # Fill in missing patients using segmentation heuristic
    patient_dirs = [
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("BraTS-")
    ]
    missing_patients = [
        d.name for d in patient_dirs if d.name not in labels
    ]

    if missing_patients:
        logger.info(
            "%d patients missing labels from metadata. "
            "Deriving from segmentation.",
            len(missing_patients),
        )
        seg_labels = derive_brats_labels_from_segmentation(data_dir)
        for pid, label in seg_labels.items():
            if pid not in labels:
                labels[pid] = label

    total = len(labels)
    hgg = sum(1 for v in labels.values() if v == 1)
    lgg = total - hgg
    logger.info(
        "Final label set: %d patients (%d HGG [%.1f%%], %d LGG [%.1f%%])",
        total,
        hgg,
        100 * hgg / max(total, 1),
        lgg,
        100 * lgg / max(total, 1),
    )

    return labels


def load_ucsf_labels(
    metadata_path: Path,
) -> dict[str, dict[str, Any]]:
    """
    Load UCSF-PDGM patient metadata from clinical CSV.

    Expected CSV columns: patient_id, grade, idh_status, mgmt_status, ...
    Grade mapping for binary classification:
        Grade II, III → LGG (0)
        Grade IV → HGG (1)

    Parameters
    ----------
    metadata_path : Path
        Path to the UCSF-PDGM clinical metadata CSV.

    Returns
    -------
    dict[str, dict]
        Mapping from patient_id to metadata dict with 'binary_label'.
    """
    metadata = {}
    grade_to_binary = {2: 0, 3: 0, 4: 1}

    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("patient_id", row.get("ID", "")).strip()
            if not pid:
                continue

            grade_raw = row.get("grade", row.get("Grade", ""))
            try:
                grade = int(grade_raw)
            except (ValueError, TypeError):
                logger.warning(
                    "Invalid grade '%s' for %s. Skipping.", grade_raw, pid
                )
                continue

            if grade not in grade_to_binary:
                logger.warning(
                    "Unknown grade %d for %s. Skipping.", grade, pid
                )
                continue

            metadata[pid] = {
                "grade": grade,
                "binary_label": grade_to_binary[grade],
                "idh": row.get("idh_status", row.get("IDH", "")),
                "mgmt": row.get("mgmt_status", row.get("MGMT", "")),
            }

    logger.info(
        "Loaded UCSF-PDGM metadata: %d patients "
        "(Grade II: %d, III: %d, IV: %d)",
        len(metadata),
        sum(1 for m in metadata.values() if m["grade"] == 2),
        sum(1 for m in metadata.values() if m["grade"] == 3),
        sum(1 for m in metadata.values() if m["grade"] == 4),
    )
    return metadata


def save_labels(labels: dict[str, int], output_path: Path) -> None:
    """
    Save label mapping to JSON for reproducibility.

    Parameters
    ----------
    labels : dict[str, int]
        Mapping from patient_id to integer label.
    output_path : Path
        Path to save the JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2, sort_keys=True)
    logger.info("Labels saved to %s (%d patients)", output_path, len(labels))
