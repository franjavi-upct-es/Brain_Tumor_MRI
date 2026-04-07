# scripts/download_data.py — Automated dataset download for BraTS 2023 and UCSF-PDGM
"""
Download clinical-grade datasets for brain tumor classification.

Supported datasets:
  - BraTS 2023 Adult Glioma: via Synapse (syn51156910). Requires Synapse
    account and acceptance of data use terms.
  - UCSF-PDGM: via The Cancer Imaging Archive (TCIA). Free access.

This script handles authentication, download, and basic verification.
Data is saved to data/raw/{dataset_name}/.

Usage:
    python scripts/download_data.py --dataset brats2023
    python scripts/download_data.py --dataset ucsf_pdgm
    python scripts/download_data.py --dataset all

References:
    - BraTS 2023: https://www.synapse.org/brats2024 (syn51156910)
    - UCSF-PDGM: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/
    - Calabrese et al. (2022). Radiology: AI, 4(6).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Project root directory (scripts/ is one level below root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_brats2023(output_dir: Path) -> None:
    """Download BraTS 2023 Adult Glioma dataset from Synapse.

    Requires:
    1. A Synapse account (free registration at synapse.org).
    2. Acceptance of BraTS 2023 data use terms.
    3. Synapse credentials set via environment variables or .synapseConfig.

    Parameters
    ----------
    output_dir : Path
        Directory to save the downloaded data.
    """
    try:
        import synapseclient
    except ImportError:
        logger.error(
            "synapseclient not installed. Run: pip install synapseclient"
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to Synapse...")

    # Authenticate via environment variables or config file
    syn = synapseclient.Synapse()
    try:
        syn.login(silent=True)
    except Exception:
        logger.error(
            "Synapse authentication failed. Set SYNAPSE_AUTH_TOKEN "
            "environment variable or configure ~/.synapseConfig. "
            "Register at: https://www.synapse.org/"
        )
        sys.exit(1)

    # BraTS 2023 Adult Glioma training data with labels
    synapse_id = "syn51156910"
    logger.info(
        "Downloading BraTS 2023 Adult Glioma (ID: %s) to %s",
        synapse_id,
        output_dir,
    )
    logger.info(
        "This may take a while (~50+ GB). Ensure sufficient disk space."
    )

    try:
        import synapseutils
    except ImportError:
        logger.error(
            "synapseutils not installed. Run: pip install synapseclient[pysftp]"
        )
        sys.exit(1)

    try:
        # Recursively sync the entire BraTS folder tree to output_dir
        synapseutils.syncFromSynapse(syn, synapse_id, path=str(output_dir))
    except Exception as e:
        logger.error("Download failed: %s", e)
        logger.info(
            "Manual download: Go to https://www.synapse.org/brats2024, "
            "accept terms, and download training data with labels."
        )
        sys.exit(1)

    logger.info("BraTS 2023 download complete: %s", output_dir)
    _verify_brats_download(output_dir)


def download_ucsf_pdgm(output_dir: Path) -> None:
    """Download UCSF-PDGM dataset from TCIA.

    The UCSF-PDGM dataset is freely available from The Cancer Imaging Archive.
    Contains 495 patients with diffuse gliomas, 11 MRI sequences per patient.

    Parameters
    ----------
    output_dir : Path
        Directory to save the downloaded data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tcia_url = "https://www.cancerimagingarchive.net/collection/ucsf-pdgm/"

    logger.info("UCSF-PDGM Dataset Download")
    logger.info("=" * 60)
    logger.info(
        "The UCSF-PDGM dataset must be downloaded manually from TCIA."
    )
    logger.info("Steps:")
    logger.info("  1. Visit: %s", tcia_url)
    logger.info("  2. Click 'Download' and use the NBIA Data Retriever.")
    logger.info("  3. Save data to: %s", output_dir)
    logger.info("")
    logger.info(
        "Alternatively, use the TCIA REST API or NBIA Data Retriever CLI."
    )
    logger.info("Dataset size: ~150 GB (495 patients, 11 sequences each).")
    logger.info("=" * 60)

    # Attempt programmatic download via TCIA REST API
    try:
        _attempt_tcia_download(output_dir)
    except Exception as e:
        logger.warning(
            "Programmatic download not available: %s. "
            "Please download manually following the instructions above.",
            e,
        )


def _attempt_tcia_download(output_dir: Path) -> None:
    """Attempt to download UCSF-PDGM via TCIA REST API.

    Parameters
    ----------
    output_dir : Path
        Output directory for downloaded data.

    Raises
    ------
    RuntimeError
        If programmatic download is not supported or fails.
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests library not available")

    # TCIA REST API base URL
    base_url = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

    # Get series list for UCSF-PDGM
    response = requests.get(
        f"{base_url}/getSeries",
        params={"Collection": "UCSF-PDGM"},
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"TCIA API returned status {response.status_code}. "
            f"Manual download required."
        )

    series_list = response.json()
    logger.info("Found %d series in UCSF-PDGM collection", len(series_list))
    logger.info(
        "Full programmatic download requires NBIA Data Retriever. "
        "Please download manually."
    )


def clean_brats_unlabeled(data_dir: Path) -> None:
    """Remove BraTS patients without ground-truth grade labels.

    BraTS 2023 includes ~1250 patients, but only ~238 have reliable
    grade labels via their cohort of origin (TCGA-GBM, TCGA-LGG,
    CPTAC-GBM, IvyGAP, ACRIN-FMISO-Brain). The remaining ~1017
    "Private Collection" patients lack grade metadata.

    This function removes unlabeled patient directories to save disk
    space and avoid noisy segmentation-based label heuristics.

    Parameters
    ----------
    data_dir : Path
        Root BraTS data directory containing patient subdirectories.
    """
    import shutil

    xlsx_candidates = [
        data_dir / "Data" / "BraTS-GLI" / "BraTS2023_2017_GLI_Mapping.xlsx",
        data_dir / "BraTS2023_2017_GLI_Mapping.xlsx",
    ]
    xlsx_path = None
    for candidate in xlsx_candidates:
        if candidate.exists():
            xlsx_path = candidate
            break

    if xlsx_path is None:
        logger.warning(
            "BraTS2023_2017_GLI_Mapping.xlsx not found. "
            "Cannot determine labeled patients. Skipping cleanup."
        )
        return

    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed. Skipping cleanup.")
        return

    df = pd.read_excel(xlsx_path)
    cohort_col = "Cohort Name (if publicly available)"
    id_col = "BraTS2023"

    if cohort_col not in df.columns or id_col not in df.columns:
        logger.warning("Unexpected columns in mapping file. Skipping cleanup.")
        return

    hgg_cohorts = {
        "TCGA-GBM", "CPTAC-GBM", "IvyGAP",
        "ACRIN-FMISO-Brain (ACRIN 6684)",
    }
    lgg_cohorts = {"TCGA-LGG"}

    labeled_ids = set()
    for _, row in df.iterrows():
        cohort = str(row[cohort_col]).strip()
        if cohort in hgg_cohorts or cohort in lgg_cohorts:
            labeled_ids.add(str(row[id_col]).strip())

    patient_dirs = [
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("BraTS-")
    ]

    to_remove = [d for d in patient_dirs if d.name not in labeled_ids]
    to_keep = [d for d in patient_dirs if d.name in labeled_ids]

    if not to_remove:
        logger.info("No unlabeled patients to remove.")
        return

    logger.info(
        "Cleaning unlabeled patients: keeping %d labeled, removing %d unlabeled",
        len(to_keep), len(to_remove),
    )

    for patient_dir in to_remove:
        shutil.rmtree(patient_dir)

    logger.info(
        "Cleanup complete. %d labeled patients remain (%d HGG cohort, %d LGG cohort)",
        len(to_keep),
        sum(1 for d in to_keep if d.name in
            {str(r[id_col]).strip() for _, r in df.iterrows()
             if str(r[cohort_col]).strip() in hgg_cohorts}),
        sum(1 for d in to_keep if d.name in
            {str(r[id_col]).strip() for _, r in df.iterrows()
             if str(r[cohort_col]).strip() in lgg_cohorts}),
    )


def _verify_brats_download(data_dir: Path) -> None:
    """Verify BraTS download by counting patient directories.

    Parameters
    ----------
    data_dir : Path
        Directory containing downloaded BraTS data.
    """
    patient_dirs = [d for d in data_dir.rglob("BraTS*") if d.is_dir()]
    n_patients = len(patient_dirs)

    if n_patients == 0:
        logger.warning(
            "No BraTS patient directories found. Check download location."
        )
    else:
        logger.info("Found %d patient directories", n_patients)

        # Verify first patient has all 4 modalities + segmentation
        sample_dir = patient_dirs[0]
        nifti_files = list(sample_dir.glob("*.nii.gz"))
        logger.info(
            "Sample patient %s has %d NIfTI files: %s",
            sample_dir.name,
            len(nifti_files),
            [f.name for f in nifti_files],
        )


def main() -> None:
    """Entry point for dataset download."""
    parser = argparse.ArgumentParser(
        description="Download BraTS 2023 and/or UCSF-PDGM datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["brats2023", "ucsf_pdgm", "all"],
        help="Dataset to download: brats2023, ucsf_pdgm, or all",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: data/raw/{dataset})",
    )
    parser.add_argument(
        "--clean-unlabeled",
        action="store_true",
        help=(
            "Remove BraTS patients without ground-truth grade "
            "labels (keeps only cohort-labeled patients)"
        ),
    )
    args = parser.parse_args()

    if args.clean_unlabeled and args.dataset in ("brats2023", "all"):
        out = (
            Path(args.output_dir)
            if args.output_dir
            else DATA_RAW_DIR / "brats2023"
        )
        clean_brats_unlabeled(out)
        return

    if args.dataset in ("brats2023", "all"):
        out = (
            Path(args.output_dir)
            if args.output_dir
            else DATA_RAW_DIR / "brats2023"
        )
        download_brats2023(out)

    if args.dataset in ("ucsf_pdgm", "all"):
        out = (
            Path(args.output_dir)
            if args.output_dir
            else DATA_RAW_DIR / "ucsf_pdgm"
        )
        download_ucsf_pdgm(out)


if __name__ == "__main__":
    main()
