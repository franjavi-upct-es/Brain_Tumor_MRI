# tests/test_data/test_label_derivation.py — Tests for grade label derivation
"""
Verifies that label derivation logic correctly maps:
- BraTS metadata CSV → HGG/LGG binary labels
- UCSF-PDGM clinical CSV → binary labels (Grade II-III=LGG, IV=HGG)
- Segmentation-based heuristic (enhancing tumor threshold)
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from src.data.label_derivation import (
    derive_brats_labels_from_metadata,
    load_ucsf_labels,
    save_labels,
)


class TestBraTSLabelDerivation:
    """Tests for BraTS grade label derivation from metadata CSV."""

    def _create_metadata_csv(self, tmpdir: Path, rows: list[dict]) -> Path:
        """Create a temporary BraTS metadata CSV."""
        csv_path = tmpdir / "name_mapping.csv"
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    def test_hgg_labels(self) -> None:
        """HGG/GBM/IV/4 should all map to label 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"BraTS2023ID": "p001", "Grade": "HGG"},
                {"BraTS2023ID": "p002", "Grade": "GBM"},
                {"BraTS2023ID": "p003", "Grade": "IV"},
                {"BraTS2023ID": "p004", "Grade": "4"},
            ]
            csv_path = self._create_metadata_csv(Path(tmpdir), rows)
            labels = derive_brats_labels_from_metadata(csv_path)

            for pid in ["p001", "p002", "p003", "p004"]:
                assert labels[pid] == 1, f"{pid} should be HGG (1)"

    def test_lgg_labels(self) -> None:
        """LGG/LOW/II/III/2/3 should all map to label 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"BraTS2023ID": "p001", "Grade": "LGG"},
                {"BraTS2023ID": "p002", "Grade": "LOW"},
                {"BraTS2023ID": "p003", "Grade": "II"},
                {"BraTS2023ID": "p004", "Grade": "III"},
                {"BraTS2023ID": "p005", "Grade": "2"},
                {"BraTS2023ID": "p006", "Grade": "3"},
            ]
            csv_path = self._create_metadata_csv(Path(tmpdir), rows)
            labels = derive_brats_labels_from_metadata(csv_path)

            for pid in ["p001", "p002", "p003", "p004", "p005", "p006"]:
                assert labels[pid] == 0, f"{pid} should be LGG (0)"

    def test_case_insensitive(self) -> None:
        """Grade strings should be case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"BraTS2023ID": "p001", "Grade": "hgg"},
                {"BraTS2023ID": "p002", "Grade": "Lgg"},
            ]
            csv_path = self._create_metadata_csv(Path(tmpdir), rows)
            labels = derive_brats_labels_from_metadata(csv_path)
            assert labels["p001"] == 1
            assert labels["p002"] == 0

    def test_missing_file_raises(self) -> None:
        """Non-existent metadata file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            derive_brats_labels_from_metadata(Path("/nonexistent/file.csv"))

    def test_unknown_grades_skipped(self) -> None:
        """Unknown grade values should be skipped (not crash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"BraTS2023ID": "p001", "Grade": "HGG"},
                {"BraTS2023ID": "p002", "Grade": "UNKNOWN"},
                {"BraTS2023ID": "p003", "Grade": ""},
            ]
            csv_path = self._create_metadata_csv(Path(tmpdir), rows)
            labels = derive_brats_labels_from_metadata(csv_path)
            assert "p001" in labels
            assert "p002" not in labels
            assert "p003" not in labels


class TestUCSFLabels:
    """Tests for UCSF-PDGM label loading."""

    def _create_ucsf_csv(self, tmpdir: Path, rows: list[dict]) -> Path:
        """Create a temporary UCSF-PDGM metadata CSV."""
        csv_path = tmpdir / "clinical_metadata.csv"
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    def test_grade_mapping(self) -> None:
        """Grade II-III → LGG (0), Grade IV → HGG (1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {
                    "patient_id": "UCSF-001",
                    "grade": "2",
                    "idh_status": "mutant",
                    "mgmt_status": "methylated",
                },
                {
                    "patient_id": "UCSF-002",
                    "grade": "3",
                    "idh_status": "mutant",
                    "mgmt_status": "unmethylated",
                },
                {
                    "patient_id": "UCSF-003",
                    "grade": "4",
                    "idh_status": "wildtype",
                    "mgmt_status": "methylated",
                },
            ]
            csv_path = self._create_ucsf_csv(Path(tmpdir), rows)
            metadata = load_ucsf_labels(csv_path)

            assert metadata["UCSF-001"]["binary_label"] == 0  # Grade II → LGG
            assert metadata["UCSF-002"]["binary_label"] == 0  # Grade III → LGG
            assert metadata["UCSF-003"]["binary_label"] == 1  # Grade IV → HGG

    def test_preserves_molecular_markers(self) -> None:
        """Should preserve IDH and MGMT status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {
                    "patient_id": "UCSF-001",
                    "grade": "4",
                    "idh_status": "wildtype",
                    "mgmt_status": "methylated",
                },
            ]
            csv_path = self._create_ucsf_csv(Path(tmpdir), rows)
            metadata = load_ucsf_labels(csv_path)
            assert metadata["UCSF-001"]["idh"] == "wildtype"
            assert metadata["UCSF-001"]["mgmt"] == "methylated"

    def test_invalid_grade_skipped(self) -> None:
        """Non-integer or invalid grades should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {
                    "patient_id": "UCSF-001",
                    "grade": "4",
                    "idh_status": "",
                    "mgmt_status": "",
                },
                {
                    "patient_id": "UCSF-002",
                    "grade": "invalid",
                    "idh_status": "",
                    "mgmt_status": "",
                },
                {
                    "patient_id": "UCSF-003",
                    "grade": "5",
                    "idh_status": "",
                    "mgmt_status": "",
                },
            ]
            csv_path = self._create_ucsf_csv(Path(tmpdir), rows)
            metadata = load_ucsf_labels(csv_path)
            assert "UCSF-001" in metadata
            assert "UCSF-002" not in metadata  # Invalid grade
            assert "UCSF-003" not in metadata  # Grade 5 not in mapping


class TestSaveLabels:
    """Tests for label saving utility."""

    def test_save_and_reload(self) -> None:
        """Saved labels should be loadable as valid JSON."""
        labels = {"p001": 1, "p002": 0, "p003": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "labels.json"
            save_labels(labels, path)

            with open(path) as f:
                loaded = json.load(f)

            assert loaded == labels

    def test_creates_parent_dirs(self) -> None:
        """Should create parent directories if they don't exist."""
        labels = {"p001": 0}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "labels.json"
            save_labels(labels, path)
            assert path.exists()
