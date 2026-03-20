# tests/test_data/test_splitter.py — Unit tests for patient-level splitter
"""
Tests for the PatientLevelSplitter class covering edge cases, error handling,
and correctness of the StratifiedGroupKFold wrapper.

These tests complement test_no_leakage.py which focuses exclusively on
leakage detection. This module covers the broader API surface.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.splitter import PatientLevelSplitter, _class_distribution


class TestPatientLevelSplitter:
    """Tests for PatientLevelSplitter correctness and edge cases."""

    def test_default_initialization(self) -> None:
        """Default splitter uses 5 folds, shuffle=True, seed=42."""
        splitter = PatientLevelSplitter()
        assert splitter.n_splits == 5
        assert splitter.shuffle is True
        assert splitter.random_state == 42

    def test_custom_initialization(self) -> None:
        """Custom parameters are stored correctly."""
        splitter = PatientLevelSplitter(
            n_splits=3, shuffle=True, random_state=123
        )
        assert splitter.n_splits == 3
        assert splitter.shuffle is True
        assert splitter.random_state == 123

    def test_creates_correct_number_of_folds(self) -> None:
        """Output contains exactly n_splits folds."""
        patient_ids = np.array(
            ["p1"] * 5 + ["p2"] * 5 + ["p3"] * 5 + ["p4"] * 5 + ["p5"] * 5
        )
        labels = np.array([0] * 10 + [1] * 15)

        for n_splits in [2, 3, 5]:
            splitter = PatientLevelSplitter(n_splits=n_splits, random_state=42)
            splits = splitter.create_splits(patient_ids, labels)
            assert len(splits) == n_splits

    def test_fold_info_structure(self) -> None:
        """Each fold dict contains all required keys."""
        patient_ids = np.array(
            ["p1"] * 3 + ["p2"] * 3 + ["p3"] * 3 + ["p4"] * 3 + ["p5"] * 3
        )
        labels = np.array([0] * 6 + [1] * 9)

        splitter = PatientLevelSplitter(n_splits=5, random_state=42)
        splits = splitter.create_splits(patient_ids, labels)

        required_keys = {
            "fold",
            "train_indices",
            "val_indices",
            "train_patients",
            "val_patients",
            "n_train_samples",
            "n_val_samples",
            "n_train_patients",
            "n_val_patients",
            "train_class_distribution",
            "val_class_distribution",
        }

        for fold_info in splits:
            assert set(fold_info.keys()) == required_keys

    def test_all_samples_covered(self) -> None:
        """Every sample index appears exactly once across train + val."""
        patient_ids = np.array(
            ["p1"] * 5 + ["p2"] * 5 + ["p3"] * 5 + ["p4"] * 5 + ["p5"] * 5
        )
        labels = np.array([0] * 10 + [1] * 15)
        n_samples = len(patient_ids)

        splitter = PatientLevelSplitter(n_splits=5, random_state=42)
        splits = splitter.create_splits(patient_ids, labels)

        for fold_info in splits:
            all_indices = sorted(
                fold_info["train_indices"] + fold_info["val_indices"]
            )
            assert all_indices == list(range(n_samples)), (
                f"Fold {fold_info['fold']}: Not all samples covered"
            )

    def test_class_distribution_counts(self) -> None:
        """Class distribution helper returns correct counts."""
        labels = np.array([0, 0, 0, 1, 1, 2])
        dist = _class_distribution(labels)

        assert dist == {"0": 3, "1": 2, "2": 1}

    def test_save_without_create_raises(self) -> None:
        """Saving before creating splits must raise RuntimeError."""
        splitter = PatientLevelSplitter()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="No splits created"):
                splitter.save_splits(Path(tmpdir) / "splits.json")

    def test_save_creates_parent_directories(self) -> None:
        """save_splits must create intermediate directories."""
        patient_ids = np.array(
            ["p1"] * 3 + ["p2"] * 3 + ["p3"] * 3 + ["p4"] * 3 + ["p5"] * 3
        )
        labels = np.array([0] * 6 + [1] * 9)

        splitter = PatientLevelSplitter(n_splits=5, random_state=42)
        splitter.create_splits(patient_ids, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "splits.json"
            splitter.save_splits(nested_path)
            assert nested_path.exists()

    def test_load_splits_returns_valid_structure(self) -> None:
        """Loaded splits JSON has correct structure."""
        patient_ids = np.array(
            ["p1"] * 3 + ["p2"] * 3 + ["p3"] * 3 + ["p4"] * 3 + ["p5"] * 3
        )
        labels = np.array([0] * 6 + [1] * 9)

        splitter = PatientLevelSplitter(n_splits=5, random_state=42)
        splitter.create_splits(patient_ids, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "splits.json"
            splitter.save_splits(path)
            loaded = PatientLevelSplitter.load_splits(path)

            assert "splitter" in loaded
            assert loaded["splitter"] == "StratifiedGroupKFold"
            assert loaded["n_splits"] == 5
            assert loaded["random_state"] == 42
            assert len(loaded["folds"]) == 5

    def test_handles_many_patients(self) -> None:
        """Splitter works correctly with large patient counts."""
        rng = np.random.RandomState(42)
        n_patients = 1000
        patient_ids = []
        labels = []

        for p in range(n_patients):
            n_slices = rng.randint(1, 20)
            label = rng.randint(0, 2)
            patient_ids.extend([f"patient_{p:05d}"] * n_slices)
            labels.extend([label] * n_slices)

        patient_ids = np.array(patient_ids)
        labels = np.array(labels)

        splitter = PatientLevelSplitter(n_splits=5, random_state=42)
        splits = splitter.create_splits(patient_ids, labels)

        assert len(splits) == 5
        for fold_info in splits:
            train_p = set(fold_info["train_patients"])
            val_p = set(fold_info["val_patients"])
            assert len(train_p & val_p) == 0
