# tests/test_data/test_no_leakage.py — CRITICAL: No data leakage verification
"""
CRITICAL TEST: Verifies that no patient appears in both training and validation
sets across all cross-validation folds.

This is THE most important test in the entire project. Data leakage through
image-level splitting is the primary methodological flaw that inflates accuracy
from ~87% to ~99% on brain MRI classification tasks.

Evidence of the problem:
    - Yagis et al. (2021, Scientific Reports): slice-level CV inflates
      accuracy by 30% (OASIS) to 48% (PPMI).
    - Wen et al. (2020, Medical Image Analysis): >50% of Alzheimer's
      classification papers had data leakage.
    - Roberts et al. (2021, Nature Machine Intelligence): 0/62 COVID ML
      studies were clinically useful — primarily due to data leakage.

This test must ALWAYS pass. Failure means the entire experimental pipeline
is compromised.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.splitter import PatientLevelSplitter


class TestNoDataLeakage:
    """
    Test suite verifying zero patient overlap between train/val splits.

    Every test in this class validates a different aspect of the leakage
    prevention mechanism. ALL must pass for the pipeline to be trusted.
    """

    def _create_synthetic_dataset(
        self,
        n_patients: int = 50,
        slices_per_patient: int = 10,
        n_classes: int = 2,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a synthetic dataset mimicking BraTS structure.

        Each patient has multiple slices, simulating the real scenario
        where multiple 2D slices come from one 3D MRI volume.

        Parameters
        ----------
        n_patients : int
            Number of unique patients.
        slices_per_patient : int
            Number of 2D slices per patient.
        n_classes : int
            Number of classes (2 for binary grading).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        tuple
            (patient_ids, labels, sample_indices) arrays.
        """
        rng = np.random.RandomState(seed)

        patient_ids = []
        labels = []

        for p in range(n_patients):
            patient_label = rng.randint(0, n_classes)
            for _ in range(slices_per_patient):
                patient_ids.append(f"patient_{p:04d}")
                labels.append(patient_label)

        return (
            np.array(patient_ids),
            np.array(labels),
            np.arange(len(patient_ids)),
        )

    def test_no_patient_overlap_in_any_fold(self) -> None:
        """
        CRITICAL: No patient appears in both train and val for any fold.

        This is the fundamental test. If this fails, ALL results from the
        pipeline are invalid due to data leakage.
        """
        patient_ids, labels, indices = self._create_synthetic_dataset(
            n_patients=100, slices_per_patient=13
        )

        splitter = PatientLevelSplitter(n_splits=5, random_state=42)
        splits = splitter.create_splits(patient_ids, labels, indices)

        for fold_info in splits:
            fold_idx = fold_info["fold"]
            train_patients = set(fold_info["train_patients"])
            val_patients = set(fold_info["val_patients"])

            overlap = train_patients & val_patients
            assert len(overlap) == 0, (
                f"DATA LEAKAGE in fold {fold_idx}! "
                f"{len(overlap)} patients in both train and val: {overlap}"
            )

    def test_all_slices_from_patient_in_same_fold(self) -> None:
        """
        All slices from one patient must be in the same fold.

        If patient_001 has 13 slices, ALL 13 must be in either train or val
        — never split across them. This prevents the model from memorizing
        patient-specific features during training and then being tested on
        different slices from the same patient.
        """
        patient_ids, labels, indices = self._create_synthetic_dataset(
            n_patients=50, slices_per_patient=13
        )

        splitter = PatientLevelSplitter(n_splits=5, random_state=42)
        splits = splitter.create_splits(patient_ids, labels, indices)

        for fold_info in splits:
            fold_idx = fold_info["fold"]
            train_idx = set(fold_info["train_indices"])
            val_idx = set(fold_info["val_indices"])

            # For each unique patient, check all their slices are together
            unique_patients = set(patient_ids)
            for patient in unique_patients:
                patient_mask = patient_ids == patient
                patient_indices = set(np.where(patient_mask)[0].tolist())

                in_train = patient_indices & train_idx
                in_val = patient_indices & val_idx

                # All slices must be in exactly one set
                assert len(in_train) == 0 or len(in_val) == 0, (
                    f"Fold {fold_idx}: Patient {patient} has "
                    f"{len(in_train)} slices in train and "
                    f"{len(in_val)} slices in val. "
                    f"ALL slices must be in the same set!"
                )

    def test_every_patient_appears_in_validation_exactly_once(self) -> None:
        """
        Each patient must appear in validation exactly once across all folds.

        In 5-fold CV, each fold uses ~20% of patients for validation. Across
        all 5 folds, every patient should be validated exactly once.
        """
        patient_ids, labels, indices = self._create_synthetic_dataset(
            n_patients=100, slices_per_patient=5
        )

        splitter = PatientLevelSplitter(n_splits=5, random_state=42)
        splits = splitter.create_splits(patient_ids, labels, indices)

        # Collect validation patients across all folds
        val_patient_counts: dict[str, int] = {}
        for fold_info in splits:
            for patient in fold_info["val_patients"]:
                val_patient_counts[patient] = (
                    val_patient_counts.get(patient, 0) + 1
                )

        # Every unique patient should appear exactly once in validation
        unique_patients = set(patient_ids)
        for patient in unique_patients:
            count = val_patient_counts.get(patient, 0)
            assert count == 1, (
                f"Patient {patient} appears in validation {count} times "
                f"(expected exactly 1 across 5 folds)"
            )

    def test_stratification_preserves_class_proportions(self) -> None:
        """
        Class proportions should be approximately maintained in each fold.

        With stratified splitting, the ratio of HGG:LGG should be similar
        in training and validation for every fold.
        """
        # Create imbalanced dataset (70% HGG, 30% LGG like real BraTS)
        rng = np.random.RandomState(42)
        n_patients = 200
        patient_ids = []
        labels = []

        for p in range(n_patients):
            patient_label = 1 if rng.random() < 0.7 else 0  # 70% HGG
            slices = rng.randint(5, 15)  # Variable slices per patient
            for _ in range(slices):
                patient_ids.append(f"patient_{p:04d}")
                labels.append(patient_label)

        patient_ids = np.array(patient_ids)
        labels = np.array(labels)

        splitter = PatientLevelSplitter(n_splits=5, random_state=42)
        splits = splitter.create_splits(patient_ids, labels)

        overall_ratio = np.mean(labels)

        for fold_info in splits:
            val_labels = labels[fold_info["val_indices"]]
            val_ratio = np.mean(val_labels)

            # Allow ±15% deviation from overall ratio (generous tolerance
            # for small fold sizes with grouped splitting)
            assert abs(val_ratio - overall_ratio) < 0.15, (
                f"Fold {fold_info['fold']}: Val class ratio {val_ratio:.3f} "
                f"deviates too much from overall ratio {overall_ratio:.3f}"
            )

    def test_no_empty_folds(self) -> None:
        """No fold should have empty training or validation sets."""
        patient_ids, labels, indices = self._create_synthetic_dataset(
            n_patients=50, slices_per_patient=10
        )

        splitter = PatientLevelSplitter(n_splits=5, random_state=42)
        splits = splitter.create_splits(patient_ids, labels, indices)

        for fold_info in splits:
            assert fold_info["n_train_samples"] > 0, (
                f"Fold {fold_info['fold']} has empty training set!"
            )
            assert fold_info["n_val_samples"] > 0, (
                f"Fold {fold_info['fold']} has empty validation set!"
            )
            assert fold_info["n_train_patients"] > 0, (
                f"Fold {fold_info['fold']} has zero training patients!"
            )
            assert fold_info["n_val_patients"] > 0, (
                f"Fold {fold_info['fold']} has zero validation patients!"
            )

    def test_splits_are_reproducible_with_same_seed(self) -> None:
        """
        Same seed must produce identical splits for reproducibility.

        This is essential for DVC pipeline reproducibility — running the
        pipeline twice with the same seed must yield the same results.
        """
        patient_ids, labels, indices = self._create_synthetic_dataset()

        splitter_a = PatientLevelSplitter(n_splits=5, random_state=42)
        splits_a = splitter_a.create_splits(patient_ids, labels, indices)

        splitter_b = PatientLevelSplitter(n_splits=5, random_state=42)
        splits_b = splitter_b.create_splits(patient_ids, labels, indices)

        for fold_a, fold_b in zip(splits_a, splits_b, strict=False):
            assert fold_a["train_indices"] == fold_b["train_indices"], (
                f"Fold {fold_a['fold']}: Train indices differ between runs "
                f"with same seed!"
            )
            assert fold_a["val_indices"] == fold_b["val_indices"], (
                f"Fold {fold_a['fold']}: Val indices differ between runs "
                f"with same seed!"
            )

    def test_different_seeds_produce_different_splits(self) -> None:
        """Different seeds should produce different splits."""
        patient_ids, labels, indices = self._create_synthetic_dataset()

        splitter_a = PatientLevelSplitter(n_splits=5, random_state=42)
        splits_a = splitter_a.create_splits(patient_ids, labels, indices)

        splitter_b = PatientLevelSplitter(n_splits=5, random_state=123)
        splits_b = splitter_b.create_splits(patient_ids, labels, indices)

        # At least one fold should have different indices
        any_different = False
        for fold_a, fold_b in zip(splits_a, splits_b, strict=False):
            if fold_a["train_indices"] != fold_b["train_indices"]:
                any_different = True
                break

        assert any_different, "Different seeds produced identical splits!"

    def test_save_and_load_preserves_splits(self) -> None:
        """Splits saved to JSON and reloaded must be identical."""
        patient_ids, labels, indices = self._create_synthetic_dataset()

        splitter = PatientLevelSplitter(n_splits=5, random_state=42)
        splits = splitter.create_splits(patient_ids, labels, indices)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_splits.json"
            splitter.save_splits(save_path)

            loaded = PatientLevelSplitter.load_splits(save_path)

            assert loaded["n_splits"] == 5
            assert loaded["random_state"] == 42
            assert len(loaded["folds"]) == 5

            for fold_orig, fold_loaded in zip(
                splits, loaded["folds"], strict=False
            ):
                assert (
                    fold_orig["train_indices"] == fold_loaded["train_indices"]
                )
                assert fold_orig["val_indices"] == fold_loaded["val_indices"]
                assert (
                    fold_orig["train_patients"]
                    == fold_loaded["train_patients"]
                )
                assert fold_orig["val_patients"] == fold_loaded["val_patients"]

    def test_leakage_detection_raises_on_overlap(self) -> None:
        """The verify_no_leakage() method must raise on patient overlap."""
        # Simulate a leaky split where patient_001 is in both sets
        train_patients = {"patient_001", "patient_002", "patient_003"}
        val_patients = {"patient_001", "patient_004", "patient_005"}

        with pytest.raises(AssertionError, match="DATA LEAKAGE"):
            PatientLevelSplitter.verify_no_leakage(
                train_patients, val_patients, fold_name="test_fold"
            )

    def test_verify_no_leakage_passes_on_clean_split(self) -> None:
        """The verify_no_leakage() method must pass with no overlap."""
        train_patients = {"patient_001", "patient_002", "patient_003"}
        val_patients = {"patient_004", "patient_005"}

        result = PatientLevelSplitter.verify_no_leakage(
            train_patients, val_patients, fold_name="test_fold"
        )
        assert result is True

    def test_mismatched_lengths_raise_error(self) -> None:
        """Mismatched patient_ids and labels lengths must raise ValueError."""
        patient_ids = np.array(["p1", "p2", "p3"])
        labels = np.array([0, 1])  # Wrong length

        splitter = PatientLevelSplitter(n_splits=2)
        with pytest.raises(ValueError, match="same length"):
            splitter.create_splits(patient_ids, labels)

    def test_single_slice_per_patient_still_works(self) -> None:
        """
        Splitting must work even with exactly one slice per patient.

        This simulates a dataset where each patient contributes exactly one
        2D slice (e.g., after selecting only the max-tumor-area slice).
        """
        patient_ids, labels, indices = self._create_synthetic_dataset(
            n_patients=100, slices_per_patient=1
        )

        splitter = PatientLevelSplitter(n_splits=5, random_state=42)
        splits = splitter.create_splits(patient_ids, labels, indices)

        for fold_info in splits:
            train_patients = set(fold_info["train_patients"])
            val_patients = set(fold_info["val_patients"])
            assert len(train_patients & val_patients) == 0
