# src/data/splitter.py — Patient-level StratifiedGroupKFold splitter
"""
Implements patient-level data splitting to prevent data leakage.

Data leakage through image-level splitting is the primary methodological flaw
exposed in this thesis. Yagis et al. (2021, Scientific Reports) demonstrated
that slice-level CV inflates accuracy by 30-48% on brain MRI data. This module
ensures ALL images from one patient remain in the same fold.

Uses sklearn.model_selection.StratifiedGroupKFold (>=1.0) which maintians class
proportions while keeping all images from the same patient group together.

References:
    - Yagis et al. (2021). Effect of data leakage in brain MRI classification
      using 2D CNNs. Scientific Reports, 11, 22544.
    - Cawley & Talbot (2010). On over-fitting in model selection. JMLR.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)


class PatientLevelSplitter:
    """
    Patient-level stratified k-fold cross-validation splitter.

    Guarantees that:
    1. All slices/images from one patient stay in the same fold.
    2. Class proportions (tumpor grades) are preserved across folds.
    3. No patient appears in both training and validation sets.

    Parameters
    ----------
    n_splits : int
      Number of cross-validation folds. Default is 5.
    shuffle : bool
      Whether to shuffle before splitting. Default is True.
    random_state : int
      Random seed for reproducibility. Default is 42.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self._splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
        self._splits: list[dict[str, Any]] | None = None

    def create_splits(
        self,
        patient_ids: np.ndarray,
        labels: np.ndarray,
        sample_indices: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """
        Create patient-level stratified k-fold splits.

        Parameters
        ----------
        patient_ids : np.ndarray
            Array of patient identifiers for each sample. All samples with the
            same patient ID will be assigned to the same fold.
        labels : np.ndarray
            Class labels for stratification (e.g., tumor grade).
        sample_indices : np.ndarray, optional
            Sample indices to split. If None, uses range(len(patient_ids)).

        Returns
        -------
        list[dict[str, Any]]
            List of fold dictionaries containing train/val indices, patient
            IDs, and class distribution statistics.

        Raises
        ------
        ValueError
            If patient_ids and labels have different lengths.
        AssertionError:
            If data leakage is detected (patient overlap between folds).
        """
        if len(patient_ids) != len(labels):
            msg = (
                f"patient_ids ({len(patient_ids)}) and labels ({len(labels)}) "
                "must have the same length."
            )
            raise ValueError(msg)

        if sample_indices is None:
            sample_indices = np.arange(len(patient_ids))

        splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            self._splitter.split(sample_indices, labels, groups=patient_ids)
        ):
            # CRITICAL VERIFICATION: no patient overlap between train and val
            train_patients = set(patient_ids[train_idx])
            val_patients = set(patient_ids[val_idx])
            overlap = train_patients & val_patients

            assert len(overlap) == 0, (
                f"DATA LEAKAGE DETECTED in fold {fold_idx}! "
                f"Patients in both train and val: {overlap}"
            )

            # Compute class distribution per fold for logging
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]

            fold_info = {
                "fold": fold_idx,
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "train_patients": sorted(str(p) for p in train_patients),
                "val_patients": sorted(str(p) for p in val_patients),
                "n_train_samples": len(train_idx),
                "n_val_samples": len(val_idx),
                "n_train_patients": len(train_patients),
                "n_val_patients": len(val_patients),
                "train_class_distribution": _class_distribution(train_labels),
                "val_class_distribution": _class_distribution(val_labels),
            }

            logger.info(
                "Fold %d: %d train patients (%d samples), "
                "%d val patients (%d samples) — NO LEAKAGE VERIFIED",
                fold_idx,
                len(train_patients),
                len(train_idx),
                len(val_patients),
                len(val_idx),
            )

            splits.append(fold_info)

        self._splits = splits
        return splits

    def save_splits(self, output_path: str | Path) -> None:
        """
        Save splits to JSON for reproducibility and DVC tracking.

        Parameters
        ----------
        output_path : str or Path
            Path to save the JSON file (e.g., data/splits/brats_5fold.json)
        """
        if self._splits is None:
            msg = "No splits created yet. Call create_splits() first."
            raise RuntimeError(msg)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "splitter": "StratifiedGroupKFold",
            "n_splits": self.n_splits,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "description": (
                "Patient-level stratified k-fold splits. All images from one "
                "patient are in the same fold to prevent data leakage."
            ),
            "folds": self._splits,
        }

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Splits saved to %s", output_path)

    @staticmethod
    def load_splits(splits_path: str | Path) -> dict[str, Any]:
        """
        Load previously saved splits from JSON.

        Parameters
        ----------
        splits_path : str or Path
            Path to the JSON file with split definitions.

        Returns
        -------
        dict[str, Any]
            Dictionary with split metadata and fold information.
        """
        with open(splits_path) as f:
            return json.load(f)

    @staticmethod
    def verify_no_leakage(
        train_patient_ids: set[str],
        val_patient_ids: set[str],
        fold_name: str = "unknown",
    ) -> bool:
        """
        Verify that no patient appears in both train and validation sets.

        This is a standalone verification function that can be called at any
        point in the pipeline as a safety check.

        Parameters
        ----------
        train_patient_ids : set[str]
            Patient IDs in the training set.
        val_patient_ids : set[str]
            Patient IDs in the validation set.
        fold_name : str
            Name/identifier for logging purposes.

        Returns
        -------
        bool
            True if not leakage is detected

        Raises
        ------
        AssertionError
            If patient overlap is detected
        """
        overlap = train_patient_ids & val_patient_ids
        assert len(overlap) == 0, (
            f"DATA LEAKAGE in {fold_name}: "
            f"{len(overlap)} patients in both sets: {overlap}"
        )
        logger.info("No leakage verified for %s", fold_name)
        return True


def _class_distribution(labels: np.ndarray) -> dict[str, int]:
    """
    Compute class frequency distribution.

    Parameters
    ----------
    labels : np.ndarray
        Array of class labels.

    Returns
    -------
    dict[str, int]
        Mapping from class label to count.
    """
    unique, counts = np.unique(labels, return_counts=True)
    return {
        str(label): int(count)
        for label, count in zip(unique, counts, strict=False)
    }
