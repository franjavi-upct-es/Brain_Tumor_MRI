# src/data/datamodule.py
# PyTorch Lightning Data Module for brain tumor classification
"""
Lightning DataModule that integrates BraTS/UCSF-PDGM datasets with
patient-level splitting and MRI-specific augmentation transforms.

Handles the complete data pipeline:
1. Load processed NIfTI data via MONAI/nibabel.
2. Apply patient-level StratifiedGroupKFold splitting (no data leakage).
3. Apply TorchIO augmentation during training only.
4. Provide DataLoaders for training, validation, and external test sets.

The DataModule if fold-aware: set the active fold before calling setup()
to get the corresponding train/val split.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.splitter import PatientLevelSplitter
from src.data.transforms import build_train_transforms, build_val_transforms

logger = logging.getLogger(__name__)


class SliceDataset(Dataset):
    """
    Dataset of 2D MRI slices with labels for classification.

    Wraps preprocessed 2D slices (already extracted from 3D volumes)
    and applies optional augmentation transforms.

    Parameters
    ----------
    slices : list[dict]
        List of sample dictionaries, each containing:
        - "image": np.ndarray or torch.Tensor of shape (C, H, W)
        - "label": int (class label)
        - "patient_id": str
        - "slice_index": int (original slice position in 3D volume)
    transform : callable, optional
        TorchIO or other transform to apply to each sample.
    """

    def __init__(
        self, slices: list[dict[str, Any]], transform: Any | None = None
    ) -> None:
        self.slices = slices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.slices[idx].copy()
        image = sample["image"]

        # Ensure tensor format
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        elif not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        # Apply transform if provided
        if self.transform is not None:
            # TorchIO expects 4D tensors (C, H, W, D); add dummy D dim
            image_4d = image.unsqueeze(-1)
            import torchio as tio

            subject = tio.Subject(image=tio.ScalarImage(tensor=image_4d))
            transformed = self.transform(subject)
            image = transformed["image"].data.squeeze(-1)

        sample["image"] = image
        sample["label"] = torch.tensor(sample["label"], dtype=torch.long)

        return sample

    def get_patient_ids(self) -> list[str]:
        """Return all patient IDs in this dataset."""
        return [s["patient_id"] for s in self.slices]

    def get_labels(self) -> np.ndarray:
        """Return all labels as a numpy array."""
        return np.array([s["label"] for s in self.slices])


class BrainTumorDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for brain tumor glioma grading.

    Manages the complete data pipeline with patient-level splitting,
    fold-aware setup, and proper augmentation isolation.

    Parameters
    ----------
    data_dir : str or Path
        Path to directory with preprocessed patient data.
    splits_path : str or Path
        Path to JSON file with pre-computed patient-level splits.
    batch_size : int
        Batch size for DataLoaders. Default 32.
    num_workers : int
        Number of DataLoader workers. Default 4.
    pin_memory : bool
        Whether to pun memory for GPU transfer. Default True.
    fold : int
        Active cross-validation fold (0 to n_splits-1). Default 0.
    train_transform_cfg : dict, optional
        Configuration for training augmentation transforms.
    """

    def __init__(
        self,
        data_dir: str | Path,
        splits_path: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        fold: int = 0,
        train_transform_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.splits_path = Path(splits_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.fold = fold
        self.train_transform_cfg = train_transform_cfg

        # Will be populated in setup()
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self._all_samples: list[dict[str, Any]] | None = None
        self._splits_data: dict[str, Any] | None = None

    def prepare_data(self) -> None:
        """
        Verify that data and splits exist.

        Called only on a single process. Does not assign state.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}. "
                f"Run preprocessing first."
            )
        if not self.splits_path.exists():
            raise FileNotFoundError(
                f"Splits file not found: {self.splits_path}. "
                f"Run split creation first."
            )

    def setup(self, stage: str | None = None) -> None:
        """
        Load data and create train/val datasets for the active fold.

        Parameters
        ----------
        stage : str, optional
            Either "fit", "validate", "test", or "predict".
        """
        # Load splits
        self._splits_data = PatientLevelSplitter.load_splits(self.splits_path)

        # Load all preprocessed sampels
        if self._all_samples is None:
            self._all_samples = self._load_all_samples()

        fold_info = self._splits_data["folds"][self.fold]
        train_indices = fold_info["train_indices"]
        val_indices = fold_info["val_indices"]

        # Build transforms
        train_transform = build_train_transforms(self.train_transform_cfg)
        val_transform = build_val_transforms()

        if stage in ("fit", None):
            train_samples = [self._all_samples[i] for i in train_indices]
            val_samples = [self._all_samples[i] for i in val_indices]

            self.train_dataset = SliceDataset(
                slices=train_samples, transform=train_transform
            )
            self.val_dataset = SliceDataset(
                slices=val_samples,
                transform=val_transform,
            )

            # Verify no leakage after dataset creation
            train_patients = set(self.train_dataset.get_patient_ids())
            val_patients = set(self.val_dataset.get_patient_ids())
            PatientLevelSplitter.verify_no_leakage(
                train_patients, val_patients, fold_name=f"fold_{self.fold}"
            )

            logger.info(
                "Fold %d setup complete: %d train samples, %d val samples",
                self.fold,
                len(self.train_dataset),
                len(self.val_dataset),
            )

        if stage in ("test", None):
            # Test dataset uses validation transform (no augmentation)
            if self.val_dataset is None:
                val_samples = [self._all_samples[i] for i in val_indices]
                self.val_dataset = SliceDataset(
                    slices=val_samples, transform=val_transform
                )

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader with shuffling."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader without shuffling."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader without shuffling."""
        dataset = (
            self.test_dataset
            if self.test_dataset is not None
            else self.val_dataset
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def set_fold(self, fold: int) -> None:
        """
        Change the active fold and reset datasets.

        Parameters
        ----------
        fold : int
            Fold index (0 to n_splits-1)
        """
        if self._splits_data is not None:
            n_folds = len(self._splits_data["folds"])
            if fold < 0 or fold >= n_folds:
                raise ValueError(f"Fold {fold} out of range [0, {n_folds}]")
        self.fold = fold
        self.train_dataset = None
        self.val_dataset = None
        logger.info("Active fold set to %d", fold)

    def set_external_test(self, test_samples: list[dict[str, Any]]) -> None:
        """
        Set an external test dataset (e.g., UCSF-PDGM).

        Parameters
        ----------
        test_samples : list[dict]
            List of sample dictionaries from the external dataset.
        """
        self.test_dataset = SliceDataset(
            slices=test_samples,
            transform=build_val_transforms(),
        )
        logger.info(
            "External test dataset set: %d samples", len(self.test_dataset)
        )

    def _load_all_samples(self) -> list[dict[str, Any]]:
        """
        Load all preprocessed 2D slices from the data directory.

        Expects a manifest JSON file in the data directory listing all
        preprocessed samples with their paths, labels, and patient IDs.

        Returns
        -------
        list[dict]
            List of sample dictionaries ready for SliceDataset.
        """
        manifest_path = self.data_dir / "manifest.json"

        if manifest_path.exists():
            return self._load_from_manifest(manifest_path)

        # Fallback: load NPZ files directly from directory structure
        return self._load_from_directory()

    def _load_from_manifest(self, manifest_path: Path) -> list[dict[str, Any]]:
        """
        Load samples listed in a manifest JSON file.

        Parameters
        ----------
        manifest_path : Path
            Path to manifest.json with sample metadata

        Returns:
        list[dict]:
            Loaded sample dictionaries.
        """
        with open(manifest_path) as f:
            manifest = json.load(f)

        samples = []
        for entry in manifest["samples"]:
            npz_path = self.data_dir / entry["file"]
            data = np.load(npz_path)

            sample = {
                "image": data["image"],  # Shape: (C, H, W)
                "label": int(entry["label"]),
                "patient_id": entry["patient_id"],
                "slice_index": entry.get("slice_index", -1),
            }
            samples.append(sample)

        logger.info("Loaded %d samples from manifest", len(samples))
        return samples

    def _load_from_directory(self) -> list[dict[str, Any]]:
        """
        Load samples from directory of NPZ files.

        Expects files named: {patient_id}_slice{idx}.npz
        Each NPZ contains 'image' array and 'label' scalar.

        Returns
        -------
        list[dict]
            Loaded sample dictionaries.
        """
        npz_files = sorted(self.data_dir.glob("*.npz"))

        if not npz_files:
            logger.warning("No .npz files found in %s", self.data_dir)
            return []

        samples = []
        for npz_path in npz_files:
            data = np.load(npz_path, allow_pickle=True)
            patient_id = str(
                data.get("patient_id", npz_path.stem.split("_slice")[0])
            )

            sample = {
                "image": data["image"],
                "label": int(data["label"]),
                "patient_id": patient_id,
                "slice_index": int(data.get("slice_index", -1)),
            }
            samples.append(sample)

        logger.info("Loaded %s samples from directory", len(samples))
        return samples
