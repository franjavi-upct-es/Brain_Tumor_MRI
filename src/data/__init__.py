# src/data/__init__.py — Data loading, preprocessing, and splitting subpackage
"""
Provides MONAI-based dataset loaders for BraTS 2023 and UCSF-PDGM,
patient-level splitting via StratifiedGroupKFold, MRI-specific transforms
using TorchIO, and a PyTorch Lightning DataModule for the training pipeline.
"""

from src.data.splitter import PatientLevelSplitter

# Lazy imports for modules with heavy dependencies (Lightning, MONAI)
# to allow test and utility code to import the splitter without requiring
# the full training stack to be installed.


def __getattr__(name: str):
    """Lazy import for heavy-dependency modules."""
    if name == "BraTSDataset":
        from src.data.brats_dataset import BraTSDataset

        return BraTSDataset
    if name == "UCSFPDGMDataset":
        from src.data.ucsf_dataset import UCSFPDGMDataset

        return UCSFPDGMDataset
    if name == "BrainTumorDataModule":
        from src.data.datamodule import BrainTumorDataModule

        return BrainTumorDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BraTSDataset",
    "UCSFPDGMDataset",
    "PatientLevelSplitter",
    "BrainTumorDataModule",
]
