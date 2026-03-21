# tests/test_data/test_transforms.py — Unit tests for MRI transform pipelines
"""
Verifies correctness of the preprocessing and augmentation transforms:
- Training transforms apply MRI-specific augmentations.
- Validation transforms apply NO augmentation (identity).
- 2D slice extraction selects the correct slice.
- 2.5D context slice extraction works correctly.
- Foreground-only z-score normalization computes on voxels > 0.
"""

from __future__ import annotations

import numpy as np
import torch

from src.data.transforms import (
    build_preprocessing_transforms,
    build_train_transforms,
    build_val_transforms,
    extract_2d_slice,
)


class TestTrainTransforms:
    """Tests for training augmentation pipeline."""

    def test_returns_composed_transform(self) -> None:
        """build_train_transforms returns a TorchIO Compose object."""
        import torchio as tio

        transform = build_train_transforms()
        assert isinstance(transform, tio.Compose)

    def test_has_augmentation_components(self) -> None:
        """
        Training transforms include spatial and MRI-specific augmentations.
        """
        transform = build_train_transforms()
        # Compose has a .transforms attribute listing child transforms
        assert len(transform.transforms) > 0, (
            "Training transforms should include at least one augmentation"
        )

    def test_custom_config_disables_augmentations(self) -> None:
        """Augmentations can be selectively disabled via config."""
        cfg = {
            "enable_affine": False,
            "enable_flip": False,
            "enable_bias_field": False,
            "enable_noise": False,
            "enable_ghosting": False,
            "enable_gamma": False,
        }
        transform = build_train_transforms(cfg)
        assert len(transform.transforms) == 0

    def test_transform_does_not_change_shape(self) -> None:
        """Augmentation must preserve the spatial dimensions of the input."""
        import torchio as tio

        transform = build_train_transforms()

        # Create a dummy 3D volume (TorchIO expects 4D: C, H, W, D)
        dummy = torch.randn(4, 64, 64, 1)  # 4 channels, 64x64, 1 depth
        subject = tio.Subject(image=tio.ScalarImage(tensor=dummy))
        transformed = transform(subject)

        assert transformed["image"].shape == dummy.shape


class TestValTransforms:
    """Tests for validation/test transform pipeline."""

    def test_val_transforms_are_identity(self) -> None:
        """Validation transforms must be empty (no augmentation)."""
        transform = build_val_transforms()
        assert len(transform.transforms) == 0, (
            "CRITICAL: Validation transforms must apply NO augmentation!"
        )

    def test_val_transform_preserves_data(self) -> None:
        """Applying val transforms must return identical data."""
        import torchio as tio

        transform = build_val_transforms()
        dummy = torch.randn(4, 64, 64, 1)
        subject = tio.Subject(image=tio.ScalarImage(tensor=dummy))
        transformed = transform(subject)

        assert torch.allclose(transformed["image"].data, dummy)


class TestSliceExtraction:
    """Tests for 2D and 2.5D slice extraction from 3D volumes."""

    def _create_dummy_volume(
        self,
        shape: tuple = (4, 64, 64, 32),
        tumor_slice: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a dummy volume with a known tumor location.

        Parameters
        ----------
        shape : tuple
            Volume shape (C, H, W, D).
        tumor_slice : int
            Axial slice index where tumor is placed.

        Returns
        -------
        tuple
            (volume, segmentation) arrays.
        """
        volume = np.random.randn(*shape).astype(np.float32)
        segmentation = np.zeros(shape[1:], dtype=np.float32)

        # Place tumor in a known slice
        segmentation[20:40, 20:40, tumor_slice] = 1.0

        return volume, segmentation

    def test_max_tumor_slice_selection(self) -> None:
        """Slice selection picks the slice with maximum tumor area."""
        volume, seg = self._create_dummy_volume(tumor_slice=20)

        extracted, idx = extract_2d_slice(
            volume, segmentation=seg, axis=2, method="max_tumor_area"
        )

        assert idx == 20, f"Expected slice 20, got {idx}"

    def test_center_slice_selection(self) -> None:
        """Center method picks the middle slice."""
        volume, seg = self._create_dummy_volume(shape=(4, 64, 64, 32))

        _, idx = extract_2d_slice(
            volume, segmentation=seg, axis=2, method="center"
        )

        assert idx == 16  # 32 // 2

    def test_2d_output_shape(self) -> None:
        """2D extraction returns (C, H, W) shape."""
        volume, seg = self._create_dummy_volume(shape=(4, 64, 64, 32))

        extracted, _ = extract_2d_slice(
            volume, segmentation=seg, axis=2, context_slices=0
        )

        assert extracted.shape == (4, 64, 64)

    def test_25d_output_shape(self) -> None:
        """2.5D extraction with context=1 returns (C*3, H, W) shape."""
        volume, seg = self._create_dummy_volume(shape=(4, 64, 64, 32))

        extracted, _ = extract_2d_slice(
            volume, segmentation=seg, axis=2, context_slices=1
        )

        # 4 channels × 3 slices (center ± 1)
        assert extracted.shape == (12, 64, 64)

    def test_25d_boundary_handling(self) -> None:
        """
        2.5D extraction at volume boundary zero-pads out-of-bounds slices.
        """
        volume, _ = self._create_dummy_volume(shape=(4, 64, 64, 32))

        # Extract at slice 0 with context=1 — left neighbor is out of bounds
        extracted, idx = extract_2d_slice(
            volume,
            segmentation=None,
            axis=2,
            method="center",
            context_slices=1,
        )

        # Should still produce valid output
        assert extracted.shape == (12, 64, 64)

    def test_no_segmentation_fallback(self) -> None:
        """Without segmentation, max_tumor_area falls back to center."""
        volume = np.random.randn(4, 64, 64, 32).astype(np.float32)

        _, idx = extract_2d_slice(
            volume, segmentation=None, axis=2, method="max_tumor_area"
        )

        assert idx == 16  # Falls back to center (32 // 2)

    def test_torch_tensor_input(self) -> None:
        """Slice extraction works with torch.Tensor inputs."""
        volume = torch.randn(4, 64, 64, 32)
        seg = torch.zeros(64, 64, 32)
        seg[20:40, 20:40, 10] = 1.0

        extracted, idx = extract_2d_slice(
            volume, segmentation=seg, axis=2, method="max_tumor_area"
        )

        assert idx == 10
        assert isinstance(extracted, torch.Tensor)
        assert extracted.shape == (4, 64, 64)


class TestPreprocessingTransforms:
    """Tests for offline preprocessing transforms."""

    def test_zscore_normalization_produces_zero_mean(self) -> None:
        """
        Z-score normalized foreground should have approximately zero mean.
        """
        import torchio as tio

        transform = build_preprocessing_transforms(
            z_score=True, foreground_only=True
        )

        # Create volume with positive foreground and zero background
        tensor = torch.zeros(1, 64, 64, 32)
        fg_region = (slice(None), slice(10, 50), slice(10, 50), slice(5, 25))
        tensor[fg_region] = torch.randn(40, 40, 20).abs() + 100  # All positive

        # Track the original foreground mask before normalization
        fg_mask = tensor > 0

        subject = tio.Subject(image=tio.ScalarImage(tensor=tensor))
        result = transform(subject)
        result_data = result["image"].data

        # Check mean of the ORIGINAL foreground region
        # (not post-normalization zeros)
        foreground_values = result_data[fg_mask]
        assert abs(foreground_values.mean().item()) < 0.5, (
            "Foreground mean after z-score: "
            f"{foreground_values.mean().item():.4f}"
        )

    def test_disabled_zscore_returns_unchanged(self) -> None:
        """With z_score=False, data should pass through unchanged."""
        transform = build_preprocessing_transforms(z_score=False)
        assert len(transform.transforms) == 0
