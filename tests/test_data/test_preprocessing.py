# tests/test_data/test_preprocessing.py — Tests for preprocessing pipeline
"""
Verifies the preprocessing functions used in the BraTS and UCSF-PDGM pipelines:
- Z-score normalization correctness (foreground-only)
- Slice selection logic
- NPZ output format
- Preprocessing statistics collection
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocessing import (
    PreprocessingStats,
    select_representative_slice,
    z_score_normalize,
)


class TestZScoreNormalization:
    """Tests for z-score normalization function."""

    def test_foreground_mean_near_zero(self) -> None:
        """Foreground voxels should have mean ≈ 0 after normalization."""
        rng = np.random.RandomState(42)
        volume = np.zeros((64, 64, 32), dtype=np.float32)
        # Place foreground with known distribution
        volume[10:50, 10:50, 5:25] = (
            rng.randn(40, 40, 20).astype(np.float32) * 200 + 1000
        )

        normalized, mean, std = z_score_normalize(volume, foreground_only=True)

        fg_mask = volume > 0
        fg_values = normalized[fg_mask]
        assert abs(fg_values.mean()) < 0.01, (
            f"Foreground mean should be ~0, got {fg_values.mean():.4f}"
        )

    def test_foreground_std_near_one(self) -> None:
        """Foreground voxels should have std ≈ 1 after normalization."""
        rng = np.random.RandomState(42)
        volume = np.zeros((64, 64, 32), dtype=np.float32)
        volume[10:50, 10:50, 5:25] = (
            rng.randn(40, 40, 20).astype(np.float32) * 200 + 1000
        )

        normalized, _, _ = z_score_normalize(volume, foreground_only=True)

        fg_mask = volume > 0
        fg_std = normalized[fg_mask].std()
        assert abs(fg_std - 1.0) < 0.01, (
            f"Foreground std should be ~1, got {fg_std:.4f}"
        )

    def test_background_remains_zero(self) -> None:
        """Background voxels (originally 0) should remain 0."""
        volume = np.zeros((64, 64, 32), dtype=np.float32)
        volume[20:40, 20:40, 10:20] = 500.0

        normalized, _, _ = z_score_normalize(volume, foreground_only=True)

        bg_mask = volume == 0
        assert np.all(normalized[bg_mask] == 0), (
            "Background voxels should remain exactly zero"
        )

    def test_all_zero_volume_unchanged(self) -> None:
        """All-zero volume should pass through unchanged."""
        volume = np.zeros((32, 32, 16), dtype=np.float32)
        normalized, mean, std = z_score_normalize(volume, foreground_only=True)
        assert np.array_equal(normalized, volume)
        assert mean == 0.0
        assert std == 1.0

    def test_constant_foreground_unchanged(self) -> None:
        """Volume with constant foreground (std=0) should not crash."""
        volume = np.zeros((32, 32, 16), dtype=np.float32)
        volume[5:25, 5:25, 3:12] = 42.0  # Constant value

        normalized, _, std = z_score_normalize(volume, foreground_only=True)
        assert std < 1e-8  # Near-zero std
        # Should return original (no division by zero)
        assert np.array_equal(normalized, volume)

    def test_returns_statistics(self) -> None:
        """Function should return (volume, mean, std) tuple."""
        volume = np.random.randn(32, 32, 16).astype(np.float32) + 100
        normalized, mean, std = z_score_normalize(
            volume, foreground_only=False
        )
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std > 0

    def test_whole_volume_normalization(self) -> None:
        """With foreground_only=False, normalize entire volume."""
        volume = np.random.randn(32, 32, 16).astype(np.float32) * 50 + 200
        normalized, _, _ = z_score_normalize(volume, foreground_only=False)
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1


class TestSliceSelection:
    """Tests for representative slice selection."""

    def test_max_tumor_area_finds_correct_slice(self) -> None:
        """Should select the slice with the most tumor voxels."""
        seg = np.zeros((64, 64, 32), dtype=np.int32)
        # Place tumor on slice 20 (largest area)
        seg[20:50, 20:50, 20] = 1
        # Smaller tumor on slice 15
        seg[25:35, 25:35, 15] = 1

        idx = select_representative_slice(
            seg, volume_depth=32, method="max_tumor_area"
        )
        assert idx == 20

    def test_center_method(self) -> None:
        """Center method should return depth // 2."""
        idx = select_representative_slice(
            None, volume_depth=155, method="center"
        )
        assert idx == 77  # 155 // 2

    def test_no_tumor_falls_back_to_center(self) -> None:
        """Empty segmentation should fall back to center slice."""
        seg = np.zeros((64, 64, 32), dtype=np.int32)
        idx = select_representative_slice(
            seg, volume_depth=32, method="max_tumor_area"
        )
        assert idx == 16  # 32 // 2

    def test_none_segmentation_uses_center(self) -> None:
        """None segmentation should use center."""
        idx = select_representative_slice(
            None, volume_depth=100, method="max_tumor_area"
        )
        assert idx == 50

    def test_invalid_method_raises(self) -> None:
        """Unknown slice method should raise ValueError."""
        seg = np.zeros((64, 64, 32), dtype=np.int32)
        seg[20:40, 20:40, 15] = 1  # Non-empty segmentation
        with pytest.raises(ValueError, match="Unknown slice selection method"):
            select_representative_slice(seg, volume_depth=32, method="invalid")


class TestPreprocessingStats:
    """Tests for PreprocessingStats dataclass."""

    def test_default_values(self) -> None:
        """Default stats should have sensible defaults."""
        stats = PreprocessingStats()
        assert stats.patient_id == ""
        assert stats.selected_slice == -1
        assert stats.n4_applied is False
        assert stats.fg_mean == 0.0

    def test_custom_values(self) -> None:
        """Stats should store custom values."""
        stats = PreprocessingStats(
            patient_id="BraTS-GLI-00001",
            original_shape=(240, 240, 155),
            selected_slice=77,
            tumor_area_on_slice=1200,
            fg_mean=500.0,
            fg_std=150.0,
        )
        assert stats.patient_id == "BraTS-GLI-00001"
        assert stats.selected_slice == 77
        assert stats.tumor_area_on_slice == 1200
