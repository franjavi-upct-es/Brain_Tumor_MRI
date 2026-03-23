# tests/test_interpretability/test_attention_validator.py — IoU validation tests
"""
Verifies the attention validation pipeline:
- IoU computation correctness.
- Dice coefficient correctness.
- Binarization at threshold = 0.5 × max.
- Single-sample validation produces correct results.
- Summary aggregation and interpretation thresholds.

This is critical because the IoU validation is the most powerful
evidence in the thesis for/against shortcut learning.
"""

from __future__ import annotations

import numpy as np

from src.interpretability.attention_validator import (
    AttentionValidationResult,
    _aggregate_results,
    compute_dice,
    compute_iou,
    validate_single_sample,
)
from src.interpretability.gradcam import binarize_heatmap


class TestIoUComputation:
    """Tests for IoU (Intersection over Union) calculation."""

    def test_perfect_overlap(self) -> None:
        """Identical masks should give IoU = 1.0."""
        mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
        assert compute_iou(mask, mask) == 1.0

    def test_no_overlap(self) -> None:
        """Completely disjoint masks should give IoU = 0.0."""
        mask_a = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
        mask_b = np.array([[0, 0, 0], [0, 0, 1]], dtype=bool)
        assert compute_iou(mask_a, mask_b) == 0.0

    def test_partial_overlap(self) -> None:
        """Known partial overlap should give correct IoU."""
        # mask_a: 4 pixels, mask_b: 4 pixels, intersection: 2, union: 6
        mask_a = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=bool)
        mask_b = np.array([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=bool)
        iou = compute_iou(mask_a, mask_b)
        assert abs(iou - 2.0 / 6.0) < 1e-6

    def test_both_empty(self) -> None:
        """Two empty masks should give IoU = 0.0."""
        mask = np.zeros((5, 5), dtype=bool)
        assert compute_iou(mask, mask) == 0.0

    def test_works_with_int_masks(self) -> None:
        """Should work with integer (0/1) masks, not just bool."""
        mask_a = np.array([[1, 0], [0, 1]], dtype=np.int32)
        mask_b = np.array([[1, 1], [0, 0]], dtype=np.int32)
        iou = compute_iou(mask_a, mask_b)
        assert 0 <= iou <= 1

    def test_symmetric(self) -> None:
        """IoU(A, B) should equal IoU(B, A)."""
        mask_a = np.random.RandomState(42).randint(0, 2, (10, 10)).astype(bool)
        mask_b = np.random.RandomState(43).randint(0, 2, (10, 10)).astype(bool)
        assert compute_iou(mask_a, mask_b) == compute_iou(mask_b, mask_a)


class TestDiceComputation:
    """Tests for Dice coefficient calculation."""

    def test_perfect_overlap_dice(self) -> None:
        """Identical masks should give Dice = 1.0."""
        mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
        assert compute_dice(mask, mask) == 1.0

    def test_no_overlap_dice(self) -> None:
        """Disjoint masks should give Dice = 0.0."""
        mask_a = np.array([[1, 0], [0, 0]], dtype=bool)
        mask_b = np.array([[0, 0], [0, 1]], dtype=bool)
        assert compute_dice(mask_a, mask_b) == 0.0

    def test_dice_greater_than_iou(self) -> None:
        """Dice should always be >= IoU for the same masks."""
        mask_a = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=bool)
        mask_b = np.array([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=bool)
        assert compute_dice(mask_a, mask_b) >= compute_iou(mask_a, mask_b)


class TestBinarization:
    """Tests for Grad-CAM heatmap binarization."""

    def test_threshold_at_half_max(self) -> None:
        """Binarization at 0.5 × max should split correctly."""
        heatmap = np.array([[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]])
        binary = binarize_heatmap(heatmap, threshold_ratio=0.5)
        # Threshold = 0.5 × 1.0 = 0.5; pixels >= 0.5 are True
        expected = np.array([[False, False, False], [True, True, True]])
        np.testing.assert_array_equal(binary, expected)

    def test_zero_heatmap(self) -> None:
        """All-zero heatmap should produce all-True (threshold = 0)."""
        heatmap = np.zeros((5, 5))
        binary = binarize_heatmap(heatmap, 0.5)
        assert binary.all()  # 0 >= 0

    def test_uniform_heatmap(self) -> None:
        """Uniform heatmap should be all True at 0.5 threshold."""
        heatmap = np.ones((5, 5)) * 0.8
        binary = binarize_heatmap(heatmap, 0.5)
        assert binary.all()  # 0.8 >= 0.5 × 0.8 = 0.4


class TestSingleSampleValidation:
    """Tests for validate_single_sample()."""

    def _create_test_data(self):
        """Create synthetic test data with known overlap."""
        h, w = 64, 64
        heatmap = np.zeros((h, w), dtype=np.float32)
        segmentation = np.zeros((h, w), dtype=np.int32)

        # Place tumor in center
        segmentation[20:44, 20:44] = 1

        # Place heatmap overlapping with tumor
        heatmap[22:42, 22:42] = 0.8

        return heatmap, segmentation

    def test_correct_prediction_positive_iou(self) -> None:
        """Overlapping heatmap and segmentation should give positive IoU."""
        heatmap, seg = self._create_test_data()
        result = validate_single_sample(
            heatmap=heatmap,
            segmentation=seg,
            predicted_class=1,
            true_class=1,
            patient_id="test_001",
        )
        assert result.iou > 0.3
        assert result.correct is True
        assert result.patient_id == "test_001"

    def test_non_overlapping_gives_zero_iou(self) -> None:
        """Heatmap on opposite side from tumor should give IoU ~ 0."""
        h, w = 64, 64
        heatmap = np.zeros((h, w), dtype=np.float32)
        heatmap[:10, :10] = 1.0  # Top-left corner

        seg = np.zeros((h, w), dtype=np.int32)
        seg[50:60, 50:60] = 1  # Bottom-right

        result = validate_single_sample(heatmap, seg, 1, 1)
        assert result.iou == 0.0

    def test_incorrect_prediction_flagged(self) -> None:
        """Incorrect predictions should have correct=False."""
        heatmap, seg = self._create_test_data()
        result = validate_single_sample(
            heatmap, seg, predicted_class=0, true_class=1
        )
        assert result.correct is False

    def test_coverage_metrics(self) -> None:
        """Coverage metrics should be in [0, 1]."""
        heatmap, seg = self._create_test_data()
        result = validate_single_sample(heatmap, seg, 1, 1)
        assert 0 <= result.heatmap_coverage <= 1
        assert 0 <= result.tumor_coverage <= 1

    def test_to_dict_serializable(self) -> None:
        """Result should be JSON-serializable."""
        import json

        result = AttentionValidationResult(
            patient_id="p001",
            predicted_class=1,
            true_class=1,
            correct=True,
            iou=0.65,
            dice=0.78,
        )
        json.dumps(result.to_dict())


class TestAggregation:
    """Tests for result aggregation and interpretation."""

    def _create_results(
        self, ious: list[float], classes: list[int] | None = None
    ):
        """Create a list of validation results with given IoU values."""
        if classes is None:
            classes = [0] * (len(ious) // 2) + [1] * (
                len(ious) - len(ious) // 2
            )
        return [
            AttentionValidationResult(
                patient_id=f"p{i:03d}",
                predicted_class=c,
                true_class=c,
                correct=True,
                iou=iou,
            )
            for i, (iou, c) in enumerate(zip(ious, classes, strict=True))
        ]

    def test_high_iou_interpretation(self) -> None:
        """Mean IoU > 0.5 should produce 'GOOD' interpretation."""
        results = self._create_results([0.6, 0.7, 0.65, 0.55])
        summary = _aggregate_results(results, correct_only=True)
        assert "GOOD" in summary.interpretation

    def test_low_iou_interpretation(self) -> None:
        """Mean IoU < 0.3 should produce 'WARNING' interpretation."""
        results = self._create_results([0.1, 0.2, 0.15, 0.25])
        summary = _aggregate_results(results, correct_only=True)
        assert "WARNING" in summary.interpretation

    def test_mixed_iou_interpretation(self) -> None:
        """Mean IoU between 0.3-0.5 should produce 'MIXED' interpretation."""
        results = self._create_results([0.35, 0.4, 0.42, 0.38])
        summary = _aggregate_results(results, correct_only=True)
        assert "MIXED" in summary.interpretation

    def test_per_class_iou(self) -> None:
        """Should compute mean IoU per class."""
        results = self._create_results(
            [0.6, 0.7, 0.3, 0.4],
            classes=[0, 0, 1, 1],
        )
        summary = _aggregate_results(results, correct_only=True)
        assert abs(summary.mean_iou_per_class[0] - 0.65) < 1e-6
        assert abs(summary.mean_iou_per_class[1] - 0.35) < 1e-6

    def test_empty_results(self) -> None:
        """Empty results should produce safe summary."""
        summary = _aggregate_results([], correct_only=True)
        assert "No samples" in summary.interpretation

    def test_summary_to_dict(self) -> None:
        """Summary should be JSON-serializable."""
        import json

        results = self._create_results([0.6, 0.7])
        summary = _aggregate_results(results, correct_only=True)
        json.dumps(summary.to_dict())
