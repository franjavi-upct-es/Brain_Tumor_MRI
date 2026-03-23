# tests/test_experiments/test_split_comparison.py — Tests for split comparison
"""
Verifies the image-level vs patient-level split comparison logic:
- Image-level splits DO produce leakage.
- Patient-level splits DO NOT produce leakage.
- Leakage detection catches overlapping patients.
- Comparison framework runs end-to-end.
"""

from __future__ import annotations

import numpy as np

from src.experiments.split_comparison import (
    SplitComparisonResult,
    create_image_level_splits,
    create_patient_level_splits,
    detect_leakage_in_splits,
    run_split_comparison,
)


def _create_multi_slice_data(
    n_patients: int = 50,
    slices_per_patient: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic data mimicking BraTS with multiple slices per patient.
    """
    rng = np.random.RandomState(seed)
    patient_ids = []
    labels = []
    for p in range(n_patients):
        label = rng.randint(0, 2)
        for _ in range(slices_per_patient):
            patient_ids.append(f"patient_{p:04d}")
            labels.append(label)
    return np.array(labels), np.array(patient_ids)


class TestImageLevelSplits:
    """Tests confirming image-level splits produce leakage."""

    def test_image_level_creates_leakage(self) -> None:
        """Image-level splits MUST produce patient overlap when patients have
        multiple slices — this is the whole point of the ablation."""
        labels, patient_ids = _create_multi_slice_data(
            n_patients=50, slices_per_patient=10
        )
        splits = create_image_level_splits(labels, n_splits=5, random_state=42)
        leakage = detect_leakage_in_splits(splits, patient_ids)

        assert leakage["has_leakage"] is True, (
            "Image-level splits with multi-slice data MUST produce leakage"
        )
        assert leakage["n_leaked_patients_total"] > 0

    def test_leakage_rate_is_substantial(self) -> None:
        """With ~10 slices/patient, nearly all patients should leak."""
        labels, patient_ids = _create_multi_slice_data(
            n_patients=50, slices_per_patient=10
        )
        splits = create_image_level_splits(labels, n_splits=5, random_state=42)
        leakage = detect_leakage_in_splits(splits, patient_ids)

        # With 10 slices per patient and 5 folds, most patients should leak
        assert leakage["leakage_rate"] > 0.5, (
            f"Expected high leakage rate, got {leakage['leakage_rate']:.2%}"
        )


class TestPatientLevelSplits:
    """Tests confirming patient-level splits prevent leakage."""

    def test_patient_level_prevents_leakage(self) -> None:
        """Patient-level splits MUST NOT produce any patient overlap."""
        labels, patient_ids = _create_multi_slice_data(
            n_patients=50, slices_per_patient=10
        )
        splits = create_patient_level_splits(
            labels, patient_ids, n_splits=5, random_state=42
        )
        leakage = detect_leakage_in_splits(splits, patient_ids)

        assert leakage["has_leakage"] is False, (
            "Patient-level splits must NEVER produce leakage"
        )
        assert leakage["n_leaked_patients_total"] == 0


class TestLeakageDetection:
    """Tests for the leakage detection function."""

    def test_detects_known_leakage(self) -> None:
        """Should detect leakage when patients are split across folds."""
        patient_ids = np.array(["p1", "p1", "p2", "p2", "p3", "p3"])
        # Manually create a leaky split: p1 in both train and val
        splits = [(np.array([0, 2, 4]), np.array([1, 3, 5]))]
        leakage = detect_leakage_in_splits(splits, patient_ids)

        assert leakage["has_leakage"] is True
        assert leakage["n_leaked_patients_total"] == 3  # All patients leak

    def test_no_leakage_when_clean(self) -> None:
        """Should report no leakage with clean patient-level splits."""
        patient_ids = np.array(["p1", "p1", "p2", "p2", "p3", "p3"])
        # Clean split: each patient fully in one set
        splits = [(np.array([0, 1, 2, 3]), np.array([4, 5]))]
        leakage = detect_leakage_in_splits(splits, patient_ids)

        assert leakage["has_leakage"] is False

    def test_per_fold_counts(self) -> None:
        """Per-fold leakage should count affected samples."""
        patient_ids = np.array(["p1"] * 5 + ["p2"] * 5)
        splits = [(np.array([0, 1, 5, 6]), np.array([2, 3, 4, 7, 8, 9]))]
        leakage = detect_leakage_in_splits(splits, patient_ids)

        assert leakage["has_leakage"] is True
        fold_info = leakage["per_fold"][0]
        assert fold_info["n_leaked_patients"] == 2


class TestRunSplitComparison:
    """Tests for the full comparison runner."""

    def test_end_to_end(self) -> None:
        """Comparison should run end-to-end with a simple evaluator."""
        labels, patient_ids = _create_multi_slice_data(
            n_patients=30, slices_per_patient=5
        )

        def simple_evaluator(train_idx, val_idx):
            # Dummy classifier: predict majority class
            from collections import Counter

            from sklearn.metrics import balanced_accuracy_score

            train_labels = labels[train_idx]
            majority = Counter(train_labels).most_common(1)[0][0]
            val_preds = np.full(len(val_idx), majority)
            return {
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels[val_idx], val_preds)
                ),
            }

        result = run_split_comparison(
            labels=labels,
            patient_ids=patient_ids,
            evaluate_fn=simple_evaluator,
            n_splits=3,
            random_state=42,
        )

        assert isinstance(result, SplitComparisonResult)
        assert "balanced_accuracy" in result.image_level_metrics
        assert "balanced_accuracy" in result.patient_level_metrics
        assert "balanced_accuracy" in result.delta
        assert result.leakage_detected is True

    def test_to_dict_serializable(self) -> None:
        """Result should be JSON-serializable."""
        import json

        result = SplitComparisonResult(
            image_level_metrics={"balanced_accuracy": 0.95},
            patient_level_metrics={"balanced_accuracy": 0.87},
            delta={"balanced_accuracy": -0.08},
            n_patients=50,
            n_samples=500,
            leakage_detected=True,
        )
        d = result.to_dict()
        json.dumps(d)  # Should not raise
