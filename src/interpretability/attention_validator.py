# src/interpretability/attention_validator.py
# Validate model attention against ground truth
"""
Validates that the model's attention (Grad-CAM) overlaps with the actual
tumor region (BraTS segmentation ground truth) by computing IoU.

This proves not just WHAT the model classifies, but WHETHER IT IS LOOKING
WHERE IT SHOULD. A model with high accuracy but low IoU is exploiting
shortcuts (scanner artifacts, background patterns, etc.).

Protocol:
    For each correct prediction in test:
        1. Generate Grad-CAM heatmap from the model.
        2. Binarize heatmap at threshold = 0.5 × max.
        3. Compare with BraTS segmentation ground truth (any tumor label > 0).
        4. Compute IoU (Intersection over Union).
        5. Report mean IoU per class.

    Interpretation:
        IoU > 0.5 → Model attends to clinically relevant tumor regions.
        0.3 < IoU < 0.5 → Partial overlap, mixed signals.
        IoU < 0.3 → Model likely uses shortcuts, NOT tumor features.

The naive Kaggle model is expected to have IoU ~ 0.1-0.2 (looks at borders).
The rigorous BraTS model is expected to have IoU > 0.5 (looks at tumor).

References:
    - DeGrave et al. (2021). AI selects shortcuts over signal. Nature MI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.interpretability.gradcam import GradCAMGenerator, binarize_heatmap

logger = logging.getLogger(__name__)


@dataclass
class AttentionValidationResult:
    """
    Result of attention validation for a single sample.

    Attributes
    ----------
    patient_id : str
        Patient identifier.
    predicted_class : int
        Model's predicted class.
    true_class : int
        Ground truth class.
    correct : bool
        Whether prediction was correct.
    iou : float
        IoU between binarized Grad-CAM and tumor segmentation.
    dice : float
        Dice coefficient (alternative overlap metric).
    heatmap_coverage : float
        Fraction of heatmap area that is "active" (above threshold).
    tumor_coverage : float
        Fraction of tumor region covered by the heatmap.
    """

    patient_id: str = ""
    predicted_class: int = -1
    true_class: int = -1
    correct: bool = False
    iou: float = 0.0
    dice: float = 0.0
    heatmap_coverage: float = 0.0
    tumor_coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "patient_id": self.patient_id,
            "predicted_class": self.predicted_class,
            "true_class": self.true_class,
            "correct": self.correct,
            "iou": round(self.iou, 4),
            "dice": round(self.dice, 4),
            "heatmap_coverage": round(self.heatmap_coverage, 4),
            "tumor_coverage": round(self.tumor_coverage, 4),
        }


@dataclass
class AttentionValidationSummary:
    """
    Aggregated attention validaiton results.

    Attributes
    ----------
    mean_iou : float
        Mean IoU across all validated samples.
    std_iou : float
        Standard deviation of IoU.
    mean_iou_per_class : dict[int, float]
        Mean IoU broken down by true class.
    n_validated : int
        Total samples validated.
    n_correct_only : int
        Samples validated (correct prediction only).
    interpretation : str
        Human-readable interpretation of the IoU result.
    per_sample : list[AttentionValidationResult]
        Per-sample results
    """

    mean_iou: float = 0.0
    std_iou: float = 0.0
    mean_iou_per_class: dict[int, float] = field(default_factory=dict)
    n_validated: int = 0
    n_correct_only: int = 0
    interpretation: str = ""
    per_sample: list[AttentionValidationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "mean_iou": round(self.mean_iou, 4),
            "std_iou": round(self.std_iou, 4),
            "mean_iou_per_class": {
                k: round(v, 4) for k, v in self.mean_iou_per_class.items()
            },
            "n_validated": self.n_validated,
            "n_correct_only": self.n_correct_only,
            "interpretation": self.interpretation,
            "per_sample": [s.to_dict() for s in self.per_sample],
        }


def compute_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """Compute Intersection over Union between two binary masks.

    Parameters
    ----------
    mask_a : np.ndarray
        First binary mask (bool or 0/1).
    mask_b : np.ndarray
        Second binary mask (bool or 0/1).

    Returns
    -------
    float
        IoU value in [0, 1]. Returns 0 if both masks are empty.
    """
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def compute_dice(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """Compute Dice coefficient between two binary masks.

    Parameters
    ----------
    mask_a : np.ndarray
        First binary mask.
    mask_b : np.ndarray
        Second binary mask.

    Returns
    -------
    float
        Dice coefficient in [0, 1].
    """
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()

    if total == 0:
        return 0.0

    return float(2 * intersection / total)


def validate_single_sample(
    heatmap: np.ndarray,
    segmentation: np.ndarray,
    predicted_class: int,
    true_class: int,
    patient_id: str = "",
    threshold_ratio: float = 0.5,
) -> AttentionValidationResult:
    """
    Validate model attention for a single sample.

    Parameters
    ----------
    heatmap : np.ndarray
        Grad-CAM heatmap of shape (H, W), values in [0, 1].
    segmentation : np.ndarray
        Ground truth tumor segmentation of shape (H, W).
        Any non-zero value indicates tumor.
    predicted_class : int
        Model's predicted class
    patient_id : str
        Patient identifier for logging.
    threshold_ratio : float
        Binarization threshold ratio (default 0.5).

    Returns
    -------
    AttentionValidationResult
        Validation result for this sample.
    """
    # Binarize heatmap
    heatmap_binary = binarize_heatmap(heatmap, threshold_ratio)

    # Create tumor mask from segmentation (any label > 0)
    tumor_mask = segmentation > 0

    # Compute overlap metrics
    iou = compute_iou(heatmap_binary, tumor_mask)
    dice = compute_dice(heatmap_binary, tumor_mask)

    # Coverage metrics
    total_pixels = heatmap_binary.size
    heatmap_area = heatmap_binary.sum()
    tumor_area = tumor_mask.sum()

    heatmap_coverage = float(heatmap_area / max(total_pixels, 1))
    tumor_coverage = float(
        np.logical_and(heatmap_binary, tumor_mask).sum() / max(tumor_area, 1)
    )

    return AttentionValidationResult(
        patient_id=patient_id,
        predicted_class=predicted_class,
        true_class=true_class,
        correct=(predicted_class == true_class),
        iou=iou,
        dice=dice,
        heatmap_coverage=heatmap_coverage,
        tumor_coverage=tumor_coverage,
    )


def validate_attention(
    model: nn.Module,
    dataloader: Any,
    gradcam_generator: GradCAMGenerator,
    correct_only: bool = True,
    max_samples: int | None = None,
    threshold_ratio: float = 0.5,
) -> AttentionValidationSummary:
    """
    Run attention validation on a dataset.

    Parameters
    ----------
    model : nn.Module
        Trained classification model.
    dataloader : DataLoader
        DataLoader yielding batches with "image", "label", "segmentation",
        and "patient_id" keys.
    gradcam_generator : GradCAMGenerator
        Configured Grad-CAM generator.
    correct_only : bool
        If True, only validate correctly predicted samples.
    max_samples : int, optional
        Maximum samples to validate (for speed).
    threshold_ratio : float
        Binarization threshold ratio.

    Returns
    -------
    AttentionValidationSummary
        Aggregated validation results.
    """
    model.eval()
    results = []
    n_processed = 0

    for batch in dataloader:
        images = batch["image"]
        labels = batch["label"]
        segmentations = batch["segmentation"]
        patient_ids = batch.get("patient_id", ["unknown"] * len(labels))

        with torch.no_grad():
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

        for i in range(len(labels)):
            if max_samples is not None and n_processed >= max_samples:
                break

            pred = int(predictions[i].item())
            true = int(labels[i].item())

            if correct_only and pred != true:
                continue

            # Generate Grad-CAM
            heatmap = gradcam_generator.generate(
                images[i].unsqueeze(0), target_class=pred
            )

            # Get segmentation slice
            seg = segmentations[i].numpy()
            if seg.ndim == 3:
                seg = seg[0]  # Remove channel dim if present

            # Validate
            result = validate_single_sample(
                heatmap=heatmap,
                segmentation=seg,
                predicted_class=pred,
                true_class=true,
                patient_id=str(patient_ids[i])
                if hasattr(patient_ids, "__getitem__")
                else "unknown",
                threshold_ratio=threshold_ratio,
            )
            results.append(result)
            n_processed += 1

        if max_samples is not None and n_processed >= max_samples:
            break

    # Aggregate
    summary = _aggregate_results(results, correct_only)
    return summary


def _aggregate_results(
    results: list[AttentionValidationResult],
    correct_only: bool,
) -> AttentionValidationSummary:
    """
    Aggregate per-sample results into a summary.

    Parameters
    ----------
    results : list[AttentionValidationResult]
        Per-sample results.
    correct_only : bool
        Whether only correct predictions were included.

    Returns
    -------
    AttentionValidationSummary
        Aggregated summary with interpretation.
    """
    if not results:
        return AttentionValidationSummary(
            interpretation="No samples validated."
        )

    ious = np.array([r.iou for r in results])
    mean_iou = float(np.mean(ious))
    std_iou = float(np.std(ious))

    # Per-class IoU
    class_ious: dict[int, list[float]] = {}
    for r in results:
        cls = r.true_class
        if cls not in class_ious:
            class_ious[cls] = []
        class_ious[cls].append(r.iou)

    mean_iou_per_class = {
        cls: float(np.mean(vals)) for cls, vals in class_ious.items()
    }

    # Interpretation
    if mean_iou > 0.5:
        interpretation = (
            f"GOOD: Mean IoU = {mean_iou:.3f} (> 0.5). "
            f"The model attends to clinically relevant tumor regions."
        )
    elif mean_iou > 0.3:
        interpretation = (
            f"MIXED: Mean IoU = {mean_iou:.3f} (0.3-0.5). "
            f"Partial overlap with tumor regions. Some shortcut "
            "usage possible."
        )
    else:
        interpretation = (
            f"WARNING: Mean IoU = {mean_iou:.3f} (< 0.3). "
            f"The model likely uses shortcuts (artifacts, background) "
            f"rather than tumor features."
        )

    n_correct = sum(1 for r in results if r.correct)

    summary = AttentionValidationSummary(
        mean_iou=mean_iou,
        std_iou=std_iou,
        mean_iou_per_class=mean_iou_per_class,
        n_validated=len(results),
        n_correct_only=n_correct,
        interpretation=interpretation,
        per_sample=results,
    )

    logger.info("Attention validation: %s", interpretation)
    logger.info(
        "  Validated %d samples (correct only: %s), per-class IoU: %s",
        len(results),
        correct_only,
        {k: f"{v:.3f}" for k, v in mean_iou_per_class.items()},
    )

    return summary
