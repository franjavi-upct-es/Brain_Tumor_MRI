# inference_utils.py - Shared helpers for calibrated tumor scoring and decisions
# -----------------------------------------------------------------------------
# Centralized utilities for:
# - stable softmax
# - converting multiclass probabilities to a single "tumor score"
# - combining multiple model outputs (ensembles)
# - applying a risk-aware triage rule around the decision threshold

from typing import Iterable, Tuple

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def tumor_score_from_probs(probs: np.ndarray, no_tumor_idx: int) -> float:
    """
    Map multiclass probabilities to a single tumor score.
    Uses the current convention: tumor_score = 1 - P(no_tumor).
    """
    probs = np.asarray(probs).reshape(-1)
    if 0 <= no_tumor_idx < probs.shape[0]:
        return float(1.0 - probs[no_tumor_idx])
    return float(np.max(probs))


def aggregate_logits(logits_list: Iterable[np.ndarray], strategy: str = "mean") -> np.ndarray:
    """
    Combine multiple logits (e.g., ensemble members or TTA replicas).
    Supported strategies: mean (default) or median.
    """
    stack = np.stack([np.asarray(l) for l in logits_list], axis=0)
    if strategy == "median":
        return np.median(stack, axis=0)
    return np.mean(stack, axis=0)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by temperature > 0 to apply calibration."""
    temperature = max(float(temperature), 1e-6)
    return logits / temperature


def risk_triage_decision(
    tumor_scores: Iterable[float],
    threshold: float,
    triage_band: float = 0.05,
    max_disagreement: float = 0.1,
) -> Tuple[str, dict]:
    """
    Apply a conservative decision rule:
        - if average score >= threshold -> 'tumor'
        - if average score <= threshold - triage_band and ensemble agrees -> 'healthy'
        - otherwise -> 'review' (manual triage)
    """
    scores = np.asarray(list(tumor_scores), dtype=float)
    if scores.size == 0:
        return "review", {
            "score": None,
            "spread": None,
            "reason": "no_scores",
        }

    avg = float(np.mean(scores))
    spread = float(np.max(scores) - np.min(scores)) if scores.size > 1 else 0.0
    lower_bound = threshold - max(triage_band, 0.0)

    decision = "tumor" if avg >= threshold else "healthy"
    reason = "threshold"

    if avg < threshold and avg >= lower_bound:
        decision = "review"
        reason = "within_triage_band"
    if spread > max_disagreement:
        decision = "review"
        reason = "ensemble_disagreement"

    return decision, {
        "score": avg,
        "spread": spread,
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "threshold": float(threshold),
        "triage_band": float(triage_band),
        "max_disagreement": float(max_disagreement),
        "reason": reason,
    }


def make_tta_layer(rotation: float = 0.05, brightness: float = 0.1, contrast: float = 0.1):
    """Lightweight TTA augmentation layer (kept conservative for MRI realism)."""
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(rotation),
            tf.keras.layers.RandomBrightness(brightness),
            tf.keras.layers.RandomContrast(contrast),
        ]
    )
