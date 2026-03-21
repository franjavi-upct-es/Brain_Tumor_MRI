# src/evaluation/metrics.py — Comprehensive evaluation metrics
"""
Metrics framework following CLAIM (Tejani et al., 2024) and TRIPOD+AI
(Collins et al., 2024) guidelines for medical AI reporting.

Primary metrics (reported in main text):
    - Balanced Accuracy (corrects for class imbalance)
    - Macro F1-Score
    - ROC-AUC per class and macro-averaged
    - Cohen's Kappa

Secondary metrics (supplementary tables):
    - Per-class sensitivity (recall) and specificity
    - Precision-Recall AUC
    - Confusion matrix with absolute values and percentages

All metrics are computed at the patient level — one prediction per patient,
not per slice — to prevent inflated estimates from correlated slices.

References:
    - Tejani et al. (2024). CLAIM checklist. Radiology: AI.
    - Collins et al. (2024). TRIPOD+AI statement. BMJ.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationReport:
    """
    Complete classification evaluation report.

    Contains all primary, secondary, and per-class metrics required
    by CLAIM and TRIPOD+AI guidelines.

    Attributes
    ----------
    balanced_accuracy : float
        Macro-averaged recall across classes.
    f1_macro : float
        Macro-averaged F1-score.
    auc_roc_macro : float
        Macro-averaged ROC-AUC.
    cohens_kappa : float
        Cohen's kappa coefficient.
    accuracy : float
        Standard (unbalanced) accuracy.
    per_class_sensitivity : dict[int, float]
        Recall per class.
    per_class_specificity : dict[int, float]
        Specificity per class.
    per_class_precision : dict[int, float]
        Precision per class.
    per_class_f1 : dict[int, float]
        F1-score per class.
    per_class_auc : dict[int, float]
        ROC-AUC per class (one-vs-rest).
    confusion_matrix_absolute : np.ndarray
        Confusion matrix with raw counts.
    confusion_matrix_normalized : np.ndarray
        Row-normalized confusion matrix (percentages).
    pr_auc_macro : float
        Macro-averaged Precision-Recall AUC.
    n_samples : int
        Total number of samples evaluated.
    n_per_class : dict[int, int]
        Sample count per class.
    """

    balanced_accuracy: float = 0.0
    f1_macro: float = 0.0
    auc_roc_macro: float = 0.0
    cohens_kappa: float = 0.0
    accuracy: float = 0.0
    per_class_sensitivity: dict[int, float] = field(default_factory=dict)
    per_class_specificity: dict[int, float] = field(default_factory=dict)
    per_class_precision: dict[int, float] = field(default_factory=dict)
    per_class_f1: dict[int, float] = field(default_factory=dict)
    per_class_auc: dict[int, float] = field(default_factory=dict)
    confusion_matrix_absolute: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    confusion_matrix_normalized: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    pr_auc_macro: float = 0.0
    n_samples: int = 0
    n_per_class: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        result = {
            "primary_metrics": {
                "balanced_accuracy": self.balanced_accuracy,
                "f1_macro": self.f1_macro,
                "auc_roc_macro": self.auc_roc_macro,
                "cohens_kappa": self.cohens_kappa,
            },
            "secondary_metrics": {
                "accuracy": self.accuracy,
                "pr_auc_macro": self.pr_auc_macro,
            },
            "per_class": {
                "sensitivity": self.per_class_sensitivity,
                "specificity": self.per_class_specificity,
                "precision": self.per_class_precision,
                "f1": self.per_class_f1,
                "auc_roc": self.per_class_auc,
            },
            "confusion_matrix": {
                "absolute": self.confusion_matrix_absolute.tolist()
                if self.confusion_matrix_absolute.size > 0
                else [],
                "normalized_percent": self.confusion_matrix_normalized.tolist()
                if self.confusion_matrix_normalized.size > 0
                else [],
            },
            "dataset_info": {
                "n_samples": self.n_samples,
                "n_per_class": self.n_per_class,
            },
        }
        return result


def compute_full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: dict[int, str] | None = None,
) -> ClassificationReport:
    """
    Compute all CLAIM-compliant evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground thruth labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted labels, shape (n_samples,).
    y_prob : np.ndarray, optional
        Predicted probabilities, shape (n_samples, n_classes).
        Required for AUC-ROC and PR-AUC computation.
    class_names : dict[int, str], optional
        Mapping from class index to name (e.g., {0: "LGG", 1: "HGG"}).

    Returns
    -------
    ClassificationReport
        Complete evaluation report.
    """
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    n_classes = len(classes)

    report = ClassificationReport()
    report.n_samples = len(y_true)
    report.n_per_class = {int(c): int(np.sum(y_true == c)) for c in classes}

    # --- Primary metrics ---
    report.balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    report.f1_macro = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    report.cohens_kappa = float(cohen_kappa_score(y_true, y_pred))
    report.accuracy = float(accuracy_score(y_true, y_pred))

    # --- Per-class metrics ---
    # Sensitivity (recall) per class
    for c in classes:
        binary_true = (y_true == c).astype(int)
        binary_pred = (y_pred == c).astype(int)

        tp = int(np.sum((binary_true == 1) & (binary_pred == 1)))
        fn = int(np.sum((binary_true == 1) & (binary_pred == 0)))
        fp = int(np.sum((binary_true == 0) & (binary_pred == 1)))
        tn = int(np.sum((binary_true == 0) & (binary_pred == 0)))

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)

        report.per_class_sensitivity[int(c)] = float(sensitivity)
        report.per_class_specificity[int(c)] = float(specificity)
        report.per_class_precision[int(c)] = float(precision)
        report.per_class_f1[int(c)] = float(f1)

    # --- AUC-ROC (requires probabilities) ---
    if y_prob is not None:
        try:
            if n_classes == 2:
                report.auc_roc_macro = float(
                    roc_auc_score(y_true, y_prob[:, 1])
                )
                report.per_class_auc[int(classes[0])] = report.auc_roc_macro
                report.per_class_auc[int(classes[1])] = report.auc_roc_macro
            else:
                report.auc_roc_macro = float(
                    roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="macro"
                    )
                )
                for c in classes:
                    binary_true = (y_true == c).astype(int)
                    report.per_class_auc[int(c)] = float(
                        roc_auc_score(binary_true, y_prob[:, c])
                    )
        except ValueError as e:
            logger.warning("Could not compute AUC-ROC: %s", e)

    # PR-AUC
    if y_prob is not None:
        try:
            pr_aucs = []
            for c in classes:
                binary_true = (y_true == c).astype(int)
                prec_arr, rec_arr, _ = precision_recall_curve(
                    binary_true, y_prob[:, c]
                )
                pr_aucs.append(float(auc(rec_arr, prec_arr)))
            report.pr_auc_macro = float(np.mean(pr_aucs))
        except ValueError as e:
            logger.warning("Could not compute PR-AUC: %s", e)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report.confusion_matrix_absolute = cm

    # Row-normalized (each row sums to 100%)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid div by zero
    report.confusion_matrix_normalized = np.round(cm / row_sums * 100, 1)

    logger.info(
        "Evaluation: bal_acc=%.4f, f1=%.4f, auc=%.4f, kappa=%.4f (n=%d)",
        report.balanced_accuracy,
        report.f1_macro,
        report.auc_roc_macro,
        report.cohens_kappa,
        report.n_samples,
    )

    return report


def compute_roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_idx: int = 1,
) -> dict[str, list[float]]:
    """
    Compute ROC curve data points for plotting.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_prob : np.ndarray
        Predicted probabilities.
    class_idx : int
        Class index for binary ROC (default: positive class = 1)

    Returns
    -------
    dict
        Dictionary with "fpr", "tpr", "thresholds", "auc" keys.
    """
    if y_prob.ndim == 2:
        probs = y_prob[:, class_idx]
    else:
        probs = y_prob

    fpr, tpr, thresholds = roc_curve(y_true, probs, pos_label=class_idx)
    auc_val = float(auc(fpr, tpr))

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": auc_val,
    }
