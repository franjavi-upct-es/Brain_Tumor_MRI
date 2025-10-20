# plots.py — Visualization & calibration utilities
# -------------------------------------------------------------------
# This module centralizes all plotting and calibration utilities used by the
# project. The intent is that you can import these functions from training
# and evaluation scripts to generate figures for the README (and for reports).
# We purposely add extensive comments so that the code reads like documentation.
#
# Conventions:
# - All file outputs are written to "reports/" (created if it doesn't exist).
# - Matplotlib is used for plotting (no seaborn for portability).
# - We NEVER set custom color palettes or styles here (to keep results reproducible
#   and avoid headaches in headless environments).

from __future__ import annotations
import os
import json
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# --------------------------- small utilities ---------------------------

def _ensure_dir(path: str) -> str:
    """Create directory `path` if it does not exist; return `path`."""
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------- training curves ---------------------------

def save_training_curves(history: Dict, out_dir: str = "reports") -> None:
    """
    Save training curves (accuracy and loss) from a Keras History.history dict.

    Parameters
    ----------
    history : Dict
        The `History.history` dictionary returned by `model.fit(...)`.
        Typically contains keys like 'loss', 'accuracy', 'val_loss', 'val_accuracy'.
    out_dir : str
        Output directory for generated figures & JSON history.
    """
    _ensure_dir(out_dir)

    # Persist the raw history (valuable for later analysis or reproductions).
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot accuracy
    if "accuracy" in history:
        plt.figure()
        plt.plot(history.get("accuracy", []), label="train_acc")
        if "val_accuracy" in history:
            plt.plot(history.get("val_accuracy", []), label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training & Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "acc_curve.png"), dpi=160)
        plt.close()

    # Plot loss
    if "loss" in history:
        plt.figure()
        plt.plot(history.get("loss", []), label="train_loss")
        if "val_loss" in history:
            plt.plot(history.get("val_loss", []), label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=160)
        plt.close()


# --------------------------- confusion matrix ---------------------------

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], normalize: bool,
                          out_path: str) -> None:
    """
    Save a (normalized) confusion matrix as a PNG.

    Parameters
    ----------
    cm : np.ndarray (C x C)
        Raw confusion matrix (counts).
    class_names : List[str]
        Class labels in order of indices used to compute cm.
    normalize : bool
        If True, normalize rows to sum to 1.0.
    out_path : str
        Where to save the PNG.
    """
    cm = np.array(cm, dtype=float)
    if normalize:
        # Divide each row by its sum to get per-class recall rates.
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-12
        cm = cm / row_sums

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix' + (' (normalized)' if normalize else ''))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations (values) on each cell for clarity.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else str(int(round(cm[i, j])))
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# --------------------------- ROC & PR curves ---------------------------

def plot_roc_pr_curves(y_true_onehot: np.ndarray, y_scores: np.ndarray,
                       class_names: List[str], out_dir: str = "reports") -> None:
    """
    One-vs-Rest ROC and PR curves for a multi-class classifier.

    Parameters
    ----------
    y_true_onehot : np.ndarray (N x C)
        One-hot ground-truth labels.
    y_scores : np.ndarray (N x C)
        Class probabilities for each sample (post-softmax, calibrated if applicable).
    class_names : List[str]
        Class labels.
    out_dir : str
        Output directory for figures.
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    _ensure_dir(out_dir)

    # ROC curves (OvR)
    plt.figure()
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=160)
    plt.close()

    # Precision-Recall curves (OvR)
    plt.figure()
    for i, cls in enumerate(class_names):
        prec, rec, _ = precision_recall_curve(y_true_onehot[:, i], y_scores[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_scores[:, i])
        plt.plot(rec, prec, label=f"{cls} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curves.png"), dpi=160)
    plt.close()


# --------------------------- calibration & reliability ---------------------------

def _bin_stats(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-bin accuracy & confidence for reliability diagrams.

    We use the standard approach of flattening multi-class probabilities
    into a binary One-vs-Rest (OvR) setup and then measuring calibration
    across confidence bins.

    Parameters
    ----------
    probs : (N x C) array of predicted probabilities
    labels : (N x C) array of one-hot labels
    n_bins : number of equal-width confidence bins on [0,1]

    Returns
    -------
    bin_acc : (n_bins,) empirical accuracy within each bin
    bin_conf : (n_bins,) average confidence within each bin
    bin_count : (n_bins,) number of samples in each bin
    bin_edges : (n_bins+1,) edges of the bins
    """
    # Flatten OvR: use the probability assigned to the true class for each sample
    true_class_indices = labels.argmax(axis=1)
    true_class_probs = probs[np.arange(probs.shape[0]), true_class_indices]

    # Predicted class for 0/1 correctness (did we get the class right?)
    pred_class_indices = probs.argmax(axis=1)
    correct = (pred_class_indices == true_class_indices).astype(float)

    # Now we bin by confidence (true_class_probs)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(true_class_probs, bin_edges[1:-1], right=False)

    bin_acc = np.zeros(n_bins, dtype=float)
    bin_conf = np.zeros(n_bins, dtype=float)
    bin_count = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = (bin_indices == b)
        if np.any(mask):
            bin_count[b] = int(mask.sum())
            bin_acc[b] = float(correct[mask].mean())
            bin_conf[b] = float(true_class_probs[mask].mean())
        else:
            bin_count[b] = 0
            bin_acc[b] = 0.0
            bin_conf[b] = 0.0

    return bin_acc, bin_conf, bin_count, bin_edges


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE = sum_b (|bin|/N) * |acc(b) - conf(b)|

    This is a widely adopted scalar summary of miscalibration, taking
    a weighted average of the absolute gap between accuracy and average
    confidence within each bin.
    """
    bin_acc, bin_conf, bin_count, _ = _bin_stats(probs, labels, n_bins=n_bins)
    N = max(1, int(bin_count.sum()))
    gaps = np.abs(bin_acc - bin_conf)
    weights = bin_count / N
    return float(np.sum(weights * gaps))


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    MCE = max_b |acc(b) - conf(b)|

    Maximum calibration error is the worst-case bin gap.
    """
    bin_acc, bin_conf, bin_count, _ = _bin_stats(probs, labels, n_bins=n_bins)
    return float(np.max(np.abs(bin_acc - bin_conf)))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Multiclass Brier score (mean squared error between one-hot labels and probs).
    Lower is better; 0 corresponds to perfect calibrated certainty.
    """
    return float(np.mean((labels - probs) ** 2))


def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray, out_path: str,
                             n_bins: int = 15) -> None:
    """
    Plot the reliability diagram (accuracy vs confidence) and the gap bars.

    We use OvR binning on the *true-class* probability to summarize calibration.
    """
    bin_acc, bin_conf, bin_count, _ = _bin_stats(probs, labels, n_bins=n_bins)

    # Bar chart of accuracy per bin vs confidence diagonal
    fig, ax = plt.subplots()
    width = 1.0 / n_bins
    centers = np.linspace(width/2, 1 - width/2, n_bins)

    # Bars for empirical accuracy
    ax.bar(centers, bin_acc, width=width*0.9, edgecolor="black", linewidth=0.5, alpha=0.8, label="Empirical accuracy")
    # Reference diagonal y=x
    ax.plot([0,1], [0,1], linestyle="--", label="Perfect calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (mean in bin)")
    ax.set_ylabel("Accuracy (in bin)")
    ax.set_title("Reliability Diagram (OvR)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_confidence_histogram(probs: np.ndarray, labels: np.ndarray, out_path: str,
                              n_bins: int = 15) -> None:
    """
    Plot the histogram of true-class confidence. Useful to understand
    whether the model mostly predicts with low, medium or high confidence.
    """
    true_class_probs = probs[np.arange(probs.shape[0]), labels.argmax(axis=1)]
    fig, ax = plt.subplots()
    ax.hist(true_class_probs, bins=n_bins, range=(0,1), alpha=0.9)
    ax.set_xlim(0,1)
    ax.set_xlabel("True-class confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Histogram (OvR)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_calibration_report(probs: np.ndarray, labels: np.ndarray,
                            out_dir: str = "reports", n_bins: int = 15) -> Dict[str, float]:
    """
    Compute and save standard calibration metrics (ECE, MCE, Brier score) and
    figures (reliability diagram and confidence histogram).

    Returns
    -------
    report : Dict[str, float]
        A dictionary with the computed scalar metrics.
    """
    _ensure_dir(out_dir)

    # Compute metrics
    ece = expected_calibration_error(probs, labels, n_bins=n_bins)
    mce = maximum_calibration_error(probs, labels, n_bins=n_bins)
    bs  = brier_score(probs, labels)

    report = {"ECE": float(ece), "MCE": float(mce), "BrierScore": float(bs)}

    # Save JSON summary
    with open(os.path.join(out_dir, "calibration_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Save figures
    plot_reliability_diagram(probs, labels, out_path=os.path.join(out_dir, "reliability_diagram.png"), n_bins=n_bins)
    plot_confidence_histogram(probs, labels, out_path=os.path.join(out_dir, "confidence_hist.png"), n_bins=n_bins)

    return report
