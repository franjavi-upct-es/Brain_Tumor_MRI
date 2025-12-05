# plots.py — Visualization & calibration utilities
# -------------------------------------------------------------------
# This module centralizes all plotting and calibration utilities used by the
# project. It includes standard training curves and now comparative plots
# to visualize the impact of fine-tuning and generalization gaps.

from __future__ import annotations
import os
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# --------------------------- small utilities ---------------------------

def _ensure_dir(path: str) -> str:
    """Create directory `path` if it does not exist; return `path`."""
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------- training curves ---------------------------

def save_training_curves(history: Dict, out_dir: str = "reports", finetune_epoch: Optional[int] = None) -> None:
    """
    Save training curves (accuracy and loss).
    
    Args:
        history: Keras history dict.
        out_dir: Output directory.
        finetune_epoch: If provided, draws a vertical line indicating where 
                        Fine-Tuning started (useful to see the adaptation phase).
    """
    _ensure_dir(out_dir)

    # Persist the raw history
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    epochs = range(len(history.get("accuracy", [])))

    # Plot accuracy
    if "accuracy" in history:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, history.get("accuracy", []), label="Train Acc")
        if "val_accuracy" in history:
            plt.plot(epochs, history.get("val_accuracy", []), label="Val Acc")
        
        if finetune_epoch is not None and finetune_epoch < len(epochs):
            plt.axvline(x=finetune_epoch, color='r', linestyle='--', alpha=0.7, label='Fine-Tuning Start')

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Pipeline Accuracy")
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "acc_curve.png"), dpi=160)
        plt.close()

    # Plot loss
    if "loss" in history:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, history.get("loss", []), label="Train Loss")
        if "val_loss" in history:
            plt.plot(epochs, history.get("val_loss", []), label="Val Loss")
            
        if finetune_epoch is not None and finetune_epoch < len(epochs):
            plt.axvline(x=finetune_epoch, color='r', linestyle='--', alpha=0.7, label='Fine-Tuning Start')

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Pipeline Loss")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=160)
        plt.close()


# --------------------------- Comparative Plots (New) ---------------------------

def plot_model_comparison(base_metrics: Dict[str, float], 
                          finetuned_metrics: Dict[str, float], 
                          out_dir: str = "reports") -> None:
    """
    Generates a grouped bar chart comparing Base Model vs Fine-Tuned Model 
    on External Data. This visualizes the 'Real World' improvement.
    
    Args:
        base_metrics: Dict like {'Accuracy': 0.85, 'Recall': 0.70, ...}
        finetuned_metrics: Dict like {'Accuracy': 0.84, 'Recall': 0.91, ...}
    """
    _ensure_dir(out_dir)
    
    metrics = list(base_metrics.keys())
    base_vals = [base_metrics[m] for m in metrics]
    ft_vals = [finetuned_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, base_vals, width, label='Base Model (External Data)', color='#1f77b4', alpha=0.8)
    rects2 = ax.bar(x + width/2, ft_vals, width, label='Fine-Tuned (External Data)', color='#2ca02c', alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Impact of Fine-Tuning on Unseen Data')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add text labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_improvement.png"), dpi=160)
    plt.close()


def plot_generalization_gap(internal_metrics: Dict[str, float], 
                            external_metrics: Dict[str, float], 
                            out_dir: str = "reports") -> None:
    """
    Visualizes the drop in performance when moving from Internal Test Set
    (where acc ~1.0) to External Data.
    """
    _ensure_dir(out_dir)
    
    metrics = list(internal_metrics.keys())
    int_vals = [internal_metrics[m] for m in metrics]
    ext_vals = [external_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, int_vals, width, label='Internal Test (Kaggle)', color='gray', alpha=0.6)
    rects2 = ax.bar(x + width/2, ext_vals, width, label='External Test (Navoneel)', color='#d62728', alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Generalization Gap: Internal vs External Data')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "generalization_gap.png"), dpi=160)
    plt.close()


def plot_comparative_roc(y_true: np.ndarray, 
                         y_probs_base: np.ndarray, 
                         y_probs_finetuned: np.ndarray, 
                         out_dir: str = "reports") -> None:
    """
    Overlays ROC curves for Base and Fine-Tuned models on the SAME external dataset.
    This assumes a Binary classification task (Tumor vs Healthy) for the external set.
    """
    _ensure_dir(out_dir)
    
    # Calculate ROC Base
    fpr_b, tpr_b, _ = roc_curve(y_true, y_probs_base)
    auc_b = auc(fpr_b, tpr_b)
    
    # Calculate ROC Fine-Tuned
    fpr_f, tpr_f, _ = roc_curve(y_true, y_probs_finetuned)
    auc_f = auc(fpr_f, tpr_f)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_b, tpr_b, linestyle='--', color='blue', label=f'Base Model (AUC={auc_b:.3f})')
    plt.plot(fpr_f, tpr_f, linestyle='-', color='green', linewidth=2, label=f'Fine-Tuned (AUC={auc_f:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Comparison on External Data')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_roc.png"), dpi=160)
    plt.close()


# --------------------------- confusion matrix ---------------------------

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], normalize: bool,
                          out_path: str) -> None:
    """
    Save a (normalized) confusion matrix as a PNG.
    """
    cm = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-12
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues' if normalize else 'Purples')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix' + (' (normalized)' if normalize else ''))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else str(int(round(val)))
            color = "white" if val > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# --------------------------- ROC & PR curves ---------------------------

def plot_roc_pr_curves(y_true_onehot: np.ndarray, y_scores: np.ndarray,
                       class_names: List[str], out_dir: str = "reports") -> None:
    """
    One-vs-Rest ROC and PR curves for a multi-class classifier.
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    _ensure_dir(out_dir)

    # ROC curves (OvR)
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        if i >= y_true_onehot.shape[1]: break
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=160)
    plt.close()

    # Precision-Recall curves (OvR)
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        if i >= y_true_onehot.shape[1]: break
        prec, rec, _ = precision_recall_curve(y_true_onehot[:, i], y_scores[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_scores[:, i])
        plt.plot(rec, prec, label=f"{cls} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (One-vs-Rest)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curves.png"), dpi=160)
    plt.close()


# --------------------------- calibration & reliability ---------------------------

def _bin_stats(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-bin accuracy & confidence for reliability diagrams."""
    true_class_indices = labels.argmax(axis=1)
    # Clip to avoid index errors if probs shape mismatches labels
    true_class_probs = probs[np.arange(probs.shape[0]), true_class_indices]

    pred_class_indices = probs.argmax(axis=1)
    correct = (pred_class_indices == true_class_indices).astype(float)

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

    return bin_acc, bin_conf, bin_count, bin_edges


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE = sum_b (|bin|/N) * |acc(b) - conf(b)|"""
    bin_acc, bin_conf, bin_count, _ = _bin_stats(probs, labels, n_bins=n_bins)
    N = max(1, int(bin_count.sum()))
    gaps = np.abs(bin_acc - bin_conf)
    weights = bin_count / N
    return float(np.sum(weights * gaps))


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """MCE = max_b |acc(b) - conf(b)|"""
    bin_acc, bin_conf, bin_count, _ = _bin_stats(probs, labels, n_bins=n_bins)
    return float(np.max(np.abs(bin_acc - bin_conf)))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score."""
    return float(np.mean((labels - probs) ** 2))


def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray, out_path: str,
                             n_bins: int = 15) -> None:
    """Plot the reliability diagram (accuracy vs confidence)."""
    bin_acc, bin_conf, bin_count, _ = _bin_stats(probs, labels, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    width = 1.0 / n_bins
    centers = np.linspace(width/2, 1 - width/2, n_bins)

    ax.bar(centers, bin_acc, width=width*0.9, edgecolor="black", linewidth=0.5, alpha=0.8, label="Empirical accuracy")
    ax.plot([0,1], [0,1], linestyle="--", color="gray", label="Perfect calibration")

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
    """Plot the histogram of true-class confidence."""
    true_class_probs = probs[np.arange(probs.shape[0]), labels.argmax(axis=1)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(true_class_probs, bins=n_bins, range=(0,1), alpha=0.9, color='purple')
    ax.set_xlim(0,1)
    ax.set_xlabel("True-class confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Histogram (OvR)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_calibration_report(probs: np.ndarray, labels: np.ndarray,
                            out_dir: str = "reports", n_bins: int = 15) -> Dict[str, float]:
    """Compute and save standard calibration metrics and figures."""
    _ensure_dir(out_dir)

    ece = expected_calibration_error(probs, labels, n_bins=n_bins)
    mce = maximum_calibration_error(probs, labels, n_bins=n_bins)
    bs  = brier_score(probs, labels)

    report = {"ECE": float(ece), "MCE": float(mce), "BrierScore": float(bs)}

    with open(os.path.join(out_dir, "calibration_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    plot_reliability_diagram(probs, labels, out_path=os.path.join(out_dir, "reliability_diagram.png"), n_bins=n_bins)
    plot_confidence_histogram(probs, labels, out_path=os.path.join(out_dir, "confidence_hist.png"), n_bins=n_bins)

    return report
