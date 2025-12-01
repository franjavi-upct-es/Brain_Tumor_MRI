# eval.py — Evaluation script (highly commented)
# ----------------------------------------------
# Loads the best checkpoint and evaluates on the test split.
# Saves artifacts to 'reports/': confusion matrices, ROC/PR curves,
# reliability diagram, confidence histogram, classification report,
# calibration metrics (ECE/MCE/Brier).

import argparse, os, json
import numpy as np
from src.utils import load_config, set_seed
from src.data import get_datasets
from sklearn.metrics import classification_report, confusion_matrix

def load_temperature(ckpt_dir: str) -> float:
    """Read temperature scalar from '{ckpt_dir}/temperature.json' if present."""
    path = os.path.join(ckpt_dir, "temperature.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return float(json.load(f).get("temperature", 1.0))
    return 1.0

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax: subtract max, exponentiate, normalize."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))
    _, val_ds, test_ds, class_names, _ = get_datasets(cfg)
    num_classes = len(class_names)

    # Load trained model (compiled=False for evaluation-internal logic)
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(cfg["train"]["checkpoint_dir"], "best.keras"), compile=False)

    # Temperature for calibrated probabilities
    T = load_temperature(cfg["train"]["checkpoint_dir"]) if cfg["inference"].get("use_calibration", True) else 1.0

    # Accumulate predictions & truths
    y_true = []          # integer class ids
    y_pred = []          # integer predicted class ids
    y_true_onehot = []   # one-hot labels
    y_scores = []        # calibrated probabilities

    for x, y in test_ds:
        logits = model.predict(x, verbose=0)
        logits = logits / T
        probs = softmax(logits, axis=1)
        y_scores.append(probs)
        y_pred += probs.argmax(axis=1).tolist()
        y_true += y.numpy().argmax(axis=1).tolist()
        y_true_onehot.append(y.numpy())

    y_scores = np.vstack(y_scores)
    y_true_onehot = np.vstack(y_true_onehot)

    # Classification report & confusion matrix
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)

    print(report)
    print(cm)

    # Save artifacts for README / reports
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "classification_report.txt"), "w") as f:
        f.write(report)

    np.savetxt(os.path.join("reports", "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    # Import plotting helpers
    from plots import (
        plot_confusion_matrix,
        plot_roc_pr_curves,
        save_calibration_report
    )

    # Save CM (raw + normalized)
    plot_confusion_matrix(cm, class_names, normalize=False, out_path=os.path.join("reports", "cm.png"))
    plot_confusion_matrix(cm, class_names, normalize=True,  out_path=os.path.join("reports", "cm_norm.png"))

    # ROC / PR curves (OvR)
    plot_roc_pr_curves(y_true_onehot, y_scores, class_names, out_dir="reports")

    # Calibration — reliability diagram, confidence hist, ECE/MCE/Brier
    calib_metrics = save_calibration_report(y_scores, y_true_onehot, out_dir="reports", n_bins=15)

    # Persist a JSON summary to reference in the README if needed
    summary = {
        "classes": class_names,
        "temperature": float(T),
        "calibration": calib_metrics,
    }
    with open(os.path.join("reports", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    main(args.config)
