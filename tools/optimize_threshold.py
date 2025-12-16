import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import yaml

from src.utils import load_config


def load_and_preprocess_image(path, img_size, preprocess_input):
    """Load and preprocess an image for the model."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    return preprocess_input(img)


def main(config_path, data_dir):
    cfg = load_config(config_path)
    img_size = cfg["data"]["image_size"]
    class_names = cfg["data"]["class_names"]
    target_recall = cfg["inference"].get("target_recall", 0.85)
    max_fp_rate = cfg["inference"].get("max_fp_rate", 0.15)

    # Index of the healthy class
    try:
        no_tumor_idx = class_names.index("no_tumor")
    except ValueError:
        print("[ERROR] 'no_tumor' not found in class_names.")
        return

    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(
        os.path.join(cfg["train"]["checkpoint_dir"], "finetuned_navoneel.keras"),
        compile=False,
    )

    if "v2" in cfg["model"]["name"]:
        preprocess = tf.keras.applications.efficientnet_v2.preprocess_input
    else:
        preprocess = tf.keras.applications.efficientnet.preprocess_input

    # 1. Load all images and labels
    data_path = Path(data_dir)
    paths = []
    y_true = []  # 0: Healthy, 1: Tumor

    print("[INFO] Loading dataset for analysis")
    # Healthy
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for p in (data_path / "no").glob(ext):
            paths.append(str(p))
            y_true.append(0)
    # Tumors
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for p in (data_path / "yes").glob(ext):
            paths.append(str(p))
            y_true.append(1)

    y_true = np.array(y_true)

    # 2. Get RAW predictions (probabilities)
    print(f"[INFO] Calculating probabilities for {len(paths)} images...")

    # Process in batches for speed
    batch_size = 32
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        lambda x: load_and_preprocess_image(x, img_size, preprocess),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    all_probs = model.predict(ds, verbose=1)

    # Apply Softmax if the model returned logits
    # If the sum of a row is not approx. 1, apply softmax
    if not np.allclose(np.sum(all_probs[0]), 1.0, atol=0.1):
        all_probs = tf.nn.softmax(all_probs).numpy()

    # 3. Calculate "Tumor Probability" (sum of all classes that are not no_tumor)
    # Prob Tumor = 1.0 - Prob(no_tumor)
    tumor_probs = 1.0 - all_probs[:, no_tumor_idx]

    # Threshold Sweep
    print("\n" + "=" * 110)
    print(
        f"{'Threshold':<10} | {'Recall (Tumor)':<15} | {'Precision':<10} | {'FP Rate':<10} | {'FP (False Alarms)':<20} | {'FN (Missed Tumors)'}"
    )
    print("=" * 110)

    best_by_constraint = None  # (recall, fp_rate, thresh, precision, fp, fn)
    best_recall_under_fp = None
    best_f1 = (0, 0.5)  # (f1, thresh)

    thresholds = np.arange(0.1, 0.95, 0.05)
    for t in thresholds:
        y_pred_t = (tumor_probs >= t).astype(int)

        # Metrics
        cm = confusion_matrix(y_true, y_pred_t)
        tn, fp, fn, tp = cm.ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(
            f"{t:.2f}       | {recall:.1%}           | {precision:.1%}      | {fp_rate:.1%}     | {fp:<20} | {fn}"
        )

        meets_constraint = (recall >= target_recall) and (fp_rate <= max_fp_rate)
        within_fp = fp_rate <= max_fp_rate

        if meets_constraint:
            if (
                best_by_constraint is None
                or recall > best_by_constraint[0]
                or (
                    recall == best_by_constraint[0]
                    and fp_rate < best_by_constraint[1]
                )
            ):
                best_by_constraint = (recall, fp_rate, t, precision, fp, fn)
        if within_fp:
            if (best_recall_under_fp is None) or (recall > best_recall_under_fp[0]):
                best_recall_under_fp = (recall, fp_rate, t, precision, fp, fn)
        if f1 > best_f1[0]:
            best_f1 = (f1, t)

    print("=" * 110)

    if best_by_constraint:
        chosen = best_by_constraint
        reason = (
            f"meets clinical target recall>={target_recall:.2f} with FP rate<={max_fp_rate:.2f}"
        )
    elif best_recall_under_fp:
        chosen = best_recall_under_fp
        reason = (
            f"max recall subject to FP rate<={max_fp_rate:.2f} (target recall not reached)"
        )
    else:
        chosen = (
            None,
            None,
            best_f1[1],
            None,
            None,
            None,
        )
        reason = "fallback to best F1 because FP constraint was violated at all thresholds"

    best_thresh = chosen[2]
    print(
        f"[RECOMMENDATION] Threshold: {best_thresh:.2f} ({reason})"
    )

    # Show detailed results with the chosen threshold
    print(f"\n--- Results with Optimized Threshold ({best_thresh:.2f}) ---")
    final_preds = (tumor_probs >= best_thresh).astype(int)
    print(classification_report(y_true, final_preds, target_names=["Healthy", "Tumor"]))

    # Update config threshold for downstream inference/eval
    cfg.setdefault("inference", {})["threshold"] = float(best_thresh)
    cfg["inference"]["last_threshold_reason"] = reason
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(
        f"\n[INFO] Updated config inference.threshold -> {best_thresh:.2f} in {config_path}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--data", default="data/external_navoneel", help="Path to external dataset"
    )
    args = p.parse_args()

    main(args.config, args.data)
