import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import yaml

from src.inference_utils import (
    aggregate_logits,
    apply_temperature,
    softmax,
)
from src.reporting import build_metadata
from src.utils import load_config


def load_temperature(ckpt_dir: Path) -> float:
    """Load temperature scalar from common calibration files if present."""
    for name in ("temperature.json", "focal_temperature.json"):
        path = Path(ckpt_dir) / name
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f) if name.endswith(".yaml") else json.load(f)
                return float(data.get("temperature", 1.0))
            except Exception:
                continue
    return 1.0


def load_and_preprocess_image(path, img_size, preprocess_input):
    """Load and preprocess an image for the model."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    return preprocess_input(img)


def main(config_path, data_dir, log_wandb=False):
    cfg = load_config(config_path)
    img_size = cfg["data"]["image_size"]
    class_names = cfg["data"]["class_names"]
    target_recall = cfg["inference"].get("target_recall", 0.85)
    max_fp_rate = cfg["inference"].get("max_fp_rate", 0.15)
    previous_threshold = cfg.get("inference", {}).get("threshold", 0.5)
    ensemble_cfg = cfg.get("inference", {}).get("ensemble", {})
    ensemble_enabled = ensemble_cfg.get("enabled", False)
    ensemble_strategy = ensemble_cfg.get("strategy", "mean")
    candidate_checkpoints = ensemble_cfg.get("checkpoints", [])
    use_calibration = cfg["inference"].get("use_calibration", True)
    tta_enabled = cfg["inference"].get("tta", False)
    tta_samples = cfg["inference"].get("tta_samples", 1)

    # Index of the healthy class
    try:
        no_tumor_idx = class_names.index("no_tumor")
    except ValueError:
        print("[ERROR] 'no_tumor' not found in class_names.")
        return

    checkpoint_dir = Path(cfg["train"]["checkpoint_dir"])
    temperature = load_temperature(checkpoint_dir) if use_calibration else 1.0

    print("[INFO] Loading models for threshold optimization...")
    models = []
    member_names = []
    member_paths = []

    if ensemble_enabled:
        for ckpt in candidate_checkpoints:
            path = checkpoint_dir / ckpt
            if path.exists():
                models.append(tf.keras.models.load_model(path, compile=False))
                member_names.append(path.name)
                member_paths.append(str(path))
                print(f"  ✓ Added ensemble member: {path}")
            else:
                print(f"  ⚠️ Missing ensemble checkpoint: {path}")

    if not models:
        fallback = [
            ("Fine-Tuned", checkpoint_dir / "finetuned_navoneel.keras"),
            ("Base", checkpoint_dir / "best.keras"),
        ]
        for name, path in fallback:
            if path.exists():
                models.append(tf.keras.models.load_model(path, compile=False))
                member_names.append(path.name)
                member_paths.append(str(path))
                print(f"  ✓ Using {name} model: {path}")
                break

    if not models:
        print("[ERROR] No trained model found. Run training first.")
        return

    model_name = "Ensemble" if len(models) > 1 else ("Fine-Tuned" if "finetuned_navoneel.keras" in member_names[0] else "Base")

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

    y_true = np.array(y_true)

    print(f"[INFO] Calculating probabilities for {len(paths)} images...")
    print(f"       Ensemble: {member_names} (strategy={ensemble_strategy})")
    print(f"       Calibration: {'on' if use_calibration else 'off'} (T={temperature:.2f})")

    # Process in batches for speed
    batch_size = 32
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        lambda x: load_and_preprocess_image(x, img_size, preprocess),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    member_logits = []
    for model in models:
        logits = model.predict(ds, verbose=1)
        logits = apply_temperature(logits, temperature) if use_calibration else logits
        member_logits.append(logits)

    agg_logits = aggregate_logits(member_logits, strategy=ensemble_strategy)
    agg_probs = softmax(agg_logits, axis=1)

    # 3. Calculate "Tumor Probability" (sum of all classes that are not no_tumor)
    # Prob Tumor = 1.0 - Prob(no_tumor)
    tumor_probs = 1.0 - agg_probs[:, no_tumor_idx]

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
    print(f"[RECOMMENDATION] Threshold: {best_thresh:.2f} ({reason})")

    # Show detailed results with the chosen threshold
    print(f"\n--- Results with Optimized Threshold ({best_thresh:.2f}) ---")
    final_preds = (tumor_probs >= best_thresh).astype(int)
    print(classification_report(y_true, final_preds, target_names=["Healthy", "Tumor"]))

    cm_final = confusion_matrix(y_true, final_preds)
    tn, fp, fn, tp = cm_final.ravel()
    fp_rate_final = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate_final = fn / (tp + fn) if (tp + fn) > 0 else 0
    recall_final = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_final = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity_final = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Update config threshold for downstream inference/eval
    cfg.setdefault("inference", {})["threshold"] = float(best_thresh)
    cfg["inference"]["last_threshold_reason"] = reason
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"\n[INFO] Updated config inference.threshold -> {best_thresh:.2f} in {config_path}")

    # Persist metadata-rich summary for reproducibility
    os.makedirs("reports", exist_ok=True)
    results = build_metadata(
        config_path,
        data_dir,
        extra={
            "source": "optimize_threshold.py",
            "model": model_name,
            "ensemble_members": member_names,
            "ensemble_strategy": ensemble_strategy,
            "model_paths": member_paths,
            "use_calibration": bool(use_calibration),
            "temperature": float(temperature),
            "preprocessing_mode": cfg.get("pipeline", {})
            .get("dataset_specific_preprocessing", {})
            .get("external", cfg.get("preprocessing", {}).get("mode", "unknown")),
            "tta": {"enabled": bool(tta_enabled), "samples": int(tta_samples)},
            "target_recall": float(target_recall),
            "max_fp_rate": float(max_fp_rate),
            "previous_threshold": float(previous_threshold),
            "recommended_threshold": float(best_thresh),
            "reason": reason,
            "counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "metrics": {
                "recall": float(recall_final),
                "precision": float(precision_final),
                "specificity": float(specificity_final),
                "fp_rate": float(fp_rate_final),
                "fn_rate": float(fn_rate_final),
            },
        },
    )
    out_path = Path("reports") / "threshold_optimization.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Threshold optimization report saved to {out_path}")

    if log_wandb:
        try:
            import wandb

            wandb.init(
                project="brain-tumor-mri-portfolio",
                job_type="threshold_opt",
                name="threshold_optimization",
                config={
                    "model": model_name,
                    "ensemble_members": member_names,
                    "ensemble_strategy": ensemble_strategy,
                    "use_calibration": bool(use_calibration),
                    "temperature": float(temperature),
                    "target_recall": float(target_recall),
                    "max_fp_rate": float(max_fp_rate),
                    "previous_threshold": float(previous_threshold),
                    "recommended_threshold": float(best_thresh),
                },
            )
            wandb.log(
                {
                    "threshold_opt/recommended": best_thresh,
                    "threshold_opt/recall": recall_final,
                    "threshold_opt/precision": precision_final,
                    "threshold_opt/specificity": specificity_final,
                    "threshold_opt/fp_rate": fp_rate_final,
                    "threshold_opt/fn_rate": fn_rate_final,
                    "threshold_opt/tn": tn,
                    "threshold_opt/fp": fp,
                    "threshold_opt/fn": fn,
                    "threshold_opt/tp": tp,
                }
            )
            wandb.finish()
        except Exception as e:
            print(f"[WARN] W&B logging failed: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--data", default="data/external_navoneel", help="Path to external dataset"
    )
    p.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log threshold search results to Weights & Biases.",
    )
    args = p.parse_args()

    main(args.config, args.data, log_wandb=args.log_wandb)
