import os
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import sys

# Ensure project root is on PYTHONPATH when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gradcam import compute_gradcam, overlay_heatmap  # noqa: E402
from src.utils import load_config  # noqa: E402


def load_and_preprocess_image(path, img_size, preprocess_input):
    """Load and preprocess an image for the model."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    return preprocess_input(img), img  # preprocessed, raw-resized


def infer_label_from_path(path: Path) -> int:
    """Infer binary label from folder name when manifest label is missing."""
    if "yes" in path.parts:
        return 1
    if "no" in path.parts:
        return 0
    # Fallback based on filename
    return 1 if "yes" in path.name.lower() else 0


def load_split_from_manifest(
    manifest_path: Path, split: str, data_dir: Path
) -> List[Dict]:
    """Load a specific split (train/val/holdout) from the manifest if it exists."""
    if not manifest_path.exists():
        return []
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read manifest {manifest_path}: {e}")
        return []

    base = Path(manifest.get("data_dir", data_dir)).resolve()
    entries = []
    for entry in manifest.get("splits", {}).get(split, []):
        rel_path = entry.get("path") if isinstance(entry, dict) else entry
        if rel_path is None:
            continue
        label = entry.get("label") if isinstance(entry, dict) else None
        full_path = (base / rel_path).resolve()
        entries.append(
            {
                "path": str(full_path),
                "label": label,
            }
        )
    return entries


def save_false_negative_audit(
    model,
    cases: List[Dict],
    preprocess_input,
    img_size: int,
    class_names: List[str],
    no_tumor_idx: int,
    out_dir: Path,
    top_k: int = 12,
):
    """
    Generate Grad-CAM overlays and metadata for the top-K high-confidence false negatives.
    """
    if not cases:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_cases = sorted(
        cases, key=lambda x: x.get("prob_no_tumor", 0.0), reverse=True
    )[:top_k]
    metadata = []

    for rank, case in enumerate(sorted_cases, 1):
        raw_pre, raw_resized = load_and_preprocess_image(
            case["path"], img_size, preprocess_input
        )
        vis_image = tf.cast(raw_resized[0], tf.uint8).numpy()

        healthy_heat = compute_gradcam(model, raw_pre, class_index=no_tumor_idx)
        tumor_cls = case.get("top_tumor_idx", None)
        tumor_heat = (
            compute_gradcam(model, raw_pre, class_index=tumor_cls)
            if tumor_cls is not None
            else None
        )

        # Save original + preprocessed view
        from PIL import Image

        Image.fromarray(vis_image).save(out_dir / f"{rank:02d}_original.png")

        # Preprocessed visualization (min-max scaled to 0-255)
        x_vis = raw_pre.numpy()[0]
        x_vis = (x_vis - x_vis.min()) / (x_vis.max() - x_vis.min() + 1e-8)
        Image.fromarray((x_vis * 255).astype(np.uint8)).save(
            out_dir / f"{rank:02d}_preprocessed.png"
        )

        # Grad-CAM overlays
        Image.fromarray(
            overlay_heatmap(vis_image, healthy_heat)
        ).save(out_dir / f"{rank:02d}_gradcam_healthy.png")

        if tumor_heat is not None:
            Image.fromarray(
                overlay_heatmap(vis_image, tumor_heat)
            ).save(out_dir / f"{rank:02d}_gradcam_tumor.png")

        metadata.append(
            {
                "path": case["path"],
                "true_label": "Tumor",
                "predicted": "Healthy",
                "prob_no_tumor": case.get("prob_no_tumor"),
                "tumor_score": case.get("tumor_score"),
                "top_tumor_class": class_names[tumor_cls]
                if tumor_cls is not None
                else None,
                "predicted_class": class_names[case["pred_idx"]],
            }
        )

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main(config_path, data_dir, save_results=True, split="full", fn_topk=12, save_fn_audit=True):
    # Verify that data exists
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ERROR] Data directory not found: {data_path}")
        print("Run first: python tools/preprocess_dataset.py")
        return

    # Load configuration and model
    cfg = load_config(config_path)
    img_size = cfg["data"]["image_size"]
    class_names = cfg["data"]["class_names"]

    try:
        no_tumor_idx = class_names.index("no_tumor")
    except ValueError:
        print("[ERROR] The 'no_tumor' class is not defined in your config.yaml")
        return

    manifest_path = Path(cfg["train"]["checkpoint_dir"]) / "navoneel_split.json"

    print(f"[INFO] Loading model from {cfg['train']['checkpoint_dir']}...")

    # Try to load fine-tuned model first, fall back to base
    finetuned_path = os.path.join(
        cfg["train"]["checkpoint_dir"], "finetuned_navoneel.keras"
    )
    base_path = os.path.join(cfg["train"]["checkpoint_dir"], "best.keras")

    if os.path.exists(finetuned_path):
        model_path = finetuned_path
        model_name = "Fine-Tuned"
        print(f"[INFO] Using fine-tuned model: {model_path}")
    elif os.path.exists(base_path):
        model_path = base_path
        model_name = "Base"
        print(f"[INFO] Using base model: {model_path}")
    else:
        print("[ERROR] No trained model found. Run training first.")
        return

    model = tf.keras.models.load_model(model_path, compile=False)

    # Preprocessing according to backbone
    if "v2" in cfg["model"]["name"]:
        preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
    else:
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    # Collect images
    images_paths = []
    true_binary_labels = []

    if split == "holdout":
        entries = load_split_from_manifest(manifest_path, "holdout", data_path)
        if entries:
            print(f"[INFO] Using holdout split from {manifest_path}")
            for entry in entries:
                images_paths.append(entry["path"])
                label = entry["label"]
                if label is None:
                    label = infer_label_from_path(Path(entry["path"]))
                true_binary_labels.append(int(label))
        else:
            print(
                f"[WARN] Holdout split requested but manifest missing. Falling back to full dataset at {data_path}"
            )
            split = "full"

    if split == "full":
        # Load 'no' class (Healthy) -> Label 0
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for p in (data_path / "no").glob(ext):
                images_paths.append(str(p))
                true_binary_labels.append(0)

        # Load 'yes' class (Tumor) -> Label 1
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for p in (data_path / "yes").glob(ext):
                images_paths.append(str(p))
                true_binary_labels.append(1)

    print(f"[INFO] Evaluating {len(images_paths)} images from {data_path} (split={split})...")

    # Inference
    pred_binary_labels = []
    pred_tumor_types = []
    pred_probs = []
    cases = []

    for path, true_label in tqdm(
        list(zip(images_paths, true_binary_labels)), desc="Evaluating"
    ):
        preprocessed, _ = load_and_preprocess_image(path, img_size, preprocess_input)
        logits = model.predict(preprocessed, verbose=0)
        probs = tf.nn.softmax(logits[0]).numpy()
        pred_idx = int(np.argmax(logits[0]))

        # Store probabilities
        pred_probs.append(probs)

        tumor_probs = np.delete(probs, no_tumor_idx)
        top_tumor_idx = int(np.argmax(tumor_probs))
        # Map back to original class index (skip no_tumor slot)
        if top_tumor_idx >= no_tumor_idx:
            top_tumor_idx += 1

        # Multiclass -> Binary mapping
        if pred_idx == no_tumor_idx:
            pred_binary_labels.append(0)
            pred_tumor_types.append("N/A")
        else:
            pred_binary_labels.append(1)
            pred_tumor_types.append(class_names[pred_idx])

        prob_no_tumor = float(probs[no_tumor_idx])
        cases.append(
            {
                "path": path,
                "true_label": int(true_label),
                "pred_binary": pred_binary_labels[-1],
                "prob_no_tumor": prob_no_tumor,
                "tumor_score": 1.0 - prob_no_tumor,
                "pred_idx": pred_idx,
                "top_tumor_idx": top_tumor_idx,
            }
        )

    # Calculate metrics
    cm = confusion_matrix(true_binary_labels, pred_binary_labels)
    tn, fp, fn, tp = cm.ravel()

    total = len(true_binary_labels)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0

    # Reports
    print("\n" + "=" * 70)
    print(f"RESULTS: EXTERNAL DATASET ({model_name} Model)")
    print("=" * 70)

    print(f"\n--- Binary Classification (Healthy vs Tumor) [{split}] ---")
    print(
        classification_report(
            true_binary_labels, pred_binary_labels, target_names=["Healthy", "Tumor"]
        )
    )

    print(f"\n--- Confusion Matrix ---")
    print(f"TN (Healthy OK): {tn} | FP (False Alarms): {fp}")
    print(f"FN (Undetected Tumors): {fn} | TP (Detected Tumors): {tp}")

    print("\n--- Clinical Metrics ---")
    print(f"Sensitivity (Recall):    {recall:.1%}  {'✓' if recall > 0.85 else '⚠️'}")
    print(f"Specificity:             {specificity:.1%}")
    print(f"Precision (PPV):         {precision:.1%}")
    print(f"Accuracy:                {accuracy:.1%}")
    print(f"False Negative Rate:     {fn_rate:.1%}  {'✓' if fn_rate < 0.15 else '⚠️'}")

    print("\n--- Predicted Tumor Types (in 'Yes' cases) ---")
    tumor_indices = [i for i, x in enumerate(true_binary_labels) if x == 1]
    from collections import Counter

    counts = Counter([pred_tumor_types[i] for i in tumor_indices])

    total_tumors = len(tumor_indices)
    for k, v in counts.items():
        if k == "N/A":
            continue
        print(f"- {k}: {v} ({v / total_tumors:.1%})")

    missed = counts.get("N/A", 0)
    if missed > 0:
        print(f"- Not detected: {missed} ({missed / total_tumors:.1%})")

    # Save results to JSON if requested
    if save_results:
        results = {
            "model": model_name,
            "model_path": model_path,
            "recall": float(recall),
            "precision": float(precision),
            "specificity": float(specificity),
            "accuracy": float(accuracy),
            "fn_rate": float(fn_rate),
            "confusion_matrix": cm.tolist(),
            "total_images": int(total),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "source": "evaluate_external.py",
            "split": split,
        }

        # Determine output filename based on model type
        suffix = "holdout" if split == "holdout" else "full"
        if model_name == "Fine-Tuned":
            output_file = os.path.join(
                cfg["train"]["checkpoint_dir"],
                f"finetuned_external_results_{suffix}.json",
            )
        else:
            output_file = os.path.join(
                cfg["train"]["checkpoint_dir"],
                f"base_external_results_{suffix}.json",
            )

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")
        print(f"   Use this file with: python tools/compare_models.py")

    # Grad-CAM audit for false negatives
    if save_fn_audit:
        fn_cases = [
            case for case in cases if case["true_label"] == 1 and case["pred_binary"] == 0
        ]
        audit_dir = Path("reports") / ("fn_audit_holdout" if split == "holdout" else "fn_audit")
        if fn_cases:
            print(f"\n[INFO] Saving Grad-CAM audit for top-{min(fn_topk, len(fn_cases))} false negatives -> {audit_dir}")
            save_false_negative_audit(
                model,
                fn_cases,
                preprocess_input,
                img_size,
                class_names,
                no_tumor_idx,
                audit_dir,
                top_k=fn_topk,
            )
        else:
            print("\n[INFO] No false negatives to audit 🎉")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--data",
        default="data/external_navoneel_medical",
        help="Path to external dataset",
    )
    p.add_argument("--no-save", action="store_true", help="Don't save results to JSON")
    p.add_argument(
        "--split",
        choices=["full", "holdout"],
        default="full",
        help="Evaluate the full external set or the holdout defined during fine-tuning.",
    )
    p.add_argument("--fn-topk", type=int, default=12, help="Number of FN cases to export Grad-CAMs for.")
    p.add_argument(
        "--no-fn-audit",
        action="store_true",
        help="Skip Grad-CAM export for false negatives.",
    )
    args = p.parse_args()

    main(
        args.config,
        args.data,
        save_results=not args.no_save,
        split=args.split,
        fn_topk=args.fn_topk,
        save_fn_audit=not args.no_fn_audit,
    )
