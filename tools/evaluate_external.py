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
from src.inference_utils import (  # noqa: E402
    aggregate_logits,
    apply_temperature,
    make_tta_layer,
    risk_triage_decision,
    softmax,
    tumor_score_from_probs,
)
from src.reporting import build_metadata  # noqa: E402


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


def load_temperature(ckpt_dir: Path) -> float:
    """Load temperature scalar from common calibration files if present."""
    for name in ("temperature.json", "focal_temperature.json"):
        path = Path(ckpt_dir) / name
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                return float(data.get("temperature", 1.0))
            except Exception:
                continue
    return 1.0


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


def main(config_path, data_dir, save_results=True, split="full", fn_topk=12, save_fn_audit=True, log_wandb=False, use_tta=None, tta_samples=None, mc_samples=None):
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
    threshold = cfg.get("inference", {}).get("threshold", 0.5)
    triage_cfg = cfg.get("inference", {}).get("triage", {})
    triage_enabled = triage_cfg.get("enabled", False)
    triage_band = float(triage_cfg.get("band", 0.05))
    triage_disagreement = float(triage_cfg.get("max_disagreement", 0.1))
    ensemble_cfg = cfg.get("inference", {}).get("ensemble", {})
    ensemble_enabled = ensemble_cfg.get("enabled", False)
    ensemble_strategy = ensemble_cfg.get("strategy", "mean")
    candidate_checkpoints = ensemble_cfg.get("checkpoints", [])
    use_calibration = cfg["inference"].get("use_calibration", True)
    tta_enabled = cfg["inference"].get("tta", False) if use_tta is None else bool(use_tta)
    tta_samples = int(cfg["inference"].get("tta_samples", 1) if tta_samples is None else tta_samples)
    mc_cfg = cfg["inference"].get("mc_dropout", {})
    mc_enabled = mc_cfg.get("enabled", False) or (mc_samples is not None and mc_samples > 0)
    mc_samples = int(mc_cfg.get("samples", 1) if mc_samples is None else mc_samples)
    last_threshold_reason = cfg["inference"].get("last_threshold_reason")

    try:
        no_tumor_idx = class_names.index("no_tumor")
    except ValueError:
        print("[ERROR] The 'no_tumor' class is not defined in your config.yaml")
        return

    manifest_path = Path(cfg["train"]["checkpoint_dir"]) / "navoneel_split.json"
    checkpoint_dir = Path(cfg["train"]["checkpoint_dir"])

    # Load ensemble members (or fall back to single model)
    models = []
    member_names = []
    member_paths = []

    if ensemble_enabled:
        for ckpt in candidate_checkpoints:
            path = checkpoint_dir / ckpt
            if path.exists():
                print(f"[INFO] Adding ensemble member: {path}")
                models.append(tf.keras.models.load_model(path, compile=False))
                member_names.append(path.name)
                member_paths.append(str(path))
            else:
                print(f"[WARN] Ensemble checkpoint not found: {path}")

    if not models:
        fallback = [
            ("Fine-Tuned", checkpoint_dir / "finetuned_navoneel.keras"),
            ("Base", checkpoint_dir / "best.keras"),
        ]
        for name, path in fallback:
            if path.exists():
                print(f"[INFO] Using {name} model: {path}")
                models.append(tf.keras.models.load_model(path, compile=False))
                member_names.append(path.name)
                member_paths.append(str(path))
                break

    if not models:
        print("[ERROR] No trained model found. Run training first.")
        return

    model_name = "Ensemble" if len(models) > 1 else ("Fine-Tuned" if "finetuned_navoneel.keras" in member_names[0] else "Base")
    primary_model = models[0]

    # Preprocessing according to backbone
    if "v2" in cfg["model"]["name"]:
        preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
    else:
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    tta_layer = make_tta_layer() if tta_enabled else None

    temperature = load_temperature(checkpoint_dir) if use_calibration else 1.0

    print(f"[INFO] Threshold={threshold:.2f} | Triage={'on' if triage_enabled else 'off'} (band={triage_band:.2f}, disagreement>{triage_disagreement:.2f})")
    print(f"[INFO] Ensemble members: {member_names} (strategy={ensemble_strategy})")
    print(f"[INFO] Calibration: {'on' if use_calibration else 'off'} (T={temperature:.2f})")
    print(f"[INFO] TTA: {'on' if tta_enabled else 'off'} (samples={tta_samples}) | MC Dropout: {'on' if mc_enabled else 'off'} (samples={mc_samples})")

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
    decisions = []

    for path, true_label in tqdm(
        list(zip(images_paths, true_binary_labels)), desc="Evaluating"
    ):
        preprocessed, _ = load_and_preprocess_image(path, img_size, preprocess_input)

        member_logits = []
        member_scores = []
        for model in models:
            logits_list = []
            passes_tta = tta_samples if tta_enabled else 1
            passes_mc = mc_samples if mc_enabled else 1
            for _ in range(passes_tta):
                x_aug = tta_layer(preprocessed, training=True) if tta_enabled else preprocessed
                for _ in range(passes_mc):
                    logits = model(x_aug, training=mc_enabled).numpy()[0]
                    logits = apply_temperature(logits, temperature) if use_calibration else logits
                    logits_list.append(logits)
            mean_logits = aggregate_logits(logits_list, strategy="mean")
            member_logits.append(mean_logits)
            probs_member = softmax(mean_logits, axis=0)
            member_scores.append(tumor_score_from_probs(probs_member, no_tumor_idx))

        agg_logits = aggregate_logits(member_logits, strategy=ensemble_strategy)
        agg_probs = softmax(agg_logits, axis=0)
        prob_no_tumor = float(agg_probs[no_tumor_idx])
        tumor_score = tumor_score_from_probs(agg_probs, no_tumor_idx)

        pred_idx = int(np.argmax(agg_logits))
        tumor_probs = np.delete(agg_probs, no_tumor_idx)
        top_tumor_idx = int(np.argmax(tumor_probs))
        # Map back to original class index (skip no_tumor slot)
        if top_tumor_idx >= no_tumor_idx:
            top_tumor_idx += 1

        if triage_enabled:
            decision, triage_info = risk_triage_decision(
                member_scores,
                threshold=threshold,
                triage_band=triage_band,
                max_disagreement=triage_disagreement,
            )
        else:
            decision = "tumor" if tumor_score >= threshold else "healthy"
            triage_info = {
                "score": tumor_score,
                "spread": 0.0,
                "min_score": tumor_score,
                "max_score": tumor_score,
                "threshold": float(threshold),
                "triage_band": float(triage_band),
                "max_disagreement": float(triage_disagreement),
                "reason": "threshold",
            }

        pred_binary = 1 if decision in ("tumor", "review") else 0
        pred_binary_labels.append(pred_binary)
        decisions.append(decision)
        pred_probs.append(agg_probs)
        pred_tumor_types.append(class_names[pred_idx] if pred_binary else "N/A")

        cases.append(
            {
                "path": path,
                "true_label": int(true_label),
                "pred_binary": pred_binary,
                "decision": decision,
                "prob_no_tumor": prob_no_tumor,
                "tumor_score": tumor_score,
                "member_scores": member_scores,
                "pred_idx": pred_idx,
                "top_tumor_idx": top_tumor_idx,
                "triage": triage_info,
            }
        )

    # Calculate metrics (review counts as "alert" to avoid FN by silence)
    cm = confusion_matrix(true_binary_labels, pred_binary_labels)
    tn, fp, fn, tp = cm.ravel()

    total = len(true_binary_labels)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0

    triage_indices = [i for i, d in enumerate(decisions) if d == "review"]
    triage_total = len(triage_indices)
    triage_tumors = sum(1 for i in triage_indices if true_binary_labels[i] == 1)
    triage_healthy = triage_total - triage_tumors
    triage_rate = triage_total / total if total > 0 else 0

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

    print("\n--- Safety / Triage ---")
    print(f"Triage rate: {triage_rate:.1%} ({triage_total}/{total})")
    print(f"  • Tumor cases triaged:   {triage_tumors}")
    print(f"  • Healthy cases triaged: {triage_healthy}")
    print(f"Threshold={threshold:.2f} (band={triage_band:.2f}, disagreement>{triage_disagreement:.2f})")

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
        results = build_metadata(
            config_path,
            data_dir,
            extra={
                "source": "evaluate_external.py",
                "split": split,
                "model": model_name,
                "model_paths": member_paths,
                "ensemble_members": member_names,
                "ensemble_strategy": ensemble_strategy,
                "use_calibration": bool(use_calibration),
                "temperature": float(temperature),
                "threshold": float(threshold),
                "last_threshold_reason": last_threshold_reason,
                "tta": {
                    "enabled": bool(tta_enabled),
                    "samples": int(tta_samples),
                },
                "preprocessing_mode": cfg.get("pipeline", {})
                .get("dataset_specific_preprocessing", {})
                .get("external", cfg.get("preprocessing", {}).get("mode", "unknown")),
                "counts": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                },
                "metrics": {
                    "recall": float(recall),
                    "precision": float(precision),
                    "specificity": float(specificity),
                    "accuracy": float(accuracy),
                    "fn_rate": float(fn_rate),
                },
                "confusion_matrix": cm.tolist(),
                "triage": {
                    "enabled": triage_enabled,
                    "band": float(triage_band),
                    "max_disagreement": float(triage_disagreement),
                    "total": triage_total,
                    "tumor_cases": triage_tumors,
                    "healthy_cases": triage_healthy,
                    "rate": float(triage_rate),
                },
                "mc_dropout": {
                    "enabled": mc_enabled,
                    "samples": mc_samples,
                },
            },
        )

        # Determine output filename based on model type
        suffix = "holdout" if split == "holdout" else "full"
        if len(models) > 1:
            filename = f"ensemble_external_results_{suffix}.json"
        elif model_name == "Fine-Tuned":
            filename = f"finetuned_external_results_{suffix}.json"
        else:
            filename = f"base_external_results_{suffix}.json"

        output_file = checkpoint_dir / filename
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")
        print(f"   Use this file with: python tools/compare_models.py")

    if log_wandb:
        try:
            import wandb

            run_name = f"external_eval_{model_name}_{split}"
            wandb.init(
                project="brain-tumor-mri-portfolio",
                job_type="external_eval",
                name=run_name,
                config={
                    "model": model_name,
                    "ensemble_members": member_names,
                    "ensemble_strategy": ensemble_strategy,
                    "use_calibration": bool(use_calibration),
                    "temperature": float(temperature),
                    "threshold": float(threshold),
                    "last_threshold_reason": last_threshold_reason,
                    "triage_enabled": triage_enabled,
                    "triage_band": triage_band,
                    "triage_max_disagreement": triage_disagreement,
                    "tta_enabled": tta_enabled,
                    "tta_samples": tta_samples,
                    "split": split,
                },
            )
            wandb.log(
                {
                    "external/recall": recall,
                    "external/specificity": specificity,
                    "external/precision": precision,
                    "external/accuracy": accuracy,
                    "external/fn_rate": fn_rate,
                    "external/triage_rate": triage_rate,
                    "external/tn": tn,
                    "external/fp": fp,
                    "external/fn": fn,
                    "external/tp": tp,
                    "external/triage_tumor": triage_tumors,
                    "external/triage_healthy": triage_healthy,
                }
            )
            wandb.finish()
        except Exception as e:
            print(f"[WARN] W&B logging failed: {e}")

    # Grad-CAM audit for false negatives
    if save_fn_audit:
        fn_cases = [
            case for case in cases if case["true_label"] == 1 and case["pred_binary"] == 0
        ]
        audit_dir = Path("reports") / ("fn_audit_holdout" if split == "holdout" else "fn_audit")
        if fn_cases:
            print(f"\n[INFO] Saving Grad-CAM audit for top-{min(fn_topk, len(fn_cases))} false negatives -> {audit_dir}")
            save_false_negative_audit(
                primary_model,
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
    p.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log metrics to Weights & Biases (uses project 'brain-tumor-mri-portfolio').",
    )
    p.add_argument("--use-tta", action="store_true", help="Enable Test Time Augmentation during evaluation.")
    p.add_argument("--tta-samples", type=int, default=None, help="Number of TTA samples (default from config).")
    p.add_argument("--mc-samples", type=int, default=None, help="Number of MC Dropout samples (default from config).")
    args = p.parse_args()

    main(
        args.config,
        args.data,
        save_results=not args.no_save,
        split=args.split,
        fn_topk=args.fn_topk,
        save_fn_audit=not args.no_fn_audit,
        log_wandb=args.log_wandb,
        use_tta=args.use_tta,
        tta_samples=args.tta_samples,
        mc_samples=args.mc_samples,
    )
