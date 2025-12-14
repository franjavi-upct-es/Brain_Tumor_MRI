import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Add root directory to path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config


def load_and_preprocess_image(path, img_size, preprocess_input):
    """Load and preprocess an image for the model."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.expand_dims(img, axis=0)
    return preprocess_input(img)


def main(config_path, data_dir, save_results=True):
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
    true_binary_labels = []  # 0 = Healthy, 1 = Tumor

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

    print(f"[INFO] Evaluating {len(images_paths)} images from {data_path}...")

    # Inference
    pred_binary_labels = []
    pred_tumor_types = []
    pred_probs = []

    for path in tqdm(images_paths, desc="Evaluating"):
        img_tensor = load_and_preprocess_image(path, img_size, preprocess_input)
        logits = model.predict(img_tensor, verbose=0)
        probs = tf.nn.softmax(logits[0]).numpy()
        pred_idx = np.argmax(logits[0])

        # Store probabilities
        pred_probs.append(probs)

        # Multiclass -> Binary mapping
        if pred_idx == no_tumor_idx:
            pred_binary_labels.append(0)
            pred_tumor_types.append("N/A")
        else:
            pred_binary_labels.append(1)
            pred_tumor_types.append(class_names[pred_idx])

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

    print("\n--- Binary Classification (Healthy vs Tumor) ---")
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
        }

        # Determine output filename based on model type
        if model_name == "Fine-Tuned":
            output_file = os.path.join(
                cfg["train"]["checkpoint_dir"], "finetuned_external_results.json"
            )
        else:
            output_file = os.path.join(
                cfg["train"]["checkpoint_dir"], "base_external_results.json"
            )

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")
        print(f"   Use this file with: python tools/compare_models.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--data",
        default="data/external_navoneel_medical",
        help="Path to external dataset",
    )
    p.add_argument("--no-save", action="store_true", help="Don't save results to JSON")
    args = p.parse_args()

    main(args.config, args.data, save_results=not args.no_save)
