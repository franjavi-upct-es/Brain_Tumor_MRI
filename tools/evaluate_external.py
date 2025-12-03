import os
import sys
import argparse
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


def main(config_path, data_dir):
    # Verify that data exists
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ERROR] Data directory not found: {data_path}")
        print("Run first: python tools/download_navoneel.py")
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
    model_path = os.path.join(cfg["train"]["checkpoint_dir"], "finetuned_navoneel.keras")
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

    for path in tqdm(images_paths):
        img_tensor = load_and_preprocess_image(
            path, img_size, preprocess_input)
        logits = model.predict(img_tensor, verbose=0)
        pred_idx = np.argmax(logits[0])

        # Multiclass -> Binary mapping
        if pred_idx == no_tumor_idx:
            pred_binary_labels.append(0)
            pred_tumor_types.append("N/A")
        else:
            pred_binary_labels.append(1)
            pred_tumor_types.append(class_names[pred_idx])

    # Reports
    print("\n" + "=" * 40)
    print("RESULTS: EXTERNAL DATASET (Navoneel)")
    print("=" * 40)

    print("\n--- Binary Classification (Healthy vs Tumor) ---")
    print(
        classification_report(
            true_binary_labels, pred_binary_labels, target_names=[
                "Healthy", "Tumor"]
        )
    )

    cm = confusion_matrix(true_binary_labels, pred_binary_labels)
    print(f"Confusion Matrix:\n{cm}")
    print(f"TN (Healthy OK): {cm[0][0]} | FP (False Alarms): {cm[0][1]}")
    print(
        f"FN (Undetected Tumors): {
            cm[1][0]} | TP (Detected Tumors): {cm[1][1]}"
    )

    print("\n--- Predicted Tumor Types (in 'Yes' cases) ---")
    tumor_indices = [i for i, x in enumerate(true_binary_labels) if x == 1]
    from collections import Counter

    counts = Counter([pred_tumor_types[i] for i in tumor_indices])

    total = len(tumor_indices)
    for k, v in counts.items():
        if k == "N/A":
            continue
        print(f"- {k}: {v} ({v / total:.1%})")

    missed = counts.get("N/A", 0)
    if missed > 0:
        print(f"- Not detected: {missed} ({missed / total:.1%})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--data", default="data/external_navoneel", help="Path to external dataset"
    )
    args = p.parse_args()

    main(args.config, args.data)
