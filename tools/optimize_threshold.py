import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Add path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config

def load_and_preprocess_image(path, img_size, preprocess_input):
    """Load and preprocess an image for the model."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    return preprocess_input(img)

def main(config_path, data_dir):
    cfg = load_config(config_path)
    img_size = cfg['data']['image_size']
    class_names = cfg['data']['class_names']

    # Index of the healthy class
    try:
        no_tumor_idx = class_names.index('no_tumor')
    except ValueError:
        print("[ERROR] 'no_tumor' not found in class_names.")
        return
    
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(os.path.join(cfg['train']['checkpoint_dir'], 'finetuned_navoneel.keras'), compile=False)

    if "v2" in cfg["model"]["name"]:
        preprocess = tf.keras.applications.efficientnet_v2.preprocess_input
    else:
        preprocess = tf.keras.applications.efficientnet.preprocess_input

    # 1. Load all images and labels
    data_path = Path(data_dir)
    paths = []
    y_true = [] # 0: Healthy, 1: Tumor

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
    ds = ds.map(lambda x: load_and_preprocess_image(x, img_size, preprocess), num_parallel_calls=tf.data.AUTOTUNE)
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
    print("\n" + "=" * 90)
    print(f"{'Threshold':<10} | {'Recall (Tumor)':<15} | {'Precision':<10} | {'FP (False Alarms)':<20} | {'FN (Missed Tumors)'}")
    print("=" * 90)

    best_f1 = 0
    best_thresh = 0.5

    thresholds = np.arange(0.1, 0.95, 0.05)
    for t in thresholds:
        y_pred_t = (tumor_probs >= t).astype(int)

        # Metrics
        cm = confusion_matrix(y_true, y_pred_t)
        tn, fp, fn, tp = cm.ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{t:.2f}       | {recall:.1%}           | {precision:.1%}      | {fp:<20} | {fn}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print("=" * 90)
    print(f"[RECOMMENDATION] The best statistical balance (F1-Score) is at threshold: {best_thresh:.2f}")

    # Show detailed results with the best threshold
    print(f"\n--- Results with Optimized Threshold ({best_thresh:.2f}) ---")
    final_preds = (tumor_probs >= best_thresh).astype(int)
    print(classification_report(y_true, final_preds, target_names=['Healthy', 'Tumor']))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--data", default="data/external_navoneel", help="Path to external dataset"
    )
    args = p.parse_args()

    main(args.config, args.data)