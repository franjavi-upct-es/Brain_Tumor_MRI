import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
import sys

# Ensure project root is on PYTHONPATH when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config, set_seed  # noqa: E402
from src.data import _get_preprocess_fn  # noqa: E402


def stratified_holdout_split(
    data_dir: Path, holdout_split: float, val_split: float, seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    Create stratified splits (train/val/holdout) for the external binary dataset.
    Splits are class-balanced and deterministic for reproducibility.
    """
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "holdout": []}

    for label_name, label_idx in (("no", 0), ("yes", 1)):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            files.extend(sorted((data_dir / label_name).glob(ext)))
        if not files:
            continue

        files = np.array(files)
        rng.shuffle(files)

        n = len(files)
        n_holdout = int(n * holdout_split)
        n_val = int((n - n_holdout) * val_split)

        holdout_files = files[:n_holdout]
        val_files = files[n_holdout : n_holdout + n_val]
        train_files = files[n_holdout + n_val :]

        for p in train_files:
            splits["train"].append({"path": str(p.relative_to(data_dir)), "label": label_idx})
        for p in val_files:
            splits["val"].append({"path": str(p.relative_to(data_dir)), "label": label_idx})
        for p in holdout_files:
            splits["holdout"].append({"path": str(p.relative_to(data_dir)), "label": label_idx})

    return splits


def make_binary_dataset(
    entries: List[Dict],
    data_dir: Path,
    batch_size: int,
    img_size: int,
    preprocess,
    augment=None,
    shuffle=False,
):
    """Build a tf.data.Dataset from manifest entries."""
    if not entries:
        return None

    paths = [str(data_dir / e["path"]) for e in entries]
    labels = [e["label"] for e in entries]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32)
        if augment is not None:
            img = augment(img, training=True)
        img = preprocess(img)
        return img, tf.cast(label, tf.float32)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def save_manifest(manifest: Dict, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_dir / "navoneel_split.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Saved split manifest -> {out_path}")


def evaluate_holdout(model, holdout_ds, no_tumor_idx: int, threshold: float, ckpt_dir: Path):
    """Quick binary evaluation on the holdout split to ensure no leakage."""
    if holdout_ds is None:
        print("[WARN] No holdout split available for evaluation.")
        return

    y_true, y_pred = [], []
    for x, y in holdout_ds:
        logits = model.predict(x, verbose=0)
        probs = tf.nn.softmax(logits, axis=1).numpy()
        tumor_scores = 1.0 - probs[:, no_tumor_idx]
        preds = (tumor_scores >= threshold).astype(int)
        y_true += y.numpy().astype(int).flatten().tolist()
        y_pred += preds.tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0

    metrics = {
        "total": int(len(y_true)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "accuracy": accuracy,
        "threshold": threshold,
    }

    out_path = ckpt_dir / "navoneel_holdout_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"[INFO] Holdout metrics -> recall={recall:.2%}, specificity={specificity:.2%}, fn={fn} (saved to {out_path})"
    )


def main(config_path, data_dir, holdout_split=0.15):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    holdout_split = cfg.get("train", {}).get("holdout_split_external", holdout_split)

    # 1. Basic configuration
    img_size = cfg["data"]["image_size"]
    batch_size = 16  # Small batch for fine-tuning
    lr = 1e-5  # VERY low Learning Rate to not break what was learned
    epochs = 10
    threshold = cfg.get("inference", {}).get("threshold", 0.5)

    class_name_orig = cfg["data"][
        "class_names"
    ]  # ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    try:
        no_tumor_idx = class_name_orig.index("no_tumor")
    except ValueError:
        print("[ERROR] 'no_tumor' not found in class_names.")
        return

    print(f"[INFO] 'no_tumor' class index: {no_tumor_idx}")

    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"[ERROR] External dataset not found at {data_dir}")
        return

    # 2. Load Navoneel Dataset with explicit holdout split
    print(f"[INFO] Loading data from {data_dir}...")

    val_split = 0.2
    splits = stratified_holdout_split(
        data_dir, holdout_split=holdout_split, val_split=val_split, seed=42
    )
    manifest = {
        "data_dir": str(data_dir.resolve()),
        "holdout_split": holdout_split,
        "val_split": val_split,
        "seed": 42,
        "splits": splits,
    }
    save_manifest(manifest, Path(cfg["train"]["checkpoint_dir"]))

    preprocess = _get_preprocess_fn(cfg["model"]["name"])

    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomBrightness(0.1),
        ]
    )

    train_ds = make_binary_dataset(
        splits["train"],
        data_dir,
        batch_size,
        img_size,
        preprocess,
        augment=augment,
        shuffle=True,
    )
    val_ds = make_binary_dataset(
        splits["val"], data_dir, batch_size, img_size, preprocess, augment=None
    )
    holdout_ds = make_binary_dataset(
        splits["holdout"], data_dir, batch_size, img_size, preprocess, augment=None
    )

    # 4. Load Pre-trained Model
    checkpoint_path = os.path.join(cfg["train"]["checkpoint_dir"], "best.keras")
    print(f"[INFO] Loading base model: {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path, compile=False)

    # Unfreeze last layers to allow adaptation
    model.trainable = True

    # 5. Define 'Hybrid Loss' (Custom Loss)
    def custom_binary_loss(y_true, y_pred_logits):
        """
        y_true: 0 (Healthy/No) or 1 (Tumor/Yes)
        y_pred_logits: Logits of the 4 classes ['glioma', 'meningioma', 'no_tumor', 'pituitary']

        Goal:
        - If y_true=0 (Healthy): We want 'no_tumor' to be high
        - If y_true=1 (Tumor): We want 'no_tumor' to be low
        """
        no_tumor_logit = y_pred_logits[:, no_tumor_idx]
        target_no_tumor = 1.0 - tf.cast(y_true, tf.float32)
        target_no_tumor = tf.reshape(target_no_tumor, [-1])
        no_tumor_logit = tf.reshape(no_tumor_logit, [-1])

        return tf.keras.losses.binary_crossentropy(
            target_no_tumor, no_tumor_logit, from_logits=True
        )

    # 6. Compile and Train
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=custom_binary_loss,
        metrics=[],  # Standard accuracy metrics don't work well here due to label format
    )

    print("[INFO] Starting Fine-Tuning...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # 7. Save the new adapted model
    save_path = os.path.join(cfg["train"]["checkpoint_dir"], "finetuned_navoneel.keras")
    model.save(save_path)
    print(f"\n[OK] Retrained model saved at: {save_path}")

    # 8. Quick holdout evaluation to ensure no leakage with adaptation
    evaluate_holdout(
        model,
        holdout_ds,
        no_tumor_idx=no_tumor_idx,
        threshold=threshold,
        ckpt_dir=Path(cfg["train"]["checkpoint_dir"]),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data", default="data/external_navoneel_medical")
    p.add_argument(
        "--holdout-split",
        type=float,
        default=0.15,
        help="Fraction of external data reserved as untouched holdout (not used for fine-tuning).",
    )
    args = p.parse_args()

    main(args.config, args.data, holdout_split=args.holdout_split)
