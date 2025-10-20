# train_kfold.py - Stratified K-Fold cross-validation training
# ------------------------------------------------------------
# This script provides a quick way to estimate performance variability by
# splitting the dataset (file-based) into K folds. For each fold, we train a
# new model and report the best validation accuracy. The final summary gives 
# mean and std across folds. This is not meant to replace a held-out test set,
# but it helps diagnose variance and overfitting.
#
# NOTE: This script reads files directly from 'data/root_dir/<class>/*' so it
# bypasses image_dataset_from_directory for flexibility
#
# Usage:
#   python -m src.train_kfold --config configs/config.yaml

import argparse, os, glob
from typing import List
from sklearn.model_selection import StratifiedKFold
import numpy as np
from utils import load_config, set_seed
from model import create_model

def collect_files(root_dir: str, class_names: List[str]):
    """Colect image file paths and their numeric labels from a class-based tree"""
    data, y = [], []
    for i, cls in enumerate(class_names):
        for p in glob.glob(os.path.join(root_dir, cls, "**", "*.*"), recursive=True):
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                data.append(p); y.append(i)
    return np.array(data), np.array(y)

def make_ds(files, labels, img_size, batch, preprocess, shuffle):
    """
    Create a simple tf.data pipeline from file paths. We keep this minimal for clarity.
    """
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if shuffle:
        ds = ds.shuffle(len(files), seed=42, reshuffle_each_iteration=False)
    
    def _load(f, y):
        img = tf.io.read_file(f)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img = preprocess(img)
        y_one = tf.one_hot(y, tf.reduce_max(labels)+1)
        return img, y_one

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def main(cfg_path):
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))
    import tensorflow as tf

    class_names = cfg["data"]["class_names"]
    files, labels = collect_files(cfg["data"]["root_dir"], class_names)

    # Preprocess according to backbone
    if "v2" in cfg["model"]["name"]:
        preprocess = tf.keras.applications.efficientnet_v2.preprocess_input
    else:
        preprocess = tf.keras.applications.efficientnet.preprocess_input

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(files, labels), start=1):
        print(f"Fold {fold}")
        train_ds = make_ds(files[train_idx], labels[train_idx], cfg["data"]["image_size"], cfg["train"]["batch_size"], preprocess, shuffle=True)
        val_ds = make_ds(files[val_idx], labels[val_idx], cfg["data"]["image_size"], cfg["train"]["batch_size"], preprocess, shuffle=False)

        steps_per_epoch = max(1, tf.data.experimental.cardinality(train_ds).numpy())
        model = create_model(cfg, len(class_names))

        # Freeze/unfreeze schedule (shorter than main train.py for speed)
        for layer in model.layers:
            if "efficientnet" in layer.name:
                layer.trainable = False
        opt = tf.keras.optimizers.AdamW(learning_rate=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=cfg["train"]["freeze_backbone_epochs"])

        for layer in model.layers:
            layer.trainable = True
        opt = tf.keras.optimizers.AdamW(learning_rate=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
        model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
        hist = model.fit(train_ds, validation_data=val_ds, epochs=cfg["train"]["epochs"])

        best = max(hist.history.get("val_accuracy", [0]))
        metrics_per_fold.append(best)
        print(f"Fold best val acc: {best}")

    print("Mean±std val acc:", float(np.mean(metrics_per_fold)), float(np.std(metrics_per_fold)))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--config", default='configs/config.yaml')
    args = p.parse_args()
    main(args.config)