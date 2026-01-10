# train.py — Training script (highly commented)
# ---------------------------------------------
# Two-stage fine-tuning for EfficientNet/EfficientNetV2:
# 1) Freeze the backbone; train the classifier head.
# 2) Unfreeze; fine-tune the whole network.
# Also computes temperature scaling (calibration) on the validation set.
# Saves curves into 'reports/'.

import argparse
import os
import json
import sys
from pathlib import Path

# GPU Configuration - must be set BEFORE importing TensorFlow
# NVIDIA GPU is device 0 (Intel iGPU doesn't count as CUDA device)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use NVIDIA GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Reduce TF logging noise

import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reporting import file_fingerprint
from src.utils import load_config, set_seed
from src.data import get_datasets
from src.model import create_model
from src.losses import get_loss_function


def build_optimizer(cfg, steps_per_epoch):
    """
    Build the optimizer for training.
    Uses AdamW (or Adam fallback), with CosineDecayRestarts if enabled.
    """
    import tensorflow as tf

    lr = cfg["train"]["lr"]
    if cfg["train"].get("use_cosine_decay", True):
        schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr,
            first_decay_steps=max(steps_per_epoch * 3, 100),
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-4,
        )
        lr = schedule
    opt_name = cfg["train"].get("optimizer", "adamw").lower()
    if opt_name == "adamw":
        try:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr, weight_decay=cfg["train"]["weight_decay"]
            )
        except Exception:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    return optimizer


def set_mixed_precision(enabled=True):
    """Enable mixed precision (float16 compute / float32 params) if supported."""
    if not enabled:
        return
    try:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        pass


def freeze_backbone(model, freeze=True):
    """
    Freeze or unfreeze the EfficientNet backbone layers.
    The final classifier Dense layer is named 'logits' and should remain trainable.
    """
    for layer in model.layers:
        if "efficientnet" in layer.name and freeze:
            layer.trainable = False
    # ensure logits is trainable
    if hasattr(model, "get_layer"):
        try:
            model.get_layer("logits").trainable = True
        except Exception:
            pass


def resolve_loss(cfg, key: str, fallback: str):
    """
    Select a loss function from config, defaulting to categorical crossentropy.
    Supports focal/tversky/weighted_bce for FN-oriented fine-tuning.
    """
    train_cfg = cfg.get("train", {})
    name = train_cfg.get(key, fallback)
    params = {}
    if name == "tversky":
        params = {
            "alpha": train_cfg.get("tversky_alpha", 0.3),
            "beta": train_cfg.get("tversky_beta", 0.7),
        }
    elif name == "weighted_bce":
        params = {"pos_weight": train_cfg.get("pos_weight", 3.0)}
    elif name == "focal":
        params = {
            "alpha": train_cfg.get("focal_alpha", 0.75),
            "gamma": train_cfg.get("focal_gamma", 2.0),
            "label_smoothing": train_cfg.get("label_smoothing", 0.0),
        }
    return get_loss_function(name, from_logits=True, **params)


def calibrate_temperature(model, val_ds, max_iters=200):
    """
    Learn temperature T>0 to minimize NLL on the validation set.
    Optimize log(T) for numerical stability; exponentiate at the end.
    """
    import tensorflow as tf
    import numpy as np

    # Collect logits and labels
    logits_list, y_list = [], []
    for x, y in val_ds:
        logits = model.predict(x, verbose=0)
        logits_list.append(logits)
        y_list.append(y.numpy())
    logits = np.concatenate(logits_list, axis=0)
    y_true = np.concatenate(y_list, axis=0)

    logT = tf.Variable(1.0, dtype=tf.float32)  # log temperature
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    for _ in range(max_iters):
        with tf.GradientTape() as tape:
            T = tf.exp(logT)
            scaled = logits / T
            loss = loss_fn(y_true, scaled)
        grads = tape.gradient(loss, [logT])
        opt.apply_gradients(zip(grads, [logT]))
    T_final = float(tf.exp(logT).numpy())
    return T_final


def init_wandb_tracking(cfg):
    """
    Initialize Weights & Biases for experiment tracking.
    Logs all configuration and allows easy comparison between runs.
    """
    # Generate unique experiment name
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg['model']['name']}_{cfg['preprocessing']['mode']}_{timestamp}"

    inf_cfg = cfg.get("inference", {})
    triage_cfg = inf_cfg.get("triage", {})
    ensemble_cfg = inf_cfg.get("ensemble", {})
    mc_cfg = inf_cfg.get("mc_dropout", {})

    dataset_fps = {
        "train": file_fingerprint(Path(cfg["data"].get("train_dir", "")))[0],
        "val": file_fingerprint(Path(cfg["data"].get("val_dir", "")))[0],
        "test": file_fingerprint(Path(cfg["data"].get("test_dir", "")))[0],
        "external": file_fingerprint(Path("data/external_navoneel_medical"))[0],
    }

    # Initialize W&B
    wandb.init(
        project="brain-tumor-mri-portfolio",  # Project name
        name=run_name,
        config={
            # Model config
            "architecture": cfg["model"]["name"],
            "pretrained": cfg["model"]["pretrained"],
            "dropout": cfg["model"]["dropout"],
            "pooling": cfg["model"]["pooling"],
            # Data config
            "image_size": cfg["data"]["image_size"],
            "num_classes": cfg["data"]["num_classes"],
            "preprocessing_mode": cfg["preprocessing"]["mode"],
            # Training config
            "learning_rate": cfg["train"]["lr"],
            "batch_size": cfg["train"]["batch_size"],
            "epochs": cfg["train"]["epochs"],
            "optimizer": cfg["train"]["optimizer"],
            "weight_decay": cfg["train"]["weight_decay"],
            "freeze_epochs": cfg["train"]["freeze_backbone_epochs"],
            "use_class_weights": cfg["train"]["use_class_weights"],
            "use_cosine_decay": cfg["train"]["use_cosine_decay"],
            # Augmentation config
            "mixup_alpha": cfg["augment"]["mixup_alpha"],
            "random_flip": cfg["augment"]["random_flip"],
            "random_rotate": cfg["augment"]["random_rotate"],
            "random_zoom": cfg["augment"]["random_zoom"],
            # Inference/decision config
            "inference_threshold": inf_cfg.get("threshold"),
            "use_calibration": inf_cfg.get("use_calibration", True),
            "tta_enabled": inf_cfg.get("tta", False),
            "tta_samples": inf_cfg.get("tta_samples", 1),
            "triage_enabled": triage_cfg.get("enabled", False),
            "triage_band": triage_cfg.get("band"),
            "triage_max_disagreement": triage_cfg.get("max_disagreement"),
            "ensemble_enabled": ensemble_cfg.get("enabled", False),
            "ensemble_strategy": ensemble_cfg.get("strategy", "mean"),
            "ensemble_members": ensemble_cfg.get("checkpoints", []),
            "mc_dropout_enabled": mc_cfg.get("enabled", False),
            "mc_dropout_samples": mc_cfg.get("samples", 1),
            # Dataset fingerprints for traceability
            "dataset_fingerprints": dataset_fps,
        },
        tags=["classification", "medical-imaging", "efficientnet", "transfer-learning"],
        notes=f"Medical-grade preprocessing: {cfg['preprocessing']['mode']}",
    )

    print(
        f"✅ W&B initialized: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}"
    )
    return wandb.config


def main(cfg_path):
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))

    # Initialize Weights & Biases tracking
    wandb_config = init_wandb_tracking(cfg)
    print("\n[INFO] Experiment tracking active on W&B")

    # Verify GPU is available
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n[INFO] ✅ Training with GPU: {len(gpus)} device(s) found")
        for gpu in gpus:
            print(f"       - {gpu}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"       Memory growth already set: {e}")
    else:
        print("\n[WARNING] ⚠️ GPU not detected. Training will be slow on CPU.")

    # Enable mixed precision for faster training on GPU
    set_mixed_precision(cfg["train"].get("mixed_precision", True))

    train_ds, val_ds, test_ds, class_names, class_weights = get_datasets(cfg)
    num_classes = len(class_names)
    steps_per_epoch = max(1, tf.data.experimental.cardinality(train_ds).numpy())

    # Build model with logits output
    model = create_model(cfg, num_classes)

    # Stage 1: freeze backbone
    freeze_backbone(model, True)
    optimizer = build_optimizer(cfg, steps_per_epoch)
    warmup_loss = resolve_loss(cfg, "loss", fallback="categorical_crossentropy")
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    model.compile(optimizer=optimizer, loss=warmup_loss, metrics=metrics)

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(cfg["train"]["checkpoint_dir"], "best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg["train"]["early_stopping_patience"],
            restore_best_weights=True,
        ),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", patience=2, factor=0.5, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(cfg["log"]["csv_log"], append=False),
        tf.keras.callbacks.TensorBoard(log_dir=cfg["log"]["tensorboard_dir"]),
        # Weights & Biases callbacks
        WandbMetricsLogger(log_freq=10),  # Log every 10 batches
        WandbModelCheckpoint(
            filepath=os.path.join(cfg["train"]["checkpoint_dir"], "best_wandb.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        ),
    ]

    print(f"[INFO] Warmup loss: {warmup_loss.name}")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["train"]["freeze_backbone_epochs"],
        callbacks=callbacks,
        class_weight=class_weights
        if cfg["train"].get("use_class_weights", False)
        else None,
    )

    # Stage 2: unfreeze backbone
    for layer in model.layers:
        layer.trainable = True

    optimizer = build_optimizer(cfg, steps_per_epoch)
    finetune_loss = resolve_loss(
        cfg,
        "recall_loss",
        fallback=cfg.get("train", {}).get("loss", "categorical_crossentropy"),
    )
    model.compile(optimizer=optimizer, loss=finetune_loss, metrics=metrics)
    print(f"[INFO] Fine-tune loss: {finetune_loss.name}")
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["train"]["epochs"],
        callbacks=callbacks,
        class_weight=class_weights
        if cfg["train"].get("use_class_weights", False)
        else None,
    )

    # Merge histories for nicer plots
    history = {}
    for k in set(list(history1.history.keys()) + list(history2.history.keys())):
        history[k] = history1.history.get(k, []) + history2.history.get(k, [])

    # Save training curves
    from src.plots import save_training_curves

    save_training_curves(history, out_dir="reports")

    print("Best val acc (stage 2):", max(history2.history.get("val_accuracy", [0])))

    # Temperature scaling calibration
    if cfg.get("calibration", {}).get("enabled", True):
        T = calibrate_temperature(
            model, val_ds, max_iters=cfg["calibration"].get("max_iters", 200)
        )
        with open(
            os.path.join(cfg["train"]["checkpoint_dir"], "temperature.json"), "w"
        ) as f:
            json.dump({"temperature": T}, f)
        print(f"[Calibration] Saved temperature: T={T:.3f}")

    # Log final metrics to W&B
    wandb.log(
        {
            "final_train_accuracy": max(history2.history.get("accuracy", [0])),
            "final_val_accuracy": max(history2.history.get("val_accuracy", [0])),
            "final_train_loss": min(history2.history.get("loss", [999])),
            "final_val_loss": min(history2.history.get("val_loss", [999])),
        }
    )

    # Save training curves as W&B artifacts
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    ax1.plot(history["accuracy"], label="Train Accuracy", linewidth=2)
    ax1.plot(history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Training Progress", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot loss
    ax2.plot(history["loss"], label="Train Loss", linewidth=2)
    ax2.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Loss Curves", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    # Log to W&B
    wandb.log({"training_curves": wandb.Image(fig)})
    plt.close(fig)

    # Finish W&B run
    wandb.finish()
    print(f"\n✅ Training complete! View results at: https://wandb.ai")


if __name__ == "__main__":
    import tensorflow as tf

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    main(args.config)
