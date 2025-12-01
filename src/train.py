# train.py — Training script (highly commented)
# ---------------------------------------------
# Two-stage fine-tuning for EfficientNet/EfficientNetV2:
# 1) Freeze the backbone; train the classifier head.
# 2) Unfreeze; fine-tune the whole network.
# Also computes temperature scaling (calibration) on the validation set.
# Saves curves into 'reports/'.

import argparse, os, json
from src.utils import load_config, set_seed
from src.data import get_datasets
from src.model import create_model

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
            first_decay_steps=max(steps_per_epoch*3, 100),
            t_mul=2.0, m_mul=0.9, alpha=1e-4
        )
        lr = schedule
    opt_name = cfg["train"].get("optimizer","adamw").lower()
    if opt_name == "adamw":
        try:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=cfg["train"]["weight_decay"])
        except Exception:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    return optimizer

def set_mixed_precision(enabled=True):
    """Enable mixed precision (float16 compute / float32 params) if supported."""
    if not enabled: return
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

def main(cfg_path):
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed",42))
    set_mixed_precision(cfg["train"].get("mixed_precision", True))

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n[INFO] ✅ Entrenando con GPU: {len(gpus)} dispositivos encontrados.")
        for gpu in gpus:
            print(f"       - {gpu}")
            # Opcional: Configurar crecimiento de memoria para evitar errores de OOM
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    else:
        print("\n[WARNING] ⚠️ GPU no detectada. El entrenamiento será lento en CPU.")

    set_mixed_precision(cfg["train"].get("mixed_precision", True))

    train_ds, val_ds, test_ds, class_names, class_weights = get_datasets(cfg)
    num_classes = len(class_names)
    steps_per_epoch = max(1, tf.data.experimental.cardinality(train_ds).numpy())

    # Build model with logits output
    model = create_model(cfg, num_classes)

    # Stage 1: freeze backbone
    freeze_backbone(model, True)
    optimizer = build_optimizer(cfg, steps_per_epoch)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(cfg["train"]["checkpoint_dir"], "best.keras"),
                                           monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                         patience=cfg["train"]["early_stopping_patience"],
                                         restore_best_weights=True),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", patience=2, factor=0.5, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(cfg["log"]["csv_log"], append=False),
        tf.keras.callbacks.TensorBoard(log_dir=cfg["log"]["tensorboard_dir"]),
    ]

    history1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=cfg["train"]["freeze_backbone_epochs"],
        callbacks=callbacks,
        class_weight=class_weights if cfg["train"].get("use_class_weights", False) else None
    )

    # Stage 2: unfreeze backbone
    for layer in model.layers:
        layer.trainable = True

    optimizer = build_optimizer(cfg, steps_per_epoch)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=cfg["train"]["epochs"],
        callbacks=callbacks,
        class_weight=class_weights if cfg["train"].get("use_class_weights", False) else None
    )

    # Merge histories for nicer plots
    history = {}
    for k in set(list(history1.history.keys()) + list(history2.history.keys())):
        history[k] = history1.history.get(k, []) + history2.history.get(k, [])

    # Save training curves
    from plots import save_training_curves
    save_training_curves(history, out_dir="reports")

    print("Best val acc (stage 2):", max(history2.history.get("val_accuracy", [0])))

    # Temperature scaling calibration
    if cfg.get("calibration",{}).get("enabled", True):
        T = calibrate_temperature(model, val_ds, max_iters=cfg["calibration"].get("max_iters", 200))
        with open(os.path.join(cfg["train"]["checkpoint_dir"], "temperature.json"), "w") as f:
            json.dump({"temperature": T}, f)
        print(f"[Calibration] Saved temperature: T={T:.3f}")

if __name__ == "__main__":
    import tensorflow as tf
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    main(args.config)
