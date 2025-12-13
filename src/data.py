# data.py - Dataset creation and augmentation pipelines (Tensorflow/Keras)
# ------------------------------------------------------------------------
# This module create tf.data pipelines from a directory tree of images.
# We support two modes:
#   1)  auto_split=True: single root 'data/train/<class>/*' with an internal
#       train/val split (validation_split).
#   2)  explicit splits: data/train, data/val, data/test subtrees.
#
# Key features:
# - On-the-fly resizing, augmentation (flip/rotate/zoom/brightness/contrast).
# - Propper pre-processing per model family (EfficientNet vs EfficientNetV2).
# - Optional class weights based on folder counts.
#
# Tips:
# - If your MRI images are grayscale, Keras loaders can output 1-channel or we
#   can convert to RGB (3-channel) explicitly; most ImagenNet backbones expect 3.

import os
from typing import Dict

from src.utils import walk_class_counts

# NOTE: Preprocessing and augmentation functions are now in model.py
# to avoid duplicating work and maximize GPU utilization.

def get_datasets(cfg: Dict):
    """
    Create (train, val, test) tf.data datasets suitable for Keras model.fit().
    Return: (train_ds, val_ds, test_ds, class_names, class_weights_dict or None)
    
    NOTE: Preprocessing and augmentation are done INSIDE the model (model.py),
    so here we only load and batch the data efficiently.
    """
    import tensorflow as tf
    AUTOTUNE = tf.data.AUTOTUNE
    img_size = cfg["data"]["image_size"]
    batch = cfg["train"]["batch_size"]
    channels = cfg["data"].get("channels", 3)
    a = cfg.get("augment", {})
    mixup_alpha = float(a.get("mixup_alpha", 0.0))

    # Load datasets - preprocessing/augmentation happens in the model
    if cfg["data"].get("auto_split", False):
        root_dir = cfg["data"]["root_dir"]
        train = tf.keras.preprocessing.image_dataset_from_directory(
            root_dir, image_size=(img_size, img_size), batch_size=batch,
            label_mode="categorical", validation_split=cfg["data"].get("valid_split", 0.2),
            subset="training", seed=42, shuffle=True, color_mode="rgb" if channels == 3 else "grayscale"
        )
        val = tf.keras.preprocessing.image_dataset_from_directory(
            root_dir, image_size=(img_size, img_size), batch_size=batch,
            label_mode="categorical", validation_split=cfg["data"].get("valid_split", 0.2),
            subset="validation", seed=42, shuffle=True, color_mode="rgb" if channels == 3 else "grayscale"
        )
        test = val
    else:
        train = tf.keras.preprocessing.image_dataset_from_directory(
            cfg["data"]["train_dir"], image_size=(img_size, img_size), batch_size=batch,
            label_mode="categorical", shuffle=True, color_mode="rgb" if channels == 3 else "grayscale"
        )
        val = tf.keras.preprocessing.image_dataset_from_directory(
            cfg["data"]["val_dir"], image_size=(img_size, img_size), batch_size=batch,
            label_mode="categorical", shuffle=True, color_mode="rgb" if channels == 3 else "grayscale"
        )
        test = tf.keras.preprocessing.image_dataset_from_directory(
            cfg["data"]["test_dir"], image_size=(img_size, img_size), batch_size=batch,
            label_mode="categorical", shuffle=True, color_mode="rgb" if channels == 3 else "grayscale"
        )

    class_names = train.class_names

    # Simple pipeline: just prefetch for GPU overlap
    # All preprocessing/augmentation is in the model graph (model.py)
    train = train.prefetch(buffer_size=AUTOTUNE)

    # Optional MixUp regularization (must be done here, before model)
    if mixup_alpha > 0:
        import tensorflow_probability as tfp
        @tf.function
        def mix_map(x, y):
            bs = tf.shape(x)[0]
            dist = tfp.distributions.Beta(mixup_alpha, mixup_alpha)
            l = tf.reshape(dist.sample([bs]), (bs,1,1,1))
            idx = tf.random.shuffle(tf.range(bs))
            x2 = tf.gather(x, idx)
            y2 = tf.gather(y, idx)
            x = x * l + x2 * (1 - l)
            y = y * tf.reshape(l, (bs, 1)) + y2 * (1 - tf.reshape(l, (bs, 1)))
            return x, y
        train = train.map(mix_map, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    val = val.prefetch(buffer_size=AUTOTUNE)
    test = test.prefetch(buffer_size=AUTOTUNE)

    # 3) Optional class weights (derived from filesystem counts for training set)
    cw = None
    if cfg["train"].get("use_class_weights", False):
        from utils import compute_class_weights
        counts = walk_class_counts(
            cfg["data"].get("root_dir") if cfg["data"].get("auto_split", False) else cfg["data"]["train_dir"],
            class_names
        )
        cw_map = compute_class_weights(counts)
        # Keras expects an index -> weight dict: {class_index: weight}
        cw = {i: float(cw_map.get(cls, 1.0)) for i, cls in enumerate(class_names)}

    return train, val, test, class_names, cw