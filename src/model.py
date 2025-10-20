# model.py — Build the classifier model (EfficientNet / EfficientNetV2)
# --------------------------------------------------------------------
# We create a Keras model that:
# - takes (H,W,C) images
# - applies a small augmentation block *inside* the graph (so it's part of the saved model)
# - applies the appropriate preprocessing for the chosen EfficientNet variant
# - attaches a Dense head that outputs logits (no softmax), which is better for calibration
#
# Why logits? For calibration and temperature scaling, it's preferable to keep
# the model output as raw scores and apply softmax externally when needed.

from typing import Dict

def create_model(cfg: Dict, num_classes: int):
    """Public entry: create a Keras model from the config."""
    return create_keras_model(cfg, num_classes)

def create_keras_model(cfg: Dict, num_classes: int):
    import tensorflow as tf
    name = (cfg["model"].get("name","efficientnet_v2_b0")).lower()
    img_size = cfg["data"]["image_size"]
    channels = cfg["data"].get("channels", 3)

    inputs = tf.keras.Input(shape=(img_size, img_size, channels), name="image")

    # In-graph augmentation: ensures saved checkpoints include the stochastic augment.
    aug = tf.keras.Sequential(name="augment", layers=[
        tf.keras.layers.RandomFlip("horizontal") if cfg["augment"].get("random_flip", True) else tf.keras.layers.Layer(),
        tf.keras.layers.RandomRotation(cfg["augment"].get("random_rotate",0.0)) if cfg["augment"].get("random_rotate",0.0) else tf.keras.layers.Layer(),
        tf.keras.layers.RandomZoom(cfg["augment"].get("random_zoom",0.0)) if cfg["augment"].get("random_zoom",0.0) else tf.keras.layers.Layer(),
        tf.keras.layers.RandomBrightness(cfg["augment"].get("random_brightness",0.0)) if cfg["augment"].get("random_brightness",0.0) else tf.keras.layers.Layer(),
        tf.keras.layers.RandomContrast(cfg["augment"].get("random_contrast",0.0)) if cfg["augment"].get("random_contrast",0.0) else tf.keras.layers.Layer(),
    ])
    x = aug(inputs, training=True)  # training=True keeps aug active even if model is later used for inference

    # Choose the appropriate EfficientNet family and preprocessing
    if "v2" in name:
        preprocess = tf.keras.applications.efficientnet_v2.preprocess_input
        if name == "efficientnet_v2_b0":
            Base = tf.keras.applications.EfficientNetV2B0
        elif name == "efficientnet_v2_b1":
            Base = tf.keras.applications.EfficientNetV2B1
        elif name == "efficientnet_v2_b2":
            Base = tf.keras.applications.EfficientNetV2B2
        elif name == "efficientnet_v2_b3":
            Base = tf.keras.applications.EfficientNetV2B3
        else:
            Base = tf.keras.applications.EfficientNetV2B0
        base = Base(include_top=False, weights="imagenet" if cfg["model"].get("pretrained",True) else None,
                    input_shape=(img_size, img_size, channels), pooling=cfg["model"].get("pooling","avg"))
    else:
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet" if cfg["model"].get("pretrained",True) else None,
                    input_shape=(img_size, img_size, channels), pooling=cfg["model"].get("pooling","avg"))

    # Preprocess inside the graph to ensure consistency with the chosen backbone
    x = preprocess(x)

    x = base(x)

    # Optional dropout can help a bit in the head
    if cfg["model"].get("dropout", 0.0) > 0:
        x = tf.keras.layers.Dropout(cfg["model"]["dropout"])(x)

    # Final classifier head: Dense logits (no activation)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)
    model = tf.keras.Model(inputs, logits, name=f"{name}_classifier")

    return model
