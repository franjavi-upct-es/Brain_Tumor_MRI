# infer.py — Single-image inference with calibrated probabilities
# ---------------------------------------------------------------
# This script loads the best checkpoint and predicts the class of a single image.
# It also applies temperature scaling (if available) to output better-calibrated
# probabilities, which is valuable for clinical decision support.
#
# Usage:
#   python src/infer.py --config configs/config.yaml --image path/to/image.jpg

import argparse, os, json, numpy as np
from utils import load_config

def softmax(x, axis=-1):
    """Stable softmax for 1D/2D arrays; subtract max to avoid overflow."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def main(cfg_path, image_path):
    cfg = load_config(cfg_path)
    infer_keras(cfg, image_path)

def infer_keras(cfg, image_path):
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    class_names = cfg["data"].get("class_names", ["no_tumor", "tumor"])

    # Load model (compiled=False for speed; we only need forward pass)
    model = tf.keras.models.load_model(os.path.join(cfg["train"]["checkpoint_dir"], "best.keras"), compile=False)

    # Load calibration temperature if available
    T = 1.0
    temp_path = os.path.join(cfg["train"]["checkpoint_dir"], "temperature.json")
    if cfg["inference"].get("use_calibration", True) and os.path.exists(temp_path):
        with open(temp_path, "r") as f:
            T = float(json.load(f).get("temperature", 1.0))

    img_size = cfg["data"]["image_size"]

    # Choose preprocess consistent with the backbone
    if "v2" in cfg["model"]["name"]:
        preprocess = tf.keras.applications.efficientnet_v2.preprocess_input
    else:
        preprocess = tf.keras.applications.efficientnet.preprocess_input

    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    x = np.array(img)[None].astype(np.float32)  # shape (1,H,W,3)
    x = preprocess(x)

    # Forward pass -> logits -> calibrated probabilities
    logits = model.predict(x, verbose=0)[0] / T
    probs = softmax(logits, axis=0)  # shape (C,)

    pred = int(np.argmax(probs))
    result = {"pred_class": class_names[pred], "probs": {cls: float(p) for cls,p in zip(class_names, probs)}}
    print(result)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--image", required=True)
    args = p.parse_args()
    main(args.config, args.image)
