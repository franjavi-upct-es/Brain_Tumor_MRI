# infer.py — Single-image inference with calibrated probabilities
# ---------------------------------------------------------------
# This script loads the best checkpoint and predicts the class of a single image.
# It also applies temperature scaling (if available) to output better-calibrated
# probabilities, which is valuable for clinical decision support.
#
# Usage:
#   python src/infer.py --config configs/config.yaml --image path/to/image.jpg

import argparse, os, json, numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def main(cfg_path, image_path, threshold=None):
    cfg = load_config(cfg_path)
    infer_keras(cfg, image_path, threshold)

def _load_temperature(ckpt_dir: str) -> float:
    """Load temperature scalar if present."""
    for name in ("temperature.json", "focal_temperature.json"):
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                return float(data.get("temperature", 1.0))
            except Exception:
                continue
    return 1.0


def infer_keras(cfg, image_path, threshold):
    import tensorflow as tf
    from PIL import Image
    
    # 1. Configuración
    class_names = cfg["data"].get("class_names", ["glioma", "meningioma", "no_tumor", "pituitary"])
    img_size = cfg["data"]["image_size"]
    
    # Intentar cargar el modelo fine-tuned si existe, si no el original
    ckpt_dir = cfg["train"]["checkpoint_dir"]
    model_name = "finetuned_navoneel.keras" if os.path.exists(os.path.join(ckpt_dir, "finetuned_navoneel.keras")) else "best.keras"
    model_path = os.path.join(ckpt_dir, model_name)
    
    print(f"[INFO] Usando modelo: {model_name}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Calibration temperature (optional)
    temperature = (
        _load_temperature(ckpt_dir)
        if cfg.get("inference", {}).get("use_calibration", True)
        else 1.0
    )

    # Buscar índice de 'no_tumor'
    try:
        no_tumor_idx = class_names.index("no_tumor")
    except ValueError:
        no_tumor_idx = -1

    # 2. Preprocesamiento
    if "v2" in cfg["model"]["name"]:
        preprocess = tf.keras.applications.efficientnet_v2.preprocess_input
    else:
        preprocess = tf.keras.applications.efficientnet.preprocess_input

    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    x = np.array(img)[None].astype(np.float32)
    x = preprocess(x)

    # 3. Predicción
    # Si tenemos temperatura guardada y usamos el modelo original, podríamos aplicarla,
    # pero para el modelo fine-tuned o thresholding manual, usamos logits directos o softmax.
    logits = model.predict(x, verbose=0)[0] / temperature
    probs = softmax(logits, axis=0)

    # 4. Decisión con Umbral
    # Probabilidad de que SEA tumor = 1 - Prob(no_tumor)
    if no_tumor_idx != -1:
        prob_tumor = 1.0 - probs[no_tumor_idx]
    else:
        # Fallback si no hay clase explícita no_tumor (raro en este proyecto)
        prob_tumor = np.max(probs)

    # Si el usuario no define umbral, usamos el de config
    if threshold is None:
        threshold = cfg.get("inference", {}).get("threshold", 0.5)

    print(f"\n--- Resultado (Umbral {threshold}) ---")
    print(f"Probabilidad de Tumor: {prob_tumor:.2%}")
    
    if prob_tumor >= threshold:
        # Es tumor. ¿De qué tipo? Buscamos el max entre las clases que NO son no_tumor
        tumor_probs_only = probs.copy()
        if no_tumor_idx != -1:
            tumor_probs_only[no_tumor_idx] = -1.0 # Anular 'no_tumor'
        
        pred_idx = np.argmax(tumor_probs_only)
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx]
        
        print(f"PREDICCIÓN: ¡TUMOR DETECTADO!")
        print(f"Tipo sospechoso: {pred_class} (Confianza del tipo: {confidence:.2%})")
    else:
        print(f"PREDICCIÓN: Sano / No Tumor")
        if no_tumor_idx != -1:
            print(f"Confianza: {probs[no_tumor_idx]:.2%}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--image", required=True)
    p.add_argument("--threshold", type=float, default=None, help="Umbral de detección de tumor (0-1). Por defecto usa el de config.")
    args = p.parse_args()
    
    main(args.config, args.image, args.threshold)
