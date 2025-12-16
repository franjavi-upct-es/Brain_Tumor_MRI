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
from src.inference_utils import (
    aggregate_logits,
    apply_temperature,
    make_tta_layer,
    risk_triage_decision,
    softmax,
    tumor_score_from_probs,
)
from src.utils import load_config

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
    inference_cfg = cfg.get("inference", {})
    triage_cfg = inference_cfg.get("triage", {})
    ensemble_cfg = inference_cfg.get("ensemble", {})
    triage_enabled = triage_cfg.get("enabled", False)
    triage_band = float(triage_cfg.get("band", 0.05))
    triage_disagreement = float(triage_cfg.get("max_disagreement", 0.1))
    ensemble_strategy = ensemble_cfg.get("strategy", "mean")
    candidate_checkpoints = ensemble_cfg.get("checkpoints", [])
    use_calibration = inference_cfg.get("use_calibration", True)
    tta_enabled = inference_cfg.get("tta", False)
    tta_samples = int(inference_cfg.get("tta_samples", 1))
    mc_cfg = inference_cfg.get("mc_dropout", {})
    mc_enabled = mc_cfg.get("enabled", False)
    mc_samples = int(mc_cfg.get("samples", 1))
    
    # Umbral manual o configurado
    if threshold is None:
        threshold = inference_cfg.get("threshold", 0.5)

    # Intentar cargar ensemble; fallback a un único modelo
    ckpt_dir = cfg["train"]["checkpoint_dir"]
    models = []
    member_names = []

    if ensemble_cfg.get("enabled", False):
        for ckpt in candidate_checkpoints:
            path = os.path.join(ckpt_dir, ckpt)
            if os.path.exists(path):
                models.append(tf.keras.models.load_model(path, compile=False))
                member_names.append(os.path.basename(path))
            else:
                print(f"[WARN] Checkpoint no encontrado para el ensemble: {path}")

    if not models:
        fallback = [
            ("finetuned_navoneel.keras", "Fine-Tuned"),
            ("best.keras", "Base"),
        ]
        for fname, label in fallback:
            path = os.path.join(ckpt_dir, fname)
            if os.path.exists(path):
                models.append(tf.keras.models.load_model(path, compile=False))
                member_names.append(fname)
                break

    if not models:
        print("[ERROR] No se encontró ningún modelo entrenado.")
        return

    print(f"[INFO] Modelos cargados: {member_names} (estrategia={ensemble_strategy})")

    # Calibration temperature (optional)
    temperature = (
        _load_temperature(ckpt_dir)
        if use_calibration
        else 1.0
    )
    print(f"[INFO] Calibración: {'on' if use_calibration else 'off'} (T={temperature:.2f})")
    print(f"[INFO] TTA: {'on' if tta_enabled else 'off'} (samples={tta_samples}) | MC Dropout: {'on' if mc_enabled else 'off'} (samples={mc_samples})")

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

    # 3. Predicción (con soporte TTA + MC Dropout)
    tta_layer = make_tta_layer() if tta_enabled else None

    def predict_with_uncertainty(model, x_input):
        logits_list = []
        passes_tta = tta_samples if tta_enabled else 1
        passes_mc = mc_samples if mc_enabled else 1
        for _ in range(passes_tta):
            x_aug = tta_layer(x_input, training=True) if tta_enabled else x_input
            for _ in range(passes_mc):
                logits = model(x_aug, training=mc_enabled).numpy()[0]
                logits = apply_temperature(logits, temperature) if use_calibration else logits
                logits_list.append(logits)
        return logits_list

    member_logits = []
    member_scores = []
    for model in models:
        logits_list = predict_with_uncertainty(model, x)
        member_logits.append(np.mean(logits_list, axis=0))
        probs_member = softmax(member_logits[-1], axis=0)
        member_scores.append(tumor_score_from_probs(probs_member, no_tumor_idx))

    agg_logits = aggregate_logits(member_logits, strategy=ensemble_strategy)
    agg_probs = softmax(agg_logits, axis=0)
    prob_tumor = tumor_score_from_probs(agg_probs, no_tumor_idx)

    # Es tumor. ¿De qué tipo? Buscamos el max entre las clases que NO son no_tumor
    tumor_probs_only = agg_probs.copy()
    if no_tumor_idx != -1:
        tumor_probs_only[no_tumor_idx] = -1.0  # Anular 'no_tumor'

    pred_idx = int(np.argmax(tumor_probs_only))
    pred_class = class_names[pred_idx]
    confidence = agg_probs[pred_idx]

    if triage_enabled:
        decision, triage_info = risk_triage_decision(
            member_scores,
            threshold=threshold,
            triage_band=triage_band,
            max_disagreement=triage_disagreement,
        )
    else:
        decision = "tumor" if prob_tumor >= threshold else "healthy"
        triage_info = {
            "score": prob_tumor,
            "spread": 0.0,
            "min_score": prob_tumor,
            "max_score": prob_tumor,
            "threshold": float(threshold),
            "triage_band": float(triage_band),
            "max_disagreement": float(triage_disagreement),
            "reason": "threshold",
        }

    print(f"\n--- Resultado (umbral {threshold:.2f}) ---")
    print(f"Probabilidad de Tumor (ensemble): {prob_tumor:.2%}")
    if len(member_scores) > 1:
        print(f"  ↳ Puntuaciones por modelo: {[round(s, 3) for s in member_scores]}")
    print(f"Decisión: {decision.upper()} (regla de riesgo: banda={triage_band}, Δ>{triage_disagreement})")
    
    if decision == "tumor":
        print(f"Tipo sospechoso: {pred_class} (Confianza del tipo: {confidence:.2%})")
    elif decision == "review":
        print(f"⚠️ Caso derivado a revisión. Motivo: {triage_info.get('reason')}")
        print(f"Sospecha principal: {pred_class} (Confianza: {confidence:.2%})")
    else:
        print(f"PREDICCIÓN: Sano / No Tumor")
        if no_tumor_idx != -1:
            print(f"Confianza: {agg_probs[no_tumor_idx]:.2%}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--image", required=True)
    p.add_argument("--threshold", type=float, default=None, help="Umbral de detección de tumor (0-1). Por defecto usa el de config.")
    args = p.parse_args()
    
    main(args.config, args.image, args.threshold)
