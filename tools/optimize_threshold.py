import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Añadir path para imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config

def load_and_preprocess_image(path, img_size, preprocess_input):
    """Carga y preprocesa una imagen para el modelo."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    return preprocess_input(img)

def main(config_path, data_dir):
    cfg = load_config(config_path)
    img_size = cfg['data']['image_size']
    class_names = cfg['data']['class_names']

    # Índice de la clase sana
    try:
        no_tumor_idx = class_names.index('no_tumor')
    except ValueError:
        print("[ERROR] 'no_tumor' no encontrado en class_names.")
        return
    
    print("[INFO] Cargando modelo...")
    model = tf.keras.models.load_model(os.path.join(cfg['train']['checkpoint_dir'], 'finetuned_navoneel.keras'), compile=False)

    if "v2" in cfg["model"]["name"]:
        preprocess = tf.keras.applications.efficientnet_v2.preprocess_input
    else:
        preprocess = tf.keras.applications.efficientnet.preprocess_input

    # 1. Cargar todas las imágenes y etiquetas
    data_path = Path(data_dir)
    paths = []
    y_true = [] # 0: Sano, 1: Tumor

    print("[INFO] Cargando dataset para análisis")
    # Sanos
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for p in (data_path / "no").glob(ext):
            paths.append(str(p))
            y_true.append(0)
    # Tumores
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for p in (data_path / "yes").glob(ext):
            paths.append(str(p))
            y_true.append(1)

    y_true = np.array(y_true)

    # 2. Obtener predicciones RAW (probabilidades)
    print(f"[INFO] Calculando probabilidades para {len(paths)} imágenes...")

    # Procesamos por lotes para velocidad
    batch_size = 32
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda x: load_and_preprocess_image(x, img_size, preprocess), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    all_probs = model.predict(ds, verbose=1)

    # Aplicar Softmax si el modelo devolvía logits 
    # Si la suma de una fila no es aprox. 1, aplicamos softmax
    if not np.allclose(np.sum(all_probs[0]), 1.0, atol=0.1):
        all_probs = tf.nn.softmax(all_probs).numpy()

    # 3. Calcular la "Probabilidad de Tumor" (suma de todas las clases que no son no_tumor)
    # Prob Tumor = 1.0 - Prob(no_tumor)
    tumor_probs = 1.0 - all_probs[:, no_tumor_idx]

    # Barrido de umbrales (Threshold Sweep)
    print("\n" + "=" * 90)
    print(f"{'Umbral':<10} | {'Recall (Tumor)':<15} | {'Precision':<10} | {'FP (Falsas Alarmas)':<20} | {'FN (Tumores Perdidos)'}")
    print("=" * 90)

    best_f1 = 0
    best_thresh = 0.5

    thresholds = np.arange(0.1, 0.95, 0.05)
    for t in thresholds:
        y_pred_t = (tumor_probs >= t).astype(int)

        # Métricas
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
    print(f"[RECOMENDACIÓN] El mejor balance estadístico (F1-Score) está en el umbral: {best_thresh:.2f}")

    # Mostrar resultados detallados con el mejor umbral
    print(f"\n--- Resultados con Umbral Optimizado ({best_thresh:.2f}) ---")
    final_preds = (tumor_probs >= best_thresh).astype(int)
    print(classification_report(y_true, final_preds, target_names=['Sano', 'Tumor']))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--data", default="data/external_navoneel", help="Ruta al dataset externo"
    )
    args = p.parse_args()

    main(args.config, args.data)