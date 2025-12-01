import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Añadimos el directorio raíz al path para importa módulo del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config


def load_and_preprocess_image(path, img_size, preprocess_input):
    """Carga y preprocesa una imagen para el modelo."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.expand_dims(img, axis=0)
    return preprocess_input(img)


def main(config_path, data_dir):
    # Verificar que los datos existen
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ERROR] No se encuentra el directorio de datos: {data_path}")
        print("Ejecuta primero: python tools/download_navoneel.py")
        return

    # Cargar configuración y modelo
    cfg = load_config(config_path)
    img_size = cfg["data"]["image_size"]
    class_names = cfg["data"]["class_names"]

    try:
        no_tumor_idx = class_names.index("no_tumor")
    except ValueError:
        print("[ERROR] La clase 'no_tumor' no está definida en tu config.yaml")
        return

    print(f"[INFO] Cargando modelo desde {cfg['train']['checkpoint_dir']}...")
    model_path = os.path.join(cfg["train"]["checkpoint_dir"], "finetuned_navoneel.keras")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Preprocesamiento según backbone
    if "v2" in cfg["model"]["name"]:
        preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
    else:
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    # Recopilar imágenes
    images_paths = []
    true_binary_labels = []  # 0 = Sano, 1 = Tumor

    # Cargar clase 'no' (Sanos) -> Etiqueta 0
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for p in (data_path / "no").glob(ext):
            images_paths.append(str(p))
            true_binary_labels.append(0)

    # Cargar clase 'yes' (Tumor) -> Etiqueta 1
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for p in (data_path / "yes").glob(ext):
            images_paths.append(str(p))
            true_binary_labels.append(1)

    print(f"[INFO] Evaluando {len(images_paths)} imágenes de {data_path}...")

    # Inferencia
    pred_binary_labels = []
    pred_tumor_types = []

    for path in tqdm(images_paths):
        img_tensor = load_and_preprocess_image(
            path, img_size, preprocess_input)
        logits = model.predict(img_tensor, verbose=0)
        pred_idx = np.argmax(logits[0])

        # Mapeo Multicalse -> Binario
        if pred_idx == no_tumor_idx:
            pred_binary_labels.append(0)
            pred_tumor_types.append("N/A")
        else:
            pred_binary_labels.append(1)
            pred_tumor_types.append(class_names[pred_idx])

    # Reportes
    print("\n" + "=" * 40)
    print("RESULTADOS: DATASET EXTERNO (Navoneel)")
    print("=" * 40)

    print("\n--- Clasificación Binaria (Sano vs Tumor) ---")
    print(
        classification_report(
            true_binary_labels, pred_binary_labels, target_names=[
                "Sano", "Tumor"]
        )
    )

    cm = confusion_matrix(true_binary_labels, pred_binary_labels)
    print(f"Matriz de Confusión:\n{cm}")
    print(f"TN (Sanos OK): {cm[0][0]} | FP (Falsas Alarmas): {cm[0][1]}")
    print(
        f"FN (Tumores No Detectados): {
            cm[1][0]} | TN (Tumores Detectados): {cm[1][1]}"
    )

    print("\n--- Tipos de Tumor Predichos (en casos 'Yes') ---")
    tumor_indices = [i for i, x in enumerate(true_binary_labels) if x == 1]
    from collections import Counter

    counts = Counter([pred_tumor_types[i] for i in tumor_indices])

    total = len(tumor_indices)
    for k, v in counts.items():
        if k == "N/A":
            continue
        print(f"- {k}: {v} ({v / total:.1%})")

    missed = counts.get("N/A", 0)
    if missed > 0:
        print(f"- No detectados: {missed} ({missed / total:.1%})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--data", default="data/external_navoneel", help="Ruta al dataset externo"
    )
    args = p.parse_args()

    main(args.config, args.data)
