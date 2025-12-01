import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# Añadir path para imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config, set_seed
from src.data import _get_preprocess_fn

def main(config_path, data_dir):
    cfg = load_config(config_path)
    set_seed(cfg.get('seed', 42))

    # 1. Configuración básica
    img_size = cfg['data']['image_size']
    batch_size = 16 # Batch pequeño para fine-tuning
    lr = 1e-5       # Learning Rate MUY bajo para no romper lo aprendido
    epochs = 10

    class_name_orig = cfg['data']['class_names'] # ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    try:
        no_tumor_idx = class_name_orig.index('no_tumor')
    except ValueError:
        print("[ERROR] 'no_tumor' no encontrado en class_names.")
        return
    
    print(f"[INFO] Índice de clase 'no_tumor': {no_tumor_idx}")

    # 2. Cargar Dataset Navoneel
    # Keras asignará: 'no' -> 0, 'yes' -> 1 (orden alfabético)
    print(f"[INFO] Cargando datos de {data_dir}...")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='binary' # 0=no, 1=yes
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='binary'
    )

    # 3. Preprocesamiento y Aumento
    preprocess = _get_preprocess_fn(cfg['model']['name'])

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomBrightness(0.1)
    ])

    def prepare(ds, apply_augment=False):
        # Mapeamos: (img, label) -> (preprocess(img), label)
        ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        if apply_augment:
            ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)
    
    train_ds = prepare(train_ds, apply_augment=True)
    val_ds = prepare(val_ds, apply_augment=False)

    # 4. Cargar Modelo Pre-entrenado
    checkpoint_path = os.path.join(cfg['train']['checkpoint_dir'], 'best.keras')
    print(f"[INFO] Cargando modelo base: {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path, compile=False)


    # Descongelar las últimas capas para permitir adaptación
    model.trainable = True

    # 5. Definir la 'Pérdida Híbrida' (Custom Loss)
    def custom_binary_loss(y_true, y_pred_logits):
        """
        y_true: 0 (Sano/No) o 1 (Tumor/Yes)
        y_pred_logits: Logits de las 4 clases ['glioma', 'meningioma', 'no_tumor', 'pituitary']

        Objetivo:
        - Si y_true=0 (Sano): Queremos que 'no_tumor' sea alto
        - Si y_true=1 (Tumor): Queremos que 'no_tumor' sea bajo
        """
        # Extraemos solo el logit correspondiente a no_tumor
        no_tumor_logit = y_pred_logits[:, no_tumor_idx]

        # El target para 'no_tumor' es el INVERSO de y_true
        # Si es Tumor (1), target 'no_tumor' = 0
        # Si es Sano (0), target 'no_tumor' = 1
        target_no_tumor = 1.0 - tf.cast(y_true, tf.float32)

        # Aplanar ambos tensores para asegurar shapes compatibles
        target_no_tumor = tf.reshape(target_no_tumor, [-1])
        no_tumor_logit = tf.reshape(no_tumor_logit, [-1])

        # Usamos Binary Crossentropy sobre ese logit específico
        # from_logits=True para estabilidad numérica
        return tf.keras.losses.binary_crossentropy(
            target_no_tumor,
            no_tumor_logit,
            from_logits=True
        )
    
    # 6. Compilar y Entrenar
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=custom_binary_loss,
        metrics=[] # Las métricas estándar de accuracy no sirven bien aquí por el formato de labels
    )

    print("[INFO] Iniciando Fine-Tuning...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # 7. Guardar el nuevo modelo adaptado
    save_path = os.path.join(cfg['train']['checkpoint_dir'], 'finetuned_navoneel.keras')
    model.save(save_path)
    print(f"\n[OK] Modelo re-entrenado guardado en: {save_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data", default="data/external_navoneel")
    args = p.parse_args()

    main(args.config, args.data)
