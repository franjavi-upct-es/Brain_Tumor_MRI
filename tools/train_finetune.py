import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config, set_seed
from src.data import _get_preprocess_fn

def main(config_path, data_dir):
    cfg = load_config(config_path)
    set_seed(cfg.get('seed', 42))

    # 1. Basic configuration
    img_size = cfg['data']['image_size']
    batch_size = 16 # Small batch for fine-tuning
    lr = 1e-5       # VERY low Learning Rate to not break what was learned
    epochs = 10

    class_name_orig = cfg['data']['class_names'] # ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    try:
        no_tumor_idx = class_name_orig.index('no_tumor')
    except ValueError:
        print("[ERROR] 'no_tumor' not found in class_names.")
        return
    
    print(f"[INFO] 'no_tumor' class index: {no_tumor_idx}")

    # 2. Load Navoneel Dataset
    # Keras will assign: 'no' -> 0, 'yes' -> 1 (alphabetical order)
    print(f"[INFO] Loading data from {data_dir}...")

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

    # 3. Preprocessing and Augmentation
    preprocess = _get_preprocess_fn(cfg['model']['name'])

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomBrightness(0.1)
    ])

    def prepare(ds, apply_augment=False):
        # Map: (img, label) -> (preprocess(img), label)
        ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        if apply_augment:
            ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)
    
    train_ds = prepare(train_ds, apply_augment=True)
    val_ds = prepare(val_ds, apply_augment=False)

    # 4. Load Pre-trained Model
    checkpoint_path = os.path.join(cfg['train']['checkpoint_dir'], 'best.keras')
    print(f"[INFO] Loading base model: {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path, compile=False)


    # Unfreeze last layers to allow adaptation
    model.trainable = True

    # 5. Define 'Hybrid Loss' (Custom Loss)
    def custom_binary_loss(y_true, y_pred_logits):
        """
        y_true: 0 (Healthy/No) or 1 (Tumor/Yes)
        y_pred_logits: Logits of the 4 classes ['glioma', 'meningioma', 'no_tumor', 'pituitary']

        Goal:
        - If y_true=0 (Healthy): We want 'no_tumor' to be high
        - If y_true=1 (Tumor): We want 'no_tumor' to be low
        """
        # Extract only the logit corresponding to no_tumor
        no_tumor_logit = y_pred_logits[:, no_tumor_idx]

        # The target for 'no_tumor' is the INVERSE of y_true
        # If it's Tumor (1), target 'no_tumor' = 0
        # If it's Healthy (0), target 'no_tumor' = 1
        target_no_tumor = 1.0 - tf.cast(y_true, tf.float32)

        # Flatten both tensors to ensure compatible shapes
        target_no_tumor = tf.reshape(target_no_tumor, [-1])
        no_tumor_logit = tf.reshape(no_tumor_logit, [-1])

        # Use Binary Crossentropy on that specific logit
        # from_logits=True for numerical stability
        return tf.keras.losses.binary_crossentropy(
            target_no_tumor,
            no_tumor_logit,
            from_logits=True
        )
    
    # 6. Compile and Train
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=custom_binary_loss,
        metrics=[] # Standard accuracy metrics don't work well here due to label format
    )

    print("[INFO] Starting Fine-Tuning...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # 7. Save the new adapted model
    save_path = os.path.join(cfg['train']['checkpoint_dir'], 'finetuned_navoneel.keras')
    model.save(save_path)
    print(f"\n[OK] Retrained model saved at: {save_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data", default="data/external_navoneel")
    args = p.parse_args()

    main(args.config, args.data)
