# gradcam.py — Grad-CAM utilities to visualize model attention
# ------------------------------------------------------------
# Grad-CAM highlights regions in the input that contribute most to the model's
# decision. For medical imaging, this serves as a sanity check and improves
# interpretability when reviewing results with domain experts.
#
# Usage (see eval.py):
# - save_batch_gradcam(model, x_batch, y_batch, class_names, out_dir)

import os
import numpy as np
import tensorflow as tf
from PIL import Image

def _find_last_conv(model) -> str:
    """
    Heuristic search for the last Conv2D layer name in a (possibly nested) model.
    Grad-CAM requieres a convolutional feature map; the 'last conv' is a good default.
    """
    # First pass: top-level layers from the end
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # Second pass: look one level deep in nested models
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    # Fallback: None (caller can handle)
    return None

def compute_gradcam(model, img_tensor, class_index=None, conv_layer_name=None):
    """
    Compute a Grad-CAM heatmap for a single image tensor.
    - model: Keras model that outputs logits
    - img_tensor: shape (1, H, W, C), preprocessed for the model
    - class_index: which class to explain; if None, uses argmax prediction
    - conv_layer_name: name of the conv layer to use; if None, auto-detect
    Returns a 2D numpy array in [0,1] of size equal to conv feature map (upsampled later).
    """
    if conv_layer_name is None:
        conv_layer_name = _find_last_conv(model)
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, logits = grad_model(img_tensor)
        if class_index is None:
            class_index = tf.argmax(logits[0])
        loss = logits[:, class_index]
    # Compute gradients of the class score w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    # Weighted sum of channels followed by ReLU
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-18)
    return heatmap.numpy()

def save_batch_gradcam(model, x_batch, y_batch, class_names, out_dir="gradcam_samples"):
    """
    Compute Grad-CAM overlays for up to the first 8 images of a batch and save PNGs.
    We assume x_batch is preprocessed; for visualization, we min-max scale to [0,255].
    """
    os.makedirs(out_dir, exist_ok=True)
    xb = x_batch.numpy()
    # Re-scale for visualization (original preprocessing could be centered/normalized)
    xb_vis = np.clip((xb - xb.min())/(xb.max()-xb.min()+1e-8)*255.0, 0, 255).astype(np.uint8)
    preds = model.predict(x_batch, verbose=0)
    for i in range(min(8, xb.shape[0])):
        cls = int(np.argmax(preds[i]))
        heat = compute_gradcam(model, x_batch[i:i+1], class_index=cls)
        ov = overlay_heatmap(xb_vis[i], heat)
        from PIL import Image
        Image.fromarray(ov).save(os.path.join(out_dir, f"sample_{i}_{class_names[cls]}.png"))
