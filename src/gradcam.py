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
import matplotlib.cm as cm


def _find_last_conv(model) -> str:
    """
    Heuristic search for the last Conv2D layer name in a (possibly nested) model.
    Grad-CAM requieres a convolutional feature map; the 'last conv' is a good default.
    """
    # First pass: top-level layers from the end
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    # Second pass: look one level deep in nested models
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub
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
    conv_layer = (
        _find_last_conv(model)
        if conv_layer_name is None
        else model.get_layer(conv_layer_name, None)
    )
    if conv_layer is None:
        raise ValueError("No convolutional layer found for Grad-CAM.")

    # Normalize input tensor shape/type
    if isinstance(img_tensor, dict):
        img_tensor = img_tensor
    else:
        img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
        if len(img_tensor.shape) == 3:  # add batch dim if missing
            img_tensor = img_tensor[None, ...]

    # Create an intermediate model that outputs both conv activations and final logits
    # This must be done outside the GradientTape context
    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output]
        )
    except (ValueError, AttributeError, KeyError) as e:
        # If model creation fails, it's likely due to graph tracing issues with loaded models
        # Fall back to using the original model and accessing layer output during forward pass
        raise ValueError(
            f"Could not create gradient model. This may happen with complex loaded models. "
            f"Original error: {e}"
        )

    # Use GradientTape to compute gradients
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(img_tensor)
        conv_outputs, logits = grad_model(img_tensor, training=False)

        if class_index is None:
            class_index = tf.argmax(logits[0])
        loss = logits[:, class_index]

    # Compute gradients of the class score w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Failed to compute gradients. The conv layer may not be in the gradient path.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    # Weighted sum of channels followed by ReLU
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-18)
    return heatmap.numpy()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35):
    """
    Overlay a Grad-CAM heatmap on top of the original image.
    Args:
        image: uint8 array (H,W,3) in [0,255]
        heatmap: float array (h,w) in [0,1]
        alpha: blending factor for the heatmap
    Returns:
        uint8 RGB image with the heatmap overlayed.
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    # Resize heatmap to image size
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap_resized = (
            tf.image.resize(
                heatmap[..., None],
                (image.shape[0], image.shape[1]),
                method="bilinear",
            )
            .numpy()
            .squeeze()
        )
    else:
        heatmap_resized = heatmap

    # Apply colormap
    colormap = cm.get_cmap("jet")
    colored = colormap(np.clip(heatmap_resized, 0, 1))[..., :3]  # drop alpha
    colored = (colored * 255).astype(np.uint8)

    # Blend
    overlay = np.clip((1 - alpha) * image + alpha * colored, 0, 255).astype(np.uint8)
    return overlay


def save_batch_gradcam(model, x_batch, y_batch, class_names, out_dir="gradcam_samples"):
    """
    Compute Grad-CAM overlays for up to the first 8 images of a batch and save PNGs.
    We assume x_batch is preprocessed; for visualization, we min-max scale to [0,255].
    """
    os.makedirs(out_dir, exist_ok=True)
    xb = x_batch.numpy()
    # Re-scale for visualization (original preprocessing could be centered/normalized)
    xb_vis = np.clip(
        (xb - xb.min()) / (xb.max() - xb.min() + 1e-8) * 255.0, 0, 255
    ).astype(np.uint8)
    preds = model.predict(x_batch, verbose=0)
    for i in range(min(8, xb.shape[0])):
        cls = int(np.argmax(preds[i]))
        heat = compute_gradcam(model, x_batch[i : i + 1], class_index=cls)
        ov = overlay_heatmap(xb_vis[i], heat)
        from PIL import Image

        Image.fromarray(ov).save(
            os.path.join(out_dir, f"sample_{i}_{class_names[cls]}.png")
        )
