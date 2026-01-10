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


def _find_last_conv(model):
    """
    Heuristic search for the last Conv2D layer name in a (possibly nested) model.
    Grad-CAM requieres a convolutional feature map; the 'last conv' is a good default.
    """
    # Scan from the end and recurse into nested Models.
    for layer in reversed(getattr(model, "layers", [])):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        if isinstance(layer, tf.keras.Model):
            found = _find_last_conv(layer)
            if found is not None:
                return found
    return None


def _find_layer_by_name(model, layer_name: str):
    for layer in getattr(model, "layers", []):
        if getattr(layer, "name", None) == layer_name:
            return layer
        if isinstance(layer, tf.keras.Model):
            found = _find_layer_by_name(layer, layer_name)
            if found is not None:
                return found
    return None


def _find_top_level_submodel_containing(model, target_layer):
    """Return the immediate child Model in `model.layers` that contains `target_layer`."""

    def contains(parent):
        for layer in getattr(parent, "layers", []):
            if layer is target_layer:
                return True
            if isinstance(layer, tf.keras.Model) and contains(layer):
                return True
        return False

    for layer in getattr(model, "layers", []):
        if isinstance(layer, tf.keras.Model) and contains(layer):
            return layer
    if contains(model):
        return model
    return None


def _call_keras_model(m, x, training=False):
    """Call a Keras model while avoiding Keras 3 single-input structure warnings."""
    if not isinstance(x, (list, tuple, dict)):
        try:
            if hasattr(m, "inputs") and len(m.inputs) == 1:
                x = [x]
        except Exception:
            pass
    return m(x, training=training)


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
        conv_layer = _find_last_conv(model)
    else:
        conv_layer = _find_layer_by_name(model, conv_layer_name)
    if conv_layer is None:
        raise ValueError("No convolutional layer found for Grad-CAM.")

    # Normalize input tensor shape/type
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
    if len(img_tensor.shape) == 3:  # add batch dim if missing
        img_tensor = img_tensor[None, ...]

    # Try the standard Functional approach first. With Keras 3, intermediate tensors
    # from nested Functional models (e.g. EfficientNet inside a wrapper) may be
    # disconnected from the outer graph and can trigger a KeyError at call-time.
    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs, outputs=[conv_layer.output, model.output]
        )
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(img_tensor)
            conv_outputs, logits = _call_keras_model(
                grad_model, img_tensor, training=False
            )
            if class_index is None:
                class_index = tf.argmax(logits[0])
            loss = logits[:, class_index]
        grads = tape.gradient(loss, conv_outputs)
    except KeyError:
        # Fallback: split into (pre -> trunk -> head) so we can read conv outputs
        # from the trunk's own graph while still computing logits consistently.
        trunk = _find_top_level_submodel_containing(model, conv_layer)
        if trunk is None or trunk is model:
            raise

        if not getattr(trunk, "_inbound_nodes", None):
            raise ValueError(
                f"Grad-CAM fallback failed: submodel '{trunk.name}' has no inbound nodes."
            )

        node = trunk._inbound_nodes[0]
        if not getattr(node, "input_tensors", None) or not getattr(node, "output_tensors", None):
            raise ValueError(
                f"Grad-CAM fallback failed: could not resolve IO tensors for submodel '{trunk.name}'."
            )

        trunk_in_tensor = node.input_tensors[0]
        trunk_out_tensor = node.output_tensors[0]

        pre_model = tf.keras.Model(model.inputs, trunk_in_tensor)
        trunk_feat_model = tf.keras.Model(
            trunk.inputs, outputs=[conv_layer.output, trunk.output]
        )
        head_model = tf.keras.Model([trunk_out_tensor], model.output)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(img_tensor)
            trunk_in = _call_keras_model(pre_model, img_tensor, training=False)
            conv_outputs, trunk_out = _call_keras_model(
                trunk_feat_model, trunk_in, training=False
            )
            logits = _call_keras_model(head_model, trunk_out, training=False)
            if class_index is None:
                class_index = tf.argmax(logits[0])
            loss = logits[:, class_index]
        grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Failed to compute gradients. The conv layer may not be in the gradient path.")

    conv_outputs = tf.cast(conv_outputs, tf.float32)
    grads = tf.cast(grads, tf.float32)

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
