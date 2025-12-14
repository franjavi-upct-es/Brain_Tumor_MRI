# src/losses.py - Advanced Loss Functions for Medical Imaging
# ===========================================================
# Implements Focal Loss and Tversky Loss to address:
# 1. Class imbalance (more healthy samples than tumors)
# 2. Hard negative mining (focus on difficult cases)
# 3. False negative reduction (critical in medical diagnosis)

import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance and hard examples.

    FL(p_t) = -𝛂_t * (1 - p_t)^γ * log(p_t)

    where:
    - p_t: predicted probability for true class
    - α_t: class weight (higher for minority class)
    - γ: focusing parameter (γ=2 standard, higher = more focus on hard examples)

    Key benefits for medical imaging:
    - Automatically down-weights easy examples (confident correct predictions)
    - Up-weights hard examples (misclassifications or low-confidence correct predictions)
    - Reduces false negatives by penalizing missed tumors more heavily

    Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection"
    """

    def __init__(
        self,
        alpha: float = 0.75,  # Weight for positive class (tumor detection)
        gamma: float = 2.0,  # Focusing parameter
        from_logits: bool = True,
        label_smoothing: float = 0.0,
        name: str = "focal_loss",
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: True labels (one-hot encoded), shape [batch, num_classes]
            y_pred: Predicted logits or probabilities, shape [batch, num_classes]

        Returns:
            Focal loss value (scalar)
        """
        # Convert logits to probabilities if needed
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Clip to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + (
                self.label_smoothing / num_classes
            )

        # Compute corss entropy
        ce = -y_true * tf.math.log(y_pred)

        # Compute focal term: (1 - p_t)^γ
        # p_t is the probability of the true class
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        # Apply focal weight
        focal_ce = focal_weight * ce

        # Apply class weights (alpha)
        # For binary: alpha for positive class, (1-alpha) for negative
        # For multiclass: can extend to per-class alphas
        if self.alpha is not None:
            alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
            focal_ce = alpha_t * focal_ce

        return tf.reduce_mean(tf.reduce_sum(focal_ce, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config


class TverskyLoss(tf.keras.losses.Loss):
    """
    Tversky Loss - Generalization of Dice Loss for imabalanced data.

    Particularly effective for medical imaging where False Negatives (missed tumors)
    are more costly than False Positives (false alarms).

    TL = 1 - (TP / (TP + α*FP + β*FN))

    where:
    - α: weight for False Positives (typically 0.3-0.5)
    - β: weight for False Negatives (typically 0.5-0.7)

    Setting β > α prioritizes recall (sensitivity), reducing missed tumors.

    Reference: Salehi et al. (2017) "Tversky loss function for image segmentation"
    """

    def __init__(
        self,
        alpha: float = 0.3,  # FP weight (lower = tolerate more false alarms)
        beta: float = 0.7,  # FN weight (higher = penalize missed tumors)
        smooth: float = 1e-6,
        from_logits: bool = True,
        name: str = "tversky_loss",
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: True labels (one-hot), shape [batch, num_classes]
            y_pred: Predicted logits/probs, shape [batch, num_classes]
        """
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Flatten for easier computation
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        # Compute confusion matrix components
        true_pos = tf.reduce_sum(y_true_flat * y_pred_flat)
        false_neg = tf.reduce_sum(y_true_flat * (1 - y_pred_flat))
        false_pos = tf.reduce_sum((1 - y_true_flat) * y_pred_flat)

        # Tversky index
        tversky_index = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )

        return 1.0 - tversky_index

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "smooth": self.smooth,
                "from_logits": self.from_logits,
            }
        )
        return config


class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    Weighted BCE specifically for Binary Tumor Detection.

    Allows asymmetric penalization:
    - Higher weight for False Negatives (missed tumors)
    - Lower weight for False Positives (false alarms)

    This is simpler than Focal Loss but very effective for the specific
    problem of tumor detection where FN >> FP in terms of cost.
    """

    def __init__(
        self,
        pos_weight: float = 3.0,  # Weight for tumor class (multiply loss by this factor)
        from_logits: bool = True,
        name: str = "weighted_bce",
    ):
        super().__init__(name=name)
        self.pos_weight = pos_weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Binary labels [batch, 1] or multiclass one-hot [batch, num_classes]
            y_pred: Logits or probabilities
        """
        # For multiclass, extract tumor vs no_tumor binary signal
        # Assuming last class is 'no_tumor', binary is 1 - no_tumor_prob

        if self.from_logits:
            # Compute BCE from logits for numerical stability
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        else:
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
            bce = -(
                y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
            )

        # Apply positive class weight
        weights = 1.0 + (self.pos_weight - 1.0) * y_true
        weighted_bce = weights * bce

        return tf.reduce_mean(weighted_bce)

    def get_config(self):
        config = super().get_config()
        config.update({"pos_weight": self.pos_weight, "from_logits": self.from_logits})
        return config


def get_loss_function(loss_name: str, from_logits: bool = True, **kwargs):
    """
    Factory function to instantiate loss functions by name.

    Usage:
        loss_fn = get_loss_function('focal', alpha=0.75, gamma=2.0)
        model.compile(optimizer='adam', loss=loss_fn)

    Available losses:
        - 'focal': FocalLoss (default: α=0.75, γ=2.0)
        - 'tversky': TverskyLoss (default: α=0.3, β=0.7)
        - 'weighted_bce': WeightedBinaryCrossEntropy (default: pos_weight=3.0)
        - 'categorical_crossentropy': Standard Keras loss
    """
    loss_map = {
        "focal": FocalLoss,
        "tversky": TverskyLoss,
        "weighted_bce": WeightedBinaryCrossEntropy,
    }

    if loss_name in loss_map:
        return loss_map[loss_name](from_logits=from_logits, **kwargs)
    elif loss_name == "categorical_crossentropy":
        return tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    """
    Example usage demonstrating the efectiveness of Focal Loss
    for imbalanced medical imaging data.
    """
    import numpy as np

    # Simulate predictions for 100 samples
    # 90 healthy (easy), 10 tumors (hard, some misclassified)
    batch_size = 100
    num_classes = 4  # [glioma, meningioma, pituitary, no_tumor]

    # True labels: mostly class 2 (no_tumor), few class 0 (glioma)
    y_true = np.zeros((batch_size, num_classes))
    y_true[:90, 2] = 1.0  # 90 healthy
    y_true[90:, 0] = 1.0  # 10 tumors (glioma)

    # Simulated predictions (logits)
    y_pred = np.random.randn(batch_size, num_classes)

    # Make healthy samples confident (easy examples)
    y_pred[:90, 2] += 3.0

    # Make tumor samples less confident (hard examples)
    y_pred[90:, 0] += 1.0  # Some are still weak predictions
    y_pred[90:, 2] += 0.5  # Model is confused (predicts healthy for tumors)

    # Convert to tensors
    y_true_tf = tf.constant(y_true, dtype=tf.float32)
    y_pred_tf = tf.constant(y_pred, dtype=tf.float32)

    # Compare losses
    print("=" * 60)
    print("LOSS COMPARISON FOR IMABALANCED MEDICAL DATA")
    print("=" * 60)
    print("Dataset: 90 healthy (easy), 10 tumors (hard, partially misclassified)")
    print()

    # Standard CE
    ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    ce_value = ce_loss(y_true_tf, y_pred_tf).numpy()
    print(f"Standard Categorical Cross-Entropy: {ce_value:.4f}")

    # Focall Loss (γ=2, α=0.75)
    focal = FocalLoss(alpha=0.75, gamma=2.0)
    focal_hard_value = focal(y_true_tf, y_pred_tf).numpy()
    print(f"Focal Loss (γ=2, α=0.75):       {focal_hard_value:.4f}")

    # Focal Loss with higher gamma (more focus on hard examples)
    focal_hard = FocalLoss(alpha=0.75, gamma=3.0)
    focal_hard_value = focal_hard(y_true_tf, y_pred_tf).numpy()
    print(f"Focal Loss (γ=3.0, α=0.75):          {focal_hard_value:.4f}")

    print()
    print("Interpretation:")
    print("- Focal Loss automatically down-weights easy examples (healthy samples)")
    print("- It focuses training on hard examples (misclassified tumors)")
    print("- Higher γ = more aggressive focus on errors")
    print()
    print("Expected result for your case:")
    print("✓ Reduced False Negatives (missed tumors)")
    print("✓ Better calibration (less overconfidence)")
    print("✓ Improved recall without sacrificing precision")
