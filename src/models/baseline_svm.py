# src/models/baseline_svm.py
# Classical ML baseline: SVM with handcrafted features
"""
Non-deep-learning baseline using SVM + radiomics-inspired features.

This baseline is mandatory to demonstrate that deep learning
adds value beyond classical approaches. It includes:
  1. Dummy classifier (majority class): absolute floor.
  2. SVM + handcrafted features: what a computational radiologist would
     use without DL. Features: intensity statistics, texture (GLCM-like),
     and shape descriptors extracted from the 2D slices.

Raghu et al. (2019, NeurIPS) found transfer learning offers surprisingly
little benefit over simple approaches on medical imaging. This baseline
tests whether that finding holds for glioma grading.

References:
    - Raghu et al. (2019). Transfusion. NeurIPS.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def extract_handcrafted_features(image: np.ndarray) -> np.ndarray:
    """
    Extract handcrafted features from a multi-channel 2D MRI slice.

    Features per channel:
        - Intensity: mean, std, median, min, max, skewness, kurtosis (7)
        - Texture: Haralick-inspired stats from gradient magnitudes (4)
        - Shape: foreground area ratio, perimeter estimate (2)
    Total: 13 features × num_channels

    Parameters
    ----------
    image : np.ndarray
        Multi-channel 2D image of shape (C, H, W).

    Returns
    -------
    np.ndarray
        1D feature vector
    """
    from scipy import ndimage, stats

    features = []

    for c in range(image.shape[0]):
        channel = image[c]
        foreground = (
            channel[channel != 0]
            if np.any(channel != 0)
            else channel.flatten()
        )

        # Intensity statistics
        features.extend(
            [
                float(np.mean(foreground)),
                float(np.std(foreground)),
                float(np.median(foreground)),
                float(np.min(foreground)),
                float(np.max(foreground)),
                float(stats.skew(foreground.flatten())),
                float(stats.kurtosis(foreground.flatten())),
            ]
        )

        # Texture: gradient-based
        grad_x = np.abs(np.diff(channel, axis=1)).mean()
        grad_y = np.abs(np.diff(channel, axis=0)).mean()
        grad_mag = ndimage.sobel(channel)
        features.extend(
            [
                float(grad_x),
                float(grad_y),
                float(np.mean(np.abs(grad_mag))),
                float(np.std(np.abs(grad_mag))),
            ]
        )

        # Shape: foreground area ratio
        fg_mask = channel > 0
        fg_ratio = (
            float(fg_mask.sum() / fg_mask.size) if fg_mask.size > 0 else 0.0
        )
        # Perimeter estimate via gradient of binary mask
        perimeter = float(ndimage.sobel(fg_mask.astype(float)).sum())
        features.extend([fg_ratio, perimeter])

    return np.array(features, dtype=np.float32)


def extract_features_from_dataset(
    samples: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract handcrafted features from all samples in a dataset.

    Parameters
    ----------
    sampels : list[dict]
        List of sample dictionaries with "image", "label", "patient_id".

    Returns
    -------
    tuple
        (features_matrix, labels, patient_ids)
    """
    feature_list = []
    labels = []
    patient_ids = []

    for sample in samples:
        image = sample["image"]
        if hasattr(image, "numpy"):
            image = image.numpy()

        feat = extract_handcrafted_features(image)
        feature_list.append(feat)
        labels.append(sample["label"])
        patient_ids.append(sample["patient_id"])

    X = np.stack(feature_list)
    y = np.array(labels)

    logger.info("Extracted features: X=%s, y=%s", X.shape, y.shape)
    return X, y, patient_ids


def train_svm_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
) -> Pipeline:
    """
    Train an SVM classifier with standardized features.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (n_samples, n_features).
    y_train : np.ndarray
        Training labels
    kernel : str
        SVM kernel type. Default "rbf".
    C : float
        Regularization parameter.
    gamma : str
        Kernel coefficient.

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline (StandardScaler + SVC).
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    train_acc = balanced_accuracy_score(y_train, pipeline.predict(X_train))
    logger.info("SVM trained: balanced_accuracy=%.4f (train)", train_acc)

    return pipeline


def train_dummy_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = "most_frequent",
) -> DummyClassifier:
    """
    Train a dummy classifier (absolute floor baseline).

    Parameters
    ----------
    X_train : np.ndarray
        Training features (unused by dummy, but needed for API).
    y_train : np.ndarray
        Training labels.
    strategy : str
        Strategy: "most_frequent" (majority class) or "stratified".

    Returns
    -------
    DummyClassifier
        Fitted dummy classifier.
    """
    dummy = DummyClassifier(strategy=strategy, random_state=42)
    dummy.fit(X_train, y_train)
    logger.info("Dummy classifier trained (strategy=%s)", strategy)
    return dummy


def evaluate_baseline(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "baseline",
) -> dict[str, float]:
    """
    Evaluate a baseline model with standard metrics.

    Parameters
    ----------
    model : sklearn estimator
        Fitted classifier with predict() and predict_proba().
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    model_name : str
        Name for logging.

    Returns
    -------
    dict[str, float]
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        ),
    }

    # AUC requires probability estimates
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                metrics["auc_roc"] = float(
                    roc_auc_score(y_test, y_proba[:, 1])
                )
            else:
                metrics["auc_roc"] = float(
                    roc_auc_score(
                        y_test, y_proba, multi_class="ovr", average="macro"
                    )
                )
        except Exception as e:
            logger.warning("Could not compute AUC for %s: %s", model_name, e)

    logger.info(
        "%s evaluation: %s",
        model_name,
        {k: f"{v:.4f}" for k, v in metrics.items()},
    )
    return metrics
