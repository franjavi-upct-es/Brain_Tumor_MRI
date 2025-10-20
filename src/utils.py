# utils.py - Small utilities for config, seeding, and class weights
# -----------------------------------------------------------------
# This module contains tiny, dependency--light helpers used across the project.
# We keep it intentionally simple so you can read/understand it in isolation
#
# Contents:
# - set_seed:       make results repeatable (to extent TF randomness allows)
# - load_config:    load YAML configuration files
# - walk_class_counts: count images per class from a folder tree
# - compute_class_weights: map raw counts to inverse-frequency weights
#
# Notes:
# - We avoid importing heavy frameworks at module import time. TF and Torch are
#   only touched inside try/except blocks to avoid hard dependencies.

import os
import random
from types import new_class

import numpy as np
import yaml
from collections import Counter
from typing import Dict, List

def set_seed(seed: int = 42) -> None:
    """
    Set random seed across common libraries to improve reproducibility.
    This cannot guarantee bitwise determinism in TF (due to CuDNN etc.),
    but it reduces run-to-run variance, which is good enough for research.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Tensorflow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    # PyTorch (no used here, but safe to seed if available)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

def load_config(path: str) -> dict:
    """Load a YAML config from disk into a Python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def walk_class_counts(root_dir: str, class_names: List[str] = None) -> Dict[str, int]:
    """
    Recursively count images per class based on subfolders:
    root_dir/<class>/*.jpg
    Only counts files with typical image extensions. This function is used to
    compute class weights (see below) in filesystem-friendly way.
    """
    counts = Counter()
    if not os.path.isdir(root_dir):
        return counts
    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        if class_names and cls not in class_names:
            # If a fixed list of classes was provided, skip foreign subfolders
            continue
        n = 0
        for _, _, files in os.walk(cls_path):
            n += sum(
                1 for x in files
                if x.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            )
        counts[cls] = n
    return counts

def compute_class_weights(counts: Dict[str, int]) -> Dict[str, float]:
    """
    Convert raw samples counts to inverse-frequency weights:
        w_c = N / (C * n_c)
    where N is total sample count, C number of classes, and n_c samples in class c.
    This simple scheme tends to work well for cross-entropy training.
    """
    if not counts:
        return {}
    total = sum(counts.values())
    n_classes = len(counts)
    weights = {}
    for cls, c in counts.items():
        weights[cls] = (total / (n_classes * c)) if c > 0 else 0.0
    return weights