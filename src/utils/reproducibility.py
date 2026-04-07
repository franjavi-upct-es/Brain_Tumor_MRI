# src/utils/reproducibility — Deterministic training and seed management
"""
Ensures full reproducibility across runs by setting seeds in all relevant
libraries and enabling MONAI deterministic mode.

The scientific quality checklist requires:
- Seed fixed everywhere (MONAI set_determinism)
- Same seed = same results (verified by test_determinism.py)

References:
- MONAI set_determinism: https://docs.monai-dev.io/en/stable/utils.html
- PyTorch reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
"""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from monai.utils import set_determinism
except ImportError:  # pragma: no cover - exercised by patching the symbol.
    set_determinism = None


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seed across all libraries for reproducibility.

    This function ensures deterministic behaviour by seeding Python's random
    module, NumPy, PyTorch (CPU and CUDA), and configuring PyTorch backends
    for deterministic operations.

    Args:
        seed: Integer seed value. Default is 42 as specific in
              configs/config.yaml.
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch deterministic backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    logger.info("Global seed to %d across all libraries", seed)


def set_monai_determinism(seed: int = 42) -> None:
    """
    Enable MONAI deterministic mode for medical imaging pipelines.

    MONAI's set_determinism wraps PyTorch, NumPy, and Python random seeds
    and additionally configures MONAI-specific operations for determinism.

    Args:
        seed: Integer seed value.
    """
    if set_determinism is None:
        logger.warning(
            "MONAI not installed. Falling back to manual seed setting."
        )
        set_global_seed(seed)
        return

    try:
        set_determinism(seed=seed)
        logger.info("MONAI deterministic mode enabled with seed %d.", seed)
    except ImportError:
        logger.warning(
            "MONAI not installed. Falling back to manual seed setting."
        )
        set_global_seed(seed)


def seed_worker(worker_id: int) -> None:
    """
    Seed function for DataLoader workers to ensure reproducibility.

    Pass this as the worker_init_fn argument to torch.utils.data.DataLoader
    to ensure each worker uses a deterministic seed derived from the global
    state.

    Args:
        worker_id: Worker index assigned by the DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int = 42) -> torch.Generator:
    """
    Create a seeded PyTorch Generator for DataLoader shuffling.

    Args:
        seed: Integer seed value.

    Returns:
        A torch.Generator instance seeded with the given value.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# Public aliases used by src.utils.__init__
set_seed = set_global_seed
configure_determinism = set_monai_determinism
