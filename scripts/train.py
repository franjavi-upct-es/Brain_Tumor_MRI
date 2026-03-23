# scripts/train.py — Training entry point for brain tumor classification
"""
Hydra-based training script for a single fold of cross-validation.

Implements the two-phase training protocol:
  Phase 1: Head-only training with frozen backbone.
  Phase 2: Gradual unfreezing with discriminative learning rates.

Usage:
    # Train fold 0 with default config (DenseNet-121 baseline)
    python scripts/train.py fold=0

    # Train with specific model and training config
    python scripts/train.py model=resnet50_2d training=focal_loss fold=2

    # Train all folds (use run_cross_validation.py instead)
    for fold in 0 1 2 3 4; do
        python scripts/train.py fold=$fold
    done
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def train_fold(
    fold: int = 0,
    backbone: str = "densenet121",
    num_classes: int = 2,
    in_channels: int = 4,
    pretrained: bool = True,
    head_hidden_dim: int = 256,
    head_dropout: float = 0.5,
    phase1_epochs: int = 10,
    phase2_epochs: int = 25,
    lr: float = 1e-3,
    lr_backbone: float = 1e-5,
    lr_head: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 4,
    patience: int = 10,
    unfreeze_every: int = 3,
    data_dir: str = "data/processed/brats2023",
    splits_path: str = "data/splits/brats_5fold.json",
    output_dir: str = "outputs",
    seed: int = 42,
    precision: int = 16,
    wandb_project: str = "brain-tumor-v2",
    wandb_mode: str = "disabled",
) -> dict:
    """Train a single fold of cross-validation.

    Parameters
    ----------
    fold : int
        Cross-validation fold index (0-4).
    backbone : str
        Backbone architecture name.
    All other parameters correspond to configs in configs/ directory.

    Returns
    -------
    dict
        Training results including best metrics and checkpoint path.
    """
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger

    from src.data.datamodule import BrainTumorDataModule
    from src.models.classifier import TumorClassifier
    from src.training.callbacks import build_callbacks
    from src.utils.reproducibility import set_monai_determinism

    # Set deterministic training
    set_monai_determinism(seed)

    logger.info("=" * 60)
    logger.info("TRAINING FOLD %d/%d", fold, 4)
    logger.info("=" * 60)
    logger.info("Backbone: %s, Pretrained: %s", backbone, pretrained)
    logger.info("Phase 1: %d epochs (head only), Phase 2: %d epochs (unfreeze)", phase1_epochs, phase2_epochs)

    # Resolve paths
    data_path = PROJECT_ROOT / data_dir
    splits_file = PROJECT_ROOT / splits_path
    fold_output = PROJECT_ROOT / output_dir / "models" / f"fold_{fold}"
    metrics_output = PROJECT_ROOT / output_dir / "metrics" / f"fold_{fold}"
    fold_output.mkdir(parents=True, exist_ok=True)
    metrics_output.mkdir(parents=True, exist_ok=True)

    # DataModule
    datamodule = BrainTumorDataModule(
        data_dir=data_path,
        splits_path=splits_file,
        batch_size=batch_size,
        num_workers=num_workers,
        fold=fold,
    )

    # --- Phase 1: Head-only training ---
    logger.info("\n--- PHASE 1: Head-only training (backbone frozen) ---")

    model_phase1 = TumorClassifier(
        backbone_name=backbone,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        freeze_backbone=True,
    )

    # W&B logger
    wandb_logger = None
    if wandb_mode != "disabled":
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=f"fold{fold}_{backbone}_phase1",
            mode=wandb_mode,
            tags=["phase1", f"fold{fold}", backbone],
        )

    callbacks_phase1 = build_callbacks(
        checkpoint_dir=str(fold_output / "phase1"),
        patience=patience,
        enable_gradual_unfreeze=False,
    )

    trainer_phase1 = pl.Trainer(
        max_epochs=phase1_epochs,
        accelerator="auto",
        precision=precision,
        callbacks=callbacks_phase1,
        logger=wandb_logger,
        enable_progress_bar=True,
        deterministic=True,
    )

    t_start = time.time()
    trainer_phase1.fit(model_phase1, datamodule)
    phase1_time = time.time() - t_start
    logger.info("Phase 1 complete in %.1f seconds", phase1_time)

    # --- Phase 2: Gradual unfreezing ---
    logger.info("\n--- PHASE 2: Gradual unfreezing ---")

    # Unfreeze backbone and set discriminative LR
    model_phase1.unfreeze_backbone()
    model_phase1.hparams.freeze_backbone = False
    model_phase1.hparams.lr_backbone = lr_backbone
    model_phase1.hparams.lr_head = lr_head

    callbacks_phase2 = build_callbacks(
        checkpoint_dir=str(fold_output / "phase2"),
        patience=patience,
        enable_gradual_unfreeze=True,
        unfreeze_start_epoch=0,  # Already unfrozen; gradual from here
        unfreeze_every_n_epochs=unfreeze_every,
    )

    if wandb_mode != "disabled":
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=f"fold{fold}_{backbone}_phase2",
            mode=wandb_mode,
            tags=["phase2", f"fold{fold}", backbone],
        )

    trainer_phase2 = pl.Trainer(
        max_epochs=phase2_epochs,
        accelerator="auto",
        precision=precision,
        callbacks=callbacks_phase2,
        logger=wandb_logger,
        enable_progress_bar=True,
        deterministic=True,
    )

    t_start = time.time()
    trainer_phase2.fit(model_phase1, datamodule)
    phase2_time = time.time() - t_start
    logger.info("Phase 2 complete in %.1f seconds", phase2_time)

    # --- Save final metrics ---
    results = {
        "fold": fold,
        "backbone": backbone,
        "phase1_epochs": phase1_epochs,
        "phase2_epochs": phase2_epochs,
        "total_training_time_seconds": phase1_time + phase2_time,
        "seed": seed,
    }

    # Extract best validation metrics from trainer
    if trainer_phase2.callback_metrics:
        for key, value in trainer_phase2.callback_metrics.items():
            if key.startswith("val/"):
                results[key] = float(value)

    metrics_path = metrics_output / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", metrics_path)
    logger.info("Metrics: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in results.items()})

    return results


def main() -> None:
    """Parse arguments and train a single fold."""
    parser = argparse.ArgumentParser(description="Train brain tumor classifier")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="densenet121")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--phase1-epochs", type=int, default=10)
    parser.add_argument("--phase2-epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="data/processed/brats2023")
    parser.add_argument("--splits-path", type=str, default="data/splits/brats_5fold.json")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-mode", type=str, default="disabled",
                        choices=["online", "offline", "disabled"])

    # Handle Hydra-style key=value args
    args, unknown = parser.parse_known_args()
    for u in unknown:
        if "=" in u:
            key, val = u.split("=", 1)
            key = key.replace("-", "_")
            if hasattr(args, key):
                attr_type = type(getattr(args, key))
                setattr(args, key, attr_type(val))

    train_fold(**vars(args))


if __name__ == "__main__":
    main()
