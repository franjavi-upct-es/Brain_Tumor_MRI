# src/utils/logging_utils.py — Weights & Biases experiment tracking helpers
"""
Utility functions for initializing and managing W&B experiment tracking.
All experiments are logged to W&B for reproducibility and comparison,
following the experiment tracking requirements.

Usage:
    wandb_logger = init_wandb_logger(cfg)
    wandb_logger.log_hyperparams(cfg)
"""

import logging
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def init_wandb_logger(
    cfg: DictConfig,
    fold: int | None = None,
    tags: list | None = None,
) -> Any:
    """
    Initialize a PyTorch Lightning W&B logger from Hydra config.

    Args:
        cfg: Hydra DictConfig containing wandb settings under cfg.wandb.
        fold: Optional cross-validation fold index to include in run name.
        tags: Optional additional tags to append to the run.

    Returns:
        A pytorch_lightning.loggers.WandbLogger instance, or a CSVLogger
        fallback if W&B is disabled or unavailable.
    """
    wandb_cfg = cfg.get("wandb", {})
    mode = wandb_cfg.get("mode", "disabled")

    if mode == "disabled":
        logger.info("W&B disabled. Using CSVLogger as fallback.")
        return _create_csv_logger(cfg, fold)

    try:
        from pytorch_lightning.loggers import WandbLogger

        # Build run name from config
        run_name = _build_run_name(cfg, fold)

        # Merge default and additional tags
        run_tags = list(wandb_cfg.get("tags", []))
        if tags:
            run_tags.extend(tags)
        if fold is not None:
            run_tags.append(f"fold_{fold}")

        wandb_logger = WandbLogger(
            project=wandb_cfg.get("project", "brain-tumor-v2"),
            entity=wandb_cfg.get("entity", None),
            name=run_name,
            tags=run_tags,
            mode=mode,
            config=OmegaConf.to_container(cfg, resolve=True),
            save_dir=cfg.paths.get("logs", "outputs/logs"),
        )

        logger.info(
            "W&B logger initialized: project=%s, run=%s",
            wandb_cfg.get("project"),
            run_name,
        )
        return wandb_logger

    except ImportError:
        logger.warning("wandb not installed. Using CSVLogger as fallback.")
        return _create_csv_logger(cfg, fold)
    except Exception as e:
        logger.warning("W&B init failed (%s). Using CSVLogger as fallback.", e)
        return _create_csv_logger(cfg, fold)


def _build_run_name(cfg: DictConfig, fold: int | None = None) -> str:
    """
    Build a descriptive run name from configuration.

    Format: {model_name}_{dataset}_{training}[_fold{N}]

    Args:
        cfg: Hydra DictConfig.
        fold: Optional fold index.

    Returns:
        A string run name.
    """
    parts = [
        cfg.get("model", {}).get("name", "unknown_model"),
        cfg.get("dataset", {}).get("name", "unknown_data"),
        cfg.get("training", {}).get("name", "unknown_train"),
    ]
    if fold is not None:
        parts.append(f"fold{fold}")
    return "_".join(parts)


def _create_csv_logger(cfg: DictConfig, fold: int | None = None) -> Any:
    """
    Create a CSV logger as fallback when W&B is unavailable.

    Args:
        cfg: HydraConfig.
        fold: Optional fold index.

    Returns:
        A pytorch_lightning.loggers.CSVLogger instance.
    """
    from pytorch_lightning.loggers import CSVLogger

    name = _build_run_name(cfg, fold)
    save_dir = cfg.paths.get("logs", "outputs/logs")

    return CSVLogger(save_dir=save_dir, name=name)


def log_config_as_artifact(wandb_logger: Any, cfg: DictConfig) -> None:
    """Log the full Hydra configuration as a W&B artifact for provenance.

    Args:
        wandb_logger: An active WandbLogger instance.
        cfg: The complete Hydra DictConfig to log.
    """
    try:
        import wandb

        if wandb.run is not None:
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            artifact = wandb.Artifact(
                name=f"config_{wandb.run.id}",
                type="configuration",
                metadata=config_dict,
            )
            wandb.run.log_artifact(artifact)
            logger.info("Configuration logged as W&B artifact.")
    except Exception as e:
        logger.debug("Could not log config artifact: %s", e)
