# tests/test_training/test_callbacks.py
"""Tests for build_callbacks and GradualUnfreezeCallback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch.nn as nn
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.training.callbacks import GradualUnfreezeCallback, build_callbacks


class TestBuildCallbacks:
    """Tests for the build_callbacks factory."""

    def test_returns_list(self) -> None:
        """Should return a list of callbacks."""
        callbacks = build_callbacks()
        assert isinstance(callbacks, list)
        assert len(callbacks) > 0

    def test_has_early_stopping(self) -> None:
        """Should include EarlyStopping."""
        callbacks = build_callbacks()
        types = [type(c) for c in callbacks]
        assert EarlyStopping in types

    def test_has_model_checkpoint(self) -> None:
        """Should include ModelCheckpoint."""
        callbacks = build_callbacks()
        types = [type(c) for c in callbacks]
        assert ModelCheckpoint in types

    def test_has_lr_monitor(self) -> None:
        """Should include LearningRateMonitor."""
        callbacks = build_callbacks()
        types = [type(c) for c in callbacks]
        assert LearningRateMonitor in types

    def test_with_gradual_unfreeze(self) -> None:
        """With enable_gradual_unfreeze=True, should include GradualUnfreezeCallback."""
        callbacks = build_callbacks(enable_gradual_unfreeze=True)
        types = [type(c) for c in callbacks]
        assert GradualUnfreezeCallback in types

    def test_without_gradual_unfreeze(self) -> None:
        """With enable_gradual_unfreeze=False, should not include GradualUnfreezeCallback."""
        callbacks = build_callbacks(enable_gradual_unfreeze=False)
        types = [type(c) for c in callbacks]
        assert GradualUnfreezeCallback not in types

    def test_custom_patience(self) -> None:
        """Custom patience should be used in EarlyStopping."""
        callbacks = build_callbacks(patience=5)
        early_stop = next(c for c in callbacks if isinstance(c, EarlyStopping))
        assert early_stop.patience == 5


class TestGradualUnfreezeCallback:
    """Tests for GradualUnfreezeCallback."""

    def _make_callback(
        self, start_epoch: int = 5, every_n: int = 2
    ) -> GradualUnfreezeCallback:
        return GradualUnfreezeCallback(
            unfreeze_start_epoch=start_epoch,
            unfreeze_every_n_epochs=every_n,
        )

    def _make_trainer(self, epoch: int) -> MagicMock:
        trainer = MagicMock()
        trainer.current_epoch = epoch
        return trainer

    def _make_module_with_groups(self) -> MagicMock:
        """Create a mock pl_module that supports get_backbone_layer_groups."""
        module = MagicMock()
        param1 = nn.Parameter(nn.Linear(10, 5).weight)
        param2 = nn.Parameter(nn.Linear(5, 2).weight)
        param1.requires_grad = False
        param2.requires_grad = False
        module.get_backbone_layer_groups.return_value = [[param1], [param2]]
        return module, param1, param2

    def test_no_action_before_start_epoch(self) -> None:
        """Should do nothing before unfreeze_start_epoch."""
        cb = self._make_callback(start_epoch=10)
        trainer = self._make_trainer(epoch=5)
        module = MagicMock()
        cb.on_train_epochs_start(trainer, module)
        # Layer groups should not be initialized
        assert cb._layer_groups is None

    def test_initializes_groups_at_start_epoch(self) -> None:
        """Should initialize layer groups at the start epoch."""
        cb = self._make_callback(start_epoch=5, every_n=2)
        trainer = self._make_trainer(epoch=5)
        module, param1, param2 = self._make_module_with_groups()
        cb.on_train_epochs_start(trainer, module)
        assert cb._layer_groups is not None

    def test_unfreezes_one_group_at_start(self) -> None:
        """Should unfreeze first group at start_epoch."""
        cb = self._make_callback(start_epoch=5, every_n=2)
        trainer = self._make_trainer(epoch=5)
        module, param1, param2 = self._make_module_with_groups()
        cb.on_train_epochs_start(trainer, module)
        assert param1.requires_grad is True
        assert cb._n_unfrozen == 1

    def test_unfreezes_progressively(self) -> None:
        """Should unfreeze more groups as epochs progress."""
        cb = self._make_callback(start_epoch=5, every_n=2)
        module, param1, param2 = self._make_module_with_groups()

        # At epoch 5: unfreeze group 0
        cb.on_train_epochs_start(self._make_trainer(5), module)
        assert cb._n_unfrozen == 1

        # At epoch 7 (5 + 2*1): unfreeze group 1
        cb.on_train_epochs_start(self._make_trainer(7), module)
        assert cb._n_unfrozen == 2

    def test_fallback_when_no_layer_groups(self) -> None:
        """Should call unfreeze_backbone if model has no get_backbone_layer_groups."""
        cb = self._make_callback(start_epoch=5, every_n=2)
        trainer = self._make_trainer(epoch=5)

        module = MagicMock(spec=[])  # No get_backbone_layer_groups
        module.unfreeze_backbone = MagicMock()
        cb.on_train_epochs_start(trainer, module)
        module.unfreeze_backbone.assert_called_once()

    def test_does_not_exceed_group_count(self) -> None:
        """Should cap unfreezing at total number of groups."""
        cb = self._make_callback(start_epoch=5, every_n=1)
        module, param1, param2 = self._make_module_with_groups()

        # Run many epochs beyond group count
        for epoch in range(5, 20):
            cb.on_train_epochs_start(self._make_trainer(epoch), module)

        assert cb._n_unfrozen <= 2  # Only 2 groups exist
