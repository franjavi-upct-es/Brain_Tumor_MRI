# tests/test_training/test_schedulers.py
"""Tests for CosineAnnealingWithWarmup and build_scheduler."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.training.schedulers import CosineAnnealingWithWarmup, build_scheduler


def _make_optimizer(lr: float = 1e-3) -> torch.optim.Optimizer:
    """Create a simple optimizer for testing."""
    model = nn.Linear(10, 2)
    return torch.optim.Adam(model.parameters(), lr=lr)


class TestCosineAnnealingWithWarmup:
    """Tests for CosineAnnealingWithWarmup scheduler."""

    def test_warmup_phase_lr_increases(self) -> None:
        """During warmup, LR should increase linearly."""
        optimizer = _make_optimizer(lr=1e-3)
        scheduler = CosineAnnealingWithWarmup(
            optimizer, warmup_epochs=5, max_epochs=20, min_lr=1e-7
        )

        lrs = []
        for epoch in range(5):
            lrs.append(scheduler.get_lr()[0])
            optimizer.step()
            scheduler.step()

        # LR should be monotonically increasing during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1], f"LR not increasing at epoch {i}"

    def test_cosine_phase_lr_decreases(self) -> None:
        """After warmup, LR should follow cosine decay."""
        optimizer = _make_optimizer(lr=1e-3)
        scheduler = CosineAnnealingWithWarmup(
            optimizer, warmup_epochs=2, max_epochs=10, min_lr=1e-7
        )

        # Skip warmup
        for _ in range(2):
            optimizer.step()
            scheduler.step()

        lrs = []
        for _ in range(5):
            lrs.append(scheduler.get_lr()[0])
            optimizer.step()
            scheduler.step()

        # LR should generally decrease in cosine phase
        assert lrs[-1] < lrs[0]

    def test_min_lr_not_exceeded(self) -> None:
        """LR should not go below min_lr."""
        min_lr = 1e-6
        optimizer = _make_optimizer(lr=1e-3)
        scheduler = CosineAnnealingWithWarmup(
            optimizer, warmup_epochs=2, max_epochs=20, min_lr=min_lr
        )

        for _ in range(20):
            lrs = scheduler.get_lr()
            for lr in lrs:
                assert lr >= min_lr - 1e-10
            optimizer.step()
            scheduler.step()

    def test_warmup_epoch_zero(self) -> None:
        """With warmup_epochs=0, should go straight to cosine."""
        optimizer = _make_optimizer(lr=1e-3)
        scheduler = CosineAnnealingWithWarmup(
            optimizer, warmup_epochs=0, max_epochs=10, min_lr=1e-7
        )
        lrs = scheduler.get_lr()
        assert len(lrs) > 0

    def test_multiple_param_groups(self) -> None:
        """Should handle multiple param groups."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(
            [
                {"params": model.weight, "lr": 1e-4},
                {"params": model.bias, "lr": 1e-3},
            ]
        )
        scheduler = CosineAnnealingWithWarmup(
            optimizer, warmup_epochs=3, max_epochs=15, min_lr=1e-7
        )
        lrs = scheduler.get_lr()
        assert len(lrs) == 2


class TestBuildScheduler:
    """Tests for build_scheduler factory."""

    def test_cosine_annealing(self) -> None:
        """Should build CosineAnnealingWithWarmup."""
        optimizer = _make_optimizer()
        scheduler = build_scheduler(
            optimizer, name="cosine_annealing", warmup_epochs=3, max_epochs=20
        )
        assert isinstance(scheduler, CosineAnnealingWithWarmup)

    def test_step_scheduler(self) -> None:
        """Should build StepLR."""
        optimizer = _make_optimizer()
        scheduler = build_scheduler(optimizer, name="step")
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_unknown_scheduler_raises(self) -> None:
        """Unknown scheduler name should raise ValueError."""
        optimizer = _make_optimizer()
        with pytest.raises(ValueError):
            build_scheduler(optimizer, name="unknown_scheduler")
