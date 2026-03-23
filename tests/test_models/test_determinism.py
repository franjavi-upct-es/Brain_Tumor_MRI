# tests/test_models/test_determinism.py — Determinism verification tests
"""
Verifies that identical seeds produce identical model outputs.

This is a checklist requirement: "Seed fixed everywhere
(MONAI set_determinism)" and "Same seed = same results."

Tests:
  1. Same seed → same model weights at initialization.
  2. Same seed → same forward pass output.
  3. Same seed → same training step loss.
"""

from __future__ import annotations

import torch

from src.utils.reproducibility import set_global_seed


class TestDeterminism:
    """Tests for reproducible model behavior."""

    def _build_model(self, seed: int):
        """Build a classifier with a specific seed."""
        from src.models.classifier import TumorClassifier

        set_global_seed(seed)
        return TumorClassifier(
            backbone_name="simple_cnn",
            num_classes=2,
            in_channels=4,
            pretrained=False,
            head_hidden_dim=64,
            head_dropout=0.0,  # Disable dropout for determinism check
            freeze_backbone=False,
        )

    def test_same_seed_same_weights(self) -> None:
        """
        Models initialized with the same seed must have identical weights.
        """
        model_a = self._build_model(seed=42)
        model_b = self._build_model(seed=42)

        for (name_a, param_a), (name_b, param_b) in zip(
            model_a.named_parameters(), model_b.named_parameters(), strict=True
        ):
            assert name_a == name_b
            assert torch.allclose(param_a, param_b), (
                f"Parameter {name_a} differs between runs with same seed!"
            )

    def test_same_seed_same_output(self) -> None:
        """
        Same seed + same input must produce identical forward pass output.
        """
        set_global_seed(42)
        model_a = self._build_model(seed=42)
        model_a.eval()

        set_global_seed(42)
        x = torch.randn(2, 4, 64, 64)

        set_global_seed(42)
        model_b = self._build_model(seed=42)
        model_b.eval()

        with torch.no_grad():
            out_a = model_a(x)
            out_b = model_b(x)

        assert torch.allclose(out_a, out_b, atol=1e-6), (
            f"Outputs differ: max delta = {(out_a - out_b).abs().max():.8f}"
        )

    def test_different_seed_different_weights(self) -> None:
        """Different seeds must produce different model weights."""
        model_a = self._build_model(seed=42)
        model_b = self._build_model(seed=123)

        any_different = False
        for (_, param_a), (_, param_b) in zip(
            model_a.named_parameters(), model_b.named_parameters(), strict=True
        ):
            if not torch.allclose(param_a, param_b):
                any_different = True
                break

        assert any_different, "Different seeds produced identical weights!"

    def test_training_step_deterministic(self) -> None:
        """Same seed + same data must produce identical training loss."""
        losses = []
        for _ in range(2):
            set_global_seed(42)
            model = self._build_model(seed=42)
            model.train()
            model.trainer = type("FakeTrainer", (), {"max_epochs": 10})()
            model.log = lambda *a, **k: None

            set_global_seed(42)
            batch = {
                "image": torch.randn(4, 4, 64, 64),
                "label": torch.tensor([0, 1, 0, 1]),
            }
            loss = model.training_step(batch, 0)
            losses.append(loss.item())

        assert abs(losses[0] - losses[1]) < 1e-6, (
            f"Training losses differ: {losses[0]:.8f} vs {losses[1]:.8f}"
        )
