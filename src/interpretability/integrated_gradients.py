# src/interpretability/integrated_gradients.py
# Integrated Gradients via Captum
"""
Computes pixel-level attribution maps using Integrated Gradients
(Sundararajan et al., 2017), providing a complementary view to Grad-CAM.

While Grad-CAM highlights regions at feature-map resolution, Integrated
Gradients produces pixel-resolution attributions by integrating gradients
along a path from a baseline (black image) to the input.

This serves as a secondary interpretability method to cross-validate
Grad-CAM findings. If both methods agree on tumor-region attention,
the evidence for clinically relevant model behavior is stronger.

References:
    - Sundararajan et al. (2017). Axiomatic Attribution for Deep
      Networks. ICML.
    - Captum IntegratedGradients: https://captum.ai
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class IntegratedGradientsGenerator:
    """
    Generates Integrated Gradients attribution maps.

    Parameters
    ----------
    model : nn.Module
        Trained classification model.
    n_steps : int
        Number of interpolation steps along the integration path.
        More steps = more accurate but slower. Default 50.
    internal_batch_size : int
        Batch size for internal computation. Default 16.
    """

    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 50,
        internal_batch_size: int = 16,
    ) -> None:
        self.model = model
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size
        self._ig = None

        try:
            from captum.attr import IntegratedGradients

            self._ig = IntegratedGradients(model)
            logger.info(
                "IntegratedGradientsGenerator initialized (n_steps=%d)",
                n_steps,
            )
        except ImportError:
            logger.warning(
                "Captum not available. Integrated Gradients will use "
                "a simplified manual implementation."
            )

    def generate(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
        baseline: torch.Tensor | None = None,
    ) -> np.ndarray:
        """
        Generate an Integrated Gradients attribution map.

        Parameters
        ----------
        image : torch.Tensor
            Input image of shape (1, C, H, W) or (C, H, W).
        target_class : int, optional
            Target class. If None, uses predicted class.
        baseline : torch.Tensor, optional
            Baseline input (default: zeros / black image).

        Returns
        -------
        np.ndarray
            Attribution map of shape (H, W), values indicate pixel importance.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        self.model.eval()

        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = int(output.argmax(dim=1).item())

        if baseline is None:
            baseline = torch.zeros_like(image)

        if self._ig is not None:
            return self._generate_captum(image, target_class, baseline)
        else:
            return self._generate_manual(image, target_class, baseline)

    def _generate_captum(
        self,
        image: torch.Tensor,
        target_class: int,
        baseline: torch.Tensor,
    ) -> np.ndarray:
        """Generate attributions using Captum's IntegratedGradients."""
        attribution = self._ig.attribute(
            image,
            baseline=baseline,
            target=target_class,
            n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size,
        )

        # Sum across channels to get per-pixel attribution
        attr_map = attribution.squeeze().detach().cpu().numpy()
        if attr_map.ndim == 3:
            attr_map = np.abs(attr_map).sum(axis=0)

        # Normalize to [0, 1]
        amax = attr_map.max()
        if amax > 0:
            attr_map = attr_map / amax

        return attr_map

    def _generate_manual(
        self,
        image: torch.Tensor,
        target_class: int,
        baseline: torch.Tensor,
    ) -> np.ndarray:
        """Simplified manual Integrated Gradients implementation."""
        image.requires_grad_(True)

        # Create interpolated inputs
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=image.device)
        scaled_inputs = torch.cat(
            [baseline + alpha * (image - baseline) for alpha in alphas]
        )

        # Compute gradients in batches
        all_grads = []
        for i in range(0, len(scaled_inputs), self.internal_batch_size):
            batch = scaled_inputs[
                i : i + self.internal_batch_size
            ].detach().requires_grad_(True)
            output = self.model(batch)
            target_scores = output[:, target_class].sum()
            target_scores.backward()
            all_grads.append(batch.grad.detach())

        grads = torch.cat(all_grads)

        # Approximate integral via trapezoidal rule
        avg_grads = (grads[:-1] + grads[1:]).mean(dim=0) / 2
        ig = ((image - baseline) * avg_grads).squeeze()

        attr_map = ig.detach().cpu().numpy()
        if attr_map.ndim == 3:
            attr_map = np.abs(attr_map).sum(axis=0)

        amax = attr_map.max()
        if amax > 0:
            attr_map = attr_map / amax

        return attr_map
