# src/interpretability/__init__.py — Model interpretability subpackage
"""
Interpretability tools for validating that models learn clinically meaningful
features. Includes Grad-CAM and Integrated Gradients via Captum, attention
validation against ground truth segmentations (IoU metric), and
publication-ready visualization generation at 300 DPI.
"""


def __getattr__(name: str):
    """Lazy imports for interpretability modules."""
    if name == "GradCAMGenerator":
        from src.interpretability.gradcam import GradCAMGenerator

        return GradCAMGenerator
    if name == "validate_attention":
        from src.interpretability.attention_validator import validate_attention

        return validate_attention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
