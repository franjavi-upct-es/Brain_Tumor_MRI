# src/models/__init__.py
"""
Classification models for brain tumor grading. Includes a Lightning classifier
module, backbone factory (DenseNet-121, ResNet-50, EfficientNet-B0),
classification heads with dropout, and a classical ML baseline (SVM).
"""


def __getattr__(name: str):
    """Lazy imports for heavy-dependency modules."""
    if name == "TumorClassifier":
        from src.models.classifier import TumorClassifier

        return TumorClassifier
    if name == "build_backbone":
        from src.models.backbones import build_backbone

        return build_backbone
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TumorClassifier", "build_backbone"]
