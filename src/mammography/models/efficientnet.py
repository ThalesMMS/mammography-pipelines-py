"""Backward-compatible EfficientNet model builders."""

from __future__ import annotations

import timm


def build_efficientnet_model(num_classes: int = 4, pretrained: bool = True, **kwargs):
    """Build an EfficientNet-B0 classifier using the legacy helper path."""
    try:
        return timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs,
        )
    except RuntimeError as exc:
        if pretrained:
            raise RuntimeError(f"Failed to download pretrained EfficientNet weights: {exc}") from exc
        raise
