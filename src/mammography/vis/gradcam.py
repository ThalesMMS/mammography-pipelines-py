#
# gradcam.py
# mammography-pipelines
#
# Wrapper module for GradCAM functionality to maintain backward compatibility with tests.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""
GradCAM wrapper module for backward compatibility.

This module provides a simplified interface to the GradCAMExplainer class
from mammography.vis.explainability.
"""

import torch
import torch.nn as nn
from typing import Optional, Union
import numpy as np

from mammography.vis.explainability import GradCAMExplainer


def apply_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> np.ndarray:
    """
    Apply Grad-CAM to generate activation heatmap for model interpretation.

    This is a convenience wrapper around GradCAMExplainer for backward compatibility.

    Args:
        model: The neural network model to explain
        input_tensor: Input image tensor (C, H, W) or (B, C, H, W)
        target_class: Target class index for visualization (None = predicted class)
        target_layer: Layer to hook for activations (None = auto-detect)
        device: Device to run on (None = auto-detect)

    Returns:
        Heatmap as numpy array (H, W) with values in [0, 1]

    Example:
        >>> model = build_model("resnet50", num_classes=4)
        >>> image = torch.randn(3, 224, 224)
        >>> heatmap = apply_gradcam(model, image, target_class=1)
    """
    explainer = GradCAMExplainer(model, target_layer=target_layer, device=device)

    # Ensure input has batch dimension
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)

    heatmap = explainer.generate_heatmap(input_tensor, target_class=target_class)
    return heatmap
