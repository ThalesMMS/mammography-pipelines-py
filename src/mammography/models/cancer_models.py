#
# cancer_models.py
# mammography-pipelines
#
# Defines MammographyModel classifier for RSNA Breast Cancer Detection.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Neural network models for breast cancer detection in mammography images.

WARNING: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

This module provides ResNet50-based architectures adapted for single-channel
mammography images. The models are designed for binary cancer classification
and include device detection utilities for optimal hardware utilization.

Components:
    - MammographyModel: Complete classifier with ResNet50 backbone
    - ViewSpecificModel: Wrapper for managing separate models per view (CC/MLO)
    - EnsemblePredictor: Combines predictions from multiple view-specific models
    - build_resnet50_classifier: Factory function for custom ResNet50 models
    - resolve_device: Utility to get optimal PyTorch device (CUDA/MPS/CPU)

Example usage:
    >>> from mammography.models.cancer_models import MammographyModel, resolve_device
    >>> import torch
    >>>
    >>> # Get optimal device
    >>> device = resolve_device()
    >>>
    >>> # Create model and move to device
    >>> model = MammographyModel()
    >>> model = model.to(device)
    >>>
    >>> # Forward pass with single-channel image (batch_size=2, channels=1, h=224, w=224)
    >>> x = torch.randn(2, 1, 224, 224).to(device)
    >>> predictions = model(x)  # Returns probabilities in range [0, 1]
    >>>
    >>> # Alternative: Build custom classifier
    >>> from mammography.models.cancer_models import build_resnet50_classifier
    >>> custom_model = build_resnet50_classifier(num_classes=1, pretrained=True)
"""

import warnings

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights, resnet50

from mammography.utils.device_detection import get_optimal_device


def resolve_device() -> torch.device:
    """Return the optimal PyTorch device for model operations.

    Returns:
        torch.device: The preferred device (CUDA, MPS, or CPU) for the running environment
    """
    device_str = get_optimal_device()
    return torch.device(device_str)


def _load_with_fallback(factory, weights, arch_name: str) -> nn.Module:
    """Load a torchvision model, falling back to random weights if pretrained download fails."""
    if weights is None:
        return factory(weights=None)
    try:
        return factory(weights=weights)
    except Exception as exc:
        warnings.warn(
            f"Failed to load pretrained weights for {arch_name}; using random init. Error: {exc}",
            RuntimeWarning,
        )
        return factory(weights=None)


def build_resnet50_classifier(num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    """Build ResNet50 adapted for single-channel mammography images.

    Args:
        num_classes: Number of output classes (default 1 for binary classification)
        pretrained: Whether to use ImageNet pretrained weights (default True)

    Returns:
        ResNet50 model with modified conv1 for single-channel input and custom fc layer
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = _load_with_fallback(resnet50, weights, "resnet50")

    # Modify first conv layer for single-channel grayscale input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Replace final fc layer with custom number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


class MammographyModel(nn.Module):
    """Classifier based solely on a ResNet50 adapted for a single channel."""

    def __init__(self):
        """Initialize ResNet50 tailored for grayscale mammography images."""
        super().__init__()

        self.rnet = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.rnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.rnet.fc = nn.Linear(self.rnet.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Process the image and return the estimated cancer probability."""

        logits = self.rnet(img)
        out = self.sigmoid(logits)
        return out


class ViewSpecificModel:
    """Wrapper managing separate model instances for different mammography views.

    This class enables view-specific training by maintaining independent models
    for each view type (e.g., CC - craniocaudal, MLO - mediolateral oblique).
    Each view can have its own trained model, allowing the system to learn
    view-specific patterns and potentially improve overall accuracy.

    Args:
        views: List of view identifiers (e.g., ['CC', 'MLO'])

    Example:
        >>> from mammography.models.cancer_models import ViewSpecificModel
        >>> model = ViewSpecificModel(views=['CC', 'MLO'])
        >>> cc_model = model.get_model('CC')
        >>> mlo_model = model.get_model('MLO')
    """

    def __init__(self, views: list):
        """Initialize view-specific model wrapper.

        Args:
            views: List of view identifiers for which to create separate models
        """
        if not views:
            raise ValueError("At least one view must be specified")

        self.views = views
        self.models = {view: None for view in views}

    def get_model(self, view: str) -> nn.Module:
        """Get the model instance for a specific view.

        Args:
            view: View identifier (e.g., 'CC', 'MLO')

        Returns:
            Model instance for the specified view, or None if not yet created

        Raises:
            KeyError: If the view is not in the configured views list
        """
        if view not in self.models:
            raise KeyError(f"View '{view}' not found. Available views: {self.views}")
        return self.models[view]

    def set_model(self, view: str, model: nn.Module) -> None:
        """Set the model instance for a specific view.

        Args:
            view: View identifier (e.g., 'CC', 'MLO')
            model: PyTorch model instance to associate with this view

        Raises:
            KeyError: If the view is not in the configured views list
            ValueError: If model is None
        """
        if view not in self.models:
            raise KeyError(f"View '{view}' not found. Available views: {self.views}")
        if model is None:
            raise ValueError(f"Cannot set None as model for view '{view}'")
        self.models[view] = model

    def predict(self, x: torch.Tensor, view: str) -> torch.Tensor:
        """Generate predictions for a specific view.

        Args:
            x: Input tensor (batch of images)
            view: View identifier for which to generate predictions

        Returns:
            Model predictions for the input

        Raises:
            KeyError: If the view is not in the configured views list
            RuntimeError: If the model for the specified view has not been created
        """
        if view not in self.models:
            raise KeyError(f"View '{view}' not found. Available views: {self.views}")

        model = self.models[view]
        if model is None:
            raise RuntimeError(f"Model for view '{view}' has not been initialized")

        return model(x)


class EnsemblePredictor:
    """Combines predictions from multiple view-specific models using ensemble methods.

    This class aggregates predictions from different mammography views to produce
    a final cancer probability score. It supports various ensemble methods including
    simple averaging, weighted averaging, and maximum pooling.

    Args:
        models: Dictionary mapping view names to model instances (e.g., {'CC': model1, 'MLO': model2})
        method: Ensemble method to use ('average', 'weighted', 'max'). Default is 'average'
        weights: Optional dictionary of weights per view for 'weighted' method

    Example:
        >>> from mammography.models.cancer_models import EnsemblePredictor, MammographyModel
        >>> import torch
        >>>
        >>> # Create models for different views
        >>> cc_model = MammographyModel()
        >>> mlo_model = MammographyModel()
        >>>
        >>> # Create ensemble predictor
        >>> ensemble = EnsemblePredictor(
        ...     models={'CC': cc_model, 'MLO': mlo_model},
        ...     method='average'
        ... )
        >>>
        >>> # Generate predictions
        >>> predictions = {'CC': torch.tensor([0.7]), 'MLO': torch.tensor([0.6])}
        >>> result = ensemble.predict(predictions)  # Returns average: 0.65
    """

    def __init__(self, models: dict, method: str = "average", weights: dict = None):
        """Initialize ensemble predictor with models and combination method.

        Args:
            models: Dictionary mapping view names to model instances
            method: Ensemble method ('average', 'weighted', 'max')
            weights: Optional weights for 'weighted' method (must sum to 1.0)

        Raises:
            ValueError: If method is unsupported or weights are invalid
        """
        if not models:
            raise ValueError("At least one model must be provided")

        supported_methods = ["average", "weighted", "max"]
        if method not in supported_methods:
            raise ValueError(f"Method '{method}' not supported. Choose from {supported_methods}")

        self.models = models
        self.method = method
        self.weights = weights

        # Validate weights if using weighted method
        if method == "weighted":
            if weights is None:
                raise ValueError("Weights must be provided for 'weighted' method")
            if set(weights.keys()) != set(models.keys()):
                raise ValueError("Weight keys must match model keys")
            weight_sum = sum(weights.values())
            if not (0.999 <= weight_sum <= 1.001):  # Tighter tolerance for floating point errors
                raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    def predict(self, view_predictions: dict) -> torch.Tensor:
        """Combine predictions from multiple views using the configured ensemble method.

        Args:
            view_predictions: Dictionary mapping view names to prediction tensors

        Returns:
            Combined prediction tensor

        Raises:
            ValueError: If view_predictions keys don't match model keys
        """
        if set(view_predictions.keys()) != set(self.models.keys()):
            raise ValueError(
                f"Prediction views {list(view_predictions.keys())} "
                f"don't match model views {list(self.models.keys())}"
            )

        predictions_list = [view_predictions[view] for view in sorted(self.models.keys())]

        if self.method == "average":
            # Simple averaging of all predictions
            stacked = torch.stack(predictions_list)
            return torch.mean(stacked, dim=0)

        elif self.method == "weighted":
            # Weighted average using provided weights
            weighted_sum = None
            for view in sorted(self.models.keys()):
                weight = self.weights[view]
                pred = view_predictions[view]

                if weighted_sum is None:
                    weighted_sum = weight * pred
                else:
                    weighted_sum = weighted_sum + (weight * pred)

            return weighted_sum

        elif self.method == "max":
            # Take maximum prediction across views
            stacked = torch.stack(predictions_list)
            return torch.max(stacked, dim=0)[0]

        else:
            raise ValueError(f"Unsupported method: {self.method}")
