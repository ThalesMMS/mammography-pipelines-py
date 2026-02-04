from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from mammography.models import cancer_models


def test_resolve_device() -> None:
    """Test that resolve_device returns a valid torch device."""
    device = cancer_models.resolve_device()
    assert isinstance(device, torch.device)
    # Device should be one of: cuda, mps, or cpu
    assert device.type in ["cuda", "mps", "cpu"]


def test_build_resnet50_classifier_without_pretrained(monkeypatch) -> None:
    """Test building ResNet50 classifier without downloading pretrained weights."""
    monkeypatch.setattr(
        cancer_models,
        "resnet50",
        lambda weights=None: torchvision.models.resnet50(weights=None),
    )

    model = cancer_models.build_resnet50_classifier(num_classes=1, pretrained=False)

    # Test input/output shape for single-channel input and binary classification
    x = torch.randn(2, 1, 224, 224)  # Batch of 2, single-channel images
    y = model(x)
    assert y.shape == (2, 1)

    # Verify modified conv1 accepts single-channel input
    assert model.conv1.in_channels == 1
    assert model.conv1.out_channels == 64

    # Verify fc layer has correct output size
    assert model.fc.out_features == 1


def test_build_resnet50_classifier_multiclass(monkeypatch) -> None:
    """Test building ResNet50 classifier with multiple classes."""
    monkeypatch.setattr(
        cancer_models,
        "resnet50",
        lambda weights=None: torchvision.models.resnet50(weights=None),
    )

    model = cancer_models.build_resnet50_classifier(num_classes=4, pretrained=False)

    # Test input/output shape for multiclass
    x = torch.randn(2, 1, 224, 224)
    y = model(x)
    assert y.shape == (2, 4)

    # Verify fc layer has correct output size
    assert model.fc.out_features == 4


def test_build_resnet50_classifier_with_pretrained(monkeypatch) -> None:
    """Test building ResNet50 classifier with pretrained weights (mocked)."""
    monkeypatch.setattr(
        cancer_models,
        "resnet50",
        lambda weights=None: torchvision.models.resnet50(weights=None),
    )

    model = cancer_models.build_resnet50_classifier(num_classes=2, pretrained=True)

    # Test forward pass
    x = torch.randn(2, 1, 224, 224)
    y = model(x)
    assert y.shape == (2, 2)


def test_load_with_fallback_success(monkeypatch) -> None:
    """Test that _load_with_fallback works when loading succeeds."""
    def mock_factory(weights=None):
        return torchvision.models.resnet50(weights=None)

    model = cancer_models._load_with_fallback(
        mock_factory,
        None,  # No weights
        "test_model"
    )
    assert model is not None


def test_load_with_fallback_failure(monkeypatch) -> None:
    """Test that _load_with_fallback falls back to random weights on error."""
    call_count = [0]

    def mock_factory(weights=None):
        call_count[0] += 1
        if weights is not None and call_count[0] == 1:
            raise RuntimeError("Network error")
        return torchvision.models.resnet50(weights=None)

    with pytest.warns(RuntimeWarning, match="Failed to load pretrained weights"):
        model = cancer_models._load_with_fallback(
            mock_factory,
            "IMAGENET1K_V2",  # Simulated weights
            "resnet50"
        )

    assert model is not None
    assert call_count[0] == 2  # Should be called twice: fail, then succeed


def test_mammography_model_initialization(monkeypatch) -> None:
    """Test MammographyModel initialization without downloading weights."""
    # Save original function to avoid recursion
    original_resnet50 = torchvision.models.resnet50

    monkeypatch.setattr(
        cancer_models.torchvision.models,
        "resnet50",
        lambda weights=None: original_resnet50(weights=None),
    )

    model = cancer_models.MammographyModel()

    # Check architecture modifications
    assert model.rnet.conv1.in_channels == 1  # Single-channel input
    assert model.rnet.fc.out_features == 1    # Binary classification
    assert hasattr(model, "sigmoid")


def test_mammography_model_forward(monkeypatch) -> None:
    """Test MammographyModel forward pass."""
    # Save original function to avoid recursion
    original_resnet50 = torchvision.models.resnet50

    monkeypatch.setattr(
        cancer_models.torchvision.models,
        "resnet50",
        lambda weights=None: original_resnet50(weights=None),
    )

    model = cancer_models.MammographyModel()
    model.eval()

    # Test forward pass
    x = torch.randn(2, 1, 224, 224)  # Single-channel images
    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 1)

    # Output should be between 0 and 1 due to sigmoid
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)


def test_mammography_model_gradients(monkeypatch) -> None:
    """Test that MammographyModel computes gradients correctly."""
    # Save original function to avoid recursion
    original_resnet50 = torchvision.models.resnet50

    monkeypatch.setattr(
        cancer_models.torchvision.models,
        "resnet50",
        lambda weights=None: original_resnet50(weights=None),
    )

    model = cancer_models.MammographyModel()
    model.train()

    # Test gradient computation
    x = torch.randn(2, 1, 224, 224, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Check that gradients are computed
    assert x.grad is not None
    assert model.rnet.fc.weight.grad is not None


def test_mammography_model_device_compatibility() -> None:
    """Test that MammographyModel can be moved to different devices."""
    model = cancer_models.MammographyModel()

    # Test CPU
    device = torch.device("cpu")
    model = model.to(device)
    x = torch.randn(1, 1, 224, 224, device=device)
    y = model(x)
    assert y.device == device
