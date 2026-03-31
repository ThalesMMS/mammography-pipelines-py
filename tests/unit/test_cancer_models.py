from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from mammography.models import cancer_models

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


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


# ---------------------------------------------------------------------------
# ViewSpecificModel tests
# ---------------------------------------------------------------------------

class _SimpleModel(torch.nn.Module):
    """Minimal deterministic model for fast view-specific tests."""

    def __init__(self, out_value: float = 0.5) -> None:
        """
        Initialize the simple deterministic model with a constant scalar output.

        Parameters:
            out_value (float): The scalar value each forward call will output for every sample (defaults to 0.5).
        """
        super().__init__()
        self.out_value = out_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce a batch-sized tensor filled with the model's constant output value.

        Parameters:
            x (torch.Tensor): Input tensor whose first dimension determines the batch size.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1) where every element equals the model's configured `out_value`.
        """
        return torch.full((x.shape[0], 1), self.out_value)


def test_view_specific_model_initialization() -> None:
    """Test ViewSpecificModel initializes with correct views."""
    vsm = cancer_models.ViewSpecificModel(views=["CC", "MLO"])

    assert vsm.views == ["CC", "MLO"]
    assert set(vsm.models.keys()) == {"CC", "MLO"}
    # Models should be None until explicitly set
    assert vsm.models["CC"] is None
    assert vsm.models["MLO"] is None


def test_view_specific_model_empty_views_raises() -> None:
    """Test that ViewSpecificModel raises ValueError for empty views list."""
    with pytest.raises(ValueError, match="At least one view must be specified"):
        cancer_models.ViewSpecificModel(views=[])


def test_view_specific_model_single_view() -> None:
    """Test ViewSpecificModel with a single view."""
    vsm = cancer_models.ViewSpecificModel(views=["CC"])

    assert vsm.views == ["CC"]
    assert list(vsm.models.keys()) == ["CC"]
    assert vsm.models["CC"] is None


def test_view_specific_model_get_model_returns_none_initially() -> None:
    """Test that get_model returns None before a model is set."""
    vsm = cancer_models.ViewSpecificModel(views=["CC", "MLO"])

    assert vsm.get_model("CC") is None
    assert vsm.get_model("MLO") is None


def test_view_specific_model_get_model_invalid_view_raises() -> None:
    """Test that get_model raises KeyError for an unregistered view."""
    vsm = cancer_models.ViewSpecificModel(views=["CC", "MLO"])

    with pytest.raises(KeyError, match="RCC"):
        vsm.get_model("RCC")


def test_view_specific_model_set_and_get_model() -> None:
    """Test that set_model stores a model and get_model retrieves it."""
    vsm = cancer_models.ViewSpecificModel(views=["CC", "MLO"])
    cc_model = _SimpleModel(out_value=0.3)

    vsm.set_model("CC", cc_model)

    retrieved = vsm.get_model("CC")
    assert retrieved is cc_model
    # MLO should still be None
    assert vsm.get_model("MLO") is None


def test_view_specific_model_set_model_invalid_view_raises() -> None:
    """Test that set_model raises KeyError for an unregistered view."""
    vsm = cancer_models.ViewSpecificModel(views=["CC", "MLO"])

    with pytest.raises(KeyError, match="RCC"):
        vsm.set_model("RCC", _SimpleModel())


def test_view_specific_model_set_model_none_raises() -> None:
    """Test that set_model raises ValueError when None is passed as model."""
    vsm = cancer_models.ViewSpecificModel(views=["CC", "MLO"])

    with pytest.raises(ValueError, match="Cannot set None"):
        vsm.set_model("CC", None)  # type: ignore[arg-type]


def test_view_specific_model_predict_returns_correct_output() -> None:
    """Test that predict calls the view model and returns its output."""
    vsm = cancer_models.ViewSpecificModel(views=["CC", "MLO"])
    vsm.set_model("CC", _SimpleModel(out_value=0.7))
    vsm.set_model("MLO", _SimpleModel(out_value=0.4))

    x = torch.randn(3, 1, 32, 32)

    cc_out = vsm.predict(x, "CC")
    mlo_out = vsm.predict(x, "MLO")

    assert cc_out.shape == (3, 1)
    assert torch.allclose(cc_out, torch.full((3, 1), 0.7))

    assert mlo_out.shape == (3, 1)
    assert torch.allclose(mlo_out, torch.full((3, 1), 0.4))


def test_view_specific_model_predict_uninitialized_model_raises() -> None:
    """Test that predict raises RuntimeError when the view model is not set."""
    vsm = cancer_models.ViewSpecificModel(views=["CC", "MLO"])
    x = torch.randn(2, 1, 32, 32)

    with pytest.raises(RuntimeError, match="has not been initialized"):
        vsm.predict(x, "CC")


def test_view_specific_model_predict_invalid_view_raises() -> None:
    """Test that predict raises KeyError for an unregistered view."""
    vsm = cancer_models.ViewSpecificModel(views=["CC", "MLO"])
    x = torch.randn(2, 1, 32, 32)

    with pytest.raises(KeyError, match="RCC"):
        vsm.predict(x, "RCC")


def test_view_specific_model_independent_models() -> None:
    """Test that each view holds an independent model instance."""
    vsm = cancer_models.ViewSpecificModel(views=["CC", "MLO"])
    cc_model = _SimpleModel(out_value=0.2)
    mlo_model = _SimpleModel(out_value=0.8)

    vsm.set_model("CC", cc_model)
    vsm.set_model("MLO", mlo_model)

    # Ensure they are distinct objects
    assert vsm.get_model("CC") is not vsm.get_model("MLO")
    assert vsm.get_model("CC") is cc_model
    assert vsm.get_model("MLO") is mlo_model


def test_view_specific_model_replace_model() -> None:
    """Test that a view model can be replaced by calling set_model again."""
    vsm = cancer_models.ViewSpecificModel(views=["CC"])
    first_model = _SimpleModel(out_value=0.1)
    second_model = _SimpleModel(out_value=0.9)

    vsm.set_model("CC", first_model)
    assert vsm.get_model("CC") is first_model

    vsm.set_model("CC", second_model)
    assert vsm.get_model("CC") is second_model


# ---------------------------------------------------------------------------
# EnsemblePredictor tests
# ---------------------------------------------------------------------------


def test_ensemble_predictor_initialization_average() -> None:
    """Test EnsemblePredictor initialization with average method."""
    cc_model = _SimpleModel(out_value=0.6)
    mlo_model = _SimpleModel(out_value=0.4)
    models = {"CC": cc_model, "MLO": mlo_model}

    ensemble = cancer_models.EnsemblePredictor(models=models, method="average")

    assert ensemble.models == models
    assert ensemble.method == "average"
    assert ensemble.weights is None


def test_ensemble_predictor_initialization_weighted() -> None:
    """Test EnsemblePredictor initialization with weighted method."""
    cc_model = _SimpleModel(out_value=0.6)
    mlo_model = _SimpleModel(out_value=0.4)
    models = {"CC": cc_model, "MLO": mlo_model}
    weights = {"CC": 0.6, "MLO": 0.4}

    ensemble = cancer_models.EnsemblePredictor(
        models=models, method="weighted", weights=weights
    )

    assert ensemble.models == models
    assert ensemble.method == "weighted"
    assert ensemble.weights == weights


def test_ensemble_predictor_initialization_max() -> None:
    """Test EnsemblePredictor initialization with max method."""
    cc_model = _SimpleModel(out_value=0.7)
    mlo_model = _SimpleModel(out_value=0.3)
    models = {"CC": cc_model, "MLO": mlo_model}

    ensemble = cancer_models.EnsemblePredictor(models=models, method="max")

    assert ensemble.models == models
    assert ensemble.method == "max"
    assert ensemble.weights is None


def test_ensemble_predictor_empty_models_raises() -> None:
    """Test that EnsemblePredictor raises ValueError for empty models dict."""
    with pytest.raises(ValueError, match="At least one model must be provided"):
        cancer_models.EnsemblePredictor(models={}, method="average")


def test_ensemble_predictor_invalid_method_raises() -> None:
    """Test that EnsemblePredictor raises ValueError for unsupported method."""
    models = {"CC": _SimpleModel()}

    with pytest.raises(ValueError, match="Method 'invalid' not supported"):
        cancer_models.EnsemblePredictor(models=models, method="invalid")


def test_ensemble_predictor_weighted_without_weights_raises() -> None:
    """Test that weighted method raises ValueError when weights not provided."""
    models = {"CC": _SimpleModel(), "MLO": _SimpleModel()}

    with pytest.raises(ValueError, match="Weights must be provided"):
        cancer_models.EnsemblePredictor(models=models, method="weighted")


def test_ensemble_predictor_weighted_mismatched_keys_raises() -> None:
    """Test that weighted method raises ValueError when weight keys don't match models."""
    models = {"CC": _SimpleModel(), "MLO": _SimpleModel()}
    weights = {"CC": 0.5, "RCC": 0.5}  # RCC doesn't match MLO

    with pytest.raises(ValueError, match="Weight keys must match model keys"):
        cancer_models.EnsemblePredictor(
            models=models, method="weighted", weights=weights
        )


def test_ensemble_predictor_weighted_invalid_sum_raises() -> None:
    """Test that weighted method raises ValueError when weights don't sum to 1.0."""
    models = {"CC": _SimpleModel(), "MLO": _SimpleModel()}
    weights = {"CC": 0.3, "MLO": 0.3}  # Sum is 0.6, not 1.0

    with pytest.raises(ValueError, match=r"Weights must sum to 1\.0"):
        cancer_models.EnsemblePredictor(
            models=models, method="weighted", weights=weights
        )


def test_ensemble_predictor_predict_average() -> None:
    """Test EnsemblePredictor predict with average method."""
    cc_model = _SimpleModel(out_value=0.6)
    mlo_model = _SimpleModel(out_value=0.4)
    models = {"CC": cc_model, "MLO": mlo_model}

    ensemble = cancer_models.EnsemblePredictor(models=models, method="average")

    # Create mock predictions
    view_predictions = {
        "CC": torch.tensor([[0.6]]),
        "MLO": torch.tensor([[0.4]]),
    }

    result = ensemble.predict(view_predictions)

    # Average of 0.6 and 0.4 should be 0.5
    assert result.shape == (1, 1)
    assert torch.allclose(result, torch.tensor([[0.5]]))


def test_ensemble_predictor_predict_weighted() -> None:
    """Test EnsemblePredictor predict with weighted method."""
    cc_model = _SimpleModel(out_value=0.8)
    mlo_model = _SimpleModel(out_value=0.2)
    models = {"CC": cc_model, "MLO": mlo_model}
    weights = {"CC": 0.7, "MLO": 0.3}

    ensemble = cancer_models.EnsemblePredictor(
        models=models, method="weighted", weights=weights
    )

    # Create mock predictions
    view_predictions = {
        "CC": torch.tensor([[0.8]]),
        "MLO": torch.tensor([[0.2]]),
    }

    result = ensemble.predict(view_predictions)

    # Weighted average: 0.7 * 0.8 + 0.3 * 0.2 = 0.56 + 0.06 = 0.62
    assert result.shape == (1, 1)
    assert torch.allclose(result, torch.tensor([[0.62]]))


def test_ensemble_predictor_predict_max() -> None:
    """Test EnsemblePredictor predict with max method."""
    cc_model = _SimpleModel(out_value=0.7)
    mlo_model = _SimpleModel(out_value=0.3)
    models = {"CC": cc_model, "MLO": mlo_model}

    ensemble = cancer_models.EnsemblePredictor(models=models, method="max")

    # Create mock predictions
    view_predictions = {
        "CC": torch.tensor([[0.7]]),
        "MLO": torch.tensor([[0.3]]),
    }

    result = ensemble.predict(view_predictions)

    # Max of 0.7 and 0.3 should be 0.7
    assert result.shape == (1, 1)
    assert torch.allclose(result, torch.tensor([[0.7]]))


def test_ensemble_predictor_predict_mismatched_views_raises() -> None:
    """Test that predict raises ValueError when prediction views don't match models."""
    models = {"CC": _SimpleModel(), "MLO": _SimpleModel()}
    ensemble = cancer_models.EnsemblePredictor(models=models, method="average")

    # Provide predictions for different views
    view_predictions = {
        "CC": torch.tensor([[0.5]]),
        "RCC": torch.tensor([[0.5]]),  # RCC instead of MLO
    }

    with pytest.raises(ValueError, match="don't match model views"):
        ensemble.predict(view_predictions)


def test_ensemble_predictor_predict_batch() -> None:
    """Test EnsemblePredictor predict with batch of predictions."""
    cc_model = _SimpleModel(out_value=0.8)
    mlo_model = _SimpleModel(out_value=0.6)
    models = {"CC": cc_model, "MLO": mlo_model}

    ensemble = cancer_models.EnsemblePredictor(models=models, method="average")

    # Create batch predictions (batch_size=3)
    view_predictions = {
        "CC": torch.tensor([[0.9], [0.7], [0.8]]),
        "MLO": torch.tensor([[0.5], [0.3], [0.6]]),
    }

    result = ensemble.predict(view_predictions)

    # Expected averages: (0.9+0.5)/2=0.7, (0.7+0.3)/2=0.5, (0.8+0.6)/2=0.7
    expected = torch.tensor([[0.7], [0.5], [0.7]])
    assert result.shape == (3, 1)
    assert torch.allclose(result, expected)


def test_ensemble_predictor_single_model() -> None:
    """Test EnsemblePredictor with a single model."""
    cc_model = _SimpleModel(out_value=0.75)
    models = {"CC": cc_model}

    ensemble = cancer_models.EnsemblePredictor(models=models, method="average")

    view_predictions = {"CC": torch.tensor([[0.75]])}

    result = ensemble.predict(view_predictions)

    # With single model, average should be the same as input
    assert result.shape == (1, 1)
    assert torch.allclose(result, torch.tensor([[0.75]]))


def test_ensemble_predictor_three_views_average() -> None:
    """Test EnsemblePredictor with three views using average method."""
    models = {
        "CC": _SimpleModel(out_value=0.6),
        "MLO": _SimpleModel(out_value=0.4),
        "LM": _SimpleModel(out_value=0.5),
    }

    ensemble = cancer_models.EnsemblePredictor(models=models, method="average")

    view_predictions = {
        "CC": torch.tensor([[0.6]]),
        "MLO": torch.tensor([[0.4]]),
        "LM": torch.tensor([[0.5]]),
    }

    result = ensemble.predict(view_predictions)

    # Average of 0.6, 0.4, 0.5 should be 0.5
    assert result.shape == (1, 1)
    assert torch.allclose(result, torch.tensor([[0.5]]))


def test_ensemble_predictor_three_views_weighted() -> None:
    """Test EnsemblePredictor with three views using weighted method."""
    models = {
        "CC": _SimpleModel(out_value=0.6),
        "MLO": _SimpleModel(out_value=0.4),
        "LM": _SimpleModel(out_value=0.2),
    }
    weights = {"CC": 0.5, "MLO": 0.3, "LM": 0.2}

    ensemble = cancer_models.EnsemblePredictor(
        models=models, method="weighted", weights=weights
    )

    view_predictions = {
        "CC": torch.tensor([[0.6]]),
        "MLO": torch.tensor([[0.4]]),
        "LM": torch.tensor([[0.2]]),
    }

    result = ensemble.predict(view_predictions)

    # Weighted average: 0.5*0.6 + 0.3*0.4 + 0.2*0.2 = 0.3 + 0.12 + 0.04 = 0.46
    expected = torch.tensor([[0.46]])
    assert result.shape == (1, 1)
    assert torch.allclose(result, expected)
