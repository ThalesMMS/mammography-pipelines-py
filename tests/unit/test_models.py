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

from mammography.models import nets

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def test_build_resnet50_without_download(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "resnet50",
        lambda *_args, **_kwargs: torchvision.models.resnet50(weights=None),
    )

    model = nets.build_model("resnet50", num_classes=4, train_backbone=False, unfreeze_last_block=True)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 4)
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_build_efficientnet_without_download(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "efficientnet_b0",
        lambda *_args, **_kwargs: torchvision.models.efficientnet_b0(weights=None),
    )

    model = nets.build_model("efficientnet_b0", num_classes=2, train_backbone=False, unfreeze_last_block=True)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 2)
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_build_efficientnet_with_fusion_without_download(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "efficientnet_b0",
        lambda *_args, **_kwargs: torchvision.models.efficientnet_b0(weights=None),
    )

    model = nets.build_model("efficientnet_b0", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8)
    x = torch.randn(2, 3, 224, 224)
    extra_features = torch.randn(2, 8)
    y = model(x, extra_features)
    assert y.shape == (2, 4)
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_efficientnet_fusion_gradient_flow(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "efficientnet_b0",
        lambda *_args, **_kwargs: torchvision.models.efficientnet_b0(weights=None),
    )

    model = nets.build_model("efficientnet_b0", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8)
    model.train()

    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    extra_features = torch.randn(2, 8, requires_grad=True)

    y = model(x, extra_features)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert extra_features.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.classifier.parameters() if p.requires_grad)


def test_efficientnet_fusion_parameter_freezing(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "efficientnet_b0",
        lambda *_args, **_kwargs: torchvision.models.efficientnet_b0(weights=None),
    )

    # Test with unfreeze_last_block=False: all backbone should be frozen
    model = nets.build_model("efficientnet_b0", num_classes=4, train_backbone=False, unfreeze_last_block=False, extra_feature_dim=8)

    # Check that all features blocks are frozen
    num_blocks = len(model.features)
    for i in range(num_blocks):
        assert not any(p.requires_grad for p in model.features[i].parameters())

    # Check that classifier is trainable
    assert all(p.requires_grad for p in model.classifier.parameters())

    # Test with unfreeze_last_block=True: only last 2 blocks should be trainable
    model = nets.build_model("efficientnet_b0", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8)

    num_blocks = len(model.features)
    # Check that early blocks are frozen
    for i in range(max(0, num_blocks - 2)):
        assert not any(p.requires_grad for p in model.features[i].parameters())

    # Check that last 2 blocks are trainable
    for i in range(max(0, num_blocks - 2), num_blocks):
        assert all(p.requires_grad for p in model.features[i].parameters())

    # Check that classifier is trainable
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_build_resnet50_with_fusion_without_download(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "resnet50",
        lambda *_args, **_kwargs: torchvision.models.resnet50(weights=None),
    )

    model = nets.build_model("resnet50", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8)
    x = torch.randn(2, 3, 224, 224)
    extra_features = torch.randn(2, 8)
    y = model(x, extra_features)
    assert y.shape == (2, 4)
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_resnet50_fusion_gradient_flow(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "resnet50",
        lambda *_args, **_kwargs: torchvision.models.resnet50(weights=None),
    )

    model = nets.build_model("resnet50", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8)
    model.train()

    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    extra_features = torch.randn(2, 8, requires_grad=True)

    y = model(x, extra_features)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert extra_features.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.classifier.parameters() if p.requires_grad)


def test_resnet50_fusion_parameter_freezing(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "resnet50",
        lambda *_args, **_kwargs: torchvision.models.resnet50(weights=None),
    )

    # Test with unfreeze_last_block=False: all backbone should be frozen
    model = nets.build_model("resnet50", num_classes=4, train_backbone=False, unfreeze_last_block=False, extra_feature_dim=8)

    # Check that early layers are frozen
    assert not any(p.requires_grad for p in model.backbone.layer1.parameters())
    assert not any(p.requires_grad for p in model.backbone.layer2.parameters())
    assert not any(p.requires_grad for p in model.backbone.layer3.parameters())
    assert not any(p.requires_grad for p in model.backbone.layer4.parameters())

    # Check that classifier is trainable
    assert all(p.requires_grad for p in model.classifier.parameters())

    # Test with unfreeze_last_block=True: only layer4 should be trainable
    model = nets.build_model("resnet50", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8)

    # Check that early layers are frozen
    assert not any(p.requires_grad for p in model.backbone.layer1.parameters())
    assert not any(p.requires_grad for p in model.backbone.layer2.parameters())
    assert not any(p.requires_grad for p in model.backbone.layer3.parameters())

    # Check that layer4 is trainable
    assert all(p.requires_grad for p in model.backbone.layer4.parameters())

    # Check that classifier is trainable
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_create_model_alias(monkeypatch) -> None:
    """Test that create_model is a valid alias for build_model."""
    monkeypatch.setattr(
        nets,
        "efficientnet_b0",
        lambda *_args, **_kwargs: torchvision.models.efficientnet_b0(weights=None),
    )

    model = nets.create_model("efficientnet_b0", num_classes=4, train_backbone=False, unfreeze_last_block=True)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 4)
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_train_backbone_efficientnet(monkeypatch) -> None:
    """Test that train_backbone=True unfreezes all backbone parameters."""
    monkeypatch.setattr(
        nets,
        "efficientnet_b0",
        lambda *_args, **_kwargs: torchvision.models.efficientnet_b0(weights=None),
    )

    model = nets.build_model("efficientnet_b0", num_classes=4, train_backbone=True, unfreeze_last_block=False)

    # Check that all backbone parameters are trainable
    assert all(p.requires_grad for p in model.backbone.parameters())
    # Check that classifier is trainable
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_train_backbone_resnet50(monkeypatch) -> None:
    """
    Verify that enabling train_backbone causes all ResNet50 backbone parameters to be trainable.
    """
    monkeypatch.setattr(
        nets,
        "resnet50",
        lambda *_args, **_kwargs: torchvision.models.resnet50(weights=None),
    )

    model = nets.build_model("resnet50", num_classes=4, train_backbone=True, unfreeze_last_block=False)

    # Check that all backbone layers are trainable
    assert all(p.requires_grad for p in model.backbone.layer1.parameters())
    assert all(p.requires_grad for p in model.backbone.layer2.parameters())
    assert all(p.requires_grad for p in model.backbone.layer3.parameters())
    assert all(p.requires_grad for p in model.backbone.layer4.parameters())
    # Check that classifier is trainable
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_invalid_architecture() -> None:
    """Test that invalid architecture raises ValueError."""
    with pytest.raises(ValueError, match=r"Architecture .* not supported"):
        nets.build_model("invalid_arch", num_classes=4)


def test_efficientnet_fusion_missing_extra_features(monkeypatch) -> None:
    """Test error when extra_features is None but extra_feature_dim > 0."""
    monkeypatch.setattr(
        nets,
        "efficientnet_b0",
        lambda *_args, **_kwargs: torchvision.models.efficientnet_b0(weights=None),
    )

    model = nets.build_model("efficientnet_b0", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8)
    x = torch.randn(2, 3, 224, 224)

    with pytest.raises(ValueError, match="Embeddings habilitados, mas extra_features=None foi recebido"):
        model(x, extra_features=None)


def test_efficientnet_fusion_batch_mismatch(monkeypatch) -> None:
    """Test error when extra_features batch size doesn't match input batch size."""
    monkeypatch.setattr(
        nets,
        "efficientnet_b0",
        lambda *_args, **_kwargs: torchvision.models.efficientnet_b0(weights=None),
    )

    model = nets.build_model("efficientnet_b0", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8)
    x = torch.randn(2, 3, 224, 224)
    extra_features = torch.randn(3, 8)  # Wrong batch size

    with pytest.raises(ValueError, match="extra_features batch mismatch"):
        model(x, extra_features)


def test_efficientnet_fusion_dimension_mismatch(monkeypatch) -> None:
    """Test error when extra_features dimension doesn't match expected size."""
    monkeypatch.setattr(
        nets,
        "efficientnet_b0",
        lambda *_args, **_kwargs: torchvision.models.efficientnet_b0(weights=None),
    )

    model = nets.build_model("efficientnet_b0", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8)
    x = torch.randn(2, 3, 224, 224)
    extra_features = torch.randn(2, 10)  # Wrong dimension

    with pytest.raises(ValueError, match="extra_features dimension mismatch"):
        model(x, extra_features)


def test_resnet50_fusion_missing_extra_features(monkeypatch) -> None:
    """Test error when extra_features is None but extra_feature_dim > 0 for ResNet50."""
    monkeypatch.setattr(
        nets,
        "resnet50",
        lambda *_args, **_kwargs: torchvision.models.resnet50(weights=None),
    )

    model = nets.build_model("resnet50", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8)
    x = torch.randn(2, 3, 224, 224)

    with pytest.raises(ValueError, match="Embeddings habilitados, mas extra_features=None foi recebido"):
        model(x, extra_features=None)


def test_load_with_fallback_success() -> None:
    """Test _load_with_fallback when weights load successfully."""
    def mock_factory(**_kwargs):
        """Return an EfficientNet-B0 with no pretrained weights."""
        return torchvision.models.efficientnet_b0(weights=None)

    model = nets._load_with_fallback(mock_factory, None, "test_arch")
    assert isinstance(model, torch.nn.Module)


def test_load_with_fallback_with_exception() -> None:
    """Test _load_with_fallback falls back to random weights on exception."""
    call_count = {"count": 0}

    def mock_factory(weights=None):
        """
        Test factory that returns an EfficientNet-B0 model and can simulate a download failure on its first invocation when `weights` is provided.

        Parameters:
            weights: If not None, the first call will raise a RuntimeError to simulate a pretrained-weights download failure; subsequent calls (or calls with weights=None) return a model.

        Returns:
            A torchvision EfficientNet-B0 model instance (torch.nn.Module).

        Raises:
            RuntimeError: If `weights` is not None and this is the factory's first invocation; used to simulate a download failure.

        Notes:
            Increments the shared `call_count["count"]` counter on every invocation.
        """
        call_count["count"] += 1
        if weights is not None and call_count["count"] == 1:
            raise RuntimeError
        return torchvision.models.efficientnet_b0(weights=None)

    with pytest.warns(RuntimeWarning, match="Failed to load pretrained weights"):
        model = nets._load_with_fallback(mock_factory, "IMAGENET1K_V1", "test_arch")

    assert isinstance(model, torch.nn.Module)
    assert call_count["count"] == 2  # Called twice: once failed, once succeeded
