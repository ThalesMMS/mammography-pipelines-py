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

from mammography.models import nets


def test_build_resnet50_without_download(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "resnet50",
        lambda weights=None: torchvision.models.resnet50(weights=None),
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
        lambda weights=None: torchvision.models.efficientnet_b0(weights=None),
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
        lambda weights=None: torchvision.models.efficientnet_b0(weights=None),
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
        lambda weights=None: torchvision.models.efficientnet_b0(weights=None),
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
        lambda weights=None: torchvision.models.efficientnet_b0(weights=None),
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
        lambda weights=None: torchvision.models.resnet50(weights=None),
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
        lambda weights=None: torchvision.models.resnet50(weights=None),
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
        lambda weights=None: torchvision.models.resnet50(weights=None),
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
