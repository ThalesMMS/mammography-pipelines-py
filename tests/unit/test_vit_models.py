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


def test_build_vit_b_16_without_download(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "vit_b_16",
        lambda weights=None: torchvision.models.vit_b_16(weights=None),
    )

    model = nets.build_model("vit_b_16", num_classes=4, train_backbone=False, unfreeze_last_block=True, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 4)
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_build_vit_b_32_without_download(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "vit_b_32",
        lambda weights=None: torchvision.models.vit_b_32(weights=None),
    )

    model = nets.build_model("vit_b_32", num_classes=2, train_backbone=False, unfreeze_last_block=True, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 2)
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_build_vit_l_16_without_download(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "vit_l_16",
        lambda weights=None: torchvision.models.vit_l_16(weights=None),
    )

    model = nets.build_model("vit_l_16", num_classes=4, train_backbone=False, unfreeze_last_block=True, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 4)
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_build_vit_with_fusion_without_download(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "vit_b_16",
        lambda weights=None: torchvision.models.vit_b_16(weights=None),
    )

    model = nets.build_model("vit_b_16", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    extra_features = torch.randn(2, 8)
    y = model(x, extra_features)
    assert y.shape == (2, 4)
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_vit_fusion_gradient_flow(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "vit_b_16",
        lambda weights=None: torchvision.models.vit_b_16(weights=None),
    )

    model = nets.build_model("vit_b_16", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8, pretrained=False)
    model.train()

    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    extra_features = torch.randn(2, 8, requires_grad=True)

    y = model(x, extra_features)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert extra_features.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.classifier.parameters() if p.requires_grad)


def test_vit_fusion_parameter_freezing(monkeypatch) -> None:
    monkeypatch.setattr(
        nets,
        "vit_b_16",
        lambda weights=None: torchvision.models.vit_b_16(weights=None),
    )

    # Test with unfreeze_last_block=False: all backbone should be frozen
    model = nets.build_model("vit_b_16", num_classes=4, train_backbone=False, unfreeze_last_block=False, extra_feature_dim=8, pretrained=False)

    # Check that encoder layers are frozen
    assert not any(p.requires_grad for p in model.backbone.encoder.layers[0].parameters())
    assert not any(p.requires_grad for p in model.backbone.encoder.layers[-1].parameters())

    # Check that classifier is trainable
    assert all(p.requires_grad for p in model.classifier.parameters())

    # Test with unfreeze_last_block=True: only last encoder block should be trainable
    model = nets.build_model("vit_b_16", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8, pretrained=False)

    # Check that early layers are frozen
    assert not any(p.requires_grad for p in model.backbone.encoder.layers[0].parameters())

    # Check that last encoder layer is trainable
    assert all(p.requires_grad for p in model.backbone.encoder.layers[-1].parameters())

    # Check that classifier is trainable
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_build_deit_small_without_download(monkeypatch) -> None:
    timm = pytest.importorskip("timm")

    def mock_create_model(model_name, pretrained=False):
        # Create a minimal mock DeiT model structure
        base_model = torchvision.models.vit_b_16(weights=None)
        # Add timm-specific attributes
        base_model.blocks = base_model.encoder.layers
        base_model.head = base_model.heads.head
        base_model.num_features = base_model.head.in_features
        # Add forward_features method
        def forward_features(self, x):
            x = self.conv_proj(x)
            x = x.flatten(2).transpose(1, 2)
            x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
            x = self.encoder(x + self.encoder.pos_embedding)
            return x
        base_model.forward_features = lambda x: forward_features(base_model, x)
        return base_model

    monkeypatch.setattr(timm, "create_model", mock_create_model)

    model = nets.build_model("deit_small", num_classes=4, train_backbone=False, unfreeze_last_block=True, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 4)
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_build_deit_base_without_download(monkeypatch) -> None:
    timm = pytest.importorskip("timm")

    def mock_create_model(model_name, pretrained=False):
        # Create a minimal mock DeiT model structure
        base_model = torchvision.models.vit_b_16(weights=None)
        # Add timm-specific attributes
        base_model.blocks = base_model.encoder.layers
        base_model.head = base_model.heads.head
        base_model.num_features = base_model.head.in_features
        # Add forward_features method
        def forward_features(self, x):
            x = self.conv_proj(x)
            x = x.flatten(2).transpose(1, 2)
            x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
            x = self.encoder(x + self.encoder.pos_embedding)
            return x
        base_model.forward_features = lambda x: forward_features(base_model, x)
        return base_model

    monkeypatch.setattr(timm, "create_model", mock_create_model)

    model = nets.build_model("deit_base", num_classes=2, train_backbone=False, unfreeze_last_block=True, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 2)
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_build_deit_with_fusion_without_download(monkeypatch) -> None:
    timm = pytest.importorskip("timm")

    def mock_create_model(model_name, pretrained=False):
        # Create a minimal mock DeiT model structure
        base_model = torchvision.models.vit_b_16(weights=None)
        # Add timm-specific attributes
        base_model.blocks = base_model.encoder.layers
        base_model.head = base_model.heads.head
        base_model.num_features = base_model.head.in_features
        # Add forward_features method
        def forward_features(self, x):
            x = self.conv_proj(x)
            x = x.flatten(2).transpose(1, 2)
            x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
            x = self.encoder(x + self.encoder.pos_embedding)
            return x
        base_model.forward_features = lambda x: forward_features(base_model, x)
        return base_model

    monkeypatch.setattr(timm, "create_model", mock_create_model)

    model = nets.build_model("deit_small", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    extra_features = torch.randn(2, 8)
    y = model(x, extra_features)
    assert y.shape == (2, 4)
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_deit_fusion_gradient_flow(monkeypatch) -> None:
    timm = pytest.importorskip("timm")

    def mock_create_model(model_name, pretrained=False):
        # Create a minimal mock DeiT model structure
        base_model = torchvision.models.vit_b_16(weights=None)
        # Add timm-specific attributes
        base_model.blocks = base_model.encoder.layers
        base_model.head = base_model.heads.head
        base_model.num_features = base_model.head.in_features
        # Add forward_features method
        def forward_features(self, x):
            x = self.conv_proj(x)
            x = x.flatten(2).transpose(1, 2)
            x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
            x = self.encoder(x + self.encoder.pos_embedding)
            return x
        base_model.forward_features = lambda x: forward_features(base_model, x)
        return base_model

    monkeypatch.setattr(timm, "create_model", mock_create_model)

    model = nets.build_model("deit_small", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8, pretrained=False)
    model.train()

    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    extra_features = torch.randn(2, 8, requires_grad=True)

    y = model(x, extra_features)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert extra_features.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.classifier.parameters() if p.requires_grad)


def test_deit_fusion_parameter_freezing(monkeypatch) -> None:
    timm = pytest.importorskip("timm")

    def mock_create_model(model_name, pretrained=False):
        # Create a minimal mock DeiT model structure
        base_model = torchvision.models.vit_b_16(weights=None)
        # Add timm-specific attributes
        base_model.blocks = base_model.encoder.layers
        base_model.head = base_model.heads.head
        base_model.num_features = base_model.head.in_features
        # Add forward_features method
        def forward_features(self, x):
            x = self.conv_proj(x)
            x = x.flatten(2).transpose(1, 2)
            x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
            x = self.encoder(x + self.encoder.pos_embedding)
            return x
        base_model.forward_features = lambda x: forward_features(base_model, x)
        return base_model

    monkeypatch.setattr(timm, "create_model", mock_create_model)

    # Test with unfreeze_last_block=False: all backbone should be frozen
    model = nets.build_model("deit_small", num_classes=4, train_backbone=False, unfreeze_last_block=False, extra_feature_dim=8, pretrained=False)

    # Check that blocks are frozen
    assert not any(p.requires_grad for p in model.backbone.blocks[0].parameters())
    assert not any(p.requires_grad for p in model.backbone.blocks[-1].parameters())

    # Check that classifier is trainable
    assert all(p.requires_grad for p in model.classifier.parameters())

    # Test with unfreeze_last_block=True: only last block should be trainable
    model = nets.build_model("deit_small", num_classes=4, train_backbone=False, unfreeze_last_block=True, extra_feature_dim=8, pretrained=False)

    # Check that early blocks are frozen
    assert not any(p.requires_grad for p in model.backbone.blocks[0].parameters())

    # Check that last block is trainable
    assert all(p.requires_grad for p in model.backbone.blocks[-1].parameters())

    # Check that classifier is trainable
    assert all(p.requires_grad for p in model.classifier.parameters())
