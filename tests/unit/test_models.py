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
    assert model.fc.weight.requires_grad


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
