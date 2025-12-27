from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")

from torch import nn
from torch.utils.data import DataLoader

from mammography.training.engine import train_one_epoch, validate


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))

    def forward(self, x: torch.Tensor, extra_features=None) -> torch.Tensor:  # noqa: ANN001
        return self.net(x)


def _make_loader() -> DataLoader:
    x = torch.randn(8, 3, 8, 8)
    y = torch.randint(0, 4, (8,))
    meta = [{"id": i} for i in range(8)]

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = [(x[i], y[i], meta[i]) for i in range(8)]
    return DataLoader(dataset, batch_size=4, collate_fn=collate)


def test_train_one_epoch_and_validate() -> None:
    model = DummyModel()
    loader = _make_loader()
    device = torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss, acc = train_one_epoch(model, loader, optimizer, device)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0

    metrics, pred_rows = validate(model, loader, device)
    assert "acc" in metrics
    assert isinstance(metrics["acc"], float)
    assert isinstance(pred_rows, list)
