from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")

from torch import nn
from torch.utils.data import DataLoader

from mammography.analysis.clustering import run_kmeans, run_pca, run_tsne, run_umap
from mammography.training.engine import extract_embeddings


class MockModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor, extra_features=None) -> torch.Tensor:  # noqa: ANN001
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _make_loader() -> DataLoader:
    x = torch.randn(20, 10, 4, 4)
    y = torch.randint(0, 2, (20,))
    meta = [{"id": i} for i in range(20)]

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    dataset = [(x[i], y[i], meta[i]) for i in range(20)]
    return DataLoader(dataset, batch_size=5, collate_fn=collate)


def test_extract_embeddings_and_projections() -> None:
    model = MockModel()
    loader = _make_loader()
    device = torch.device("cpu")

    feats, metas = extract_embeddings(model, loader, device, amp_enabled=False)
    assert feats.shape == (20, 10)
    assert len(metas) == 20

    pca = run_pca(feats, n_components=2)
    assert pca.shape == (20, 2)

    tsne = run_tsne(feats, n_components=2, perplexity=5)
    assert tsne.shape == (20, 2)

    labels, _ = run_kmeans(feats, k=3)
    assert labels.shape == (20,)

    try:
        umap_emb = run_umap(feats, n_components=2)
    except ImportError:
        pytest.skip("umap-learn not installed")
    else:
        assert umap_emb.shape == (20, 2)
