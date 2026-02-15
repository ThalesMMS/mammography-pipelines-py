import numpy as np
import pandas as pd
import pytest

from mammography.commands import embeddings_baselines


def _write_embeddings(tmp_path, labels):
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    features = np.arange(len(labels) * 4, dtype=np.float32).reshape(len(labels), 4)
    np.save(embeddings_dir / "features.npy", features)
    metadata = pd.DataFrame({"label": labels})
    metadata.to_csv(embeddings_dir / "metadata.csv", index=False)
    return embeddings_dir


def test_load_embeddings_limits_samples_stratified(tmp_path):
    labels = ["A", "A", "A", "B", "B", "B"]
    embeddings_dir = _write_embeddings(tmp_path, labels)

    features, y, meta = embeddings_baselines.load_embeddings(
        embeddings_dir, max_samples=4
    )

    assert features.shape[0] == 4
    assert len(y) == 4
    assert len(meta) == 4

    unique, counts = np.unique(y, return_counts=True)
    counts_by_label = dict(zip(unique, counts))
    assert counts_by_label[1] == 2
    assert counts_by_label[2] == 2


def test_load_embeddings_max_samples_too_small_raises(tmp_path):
    labels = ["A", "A", "B", "B"]
    embeddings_dir = _write_embeddings(tmp_path, labels)

    with pytest.raises(ValueError, match="max_samples"):
        embeddings_baselines.load_embeddings(embeddings_dir, max_samples=1)
