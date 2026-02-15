from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from mammography.commands.embeddings_baselines import (
    compute_handcrafted_features,
    load_embeddings,
)


def _write_image(path: Path, value: int) -> None:
    data = np.full((8, 8), value, dtype=np.uint8)
    Image.fromarray(data, mode="L").save(path)


def test_compute_handcrafted_features_caps_components(tmp_path: Path) -> None:
    image_paths = []
    for idx in range(3):
        img_path = tmp_path / f"img_{idx}.png"
        _write_image(img_path, value=idx * 20)
        image_paths.append(img_path)

    metadata = pd.DataFrame(
        {
            "path": [str(path) for path in image_paths],
            "raw_label": [1, 2, 3],
        }
    )
    cache_path = tmp_path / "classic_features.npy"

    features = compute_handcrafted_features(
        metadata,
        cache_path,
        img_size=8,
        embeddings_dir=tmp_path,
        pca_svd_solver="full",
    )

    assert features.shape == (3, 15)
    assert cache_path.exists()

    cached = compute_handcrafted_features(
        metadata,
        cache_path,
        img_size=8,
        embeddings_dir=tmp_path,
        pca_svd_solver="full",
    )
    assert np.allclose(features, cached, equal_nan=True)


def test_load_embeddings_sanitizes_non_finite(tmp_path: Path) -> None:
    features = np.array([[1.0, np.nan], [np.inf, -np.inf]], dtype=np.float32)
    np.save(tmp_path / "features.npy", features)
    pd.DataFrame({"raw_label": [1, 2]}).to_csv(tmp_path / "metadata.csv", index=False)

    loaded, labels, meta = load_embeddings(tmp_path)

    assert loaded.shape == (2, 2)
    assert np.isfinite(loaded).all()
    assert labels.tolist() == [2, 3]
    assert len(meta) == 2
