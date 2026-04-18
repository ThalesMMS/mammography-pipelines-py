from __future__ import annotations

import numpy as np
import pytest

from mammography.vis.dimensionality import (
    build_pca,
    project_pca,
    project_umap,
    resolve_tsne_perplexity,
)


def test_resolve_tsne_perplexity_clamps_to_sample_count() -> None:
    assert resolve_tsne_perplexity(10, 30.0) == 3.0
    assert resolve_tsne_perplexity(1, 30.0) == 1.0


def test_build_pca_clips_component_count() -> None:
    features = np.ones((3, 5), dtype=float)

    pca = build_pca(features, n_components=10, seed=42)

    assert pca.n_components == 3


def test_project_pca_returns_embedding_and_model() -> None:
    features = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    embedding, pca = project_pca(features, n_components=2, seed=42)

    assert embedding.shape == (4, 2)
    assert pca.n_components == 2


class _PreFittedReducer:
    def __init__(self):
        self.transform_calls = 0

    def transform(self, features):
        self.transform_calls += 1
        return features[:, :2] + 1.0

    def fit_transform(self, features):
        raise AssertionError("pre-fitted reducer should not be refit")


def test_project_pca_uses_transform_for_prefitted_model() -> None:
    features = np.arange(12, dtype=float).reshape(4, 3)
    model = _PreFittedReducer()

    embedding, returned_model = project_pca(features, model=model)

    assert returned_model is model
    assert model.transform_calls == 1
    np.testing.assert_array_equal(embedding, features[:, :2] + 1.0)


def test_project_pca_rejects_params_with_prefitted_model() -> None:
    features = np.arange(12, dtype=float).reshape(4, 3)

    with pytest.raises(ValueError, match="pca_params"):
        project_pca(features, model=_PreFittedReducer(), pca_params={"n_components": 1})


def test_project_umap_uses_transform_for_prefitted_model() -> None:
    features = np.arange(12, dtype=float).reshape(4, 3)
    model = _PreFittedReducer()

    embedding, returned_model = project_umap(features, model=model)

    assert returned_model is model
    assert model.transform_calls == 1
    np.testing.assert_array_equal(embedding, features[:, :2] + 1.0)
