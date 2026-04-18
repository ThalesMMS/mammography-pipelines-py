from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from mammography.models.embeddings.embedding_vector import EmbeddingVector
from mammography.utils.embeddings import extract_embedding_matrix


class _TensorLike:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def test_extract_embedding_matrix_stacks_numpy_embeddings() -> None:
    vectors = [
        SimpleNamespace(embedding=np.array([1.0, 2.0])),
        SimpleNamespace(embedding=np.array([3.0, 4.0])),
    ]

    matrix = extract_embedding_matrix(vectors)

    assert matrix is not None
    np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_extract_embedding_matrix_accepts_tensor_like_embeddings() -> None:
    vectors = [
        SimpleNamespace(embedding=_TensorLike([1.0, 2.0])),
        SimpleNamespace(embedding=_TensorLike([3.0, 4.0])),
    ]

    matrix = extract_embedding_matrix(vectors)

    assert matrix is not None
    np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_embedding_vector_property_returns_independent_copy() -> None:
    embedding = torch.arange(128, dtype=torch.float32)
    vector = EmbeddingVector("image-1", embedding).vector

    vector[0] = 999.0

    assert embedding[0].item() == 0.0
    assert isinstance(vector, np.ndarray)


def test_embedding_vector_default_model_config_is_copied() -> None:
    config = EmbeddingVector.default_model_config()
    config["model_name"] = "custom"

    assert EmbeddingVector.DEFAULT_MODEL_CONFIG["model_name"] == "resnet50_pretrained"
    with pytest.raises(TypeError):
        EmbeddingVector.DEFAULT_MODEL_CONFIG["model_name"] = "custom"

    embedding = torch.arange(128, dtype=torch.float32)
    vector = EmbeddingVector("image-1", embedding)

    assert vector.model_config["model_name"] == "resnet50_pretrained"
