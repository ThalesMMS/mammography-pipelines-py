"""Shared embedding-vector conversion helpers."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


def _embedding_to_numpy(embedding: Any) -> np.ndarray:
    if hasattr(embedding, "detach"):
        embedding = embedding.detach()
    if hasattr(embedding, "cpu"):
        embedding = embedding.cpu()
    if hasattr(embedding, "numpy"):
        return np.asarray(embedding.numpy())
    return np.asarray(embedding)


def extract_embedding_matrix(
    embedding_vectors: Sequence[Any],
    *,
    logger: logging.Logger | None = None,
) -> np.ndarray | None:
    """Extract a stacked embedding matrix from embedding-vector objects."""
    active_logger = logger or LOGGER
    try:
        embeddings = [
            _embedding_to_numpy(embedding_vector.embedding)
            for embedding_vector in embedding_vectors
        ]
        return np.vstack(embeddings)
    except Exception as exc:
        active_logger.error("Error extracting embedding matrix: %s", exc)
        return None
