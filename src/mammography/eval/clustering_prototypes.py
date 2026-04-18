# ruff: noqa
"""
Clustering evaluation module for mammography embedding analysis.

This module provides comprehensive evaluation capabilities for clustering
results including quality metrics, sanity checks, and visual prototype
selection for the breast density exploration pipeline.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- Clustering evaluation assesses quality without ground truth labels
- Sanity checks ensure clinical relevance and catch obvious failures
- Visual prototypes enable qualitative validation of clustering results
- Multiple metrics provide comprehensive quality assessment

Author: Research Team
Version: 1.0.0
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..clustering.clustering_result import ClusteringResult
from ..io.dicom import MammographyImage
from ..models.embeddings.embedding_vector import EmbeddingVector
from ..utils.embeddings import extract_embedding_matrix

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


class ClusteringPrototypeMixin:
    def _select_visual_prototypes(
        self,
        clustering_result: ClusteringResult,
        embedding_vectors: List[EmbeddingVector],
        _mammography_images: Optional[List[MammographyImage]] = None,
    ) -> Dict[str, Any]:
        """
        Select representative visual prototype samples for each cluster in a clustering result.

        The returned dictionary is keyed by "cluster_{cluster_id}" and each value contains:
        - `prototype_indices` (list[int]): indices of selected samples in the original input list,
        - `prototype_image_ids` (list): corresponding image IDs from the provided embedding vectors,
        - `cluster_size` (int): number of samples assigned to the cluster,
        - `selection_method` (str): name of the method used to select prototypes.

        Parameters:
            clustering_result: ClusteringResult containing per-sample cluster labels.
            embedding_vectors: List of EmbeddingVector instances corresponding to the original samples.
            _mammography_images: Optional list of MammographyImage instances (accepted but not used).

        Returns:
            dict: Mapping of cluster keys to prototype selection results. If embeddings cannot be extracted or an error occurs during selection, an empty or partially filled dictionary may be returned. Noise samples labeled `-1` are skipped.
        """
        visual_prototypes = {}

        try:
            cluster_labels = _to_numpy(clustering_result.cluster_labels)
            n_samples_per_cluster = self.config["visual_prototypes"][
                "n_samples_per_cluster"
            ]
            selection_method = self.config["visual_prototypes"]["selection_method"]

            # Extract embedding matrix
            embedding_matrix = self._extract_embedding_matrix(embedding_vectors)
            if embedding_matrix is None:
                return visual_prototypes
            rng = getattr(self, "rng", None)
            if rng is None:
                rng = np.random.default_rng(self.config.get("seed", 42))
                self.rng = rng

            # Select prototypes for each cluster
            for cluster_id in np.unique(cluster_labels):
                if cluster_id == -1:  # Skip noise points in HDBSCAN
                    continue

                resolved_selection = selection_method
                cluster_mask = cluster_labels == cluster_id
                cluster_embeddings = embedding_matrix[cluster_mask]
                cluster_indices = np.nonzero(cluster_mask)[0]

                if len(cluster_indices) == 0:
                    continue

                # Select prototypes based on method
                if selection_method == "centroid_distance":
                    prototype_indices = self._select_prototypes_by_centroid_distance(
                        cluster_embeddings, cluster_indices, n_samples_per_cluster
                    )
                elif selection_method == "random":
                    prototype_indices = self._select_prototypes_randomly(
                        cluster_indices, n_samples_per_cluster, rng
                    )
                else:
                    logger.warning(
                        "Unknown prototype selection method: %s; using fallback:first_n",
                        selection_method,
                    )
                    prototype_indices = cluster_indices[:n_samples_per_cluster]
                    resolved_selection = "fallback:first_n"

                # Store prototype information
                visual_prototypes[f"cluster_{cluster_id}"] = {
                    "prototype_indices": prototype_indices.tolist(),
                    "prototype_image_ids": [
                        embedding_vectors[i].image_id for i in prototype_indices
                    ],
                    "cluster_size": len(cluster_indices),
                    "selection_method": resolved_selection,
                }

            logger.info("Successfully selected visual prototypes")
            return visual_prototypes

        except Exception as e:
            logger.exception(f"Error selecting visual prototypes: {e!s}")
            return visual_prototypes

    def _select_prototypes_by_centroid_distance(
        self,
        cluster_embeddings: np.ndarray,
        cluster_indices: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Selects up to `n_samples` cluster members whose embeddings are closest to the cluster centroid.

        Parameters:
            cluster_embeddings (np.ndarray): Embedding vectors for samples in the cluster (shape: [n_samples_in_cluster, embedding_dim]).
            cluster_indices (np.ndarray): Original dataset indices corresponding to rows in `cluster_embeddings`.
            n_samples (int): Maximum number of prototypes to select.

        Returns:
            np.ndarray: Array of selected original sample indices (subset of `cluster_indices`), ordered by proximity to the centroid.
        """
        # Compute cluster centroid
        centroid = cluster_embeddings.mean(axis=0)

        # Compute distances to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

        # Select closest samples
        closest_indices = np.argsort(distances)[:n_samples]

        return cluster_indices[closest_indices]

    def _select_prototypes_randomly(
        self,
        cluster_indices: np.ndarray,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Randomly selects up to n_samples indices from the provided cluster indices.

        Parameters:
            cluster_indices (np.ndarray): Array of original dataset indices that belong to the cluster.
            n_samples (int): Maximum number of prototype indices to select.
            rng (np.random.Generator | None): Optional generator shared across cluster selections.

        Returns:
            np.ndarray: Selected prototype indices (subset of `cluster_indices`) chosen without replacement.
        """
        n_available = len(cluster_indices)
        n_select = min(n_samples, n_available)

        if rng is None:
            rng = np.random.default_rng(self.config.get("seed", 42))
        selected_indices = rng.choice(cluster_indices, size=n_select, replace=False)
        return selected_indices
