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
from typing import Any, Dict, List

import numpy as np
import torch

from ..clustering.clustering_result import ClusteringResult
from ..models.embeddings.embedding_vector import EmbeddingVector

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


class ClusteringMetricsMixin:
    def _compute_quality_metrics(
        self,
        clustering_result: ClusteringResult,
        embedding_vectors: List[EmbeddingVector],
    ) -> Dict[str, Any]:
        """
        Assemble label-free clustering quality metrics from a clustering result and corresponding embeddings.

        Computes per-cluster statistics, distance-based metrics (intra/inter-cluster and their ratio), centroid separation summaries, and—when centroids are provided—point-to-centroid stability metrics. Noise points labeled `-1` are ignored where appropriate; if embeddings cannot be extracted or there is insufficient cluster structure some metric sections may be empty.

        Parameters:
            clustering_result: ClusteringResult containing at least `cluster_labels` and optional `centroids`.
            embedding_vectors: Sequence of embedding objects used to build the embedding matrix via `self._extract_embedding_matrix`.

        Returns:
            Dict[str, Any]: A dictionary with any of the following keys when available:
                - "cluster_statistics": per-cluster size and embedding summaries
                - "distance_metrics": mean/std of intra- and inter-cluster distances and separation ratio
                - "separation_metrics": centroid pairwise distance summaries (mean/min/max/std)
                - "stability_metrics": point-to-assigned-centroid distance summaries (mean/std/max)
        """
        quality_metrics = {}

        try:
            # Extract embedding matrix
            embedding_matrix = self._extract_embedding_matrix(embedding_vectors)
            if embedding_matrix is None:
                return quality_metrics

            cluster_labels = _to_numpy(clustering_result.cluster_labels)

            # Compute cluster statistics
            cluster_stats = self._compute_cluster_statistics(
                cluster_labels, embedding_matrix
            )
            quality_metrics["cluster_statistics"] = cluster_stats

            # Compute intra-cluster and inter-cluster distances
            distance_metrics = self._compute_distance_metrics(
                embedding_matrix, cluster_labels
            )
            quality_metrics["distance_metrics"] = distance_metrics

            # Compute cluster separation metrics
            separation_metrics = self._compute_separation_metrics(
                embedding_matrix, cluster_labels
            )
            quality_metrics["separation_metrics"] = separation_metrics

            # Compute cluster stability (if centroids available)
            if clustering_result.centroids is not None:
                stability_metrics = self._compute_stability_metrics(
                    embedding_matrix, cluster_labels, clustering_result.centroids
                )
                quality_metrics["stability_metrics"] = stability_metrics

            logger.info("Successfully computed quality metrics")
            return quality_metrics

        except (ValueError, TypeError, KeyError) as e:
            logger.exception("Error computing quality metrics: %s", e)
            return quality_metrics
        except Exception:
            logger.exception("Unexpected error computing quality metrics")
            raise

    def _compute_cluster_statistics(
        self, cluster_labels: np.ndarray, embedding_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute per-cluster summary statistics for each non-noise cluster.

        For each cluster label (excluding `-1` which is treated as noise), returns the count of members and vector-wise summary statistics of the cluster's embeddings.

        Parameters:
            cluster_labels (np.ndarray): 1D array of integer cluster labels for each embedding; label `-1` denotes noise.
            embedding_matrix (np.ndarray): 2D array of shape (n_samples, n_features) containing embedding vectors.

        Returns:
            Dict[str, Any]: Mapping "cluster_{label}" -> {
                "size": int count of points in the cluster,
                "mean_embedding": list of per-dimension means,
                "std_embedding": list of per-dimension standard deviations,
                "min_embedding": list of per-dimension minima,
                "max_embedding": list of per-dimension maxima
            } for each non-noise cluster.
        """
        cluster_stats = {}

        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue

            cluster_mask = cluster_labels == label
            cluster_embeddings = embedding_matrix[cluster_mask]

            cluster_stats[f"cluster_{label}"] = {
                "size": int(np.sum(cluster_mask)),
                "mean_embedding": cluster_embeddings.mean(axis=0).tolist(),
                "std_embedding": cluster_embeddings.std(axis=0).tolist(),
                "min_embedding": cluster_embeddings.min(axis=0).tolist(),
                "max_embedding": cluster_embeddings.max(axis=0).tolist(),
            }

        return cluster_stats

    def _compute_distance_metrics(
        self, embedding_matrix: np.ndarray, cluster_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Summarizes intra- and inter-cluster Euclidean distance statistics for non-noise clusters.

        Excludes points with label -1. Computes per-pairpoint intra-cluster distances (upper-triangular off-diagonal) and all pairwise inter-cluster distances between distinct clusters. When available, returns mean and standard deviation for intra- and inter-cluster distances and a separation ratio (mean_inter / mean_intra).

        Returns:
            distance_metrics (Dict[str, float]): Dictionary that may contain:
                - "mean_intra_cluster_distance": mean of all intra-cluster distances
                - "std_intra_cluster_distance": standard deviation of intra-cluster distances
                - "mean_inter_cluster_distance": mean of all inter-cluster distances
                - "std_inter_cluster_distance": standard deviation of inter-cluster distances
                - "separation_ratio": ratio of mean inter-cluster distance to mean intra-cluster distance
        """
        distance_metrics = {}

        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]  # Remove noise points

        # Compute intra-cluster distances
        intra_cluster_distances = []
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_embeddings = embedding_matrix[cluster_mask]

            if len(cluster_embeddings) > 1:
                # Compute pairwise distances within cluster
                distances = np.linalg.norm(
                    cluster_embeddings[:, np.newaxis] - cluster_embeddings, axis=2
                )
                intra_cluster_distances.extend(
                    distances[np.triu_indices_from(distances, k=1)]
                )

        if intra_cluster_distances:
            distance_metrics["mean_intra_cluster_distance"] = float(
                np.mean(intra_cluster_distances)
            )
            distance_metrics["std_intra_cluster_distance"] = float(
                np.std(intra_cluster_distances)
            )

        if len(unique_labels) < 2:
            return distance_metrics

        # Compute inter-cluster distances
        inter_cluster_distances = []
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i + 1 :]:
                cluster1_mask = cluster_labels == label1
                cluster2_mask = cluster_labels == label2

                cluster1_embeddings = embedding_matrix[cluster1_mask]
                cluster2_embeddings = embedding_matrix[cluster2_mask]

                # Compute distances between clusters
                distances = np.linalg.norm(
                    cluster1_embeddings[:, np.newaxis] - cluster2_embeddings, axis=2
                )
                inter_cluster_distances.extend(distances.flatten())

        # Compute summary statistics
        if inter_cluster_distances:
            distance_metrics["mean_inter_cluster_distance"] = float(
                np.mean(inter_cluster_distances)
            )
            distance_metrics["std_inter_cluster_distance"] = float(
                np.std(inter_cluster_distances)
            )

        if intra_cluster_distances and inter_cluster_distances:
            mean_inter = distance_metrics["mean_inter_cluster_distance"]
            mean_intra = distance_metrics["mean_intra_cluster_distance"]
            if np.isfinite(mean_intra) and mean_intra != 0.0:
                distance_metrics["separation_ratio"] = mean_inter / mean_intra
            else:
                distance_metrics["separation_ratio"] = float("inf")

        return distance_metrics

    def _compute_separation_metrics(
        self, embedding_matrix: np.ndarray, cluster_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute summary statistics of pairwise centroid distances to quantify cluster separation.

        This method ignores noise points labeled -1 and requires at least two non-noise clusters; when those conditions are not met it returns an empty dict. The returned metrics describe the distribution of Euclidean distances between cluster centroids.

        Parameters:
            embedding_matrix (np.ndarray): 2D array of sample embeddings (n_samples x n_features).
            cluster_labels (np.ndarray): Integer cluster labels for each sample; label -1 is treated as noise and excluded.

        Returns:
            Dict[str, float]: Dictionary containing:
                - "mean_centroid_distance": mean of all off-diagonal centroid-to-centroid distances.
                - "min_centroid_distance": minimum off-diagonal centroid distance.
                - "max_centroid_distance": maximum off-diagonal centroid distance.
                - "std_centroid_distance": standard deviation of off-diagonal centroid distances.
        """
        separation_metrics = {}

        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]  # Remove noise points

        if len(unique_labels) < 2:
            return separation_metrics

        # Compute cluster centroids
        centroids = []
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            centroid = embedding_matrix[cluster_mask].mean(axis=0)
            centroids.append(centroid)

        centroids = np.array(centroids)

        # Compute pairwise distances between centroids
        centroid_distances = np.linalg.norm(
            centroids[:, np.newaxis] - centroids, axis=2
        )

        # Remove diagonal (distance to self)
        mask = np.ones_like(centroid_distances, dtype=bool)
        np.fill_diagonal(mask, False)
        centroid_distances = centroid_distances[mask]

        separation_metrics["mean_centroid_distance"] = float(
            np.mean(centroid_distances)
        )
        separation_metrics["min_centroid_distance"] = float(np.min(centroid_distances))
        separation_metrics["max_centroid_distance"] = float(np.max(centroid_distances))
        separation_metrics["std_centroid_distance"] = float(np.std(centroid_distances))

        return separation_metrics

    def _compute_stability_metrics(
        self,
        embedding_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        centroids: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute point-to-centroid distance statistics for non-noise cluster assignments.

        Parameters:
            embedding_matrix (np.ndarray): 2D array of sample embeddings aligned with cluster_labels by index.
            cluster_labels (np.ndarray): 1D array of integer cluster labels; label `-1` is treated as noise and skipped.
            centroids (torch.Tensor): Tensor of cluster centroids ordered to match the sorted non-noise labels in `valid_labels`; row `i` corresponds to `valid_labels[i]`.

        Returns:
            Dict[str, float]: Dictionary containing:
                - "mean_point_to_centroid_distance": mean Euclidean distance from points to their assigned centroids.
                - "std_point_to_centroid_distance": standard deviation of those distances.
                - "max_point_to_centroid_distance": maximum of those distances.
            Returns an empty dict if no non-noise points are present.
        """
        stability_metrics = {}

        centroids_np = _to_numpy(centroids)
        valid_labels = [
            int(label) for label in np.unique(cluster_labels) if int(label) != -1
        ]
        if len(centroids_np) < len(valid_labels):
            logger.warning(
                "Centroid rows (%d) fewer than non-noise cluster labels (%d); "
                "skipping labels without centroids.",
                len(centroids_np),
                len(valid_labels),
            )
        label_to_row = {
            label: row_idx
            for row_idx, label in enumerate(valid_labels)
            if row_idx < len(centroids_np)
        }

        # Compute distances from each point to its assigned centroid
        point_to_centroid_distances = []
        for i, label in enumerate(cluster_labels):
            label_key = int(label)
            if label_key != -1 and label_key in label_to_row:  # Skip noise points
                centroid = centroids_np[label_to_row[label_key]]
                distance = np.linalg.norm(embedding_matrix[i] - centroid)
                point_to_centroid_distances.append(distance)

        if point_to_centroid_distances:
            stability_metrics["mean_point_to_centroid_distance"] = float(
                np.mean(point_to_centroid_distances)
            )
            stability_metrics["std_point_to_centroid_distance"] = float(
                np.std(point_to_centroid_distances)
            )
            stability_metrics["max_point_to_centroid_distance"] = float(
                np.max(point_to_centroid_distances)
            )

        return stability_metrics

    def _analyze_cluster_sizes(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        Summarize cluster counts and size-distribution statistics while flagging size-related issues.

        Parameters:
            cluster_labels (np.ndarray): 1D array of cluster assignments for each sample; label `-1` is treated as noise and excluded from size statistics.

        Returns:
            Dict[str, Any]: Analysis dictionary containing:
                - "n_clusters" (int): number of non-noise clusters.
                - "cluster_sizes" (Dict[str, int]): per-cluster counts keyed by "cluster_{label}".
                - "size_statistics" (Dict[str, float|int]): distribution summaries with keys
                    "mean_size", "std_size", "min_size", "max_size", and "median_size".
                - "size_balance" (Dict[str, float]): balance metrics with keys
                    "coefficient_of_variation" and "gini_coefficient".
                - "potential_issues" (List[str]): detected issues such as very small clusters (<5 samples),
                    overly large clusters (>80% of data), or high imbalance (coefficient of variation > 1.0).
        """
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)

        # Remove noise points (-1) from analysis
        valid_mask = unique_labels != -1
        valid_labels = unique_labels[valid_mask]
        valid_counts = counts[valid_mask]

        if valid_counts.size == 0:
            return {
                "n_clusters": 0,
                "cluster_sizes": {},
                "size_statistics": {
                    "mean_size": 0.0,
                    "std_size": 0.0,
                    "min_size": 0,
                    "max_size": 0,
                    "median_size": 0.0,
                },
                "size_balance": {
                    "coefficient_of_variation": 0.0,
                    "gini_coefficient": 0.0,
                },
                "potential_issues": ["all points labeled as noise"],
            }

        mean_size = float(np.mean(valid_counts))
        size_analysis = {
            "n_clusters": len(valid_labels),
            "cluster_sizes": {
                f"cluster_{label}": int(count)
                for label, count in zip(valid_labels, valid_counts, strict=False)
            },
            "size_statistics": {
                "mean_size": mean_size,
                "std_size": float(np.std(valid_counts)),
                "min_size": int(np.min(valid_counts)),
                "max_size": int(np.max(valid_counts)),
                "median_size": float(np.median(valid_counts)),
            },
            "size_balance": {
                "coefficient_of_variation": float(np.std(valid_counts) / mean_size),
                "gini_coefficient": self._compute_gini_coefficient(valid_counts),
            },
        }

        # Check for potential issues
        total_samples = len(cluster_labels)
        size_analysis["potential_issues"] = []

        if np.min(valid_counts) < 5:
            size_analysis["potential_issues"].append(
                "Some clusters have very few samples (< 5)"
            )

        if np.max(valid_counts) > total_samples * 0.8:
            size_analysis["potential_issues"].append(
                "Some clusters are very large (> 80% of data)"
            )

        if size_analysis["size_balance"]["coefficient_of_variation"] > 1.0:
            size_analysis["potential_issues"].append("High cluster size imbalance")

        return size_analysis

    def _compute_gini_coefficient(self, values: np.ndarray) -> float:
        """
        Compute the Gini coefficient for a distribution of non-negative values representing cluster sizes.

        Parameters:
            values (np.ndarray): 1-D array of non-negative cluster sizes; must contain at least one element and have a positive sum.

        Returns:
            float: Gini coefficient between 0.0 and 1.0, where 0.0 indicates perfect equality and values closer to 1.0 indicate greater inequality.
        """
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        total = cumsum[-1]
        if total == 0:
            return 0.0

        return (n + 1 - 2 * np.sum(cumsum) / total) / n
