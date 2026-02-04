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

from collections import Counter
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..clustering.clustering_result import ClusteringResult
from ..io.dicom import MammographyImage
from ..models.embeddings.embedding_vector import EmbeddingVector

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """
    Clustering evaluator for mammography embedding analysis.

    This class provides methods for evaluating clustering results including
    quality metrics computation, sanity checks, and visual prototype selection.
    It ensures comprehensive assessment of clustering quality and clinical relevance.

    Educational Notes:
    - Evaluation metrics assess clustering quality without ground truth
    - Sanity checks ensure clinical relevance and catch obvious failures
    - Visual prototypes enable qualitative validation
    - Multiple assessment methods provide comprehensive evaluation

    Attributes:
        config: Evaluation configuration dictionary
        supported_metrics: List of supported evaluation metrics
        sanity_check_methods: List of available sanity check methods
    """

    # Supported evaluation metrics
    SUPPORTED_METRICS = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "ari",
        "nmi",
        "homogeneity",
        "completeness",
        "v_measure",
    ]

    # Available sanity check methods
    SANITY_CHECK_METHODS = [
        "intensity_histograms",
        "projection_distribution",
        "laterality_distribution",
        "visual_prototypes",
        "cluster_size_analysis",
        "embedding_statistics",
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize clustering evaluator with configuration.

        Args:
            config: Evaluation configuration dictionary

        Educational Note: Configuration validation ensures all required
        parameters are present and valid for evaluation.
        """
        self.config = self._validate_config(config)

        # Set random seed for reproducibility
        if "seed" in self.config:
            torch.manual_seed(self.config["seed"])
            np.random.seed(self.config["seed"])

        logger.info(f"Initialized ClusteringEvaluator with config: {self.config}")

    def evaluate_clustering(
        self,
        clustering_result: ClusteringResult,
        embedding_vectors: List[EmbeddingVector],
        mammography_images: Optional[List[MammographyImage]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive clustering evaluation.

        Educational Note: This method demonstrates the complete evaluation
        pipeline including metrics computation, sanity checks, and analysis.

        Args:
            clustering_result: ClusteringResult instance to evaluate
            embedding_vectors: List of EmbeddingVector instances
            mammography_images: Optional list of MammographyImage instances for sanity checks

        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        evaluation_results = {
            "clustering_result": clustering_result,
            "evaluation_timestamp": datetime.now().isoformat(),
            "config": self.config,
        }

        try:
            # Compute additional quality metrics
            quality_metrics = self._compute_quality_metrics(
                clustering_result, embedding_vectors
            )
            evaluation_results["quality_metrics"] = quality_metrics

            # Perform sanity checks
            sanity_checks = self._perform_sanity_checks(
                clustering_result, embedding_vectors, mammography_images
            )
            evaluation_results["sanity_checks"] = sanity_checks

            # Select visual prototypes
            visual_prototypes = self._select_visual_prototypes(
                clustering_result, embedding_vectors, mammography_images
            )
            evaluation_results["visual_prototypes"] = visual_prototypes

            # Generate evaluation summary
            evaluation_summary = self._generate_evaluation_summary(evaluation_results)
            evaluation_results["summary"] = evaluation_summary

            logger.info("Successfully completed clustering evaluation")
            return evaluation_results

        except Exception as e:
            logger.error(f"Error in clustering evaluation: {e!s}")
            evaluation_results["error"] = str(e)
            return evaluation_results

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate evaluation configuration.

        Educational Note: Configuration validation ensures all required
        parameters are present and within valid ranges.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Dict[str, Any]: Validated configuration
        """
        # Set default values for optional parameters
        config.setdefault("metrics", self.SUPPORTED_METRICS)
        config.setdefault("sanity_checks", self.SANITY_CHECK_METHODS)
        config.setdefault(
            "visual_prototypes",
            {"n_samples_per_cluster": 4, "selection_method": "centroid_distance"},
        )
        config.setdefault("seed", 42)

        # Validate metrics
        for metric in config["metrics"]:
            if metric not in self.SUPPORTED_METRICS:
                logger.warning(f"Unknown metric: {metric}")

        # Validate sanity checks
        for check in config["sanity_checks"]:
            if check not in self.SANITY_CHECK_METHODS:
                logger.warning(f"Unknown sanity check: {check}")

        return config

    def _compute_quality_metrics(
        self,
        clustering_result: ClusteringResult,
        embedding_vectors: List[EmbeddingVector],
    ) -> Dict[str, Any]:
        """
        Compute additional quality metrics for clustering results.

        Educational Note: Quality metrics assess clustering performance
        from different perspectives without requiring ground truth labels.

        Args:
            clustering_result: ClusteringResult instance
            embedding_vectors: List of EmbeddingVector instances

        Returns:
            Dict[str, Any]: Quality metrics dictionary
        """
        quality_metrics = {}

        try:
            # Extract embedding matrix
            embedding_matrix = self._extract_embedding_matrix(embedding_vectors)
            if embedding_matrix is None:
                return quality_metrics

            cluster_labels = clustering_result.cluster_labels.numpy()

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

        except Exception as e:
            logger.error(f"Error computing quality metrics: {e!s}")
            return quality_metrics

    def _perform_sanity_checks(
        self,
        clustering_result: ClusteringResult,
        embedding_vectors: List[EmbeddingVector],
        mammography_images: Optional[List[MammographyImage]] = None,
    ) -> Dict[str, Any]:
        """
        Perform sanity checks on clustering results.

        Educational Note: Sanity checks ensure clustering results are
        clinically relevant and catch obvious failures or biases.

        Args:
            clustering_result: ClusteringResult instance
            embedding_vectors: List of EmbeddingVector instances
            mammography_images: Optional list of MammographyImage instances

        Returns:
            Dict[str, Any]: Sanity check results
        """
        sanity_checks = {}

        try:
            cluster_labels = clustering_result.cluster_labels.numpy()

            # Check cluster size distribution
            if "cluster_size_analysis" in self.config["sanity_checks"]:
                size_analysis = self._analyze_cluster_sizes(cluster_labels)
                sanity_checks["cluster_size_analysis"] = size_analysis

            # Check embedding statistics
            if "embedding_statistics" in self.config["sanity_checks"]:
                embedding_stats = self._analyze_embedding_statistics(
                    embedding_vectors, cluster_labels
                )
                sanity_checks["embedding_statistics"] = embedding_stats

            # Check projection distribution (if mammography images available)
            if (
                mammography_images
                and "projection_distribution" in self.config["sanity_checks"]
            ):
                projection_analysis = self._analyze_projection_distribution(
                    mammography_images, cluster_labels
                )
                sanity_checks["projection_distribution"] = projection_analysis

            # Check laterality distribution (if mammography images available)
            if (
                mammography_images
                and "laterality_distribution" in self.config["sanity_checks"]
            ):
                laterality_analysis = self._analyze_laterality_distribution(
                    mammography_images, cluster_labels
                )
                sanity_checks["laterality_distribution"] = laterality_analysis

            # Check intensity histograms (if mammography images available)
            if (
                mammography_images
                and "intensity_histograms" in self.config["sanity_checks"]
            ):
                intensity_analysis = self._analyze_intensity_histograms(
                    mammography_images, cluster_labels
                )
                sanity_checks["intensity_histograms"] = intensity_analysis

            logger.info("Successfully completed sanity checks")
            return sanity_checks

        except Exception as e:
            logger.error(f"Error in sanity checks: {e!s}")
            return sanity_checks

    def _select_visual_prototypes(
        self,
        clustering_result: ClusteringResult,
        embedding_vectors: List[EmbeddingVector],
        _mammography_images: Optional[List[MammographyImage]] = None,
    ) -> Dict[str, Any]:
        """
        Select visual prototypes for each cluster.

        Educational Note: Visual prototypes enable qualitative validation
        of clustering results by examining representative images.

        Args:
            clustering_result: ClusteringResult instance
            embedding_vectors: List of EmbeddingVector instances
            mammography_images: Optional list of MammographyImage instances

        Returns:
            Dict[str, Any]: Visual prototype selection results
        """
        visual_prototypes = {}

        try:
            cluster_labels = clustering_result.cluster_labels.numpy()
            n_samples_per_cluster = self.config["visual_prototypes"][
                "n_samples_per_cluster"
            ]
            selection_method = self.config["visual_prototypes"]["selection_method"]

            # Extract embedding matrix
            embedding_matrix = self._extract_embedding_matrix(embedding_vectors)
            if embedding_matrix is None:
                return visual_prototypes

            # Select prototypes for each cluster
            for cluster_id in np.unique(cluster_labels):
                if cluster_id == -1:  # Skip noise points in HDBSCAN
                    continue

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
                        cluster_indices, n_samples_per_cluster
                    )
                else:
                    logger.warning(
                        f"Unknown prototype selection method: {selection_method}"
                    )
                    prototype_indices = cluster_indices[:n_samples_per_cluster]

                # Store prototype information
                visual_prototypes[f"cluster_{cluster_id}"] = {
                    "prototype_indices": prototype_indices.tolist(),
                    "prototype_image_ids": [
                        embedding_vectors[i].image_id for i in prototype_indices
                    ],
                    "cluster_size": len(cluster_indices),
                    "selection_method": selection_method,
                }

            logger.info("Successfully selected visual prototypes")
            return visual_prototypes

        except Exception as e:
            logger.error(f"Error selecting visual prototypes: {e!s}")
            return visual_prototypes

    def _extract_embedding_matrix(
        self, embedding_vectors: List[EmbeddingVector]
    ) -> Optional[np.ndarray]:
        """
        Extract embedding matrix from embedding vectors.

        Educational Note: This method converts embedding vectors to a matrix
        format suitable for analysis and computation.

        Args:
            embedding_vectors: List of EmbeddingVector instances

        Returns:
            np.ndarray: Embedding matrix, None if failed
        """
        try:
            embeddings = [emb.embedding.cpu().numpy() for emb in embedding_vectors]
            embedding_matrix = np.vstack(embeddings)
            return embedding_matrix
        except Exception as e:
            logger.error(f"Error extracting embedding matrix: {e!s}")
            return None

    def _compute_cluster_statistics(
        self, cluster_labels: np.ndarray, embedding_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute cluster statistics.

        Educational Note: Cluster statistics provide insights into
        the distribution and characteristics of each cluster.

        Args:
            cluster_labels: Cluster assignments
            embedding_matrix: Embedding matrix

        Returns:
            Dict[str, Any]: Cluster statistics
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
        Compute intra-cluster and inter-cluster distance metrics.

        Educational Note: Distance metrics assess cluster compactness
        and separation, key indicators of clustering quality.

        Args:
            embedding_matrix: Embedding matrix
            cluster_labels: Cluster assignments

        Returns:
            Dict[str, float]: Distance metrics
        """
        distance_metrics = {}

        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]  # Remove noise points

        if len(unique_labels) < 2:
            return distance_metrics

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
        if intra_cluster_distances:
            distance_metrics["mean_intra_cluster_distance"] = float(
                np.mean(intra_cluster_distances)
            )
            distance_metrics["std_intra_cluster_distance"] = float(
                np.std(intra_cluster_distances)
            )

        if inter_cluster_distances:
            distance_metrics["mean_inter_cluster_distance"] = float(
                np.mean(inter_cluster_distances)
            )
            distance_metrics["std_inter_cluster_distance"] = float(
                np.std(inter_cluster_distances)
            )

        if intra_cluster_distances and inter_cluster_distances:
            distance_metrics["separation_ratio"] = (
                distance_metrics["mean_inter_cluster_distance"]
                / distance_metrics["mean_intra_cluster_distance"]
            )

        return distance_metrics

    def _compute_separation_metrics(
        self, embedding_matrix: np.ndarray, cluster_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute cluster separation metrics.

        Educational Note: Separation metrics assess how well-separated
        clusters are from each other.

        Args:
            embedding_matrix: Embedding matrix
            cluster_labels: Cluster assignments

        Returns:
            Dict[str, float]: Separation metrics
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
        Compute cluster stability metrics.

        Educational Note: Stability metrics assess how stable cluster
        assignments are relative to cluster centroids.

        Args:
            embedding_matrix: Embedding matrix
            cluster_labels: Cluster assignments
            centroids: Cluster centroids

        Returns:
            Dict[str, float]: Stability metrics
        """
        stability_metrics = {}

        centroids_np = centroids.numpy()

        # Compute distances from each point to its assigned centroid
        point_to_centroid_distances = []
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Skip noise points
                centroid = centroids_np[label]
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
        Analyze cluster size distribution.

        Educational Note: Cluster size analysis helps identify
        potential issues like imbalanced clusters or outliers.

        Args:
            cluster_labels: Cluster assignments

        Returns:
            Dict[str, Any]: Cluster size analysis
        """
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)

        # Remove noise points (-1) from analysis
        valid_mask = unique_labels != -1
        valid_labels = unique_labels[valid_mask]
        valid_counts = counts[valid_mask]

        size_analysis = {
            "n_clusters": len(valid_labels),
            "cluster_sizes": {
                f"cluster_{label}": int(count)
                for label, count in zip(valid_labels, valid_counts, strict=False)
            },
            "size_statistics": {
                "mean_size": float(np.mean(valid_counts)),
                "std_size": float(np.std(valid_counts)),
                "min_size": int(np.min(valid_counts)),
                "max_size": int(np.max(valid_counts)),
                "median_size": float(np.median(valid_counts)),
            },
            "size_balance": {
                "coefficient_of_variation": float(
                    np.std(valid_counts) / np.mean(valid_counts)
                ),
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
        Compute Gini coefficient for cluster size distribution.

        Educational Note: Gini coefficient measures inequality in cluster sizes,
        with 0 indicating perfect equality and 1 indicating maximum inequality.

        Args:
            values: Array of cluster sizes

        Returns:
            float: Gini coefficient
        """
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)

        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def _analyze_embedding_statistics(
        self, embedding_vectors: List[EmbeddingVector], cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze embedding statistics across clusters.

        Educational Note: Embedding statistics help understand
        the distribution of features within and across clusters.

        Args:
            embedding_vectors: List of EmbeddingVector instances
            cluster_labels: Cluster assignments

        Returns:
            Dict[str, Any]: Embedding statistics analysis
        """
        embedding_matrix = self._extract_embedding_matrix(embedding_vectors)
        if embedding_matrix is None:
            return {}

        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]  # Remove noise points

        embedding_stats = {
            "global_statistics": {
                "mean": float(np.mean(embedding_matrix)),
                "std": float(np.std(embedding_matrix)),
                "min": float(np.min(embedding_matrix)),
                "max": float(np.max(embedding_matrix)),
            },
            "cluster_statistics": {},
        }

        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_embeddings = embedding_matrix[cluster_mask]

            embedding_stats["cluster_statistics"][f"cluster_{label}"] = {
                "mean": float(np.mean(cluster_embeddings)),
                "std": float(np.std(cluster_embeddings)),
                "min": float(np.min(cluster_embeddings)),
                "max": float(np.max(cluster_embeddings)),
            }

        return embedding_stats

    def _analyze_projection_distribution(
        self, mammography_images: List[MammographyImage], cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze projection type distribution across clusters.

        Educational Note: This analysis ensures clusters don't simply
        separate by projection type (CC vs MLO), which would indicate
        a trivial clustering solution.

        Args:
            mammography_images: List of MammographyImage instances
            cluster_labels: Cluster assignments

        Returns:
            Dict[str, Any]: Projection distribution analysis
        """
        if len(mammography_images) != len(cluster_labels):
            logger.warning("Mismatch between mammography images and cluster labels")
            return {}

        # Create mapping from image ID to projection type
        image_id_to_projection = {
            img.instance_id: img.projection_type for img in mammography_images
        }

        # Analyze projection distribution per cluster
        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]  # Remove noise points

        projection_analysis = {
            "cluster_projection_distributions": {},
            "overall_projection_distribution": {},
            "projection_balance_analysis": {},
        }

        # Overall projection distribution
        all_projections = [img.projection_type for img in mammography_images]
        projection_counts = Counter(all_projections)
        total_images = len(all_projections)

        for projection, count in projection_counts.items():
            projection_analysis["overall_projection_distribution"][projection] = {
                "count": count,
                "percentage": (count / total_images) * 100,
            }

        # Per-cluster projection distribution
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_image_ids = [
                mammography_images[i].instance_id for i in np.nonzero(cluster_mask)[0]
            ]
            cluster_projections = [
                image_id_to_projection[img_id] for img_id in cluster_image_ids
            ]

            cluster_projection_counts = Counter(cluster_projections)
            cluster_size = len(cluster_projections)

            projection_analysis["cluster_projection_distributions"][
                f"cluster_{label}"
            ] = {}
            for projection, count in cluster_projection_counts.items():
                projection_analysis["cluster_projection_distributions"][
                    f"cluster_{label}"
                ][projection] = {
                    "count": count,
                    "percentage": (count / cluster_size) * 100,
                }

        # Check for projection bias
        projection_analysis["potential_issues"] = []
        for label in unique_labels:
            cluster_dist = projection_analysis["cluster_projection_distributions"][
                f"cluster_{label}"
            ]
            for projection, stats in cluster_dist.items():
                if stats["percentage"] > 90:  # More than 90% of one projection type
                    projection_analysis["potential_issues"].append(
                        f"Cluster {label} is heavily biased toward {projection} projection ({stats['percentage']:.1f}%)"
                    )

        return projection_analysis

    def _analyze_laterality_distribution(
        self, mammography_images: List[MammographyImage], cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze laterality distribution across clusters.

        Educational Note: This analysis ensures clusters don't simply
        separate by laterality (L vs R), which would indicate
        a trivial clustering solution.

        Args:
            mammography_images: List of MammographyImage instances
            cluster_labels: Cluster assignments

        Returns:
            Dict[str, Any]: Laterality distribution analysis
        """
        if len(mammography_images) != len(cluster_labels):
            logger.warning("Mismatch between mammography images and cluster labels")
            return {}

        # Create mapping from image ID to laterality
        image_id_to_laterality = {
            img.instance_id: img.laterality for img in mammography_images
        }

        # Analyze laterality distribution per cluster
        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]  # Remove noise points

        laterality_analysis = {
            "cluster_laterality_distributions": {},
            "overall_laterality_distribution": {},
            "laterality_balance_analysis": {},
        }

        # Overall laterality distribution
        all_lateralities = [img.laterality for img in mammography_images]
        laterality_counts = Counter(all_lateralities)
        total_images = len(all_lateralities)

        for laterality, count in laterality_counts.items():
            laterality_analysis["overall_laterality_distribution"][laterality] = {
                "count": count,
                "percentage": (count / total_images) * 100,
            }

        # Per-cluster laterality distribution
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_image_ids = [
                mammography_images[i].instance_id for i in np.nonzero(cluster_mask)[0]
            ]
            cluster_lateralities = [
                image_id_to_laterality[img_id] for img_id in cluster_image_ids
            ]

            cluster_laterality_counts = Counter(cluster_lateralities)
            cluster_size = len(cluster_lateralities)

            laterality_analysis["cluster_laterality_distributions"][
                f"cluster_{label}"
            ] = {}
            for laterality, count in cluster_laterality_counts.items():
                laterality_analysis["cluster_laterality_distributions"][
                    f"cluster_{label}"
                ][laterality] = {
                    "count": count,
                    "percentage": (count / cluster_size) * 100,
                }

        # Check for laterality bias
        laterality_analysis["potential_issues"] = []
        for label in unique_labels:
            cluster_dist = laterality_analysis["cluster_laterality_distributions"][
                f"cluster_{label}"
            ]
            for laterality, stats in cluster_dist.items():
                if stats["percentage"] > 90:  # More than 90% of one laterality
                    laterality_analysis["potential_issues"].append(
                        f"Cluster {label} is heavily biased toward {laterality} laterality ({stats['percentage']:.1f}%)"
                    )

        return laterality_analysis

    def _analyze_intensity_histograms(
        self, _mammography_images: List[MammographyImage], _cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze intensity histograms across clusters.

        Educational Note: This analysis helps understand whether
        clusters capture different intensity patterns in the images.

        Args:
            mammography_images: List of MammographyImage instances
            cluster_labels: Cluster assignments

        Returns:
            Dict[str, Any]: Intensity histogram analysis
        """
        # This is a placeholder for intensity histogram analysis
        # In a full implementation, this would load and analyze pixel data
        intensity_analysis = {
            "note": "Intensity histogram analysis requires pixel data access",
            "recommendation": "Implement pixel data loading for full intensity analysis",
        }

        return intensity_analysis

    def _select_prototypes_by_centroid_distance(
        self,
        cluster_embeddings: np.ndarray,
        cluster_indices: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Select prototypes based on distance to cluster centroid.

        Educational Note: Centroid-based selection chooses samples
        closest to the cluster center as representative prototypes.

        Args:
            cluster_embeddings: Embeddings for the cluster
            cluster_indices: Indices of cluster samples
            n_samples: Number of prototypes to select

        Returns:
            np.ndarray: Selected prototype indices
        """
        # Compute cluster centroid
        centroid = cluster_embeddings.mean(axis=0)

        # Compute distances to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

        # Select closest samples
        closest_indices = np.argsort(distances)[:n_samples]

        return cluster_indices[closest_indices]

    def _select_prototypes_randomly(
        self, cluster_indices: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """
        Select prototypes randomly from cluster.

        Educational Note: Random selection provides unbiased
        representative samples from each cluster.

        Args:
            cluster_indices: Indices of cluster samples
            n_samples: Number of prototypes to select

        Returns:
            np.ndarray: Selected prototype indices
        """
        n_available = len(cluster_indices)
        n_select = min(n_samples, n_available)

        rng = np.random.default_rng(self.config.get("seed", 42))
        selected_indices = rng.choice(cluster_indices, size=n_select, replace=False)
        return selected_indices

    def _generate_evaluation_summary(
        self, evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate evaluation summary.

        Educational Note: This summary provides a concise overview
        of the clustering evaluation results.

        Args:
            evaluation_results: Complete evaluation results

        Returns:
            Dict[str, Any]: Evaluation summary
        """
        summary = {
            "evaluation_timestamp": evaluation_results["evaluation_timestamp"],
            "clustering_algorithm": evaluation_results["clustering_result"].algorithm,
            "n_clusters": len(
                np.unique(evaluation_results["clustering_result"].cluster_labels)
            ),
            "n_samples": len(evaluation_results["clustering_result"].cluster_labels),
        }

        # Add quality metrics summary
        if "quality_metrics" in evaluation_results:
            quality_metrics = evaluation_results["quality_metrics"]
            summary["quality_metrics"] = {
                "cluster_statistics_available": "cluster_statistics" in quality_metrics,
                "distance_metrics_available": "distance_metrics" in quality_metrics,
                "separation_metrics_available": "separation_metrics" in quality_metrics,
            }

        # Add sanity checks summary
        if "sanity_checks" in evaluation_results:
            sanity_checks = evaluation_results["sanity_checks"]
            summary["sanity_checks"] = {
                "checks_performed": list(sanity_checks.keys()),
                "potential_issues_found": sum(
                    len(check.get("potential_issues", []))
                    for check in sanity_checks.values()
                    if isinstance(check, dict) and "potential_issues" in check
                ),
            }

        # Add visual prototypes summary
        if "visual_prototypes" in evaluation_results:
            visual_prototypes = evaluation_results["visual_prototypes"]
            summary["visual_prototypes"] = {
                "clusters_with_prototypes": len(visual_prototypes),
                "total_prototypes": sum(
                    len(prototype["prototype_indices"])
                    for prototype in visual_prototypes.values()
                ),
            }

        return summary


def create_clustering_evaluator(config: Dict[str, Any]) -> ClusteringEvaluator:
    """
    Factory function to create a ClusteringEvaluator instance.

    Educational Note: This factory function provides a convenient way
    to create ClusteringEvaluator instances with validated configurations.

    Args:
        config: Evaluation configuration dictionary

    Returns:
        ClusteringEvaluator: Configured ClusteringEvaluator instance
    """
    return ClusteringEvaluator(config)


def evaluate_clustering(
    clustering_result: ClusteringResult,
    embedding_vectors: List[EmbeddingVector],
    config: Dict[str, Any],
    mammography_images: Optional[List[MammographyImage]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate clustering results.

    Educational Note: This function provides a simple interface for
    clustering evaluation without creating a ClusteringEvaluator instance.

    Args:
        clustering_result: ClusteringResult instance to evaluate
        embedding_vectors: List of EmbeddingVector instances
        config: Evaluation configuration dictionary
        mammography_images: Optional list of MammographyImage instances

    Returns:
        Dict[str, Any]: Evaluation results
    """
    evaluator = create_clustering_evaluator(config)
    return evaluator.evaluate_clustering(
        clustering_result, embedding_vectors, mammography_images
    )
