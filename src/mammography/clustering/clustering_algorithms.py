"""
Clustering algorithms module for mammography embedding analysis.

This module provides comprehensive clustering capabilities including K-means,
Gaussian Mixture Models (GMM), and HDBSCAN algorithms with PCA dimensionality
reduction and evaluation metrics computation.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- Clustering algorithms discover patterns in high-dimensional embeddings
- PCA dimensionality reduction improves clustering performance
- Multiple algorithms enable comparison and robustness assessment
- Evaluation metrics assess clustering quality without ground truth

Author: Research Team
Version: 1.0.0
"""

from datetime import datetime
import logging
import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import hdbscan
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import torch

from ..models.embeddings.embedding_vector import EmbeddingVector
from ..utils.numpy_warnings import (
    resolve_pca_svd_solver,
    suppress_numpy_matmul_warnings,
)
from .clustering_result import ClusteringResult, create_clustering_result_from_algorithm

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


class ClusteringAlgorithms:
    """
    Clustering algorithms for mammography embedding analysis.

    This class provides methods for applying various clustering algorithms
    to mammography embeddings, including dimensionality reduction and
    evaluation metrics computation.

    Educational Notes:
    - Different algorithms have different strengths and assumptions
    - PCA dimensionality reduction improves clustering performance
    - Evaluation metrics assess clustering quality without ground truth
    - Multiple algorithms enable comparison and robustness assessment

    Attributes:
        config: Clustering configuration dictionary
        pca_model: PCA model for dimensionality reduction
        scaler: StandardScaler for feature normalization
        supported_algorithms: List of supported clustering algorithms
    """

    # Supported clustering algorithms
    SUPPORTED_ALGORITHMS: ClassVar[List[str]] = ["kmeans", "gmm", "hdbscan", "agglomerative"]

    # Required evaluation metrics
    REQUIRED_METRICS: ClassVar[List[str]] = ["silhouette", "davies_bouldin", "calinski_harabasz"]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize clustering algorithms with configuration.

        Args:
            config: Clustering configuration dictionary

        Educational Note: Configuration validation ensures all required
        parameters are present and valid for clustering.

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = self._validate_config(config)

        # Set random seed for reproducibility
        if "seed" in self.config:
            torch.manual_seed(self.config["seed"])
            np.random.seed(self.config["seed"])

        # Initialize PCA model
        self.pca_model = None

        # Initialize scaler
        self.scaler = StandardScaler()

        logger.info(f"Initialized ClusteringAlgorithms with config: {self.config}")

    def cluster_embeddings(
        self, embedding_vectors: List[EmbeddingVector]
    ) -> Optional[ClusteringResult]:
        """
        Cluster embeddings using the configured algorithm.

        Educational Note: This method demonstrates the complete clustering
        pipeline including dimensionality reduction, clustering, and evaluation.

        Args:
            embedding_vectors: List of EmbeddingVector instances

        Returns:
            ClusteringResult: Clustering result if successful, None otherwise

        Raises:
            ValueError: If embedding vectors are invalid
        """
        start_time = time.time()

        try:
            # Validate input embeddings
            if not self._validate_embeddings(embedding_vectors):
                return None

            # Extract embedding matrix
            embedding_matrix = self._extract_embedding_matrix(embedding_vectors)
            if embedding_matrix is None:
                return None

            # Apply dimensionality reduction
            reduced_embeddings = self._apply_dimensionality_reduction(embedding_matrix)
            if reduced_embeddings is None:
                return None

            # Apply clustering algorithm
            cluster_labels, centroids, uncertainty_scores = (
                self._apply_clustering_algorithm(reduced_embeddings)
            )
            if cluster_labels is None:
                return None

            # Compute evaluation metrics
            metrics = self._compute_evaluation_metrics(
                reduced_embeddings, cluster_labels
            )
            if metrics is None:
                return None

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create experiment ID
            experiment_id = self._generate_experiment_id()

            # Extract embedding IDs
            embedding_ids = [emb.image_id for emb in embedding_vectors]

            # Create ClusteringResult instance
            clustering_result = create_clustering_result_from_algorithm(
                experiment_id=experiment_id,
                algorithm=self.config["algorithm"],
                cluster_labels=torch.tensor(cluster_labels, dtype=torch.long),
                hyperparameters=self.config.get("hyperparameters", {}),
                metrics=metrics,
                centroids=centroids,
                uncertainty_scores=uncertainty_scores,
                embedding_ids=embedding_ids,
                processing_time=processing_time,
            )

            logger.info(
                f"Successfully clustered {len(embedding_vectors)} embeddings using {self.config['algorithm']} in {processing_time:.3f}s"
            )
            return clustering_result

        except Exception as e:
            logger.error(f"Error clustering embeddings: {e!s}")
            return None

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate clustering configuration.

        Educational Note: Configuration validation ensures all required
        parameters are present and within valid ranges.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Dict[str, Any]: Validated configuration

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required parameters
        required_params = ["algorithm", "pca_dimensions"]
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required configuration parameter: {param}")

        # Validate algorithm
        algorithm = config["algorithm"]
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

        # Validate PCA dimensions
        pca_dimensions = config["pca_dimensions"]
        if not isinstance(pca_dimensions, int) or pca_dimensions <= 0:
            raise ValueError("pca_dimensions must be a positive integer")

        if pca_dimensions < 2:
            raise ValueError("pca_dimensions must be at least 2")

        # Set default values for optional parameters
        config.setdefault("n_clusters", 4)
        config.setdefault("evaluation_metrics", self.REQUIRED_METRICS)
        config.setdefault("seed", 42)
        config.setdefault("hyperparameters", {})
        config.setdefault("pca_svd_solver", "auto")

        # Set algorithm-specific default hyperparameters
        algorithm = config["algorithm"]
        hyperparameters = config["hyperparameters"]

        if algorithm == "kmeans":
            hyperparameters.setdefault("n_clusters", config["n_clusters"])
            hyperparameters.setdefault("random_state", config["seed"])
            hyperparameters.setdefault("n_init", 10)
            hyperparameters.setdefault("max_iter", 300)

        elif algorithm == "gmm":
            hyperparameters.setdefault("n_components", config["n_clusters"])
            hyperparameters.setdefault("random_state", config["seed"])
            hyperparameters.setdefault("covariance_type", "full")
            hyperparameters.setdefault("max_iter", 100)

        elif algorithm == "hdbscan":
            hyperparameters.setdefault("min_cluster_size", 10)
            hyperparameters.setdefault("min_samples", 5)
            hyperparameters.setdefault("random_state", config["seed"])

        elif algorithm == "agglomerative":
            hyperparameters.setdefault("n_clusters", config["n_clusters"])
            hyperparameters.setdefault("linkage", "ward")

        config["hyperparameters"] = hyperparameters

        return config

    def _validate_embeddings(self, embedding_vectors: List[EmbeddingVector]) -> bool:
        """
        Validate embedding vectors for clustering.

        Educational Note: Embedding validation ensures all vectors
        are in the correct format and have consistent dimensions.

        Args:
            embedding_vectors: List of EmbeddingVector instances

        Returns:
            bool: True if embeddings are valid, False otherwise
        """
        if not embedding_vectors:
            logger.error("No embedding vectors provided")
            return False

        # Check for None values
        if any(emb is None for emb in embedding_vectors):
            logger.error("Some embedding vectors are None")
            return False

        # Check embedding dimensions
        expected_dim = embedding_vectors[0].embedding.shape[0]
        for i, emb in enumerate(embedding_vectors):
            if emb.embedding.shape[0] != expected_dim:
                logger.error(
                    f"Embedding {i} has inconsistent dimension: {emb.embedding.shape[0]} vs {expected_dim}"
                )
                return False

        # Check for minimum number of embeddings
        if len(embedding_vectors) < 2:
            logger.error("Need at least 2 embeddings for clustering")
            return False

        return True

    def _extract_embedding_matrix(
        self, embedding_vectors: List[EmbeddingVector]
    ) -> Optional[np.ndarray]:
        """
        Extract embedding matrix from embedding vectors.

        Educational Note: This method converts a list of embedding vectors
        into a matrix format suitable for clustering algorithms.
        Optimized to use torch.stack before numpy conversion for efficiency.

        Args:
            embedding_vectors: List of EmbeddingVector instances

        Returns:
            np.ndarray: Embedding matrix (n_samples, n_features), None if failed
        """
        try:
            # Stack tensors first (more efficient than individual conversions)
            embeddings_tensor = torch.stack(
                [emb.embedding.cpu() for emb in embedding_vectors], dim=0
            )

            # Convert to numpy once
            embedding_matrix = embeddings_tensor.numpy()

            logger.debug(
                f"Extracted embedding matrix with shape: {embedding_matrix.shape}"
            )
            return embedding_matrix

        except Exception as e:
            logger.error(f"Error extracting embedding matrix: {e!s}")
            return None

    def _apply_dimensionality_reduction(
        self, embedding_matrix: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Apply PCA dimensionality reduction to embeddings.

        Educational Note: PCA reduces dimensionality while preserving
        maximum variance, improving clustering performance and efficiency.

        Args:
            embedding_matrix: Input embedding matrix

        Returns:
            np.ndarray: Reduced embedding matrix, None if failed
        """
        try:
            pca_dimensions = self.config["pca_dimensions"]

            # Check if PCA dimensions are valid
            if pca_dimensions >= embedding_matrix.shape[1]:
                logger.warning(
                    f"PCA dimensions ({pca_dimensions}) >= embedding dimensions ({embedding_matrix.shape[1]}), skipping PCA"
                )
                return embedding_matrix

            # Initialize PCA model if not already done
            if self.pca_model is None:
                solver = resolve_pca_svd_solver(
                    embedding_matrix.shape[0],
                    embedding_matrix.shape[1],
                    pca_dimensions,
                    self.config.get("pca_svd_solver"),
                )
                self.pca_model = PCA(
                    n_components=pca_dimensions,
                    random_state=self.config["seed"],
                    svd_solver=solver,
                )

            # Fit PCA and transform embeddings
            with suppress_numpy_matmul_warnings():
                reduced_embeddings = self.pca_model.fit_transform(embedding_matrix)

            # Log explained variance
            explained_variance_ratio = np.sum(self.pca_model.explained_variance_ratio_)
            logger.info(
                f"Applied PCA: {embedding_matrix.shape[1]} -> {pca_dimensions} dimensions, explained variance: {explained_variance_ratio:.3f}"
            )

            return reduced_embeddings

        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {e!s}")
            return None

    def _apply_clustering_algorithm(
        self, embeddings: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply the configured clustering algorithm.

        Educational Note: Different algorithms have different strengths:
        - K-means: Fast, works well with spherical clusters
        - GMM: Handles overlapping clusters, provides uncertainty
        - HDBSCAN: Discovers natural cluster structure
        - Agglomerative: Hierarchical clustering

        Args:
            embeddings: Reduced embedding matrix

        Returns:
            Tuple of (cluster_labels, centroids, uncertainty_scores)
        """
        algorithm = self.config["algorithm"]
        hyperparameters = self.config["hyperparameters"]

        try:
            if algorithm == "kmeans":
                return self._apply_kmeans(embeddings, hyperparameters)
            elif algorithm == "gmm":
                return self._apply_gmm(embeddings, hyperparameters)
            elif algorithm == "hdbscan":
                return self._apply_hdbscan(embeddings, hyperparameters)
            elif algorithm == "agglomerative":
                return self._apply_agglomerative(embeddings, hyperparameters)
            else:
                raise ValueError(f"Unknown clustering algorithm: {algorithm}")

        except Exception as e:
            logger.error(f"Error applying clustering algorithm {algorithm}: {e!s}")
            return None, None, None

    def _apply_kmeans(
        self, embeddings: np.ndarray, hyperparameters: Dict[str, Any]
    ) -> Tuple[np.ndarray, torch.Tensor, None]:
        """
        Apply K-means clustering.

        Educational Note: K-means is a centroid-based algorithm that
        partitions data into k clusters by minimizing within-cluster variance.

        Args:
            embeddings: Input embedding matrix
            hyperparameters: K-means hyperparameters

        Returns:
            Tuple of (cluster_labels, centroids, None)
        """
        # Apply K-means
        kmeans = KMeans(**hyperparameters)
        with suppress_numpy_matmul_warnings():
            cluster_labels = kmeans.fit_predict(embeddings)

        # Get centroids
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        logger.info(
            f"Applied K-means clustering with {hyperparameters['n_clusters']} clusters"
        )
        return cluster_labels, centroids, None

    def _apply_gmm(
        self, embeddings: np.ndarray, hyperparameters: Dict[str, Any]
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Apply Gaussian Mixture Model clustering.

        Educational Note: GMM assumes data is generated from a mixture
        of Gaussian distributions and provides uncertainty measures.

        Args:
            embeddings: Input embedding matrix
            hyperparameters: GMM hyperparameters

        Returns:
            Tuple of (cluster_labels, centroids, uncertainty_scores)
        """
        # Apply GMM
        gmm = GaussianMixture(**hyperparameters)
        with suppress_numpy_matmul_warnings():
            cluster_labels = gmm.fit_predict(embeddings)

        # Get centroids (means of components)
        centroids = torch.tensor(gmm.means_, dtype=torch.float32)

        # Get uncertainty scores (negative log-likelihood)
        with suppress_numpy_matmul_warnings():
            uncertainty_scores = torch.tensor(
                -gmm.score_samples(embeddings), dtype=torch.float32
            )

        logger.info(
            f"Applied GMM clustering with {hyperparameters['n_components']} components"
        )
        return cluster_labels, centroids, uncertainty_scores

    def _apply_hdbscan(
        self, embeddings: np.ndarray, hyperparameters: Dict[str, Any]
    ) -> Tuple[np.ndarray, None, torch.Tensor]:
        """
        Apply HDBSCAN clustering.

        Educational Note: HDBSCAN is a density-based algorithm that
        discovers clusters of varying densities and provides uncertainty measures.

        Args:
            embeddings: Input embedding matrix
            hyperparameters: HDBSCAN hyperparameters

        Returns:
            Tuple of (cluster_labels, None, uncertainty_scores)
        """
        # Apply HDBSCAN
        hdbscan_clusterer = hdbscan.HDBSCAN(**hyperparameters)
        with suppress_numpy_matmul_warnings():
            cluster_labels = hdbscan_clusterer.fit_predict(embeddings)

        # Get uncertainty scores (outlier scores)
        uncertainty_scores = torch.tensor(
            hdbscan_clusterer.outlier_scores_, dtype=torch.float32
        )

        # Handle case where no clusters are found
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logger.info(f"Applied HDBSCAN clustering with {n_clusters} clusters")

        return cluster_labels, None, uncertainty_scores

    def _apply_agglomerative(
        self, embeddings: np.ndarray, hyperparameters: Dict[str, Any]
    ) -> Tuple[np.ndarray, None, None]:
        """
        Apply Agglomerative clustering.

        Educational Note: Agglomerative clustering builds clusters
        hierarchically by merging the closest clusters iteratively.

        Args:
            embeddings: Input embedding matrix
            hyperparameters: Agglomerative hyperparameters

        Returns:
            Tuple of (cluster_labels, None, None)
        """
        # Apply Agglomerative clustering
        agglomerative = AgglomerativeClustering(**hyperparameters)
        with suppress_numpy_matmul_warnings():
            cluster_labels = agglomerative.fit_predict(embeddings)

        logger.info(
            f"Applied Agglomerative clustering with {hyperparameters['n_clusters']} clusters"
        )
        return cluster_labels, None, None

    def _compute_evaluation_metrics(
        self, embeddings: np.ndarray, cluster_labels: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Compute evaluation metrics for clustering results.

        Educational Note: These metrics assess clustering quality without
        ground truth labels, helping compare different algorithms.

        Args:
            embeddings: Reduced embedding matrix
            cluster_labels: Cluster assignments

        Returns:
            Dict[str, float]: Evaluation metrics, None if failed
        """
        try:
            metrics = {}

            # Check if we have enough clusters for metrics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            if n_clusters < 2:
                logger.warning("Not enough clusters for evaluation metrics")
                return None

            # Compute Silhouette score
            if "silhouette" in self.config["evaluation_metrics"]:
                try:
                    with suppress_numpy_matmul_warnings():
                        metrics["silhouette"] = silhouette_score(
                            embeddings, cluster_labels
                        )
                except Exception as e:
                    logger.warning(f"Could not compute Silhouette score: {e!s}")
                    metrics["silhouette"] = -1.0

            # Compute Davies-Bouldin score
            if "davies_bouldin" in self.config["evaluation_metrics"]:
                try:
                    with suppress_numpy_matmul_warnings():
                        metrics["davies_bouldin"] = davies_bouldin_score(
                            embeddings, cluster_labels
                        )
                except Exception as e:
                    logger.warning(f"Could not compute Davies-Bouldin score: {e!s}")
                    metrics["davies_bouldin"] = float("inf")

            # Compute Calinski-Harabasz score
            if "calinski_harabasz" in self.config["evaluation_metrics"]:
                try:
                    with suppress_numpy_matmul_warnings():
                        metrics["calinski_harabasz"] = calinski_harabasz_score(
                            embeddings, cluster_labels
                        )
                except Exception as e:
                    logger.warning(f"Could not compute Calinski-Harabasz score: {e!s}")
                    metrics["calinski_harabasz"] = 0.0

            logger.info(f"Computed evaluation metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error computing evaluation metrics: {e!s}")
            return None

    def _generate_experiment_id(self) -> str:
        """
        Generate unique experiment ID for clustering result.

        Educational Note: Experiment IDs enable tracking and comparison
        of different clustering experiments.

        Returns:
            str: Unique experiment ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algorithm = self.config["algorithm"]
        pca_dims = self.config["pca_dimensions"]
        n_clusters = self.config.get("n_clusters", "auto")

        return f"{algorithm}_pca{pca_dims}_k{n_clusters}_{timestamp}"

    def get_pca_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the PCA model.

        Educational Note: PCA information helps understand the dimensionality
        reduction process and explained variance.

        Returns:
            Dict[str, Any]: PCA information, None if PCA not fitted
        """
        if self.pca_model is None:
            return None

        return {
            "n_components": self.pca_model.n_components_,
            "explained_variance_ratio": self.pca_model.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(
                np.sum(self.pca_model.explained_variance_ratio_)
            ),
            "singular_values": self.pca_model.singular_values_.tolist(),
        }


def create_clustering_algorithms(config: Dict[str, Any]) -> ClusteringAlgorithms:
    """
    Factory function to create a ClusteringAlgorithms instance.

    Educational Note: This factory function provides a convenient way
    to create ClusteringAlgorithms instances with validated configurations.

    Args:
        config: Clustering configuration dictionary

    Returns:
        ClusteringAlgorithms: Configured ClusteringAlgorithms instance
    """
    return ClusteringAlgorithms(config)


def cluster_embeddings(
    embedding_vectors: List[EmbeddingVector], config: Dict[str, Any]
) -> Optional[ClusteringResult]:
    """
    Convenience function to cluster embeddings.

    Educational Note: This function provides a simple interface for
    clustering embeddings without creating a ClusteringAlgorithms instance.

    Args:
        embedding_vectors: List of EmbeddingVector instances
        config: Clustering configuration dictionary

    Returns:
        ClusteringResult: Clustering result if successful, None otherwise
    """
    clusterer = create_clustering_algorithms(config)
    return clusterer.cluster_embeddings(embedding_vectors)
