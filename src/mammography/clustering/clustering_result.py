"""
ClusteringResult model for clustering algorithm results representation.

This module defines the data structure for representing results from
clustering algorithms applied to ResNet-50 embeddings, including
evaluation metrics and uncertainty measures.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- This model represents the fourth step in our unsupervised learning pipeline
- It captures clustering results from various algorithms (K-means, GMM, HDBSCAN)
- Evaluation metrics enable comparison of clustering quality
- Uncertainty measures provide confidence estimates for cluster assignments

Author: Research Team
Version: 1.0.0
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


class ClusteringResult:
    """
    Represents results from clustering algorithm applied to embeddings.

    This class encapsulates clustering results from various algorithms,
    including cluster assignments, centroids, uncertainty measures,
    and evaluation metrics. It serves as the final output of our
    unsupervised learning pipeline.

    Educational Notes:
    - Clustering algorithms: K-means, GMM, HDBSCAN
    - Evaluation metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz
    - Uncertainty measures: Distance to centroids, probability distributions
    - Result validation: Ensures clustering quality and consistency

    Attributes:
        experiment_id (str): Unique identifier for clustering experiment
        algorithm (str): Clustering algorithm used
        cluster_labels (torch.Tensor): Cluster assignments for each embedding
        centroids (torch.Tensor): Cluster centroids (if applicable)
        uncertainty_scores (torch.Tensor): Uncertainty measures (if applicable)
        hyperparameters (dict): Algorithm-specific parameters
        metrics (dict): Evaluation metrics (Silhouette, DB, CH)
        created_at (datetime): Timestamp of clustering
    """

    # Define valid clustering algorithms
    VALID_ALGORITHMS = ["kmeans", "gmm", "hdbscan", "agglomerative"]

    # Define required evaluation metrics
    REQUIRED_METRICS = ["silhouette", "davies_bouldin", "calinski_harabasz"]

    # Define optional evaluation metrics
    OPTIONAL_METRICS = ["ari", "nmi", "homogeneity", "completeness", "v_measure"]

    def __init__(
        self,
        experiment_id: str,
        algorithm: str,
        cluster_labels: torch.Tensor,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        centroids: Optional[torch.Tensor] = None,
        uncertainty_scores: Optional[torch.Tensor] = None,
        embedding_ids: Optional[List[str]] = None,
        processing_time: Optional[float] = None,
        created_at: Optional[datetime] = None,
    ):
        """
        Initialize a ClusteringResult instance.

        Args:
            experiment_id: Unique identifier for clustering experiment
            algorithm: Clustering algorithm used
            cluster_labels: Cluster assignments for each embedding
            hyperparameters: Algorithm-specific parameters
            metrics: Evaluation metrics dictionary
            centroids: Cluster centroids (if applicable)
            uncertainty_scores: Uncertainty measures (if applicable)
            embedding_ids: List of embedding IDs (for traceability)
            processing_time: Time taken for clustering (seconds)
            created_at: Timestamp of clustering (default: now)

        Raises:
            ValueError: If validation rules are violated
            TypeError: If data types are incorrect
        """
        # Initialize core attributes with validation
        self.experiment_id = self._validate_experiment_id(experiment_id)
        self.algorithm = self._validate_algorithm(algorithm)
        self.cluster_labels = self._validate_cluster_labels(cluster_labels)
        self.hyperparameters = self._validate_hyperparameters(hyperparameters)
        self.metrics = self._validate_metrics(metrics)
        self.centroids = centroids  # Optional, validated separately
        self.uncertainty_scores = uncertainty_scores  # Optional, validated separately
        self.embedding_ids = embedding_ids or []
        self.processing_time = processing_time or 0.0
        self.created_at = created_at or datetime.now()

        # Initialize tracking attributes
        self.validation_errors: List[str] = []
        self.updated_at = datetime.now()

        # Validate optional attributes if provided
        if self.centroids is not None:
            self._validate_centroids()

        if self.uncertainty_scores is not None:
            self._validate_uncertainty_scores()

        # Validate consistency between components
        self._validate_result_consistency()

        # Log creation for educational purposes
        logger.info(
            f"Created ClusteringResult: {self.experiment_id} using {self.algorithm}"
        )

    def _validate_experiment_id(self, experiment_id: str) -> str:
        """
        Validate experiment ID.

        Educational Note: Experiment ID enables tracking and comparison
        of different clustering experiments and their results.

        Args:
            experiment_id: Experiment identifier to validate

        Returns:
            str: Validated experiment ID

        Raises:
            ValueError: If experiment ID is invalid
            TypeError: If experiment ID is not a string
        """
        if not isinstance(experiment_id, str):
            raise TypeError(
                f"experiment_id must be a string, got {type(experiment_id)}"
            )

        if not experiment_id.strip():
            raise ValueError("experiment_id cannot be empty or whitespace")

        return experiment_id.strip()

    def _validate_algorithm(self, algorithm: str) -> str:
        """
        Validate clustering algorithm.

        Educational Note: Algorithm validation ensures only supported
        clustering methods are used in our pipeline.

        Args:
            algorithm: Algorithm name to validate

        Returns:
            str: Validated algorithm name

        Raises:
            ValueError: If algorithm is invalid
            TypeError: If algorithm is not a string
        """
        if not isinstance(algorithm, str):
            raise TypeError(f"algorithm must be a string, got {type(algorithm)}")

        if algorithm not in self.VALID_ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {self.VALID_ALGORITHMS}, got {algorithm}"
            )

        return algorithm

    def _validate_cluster_labels(self, cluster_labels: torch.Tensor) -> torch.Tensor:
        """
        Validate cluster labels.

        Educational Note: Cluster labels validation ensures proper format
        for downstream analysis and visualization.

        Args:
            cluster_labels: Cluster assignments to validate

        Returns:
            torch.Tensor: Validated cluster labels

        Raises:
            ValueError: If cluster labels are invalid
            TypeError: If cluster labels are not a tensor
        """
        if not isinstance(cluster_labels, torch.Tensor):
            raise TypeError(
                f"cluster_labels must be a torch.Tensor, got {type(cluster_labels)}"
            )

        # Check tensor dimensions (should be 1D)
        if cluster_labels.ndim != 1:
            raise ValueError(f"cluster_labels must be 1D, got {cluster_labels.ndim}D")

        # Check for valid cluster labels (non-negative integers)
        if cluster_labels.dtype not in [torch.int32, torch.int64, torch.long]:
            logger.warning(
                f"Converting cluster_labels from {cluster_labels.dtype} to long"
            )
            cluster_labels = cluster_labels.long()

        # Check for negative cluster labels (except -1 for HDBSCAN noise)
        if self.algorithm != "hdbscan" and torch.any(cluster_labels < 0):
            raise ValueError("cluster_labels cannot contain negative values")

        # Check for reasonable number of clusters
        n_clusters = len(torch.unique(cluster_labels))
        if n_clusters > len(cluster_labels) // 2:
            logger.warning(
                f"Number of clusters ({n_clusters}) seems high relative to data size ({len(cluster_labels)})"
            )

        return cluster_labels

    def _validate_hyperparameters(
        self, hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate hyperparameters.

        Educational Note: Hyperparameter validation ensures algorithm-specific
        parameters are present and valid for reproducibility.

        Args:
            hyperparameters: Hyperparameters dictionary to validate

        Returns:
            Dict[str, Any]: Validated hyperparameters

        Raises:
            ValueError: If hyperparameters are invalid
            TypeError: If hyperparameters are not a dictionary
        """
        if not isinstance(hyperparameters, dict):
            raise TypeError(
                f"hyperparameters must be a dictionary, got {type(hyperparameters)}"
            )

        # Validate algorithm-specific hyperparameters
        if self.algorithm == "kmeans":
            required_params = ["n_clusters", "random_state"]
            for param in required_params:
                if param not in hyperparameters:
                    raise ValueError(
                        f"Missing required hyperparameter for K-means: {param}"
                    )

        elif self.algorithm == "gmm":
            required_params = ["n_components", "random_state"]
            for param in required_params:
                if param not in hyperparameters:
                    raise ValueError(
                        f"Missing required hyperparameter for GMM: {param}"
                    )

        elif self.algorithm == "hdbscan":
            required_params = ["min_cluster_size", "min_samples"]
            for param in required_params:
                if param not in hyperparameters:
                    raise ValueError(
                        f"Missing required hyperparameter for HDBSCAN: {param}"
                    )

        return hyperparameters

    def _validate_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Validate evaluation metrics.

        Educational Note: Metrics validation ensures all required
        evaluation measures are present and within valid ranges.

        Args:
            metrics: Metrics dictionary to validate

        Returns:
            Dict[str, float]: Validated metrics

        Raises:
            ValueError: If metrics are invalid
            TypeError: If metrics are not a dictionary
        """
        if not isinstance(metrics, dict):
            raise TypeError(f"metrics must be a dictionary, got {type(metrics)}")

        # Check for required metrics
        for metric in self.REQUIRED_METRICS:
            if metric not in metrics:
                raise ValueError(f"Missing required metric: {metric}")

            if not isinstance(metrics[metric], (int, float)):
                raise TypeError(
                    f"Metric {metric} must be a number, got {type(metrics[metric])}"
                )

            # Validate metric ranges
            if metric == "silhouette":
                if not (-1 <= metrics[metric] <= 1):
                    raise ValueError(
                        f"Silhouette score must be between -1 and 1, got {metrics[metric]}"
                    )

            elif metric == "davies_bouldin":
                if metrics[metric] < 0:
                    raise ValueError(
                        f"Davies-Bouldin score must be non-negative, got {metrics[metric]}"
                    )

            elif metric == "calinski_harabasz":
                if metrics[metric] < 0:
                    raise ValueError(
                        f"Calinski-Harabasz score must be non-negative, got {metrics[metric]}"
                    )

        return metrics

    def _validate_centroids(self) -> None:
        """
        Validate cluster centroids.

        Educational Note: Centroids validation ensures proper format
        for algorithms that produce cluster centers.

        Raises:
            ValueError: If centroids are invalid
        """
        if not isinstance(self.centroids, torch.Tensor):
            raise TypeError(
                f"centroids must be a torch.Tensor, got {type(self.centroids)}"
            )

        # Check tensor dimensions (should be 2D: n_clusters x n_features)
        if self.centroids.ndim != 2:
            raise ValueError(f"centroids must be 2D, got {self.centroids.ndim}D")

        # Check number of clusters matches labels
        n_clusters_labels = len(torch.unique(self.cluster_labels))
        n_clusters_centroids = self.centroids.shape[0]

        if n_clusters_centroids != n_clusters_labels:
            error_msg = f"Number of centroids ({n_clusters_centroids}) doesn't match unique labels ({n_clusters_labels})"
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)

    def _validate_uncertainty_scores(self) -> None:
        """
        Validate uncertainty scores.

        Educational Note: Uncertainty validation ensures proper format
        for algorithms that provide confidence measures.

        Raises:
            ValueError: If uncertainty scores are invalid
        """
        if not isinstance(self.uncertainty_scores, torch.Tensor):
            raise TypeError(
                f"uncertainty_scores must be a torch.Tensor, got {type(self.uncertainty_scores)}"
            )

        # Check tensor dimensions (should be 1D)
        if self.uncertainty_scores.ndim != 1:
            raise ValueError(
                f"uncertainty_scores must be 1D, got {self.uncertainty_scores.ndim}D"
            )

        # Check length matches cluster labels
        if len(self.uncertainty_scores) != len(self.cluster_labels):
            error_msg = f"uncertainty_scores length ({len(self.uncertainty_scores)}) doesn't match cluster_labels length ({len(self.cluster_labels)})"
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)

        # Check for valid uncertainty values (0 to 1 for probabilities)
        if torch.any(self.uncertainty_scores < 0) or torch.any(
            self.uncertainty_scores > 1
        ):
            logger.warning("uncertainty_scores contains values outside [0, 1] range")

    def _validate_result_consistency(self) -> None:
        """
        Validate consistency between different result components.

        Educational Note: Consistency validation ensures all components
        of the clustering result are compatible and coherent.

        Raises:
            ValueError: If components are inconsistent
        """
        # Check embedding_ids length matches cluster_labels
        if self.embedding_ids and len(self.embedding_ids) != len(self.cluster_labels):
            error_msg = f"embedding_ids length ({len(self.embedding_ids)}) doesn't match cluster_labels length ({len(self.cluster_labels)})"
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)

        # Check for reasonable cluster distribution
        unique_labels, counts = torch.unique(self.cluster_labels, return_counts=True)
        min_cluster_size = torch.min(counts).item()
        max_cluster_size = torch.max(counts).item()

        if min_cluster_size < 2:
            logger.warning(
                f"Some clusters have very few samples (minimum: {min_cluster_size})"
            )

        if max_cluster_size > len(self.cluster_labels) * 0.8:
            logger.warning(
                f"Some clusters are very large (maximum: {max_cluster_size})"
            )

    def get_cluster_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for each cluster.

        Educational Note: Cluster summary provides insights into
        the distribution and characteristics of each cluster.

        Returns:
            Dict[str, Any]: Dictionary containing cluster summaries
        """
        unique_labels, counts = torch.unique(self.cluster_labels, return_counts=True)

        cluster_summary = {}
        for label, count in zip(unique_labels, counts, strict=False):
            cluster_summary[f"cluster_{label.item()}"] = {
                "size": count.item(),
                "percentage": (count.item() / len(self.cluster_labels)) * 100,
                "label": label.item(),
            }

        return cluster_summary

    def get_result_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of clustering results.

        Educational Note: This summary provides a complete overview
        of the clustering experiment for analysis and reporting.

        Returns:
            Dict[str, Any]: Dictionary containing result summary
        """
        return {
            "experiment_id": self.experiment_id,
            "algorithm": self.algorithm,
            "n_clusters": len(torch.unique(self.cluster_labels)),
            "n_samples": len(self.cluster_labels),
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "cluster_summary": self.get_cluster_summary(),
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat(),
            "validation_errors": self.validation_errors,
        }

    def save_result(self, file_path: Union[str, Path]) -> bool:
        """
        Save clustering result to file.

        Educational Note: Result saving enables persistence of clustering
        experiments for later analysis and comparison.

        Args:
            file_path: Path where to save the result

        Returns:
            bool: True if saving successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save clustering result data
            torch.save(
                {
                    "experiment_id": self.experiment_id,
                    "algorithm": self.algorithm,
                    "cluster_labels": self.cluster_labels,
                    "centroids": self.centroids,
                    "uncertainty_scores": self.uncertainty_scores,
                    "hyperparameters": self.hyperparameters,
                    "metrics": self.metrics,
                    "embedding_ids": self.embedding_ids,
                    "processing_time": self.processing_time,
                    "created_at": self.created_at.isoformat(),
                    "validation_errors": self.validation_errors,
                },
                file_path,
            )

            logger.info(f"Saved ClusteringResult to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving ClusteringResult to {file_path}: {e!s}")
            return False

    @classmethod
    def load_result(cls, file_path: Union[str, Path]) -> "ClusteringResult":
        """
        Load clustering result from file.

        Educational Note: This class method enables loading of previously
        saved clustering results for analysis and comparison.

        Args:
            file_path: Path to the saved result file

        Returns:
            ClusteringResult: Loaded instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is corrupted or invalid
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Clustering result file not found: {file_path}"
                )

            # Load clustering result data
            data = torch.load(file_path, map_location="cpu")

            # Parse creation timestamp
            created_at = (
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else None
            )

            # Create ClusteringResult instance
            clustering_result = cls(
                experiment_id=data["experiment_id"],
                algorithm=data["algorithm"],
                cluster_labels=data["cluster_labels"],
                hyperparameters=data["hyperparameters"],
                metrics=data["metrics"],
                centroids=data.get("centroids"),
                uncertainty_scores=data.get("uncertainty_scores"),
                embedding_ids=data.get("embedding_ids", []),
                processing_time=data.get("processing_time", 0.0),
                created_at=created_at,
            )

            # Restore validation errors if any
            if data.get("validation_errors"):
                clustering_result.validation_errors = data["validation_errors"]

            logger.info(f"Loaded ClusteringResult from {file_path}")
            return clustering_result

        except Exception as e:
            raise ValueError(f"Error loading ClusteringResult from {file_path}: {e!s}")

    def __repr__(self) -> str:
        """String representation for debugging and logging."""
        return (
            f"ClusteringResult("
            f"experiment_id='{self.experiment_id}', "
            f"algorithm='{self.algorithm}', "
            f"n_clusters={len(torch.unique(self.cluster_labels))}, "
            f"n_samples={len(self.cluster_labels)})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Clustering Result: {self.experiment_id}\n"
            f"Algorithm: {self.algorithm}\n"
            f"Clusters: {len(torch.unique(self.cluster_labels))}\n"
            f"Samples: {len(self.cluster_labels)}\n"
            f"Silhouette: {self.metrics.get('silhouette', 'N/A'):.3f}\n"
            f"Davies-Bouldin: {self.metrics.get('davies_bouldin', 'N/A'):.3f}\n"
            f"Calinski-Harabasz: {self.metrics.get('calinski_harabasz', 'N/A'):.3f}"
        )


def create_clustering_result_from_algorithm(
    experiment_id: str,
    algorithm: str,
    cluster_labels: torch.Tensor,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, float],
    centroids: Optional[torch.Tensor] = None,
    uncertainty_scores: Optional[torch.Tensor] = None,
    embedding_ids: Optional[List[str]] = None,
    processing_time: float = 0.0,
) -> ClusteringResult:
    """
    Create a ClusteringResult instance from algorithm output.

    Educational Note: This factory function demonstrates how to create
    ClusteringResult instances from clustering algorithm outputs,
    enabling standardized result creation.

    Args:
        experiment_id: Unique experiment identifier
        algorithm: Clustering algorithm used
        cluster_labels: Cluster assignments
        hyperparameters: Algorithm parameters
        metrics: Evaluation metrics
        centroids: Cluster centroids (if applicable)
        uncertainty_scores: Uncertainty measures (if applicable)
        embedding_ids: List of embedding IDs
        processing_time: Time taken for clustering

    Returns:
        ClusteringResult: Created instance

    Raises:
        ValueError: If any parameter is invalid
    """
    # Create ClusteringResult instance
    clustering_result = ClusteringResult(
        experiment_id=experiment_id,
        algorithm=algorithm,
        cluster_labels=cluster_labels,
        hyperparameters=hyperparameters,
        metrics=metrics,
        centroids=centroids,
        uncertainty_scores=uncertainty_scores,
        embedding_ids=embedding_ids,
        processing_time=processing_time,
    )

    return clustering_result
