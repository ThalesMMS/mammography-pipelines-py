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

from collections import Counter
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..clustering.clustering_result import ClusteringResult
from ..io.dicom import MammographyImage
from ..models.embeddings.embedding_vector import EmbeddingVector
from ..utils.embeddings import extract_embedding_matrix
from .clustering_analysis import ClusteringAnalysisMixin
from .clustering_metrics import ClusteringMetricsMixin
from .clustering_prototypes import ClusteringPrototypeMixin

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


class ClusteringEvaluator(
    ClusteringMetricsMixin, ClusteringAnalysisMixin, ClusteringPrototypeMixin
):
    """Clustering evaluator for mammography embedding analysis.

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
        self.rng = np.random.default_rng(self.config.get("seed", 42))

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
        Validate and normalize the evaluator configuration by applying defaults and warning about unsupported entries.

        Parameters:
            config (Dict[str, Any]): Configuration options for the evaluator. May include keys
                "metrics", "sanity_checks", "visual_prototypes", and "seed". Missing keys will be
                populated with module defaults.

        Returns:
            Dict[str, Any]: The validated configuration dictionary with defaults applied. Unknown
            metric or sanity-check names will be left in the config but logged as warnings.
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

    def _generate_evaluation_summary(
        self, evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Builds a concise summary of a clustering evaluation suitable for quick reporting.

        Parameters:
            evaluation_results (Dict[str, Any]): Full results produced by evaluate_clustering; must include
                "evaluation_timestamp" and "clustering_result" with attributes `algorithm` and
                `cluster_labels`. May also contain optional keys "quality_metrics", "sanity_checks",
                and "visual_prototypes".

        Returns:
            Dict[str, Any]: Summary dictionary containing:
                - evaluation_timestamp: the evaluation timestamp from the input results.
                - clustering_algorithm: name of the clustering algorithm.
                - n_clusters: number of unique cluster labels.
                - n_samples: total number of samples evaluated.
                - quality_metrics (optional): flags indicating presence of `cluster_statistics`,
                  `distance_metrics`, and `separation_metrics`.
                - sanity_checks (optional): list of performed checks and a count of detected
                  `potential_issues`.
                - visual_prototypes (optional): counts of clusters with prototypes and total prototypes.
        """
        labels = evaluation_results["clustering_result"].cluster_labels
        unique_clusters = np.unique(labels)
        unique_clusters = unique_clusters[unique_clusters != -1]
        summary = {
            "evaluation_timestamp": evaluation_results["evaluation_timestamp"],
            "clustering_algorithm": evaluation_results["clustering_result"].algorithm,
            "n_clusters": len(unique_clusters),
            "n_samples": len(labels),
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
                    len(prototype.get("prototype_indices", []))
                    for prototype in visual_prototypes.values()
                    if isinstance(prototype, dict)
                ),
            }

        return summary


def create_clustering_evaluator(config: Dict[str, Any]) -> ClusteringEvaluator:
    """
    Create a configured ClusteringEvaluator.

    Parameters:
        config (Dict[str, Any]): Evaluation configuration dictionary; will be validated and defaults applied.

    Returns:
        ClusteringEvaluator: ClusteringEvaluator instance configured with the provided `config`.
    """
    return ClusteringEvaluator(config)


def evaluate_clustering(
    clustering_result: ClusteringResult,
    embedding_vectors: List[EmbeddingVector],
    config: Dict[str, Any],
    mammography_images: Optional[List[MammographyImage]] = None,
) -> Dict[str, Any]:
    """
    Evaluate clustering results and return a structured evaluation dictionary.

    Runs the full clustering evaluation pipeline according to `config`, producing quality metrics, sanity checks, visual prototypes, and a summary.

    Parameters:
        clustering_result: The clustering result object to evaluate.
        embedding_vectors: The embedding vectors corresponding to the clustered samples.
        config: Evaluation configuration dictionary (controls metrics, sanity checks, prototype selection, seed, etc.).
        mammography_images: Optional list of mammography images used by sanity checks or prototype selection when available.

    Returns:
        A dictionary with evaluation outputs (includes `clustering_result`, `evaluation_timestamp`, `config`, and, when produced, `quality_metrics`, `sanity_checks`, `visual_prototypes`, `summary`; on failure the dictionary will include an `error` string).
    """
    evaluator = create_clustering_evaluator(config)
    return evaluator.evaluate_clustering(
        clustering_result, embedding_vectors, mammography_images
    )
