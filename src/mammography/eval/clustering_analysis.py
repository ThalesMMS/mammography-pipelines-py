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

# Configure logging for educational purposes
logger = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


class ClusteringAnalysisMixin:
    """Sanity-check analyses for clustering evaluation.

    This mixin expects the composed evaluator to also provide
    ``_analyze_cluster_sizes``; ClusteringEvaluator supplies that method through
    ClusteringMetricsMixin.
    """

    def _perform_sanity_checks(
        self,
        clustering_result: ClusteringResult,
        embedding_vectors: List[EmbeddingVector],
        mammography_images: Optional[List[MammographyImage]] = None,
    ) -> Dict[str, Any]:
        """
        Run the configured sanity checks on clustering results, embedding vectors, and optional mammography image metadata.

        Performs a set of analyses enabled in self.config["sanity_checks"] and aggregates their outputs into a dictionary. Image-based analyses are only run when mammography_images is provided. Any errors during checks are logged and the function returns whatever results were collected up to the failure.

        Parameters:
            clustering_result (ClusteringResult): Result object containing cluster_labels.
            embedding_vectors (List[EmbeddingVector]): Embeddings corresponding to items in clustering_result.
            mammography_images (Optional[List[MammographyImage]]): Optional image metadata used by projection, laterality, and intensity checks.

        Returns:
            Dict[str, Any]: Mapping of enabled sanity-check names to their results. Possible keys include:
                - "cluster_size_analysis"
                - "embedding_statistics"
                - "projection_distribution"
                - "laterality_distribution"
                - "intensity_histograms"
            Values are analysis-specific dictionaries; when an analysis is skipped or fails, its key will be absent or map to an empty dict.
        """
        sanity_checks = {}

        try:
            cluster_labels = _to_numpy(clustering_result.cluster_labels)

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

        except Exception:
            logger.exception("Error in sanity checks")
            return sanity_checks

    def _extract_embedding_matrix(
        self, embedding_vectors: List[EmbeddingVector]
    ) -> Optional[np.ndarray]:
        """
        Convert a list of EmbeddingVector objects into a NumPy embedding matrix.

        Parameters:
            embedding_vectors (List[EmbeddingVector]): EmbeddingVector instances to convert; the input order is preserved and becomes the corresponding row order in the output matrix.

        Returns:
            np.ndarray or None: A 2D NumPy array where each row is an embedding, or `None` if extraction failed.
        """
        return extract_embedding_matrix(embedding_vectors, logger=logger)

    def _analyze_embedding_statistics(
        self, embedding_vectors: List[EmbeddingVector], cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute global and per-cluster summary statistics (mean, std, min, max) for the provided embeddings.

        If embedding matrix extraction fails, an empty dict is returned.

        Parameters:
            embedding_vectors (List[EmbeddingVector]): EmbeddingVector objects to be converted into an embedding matrix.
            cluster_labels (np.ndarray): Array of cluster assignments for each embedding; entries equal to -1 are treated as noise and excluded from per-cluster statistics.

        Returns:
            Dict[str, Any]: A dictionary with the following structure:
                {
                    "global_statistics": {"mean": float, "std": float, "min": float, "max": float},
                    "cluster_statistics": {
                        "cluster_<label>": {"mean": float, "std": float, "min": float, "max": float},
                        ...
                    }
                }
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

    def _analyze_attribute_distribution(
        self,
        mammography_images: List[MammographyImage],
        cluster_labels: np.ndarray,
        *,
        attribute_name: str,
        cluster_key: str,
        overall_key: str,
        issue_label: str,
    ) -> Dict[str, Any]:
        """
        Summarize an image metadata attribute overall and per cluster.
        """
        if len(mammography_images) != len(cluster_labels):
            logger.warning("Mismatch between mammography images and cluster labels")
            return {}

        image_id_to_attr = {
            img.instance_id: getattr(img, attribute_name) for img in mammography_images
        }
        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]  # Remove noise points

        analysis = {
            cluster_key: {},
            overall_key: {},
        }

        all_values = [getattr(img, attribute_name) for img in mammography_images]
        value_counts = Counter(all_values)
        total_images = len(all_values)

        for value, count in value_counts.items():
            analysis[overall_key][value] = {
                "count": count,
                "percentage": (count / total_images) * 100,
            }

        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_image_ids = [
                mammography_images[i].instance_id for i in np.nonzero(cluster_mask)[0]
            ]
            cluster_values = [image_id_to_attr[img_id] for img_id in cluster_image_ids]

            cluster_value_counts = Counter(cluster_values)
            cluster_size = len(cluster_values)

            analysis[cluster_key][f"cluster_{label}"] = {}
            for value, count in cluster_value_counts.items():
                analysis[cluster_key][f"cluster_{label}"][value] = {
                    "count": count,
                    "percentage": (count / cluster_size) * 100,
                }

        analysis["potential_issues"] = []
        for label in unique_labels:
            cluster_dist = analysis[cluster_key][f"cluster_{label}"]
            for value, stats in cluster_dist.items():
                if stats["percentage"] > 90:
                    analysis["potential_issues"].append(
                        f"Cluster {label} is heavily biased toward {value} {issue_label} ({stats['percentage']:.1f}%)"
                    )

        return analysis

    def _analyze_projection_distribution(
        self, mammography_images: List[MammographyImage], cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Summarizes projection-type distributions overall and per cluster and flags clusters dominated by a single projection type.
        """
        return self._analyze_attribute_distribution(
            mammography_images,
            cluster_labels,
            attribute_name="projection_type",
            cluster_key="cluster_projection_distributions",
            overall_key="overall_projection_distribution",
            issue_label="projection",
        )

    def _analyze_laterality_distribution(
        self, mammography_images: List[MammographyImage], cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Summarizes laterality (e.g., left/right) distributions overall and per cluster and flags clusters dominated by a single laterality.
        """
        return self._analyze_attribute_distribution(
            mammography_images,
            cluster_labels,
            attribute_name="laterality",
            cluster_key="cluster_laterality_distributions",
            overall_key="overall_laterality_distribution",
            issue_label="laterality",
        )

    def _analyze_intensity_histograms(
        self, _mammography_images: List[MammographyImage], _cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Provide a placeholder indicating that intensity histogram analysis requires access to image pixel data.

        Parameters:
            _mammography_images (List[MammographyImage]): List of images aligned with `_cluster_labels`.
            _cluster_labels (np.ndarray): Cluster assignments corresponding index-wise to `_mammography_images`.

        Returns:
            Dict[str, Any]: A dictionary with a `note` explaining pixel-data requirement and a `recommendation` for implementing pixel loading to perform full intensity histogram analysis.
        """
        # This is a placeholder for intensity histogram analysis
        # In a full implementation, this would load and analyze pixel data
        intensity_analysis = {
            "note": "Intensity histogram analysis requires pixel data access",
            "recommendation": "Implement pixel data loading for full intensity analysis",
        }

        return intensity_analysis
