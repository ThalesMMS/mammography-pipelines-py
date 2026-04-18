"""
Cluster visualization module for mammography embedding analysis.

This module provides comprehensive visualization capabilities for clustering
results including UMAP plots, cluster montages, and metrics visualization
for the breast density exploration pipeline.

DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- UMAP provides non-linear dimensionality reduction for visualization
- Cluster montages enable qualitative validation of clustering results
- Metrics plots help assess clustering quality and algorithm comparison
- Publication-ready visualizations support research dissemination

Author: Research Team
Version: 1.0.0
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from ..clustering.clustering_result import ClusteringResult
from ..io.dicom import MammographyImage
from ..models.embeddings.embedding_vector import EmbeddingVector
from ..utils.embeddings import extract_embedding_matrix
from .cluster_montage import ClusterMontageMixin
from .cluster_plots import ClusterPlotMixin

# Configure logging for educational purposes
logger = logging.getLogger(__name__)

# Set matplotlib style for publication-ready plots
plt.style.use("default")
sns.set_palette("husl")

# Constants
RESEARCH_DISCLAIMER_TEXT = "Research purposes only - NOT for clinical use"


class ClusterVisualizer(ClusterPlotMixin, ClusterMontageMixin):
    """Cluster visualizer for mammography embedding analysis.

    This class provides methods for creating comprehensive visualizations
    of clustering results including UMAP plots, cluster montages, and
    metrics visualization for research and educational purposes.

    Educational Notes:
    - UMAP provides non-linear dimensionality reduction for visualization
    - Cluster montages enable qualitative validation of clustering results
    - Metrics plots help assess clustering quality and algorithm comparison
    - Publication-ready visualizations support research dissemination

    Attributes:
        config: Visualization configuration dictionary
        umap_model: UMAP model for dimensionality reduction
        umap_model_2d: Cached 2D UMAP reducer
        umap_model_3d: Cached 3D UMAP reducer
        pca_model: PCA model for linear dimensionality reduction
    """

    SUPPORTED_VISUALIZATIONS: ClassVar[tuple[str, ...]] = (
        "umap_2d",
        "umap_3d",
        "pca_2d",
        "cluster_montage",
        "metrics_plot",
        "cluster_size_plot",
        "embedding_heatmap",
    )

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cluster visualizer with configuration.

        Args:
            config: Visualization configuration dictionary

        Educational Note: Configuration validation ensures all required
        parameters are present and valid for visualization.
        """
        self.config = self._validate_config(config)

        # Set random seed for reproducibility
        if "seed" in self.config:
            torch.manual_seed(self.config["seed"])
            np.random.seed(self.config["seed"])

        # Initialize UMAP model
        self.umap_model = None
        self.umap_model_2d = None
        self.umap_model_3d = None

        # Initialize PCA model
        self.pca_model = None

        logger.info(f"Initialized ClusterVisualizer with config: {self.config}")

    def create_visualizations(
        self,
        clustering_result: ClusteringResult,
        embedding_vectors: List[EmbeddingVector],
        mammography_images: Optional[List[MammographyImage]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Create comprehensive visualizations for clustering results.

        Educational Note: This method demonstrates the complete visualization
        pipeline including dimensionality reduction, plotting, and saving.

        Args:
            clustering_result: ClusteringResult instance to visualize
            embedding_vectors: List of EmbeddingVector instances
            mammography_images: Optional list of MammographyImage instances
            output_dir: Optional output directory for saving plots. When
                omitted, plots are displayed interactively, no files are
                written, and plot helpers return None.

        Returns:
            Dict[str, Any]: Visualization results. The `output_files` mapping is
            populated only when a concrete `output_dir` is provided and files
            are written successfully.
        """
        visualization_results = {
            "clustering_result": clustering_result,
            "visualization_timestamp": datetime.now().isoformat(),
            "config": self.config,
            "output_files": {},
        }

        try:
            # Extract embedding matrix
            embedding_matrix = self._extract_embedding_matrix(embedding_vectors)
            if embedding_matrix is None:
                return visualization_results

            cluster_labels = clustering_result.cluster_labels.numpy()

            # Create UMAP 2D visualization
            if "umap_2d" in self.config["visualizations"]:
                umap_2d_path = self._create_umap_2d_plot(
                    embedding_matrix, cluster_labels, output_dir
                )
                if umap_2d_path:
                    visualization_results["output_files"]["umap_2d"] = str(umap_2d_path)

            # Create UMAP 3D visualization
            if "umap_3d" in self.config["visualizations"]:
                umap_3d_path = self._create_umap_3d_plot(
                    embedding_matrix, cluster_labels, output_dir
                )
                if umap_3d_path:
                    visualization_results["output_files"]["umap_3d"] = str(umap_3d_path)

            # Create PCA 2D visualization
            if "pca_2d" in self.config["visualizations"]:
                pca_2d_path = self._create_pca_2d_plot(
                    embedding_matrix, cluster_labels, output_dir
                )
                if pca_2d_path:
                    visualization_results["output_files"]["pca_2d"] = str(pca_2d_path)

            # Create cluster montage
            if (
                "cluster_montage" in self.config["visualizations"]
                and mammography_images
            ):
                montage_path = self._create_cluster_montage(
                    clustering_result, mammography_images, output_dir
                )
                if montage_path:
                    visualization_results["output_files"]["cluster_montage"] = str(
                        montage_path
                    )

            # Create metrics plot
            if "metrics_plot" in self.config["visualizations"]:
                metrics_path = self._create_metrics_plot(clustering_result, output_dir)
                if metrics_path:
                    visualization_results["output_files"]["metrics_plot"] = str(
                        metrics_path
                    )

            # Create cluster size plot
            if "cluster_size_plot" in self.config["visualizations"]:
                size_path = self._create_cluster_size_plot(cluster_labels, output_dir)
                if size_path:
                    visualization_results["output_files"]["cluster_size_plot"] = str(
                        size_path
                    )

            # Create embedding heatmap
            if "embedding_heatmap" in self.config["visualizations"]:
                heatmap_path = self._create_embedding_heatmap(
                    embedding_matrix, cluster_labels, output_dir
                )
                if heatmap_path:
                    visualization_results["output_files"]["embedding_heatmap"] = str(
                        heatmap_path
                    )

            logger.info("Successfully created all visualizations")
            return visualization_results

        except Exception as e:
            logger.error(f"Error creating visualizations: {e!s}")
            visualization_results["error"] = str(e)
            return visualization_results

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate visualization configuration.

        Educational Note: Configuration validation ensures all required
        parameters are present and within valid ranges.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Dict[str, Any]: Validated configuration
        """
        # Set default values for optional parameters
        config.setdefault("visualizations", list(self.SUPPORTED_VISUALIZATIONS))
        config.setdefault(
            "umap_params",
            {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "n_components": 2,
                "metric": "euclidean",
                "random_state": 42,
            },
        )
        config.setdefault(
            "pca_params", {"n_components": 2, "random_state": 42, "svd_solver": "auto"}
        )
        config.setdefault(
            "plot_params",
            {"figsize": (10, 8), "dpi": 300, "style": "whitegrid", "palette": "husl"},
        )
        config.setdefault(
            "montage_params",
            {"n_samples_per_cluster": 4, "image_size": (224, 224), "grid_size": (2, 2)},
        )
        config.setdefault("seed", 42)

        # Validate visualizations
        for viz in config["visualizations"]:
            if viz not in self.SUPPORTED_VISUALIZATIONS:
                logger.warning(f"Unknown visualization type: {viz}")

        return config

    def _extract_embedding_matrix(
        self, embedding_vectors: List[EmbeddingVector]
    ) -> Optional[np.ndarray]:
        """
        Convert a list of embedding vectors into a 2D NumPy array suitable for downstream visualization.

        Parameters:
            embedding_vectors (List[EmbeddingVector]): Sequence of embedding objects to convert.

        Returns:
            np.ndarray | None: 2D array of shape (n_samples, n_features) containing embeddings, or `None` if extraction fails.
        """
        return extract_embedding_matrix(embedding_vectors, logger=logger)


def create_cluster_visualizer(config: Dict[str, Any]) -> ClusterVisualizer:
    """
    Create a ClusterVisualizer configured from the provided visualization configuration.

    Parameters:
        config (Dict[str, Any]): Visualization configuration dictionary; it will be validated and augmented with defaults.

    Returns:
        ClusterVisualizer: Instance configured according to `config`.
    """
    return ClusterVisualizer(config)


def visualize_clustering(
    clustering_result: ClusteringResult,
    embedding_vectors: List[EmbeddingVector],
    config: Dict[str, Any],
    mammography_images: Optional[List[MammographyImage]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Visualize clustering results and return metadata about generated outputs.

    Parameters:
        clustering_result (ClusteringResult): Clustering output containing sample labels and metrics to visualize.
        embedding_vectors (List[EmbeddingVector]): Embedding vectors used for dimensionality reduction and plots.
        config (Dict[str, Any]): Visualization options controlling enabled plots and plotting parameters.
        mammography_images (Optional[List[MammographyImage]]): Optional list of images used for cluster montages; required for the `"cluster_montage"` visualization.
        output_dir (Optional[Union[str, Path]]): Optional directory where generated plot files will be saved. If omitted, plots are displayed interactively and no files are written.

    Returns:
        Dict[str, Any]: A dictionary containing visualization metadata and outputs, including:
            - `clustering_result`: the provided clustering result
            - `visualization_timestamp`: ISO-formatted timestamp of the run
            - `config`: effective configuration used
            - `output_files`: mapping of visualization identifiers to produced file paths; populated only when `output_dir` is provided
            - `error` (optional): error message if visualization failed
    """
    visualizer = create_cluster_visualizer(config)
    return visualizer.create_visualizations(
        clustering_result, embedding_vectors, mammography_images, output_dir
    )
