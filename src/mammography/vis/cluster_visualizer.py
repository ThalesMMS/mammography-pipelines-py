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
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import torch
import umap

from ..clustering.clustering_result import ClusteringResult
from ..io_dicom.mammography_image import MammographyImage
from ..models.embeddings.embedding_vector import EmbeddingVector
from ..utils.numpy_warnings import suppress_numpy_matmul_warnings, resolve_pca_svd_solver

# Configure logging for educational purposes
logger = logging.getLogger(__name__)

# Set matplotlib style for publication-ready plots
plt.style.use("default")
sns.set_palette("husl")

# Constants
RESEARCH_DISCLAIMER_TEXT = "Research purposes only - NOT for clinical use"


class ClusterVisualizer:
    """
    Cluster visualizer for mammography embedding analysis.

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
        pca_model: PCA model for linear dimensionality reduction
    """

    # Supported visualization types
    SUPPORTED_VISUALIZATIONS = [
        "umap_2d",
        "umap_3d",
        "pca_2d",
        "cluster_montage",
        "metrics_plot",
        "cluster_size_plot",
        "embedding_heatmap",
    ]

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
            output_dir: Optional output directory for saving plots

        Returns:
            Dict[str, Any]: Visualization results and file paths
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
        config.setdefault("visualizations", self.SUPPORTED_VISUALIZATIONS)
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
        Extract embedding matrix from embedding vectors.

        Educational Note: This method converts embedding vectors to a matrix
        format suitable for dimensionality reduction and visualization.

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

    def _create_umap_2d_plot(
        self,
        embedding_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        Create 2D UMAP visualization.

        Educational Note: UMAP provides non-linear dimensionality reduction
        that preserves both local and global structure of the data.

        Args:
            embedding_matrix: Input embedding matrix
            cluster_labels: Cluster assignments
            output_dir: Optional output directory

        Returns:
            Path: Path to saved plot, None if failed
        """
        try:
            # Fit UMAP model
            umap_params = self.config["umap_params"].copy()
            umap_params["n_components"] = 2

            if self.umap_model is None:
                self.umap_model = umap.UMAP(**umap_params)

            # Transform embeddings
            with suppress_numpy_matmul_warnings():
                umap_embeddings = self.umap_model.fit_transform(embedding_matrix)

            # Create plot
            plt.figure(figsize=self.config["plot_params"]["figsize"])

            # Plot clusters
            unique_labels = np.unique(cluster_labels)
            colors = sns.color_palette(
                self.config["plot_params"]["palette"], len(unique_labels)
            )

            for i, label in enumerate(unique_labels):
                if label == -1:  # Noise points
                    mask = cluster_labels == label
                    plt.scatter(
                        umap_embeddings[mask, 0],
                        umap_embeddings[mask, 1],
                        c="black",
                        marker="x",
                        s=20,
                        alpha=0.6,
                        label="Noise",
                    )
                else:
                    mask = cluster_labels == label
                    plt.scatter(
                        umap_embeddings[mask, 0],
                        umap_embeddings[mask, 1],
                        c=[colors[i]],
                        label=f"Cluster {label}",
                        s=30,
                        alpha=0.7,
                    )

            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.title("UMAP 2D Visualization of Mammography Embeddings")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add research disclaimer
            plt.figtext(
                0.02,
                0.02,
                RESEARCH_DISCLAIMER_TEXT,
                fontsize=8,
                style="italic",
                alpha=0.7,
            )

            # Save plot
            if output_dir:
                output_path = (
                    Path(output_dir)
                    / f"umap_2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    output_path,
                    dpi=self.config["plot_params"]["dpi"],
                    bbox_inches="tight",
                )
                plt.close()
                return output_path
            else:
                plt.show()
                return None

        except Exception as e:
            logger.error(f"Error creating UMAP 2D plot: {e!s}")
            return None

    def _create_umap_3d_plot(
        self,
        embedding_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        Create 3D UMAP visualization.

        Educational Note: 3D UMAP provides additional dimensionality
        for better separation of complex cluster structures.

        Args:
            embedding_matrix: Input embedding matrix
            cluster_labels: Cluster assignments
            output_dir: Optional output directory

        Returns:
            Path: Path to saved plot, None if failed
        """
        try:
            # Fit UMAP model
            umap_params = self.config["umap_params"].copy()
            umap_params["n_components"] = 3

            umap_3d_model = umap.UMAP(**umap_params)
            with suppress_numpy_matmul_warnings():
                umap_embeddings = umap_3d_model.fit_transform(embedding_matrix)

            # Create 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

            # Plot clusters
            unique_labels = np.unique(cluster_labels)
            colors = sns.color_palette(
                self.config["plot_params"]["palette"], len(unique_labels)
            )

            for i, label in enumerate(unique_labels):
                if label == -1:  # Noise points
                    mask = cluster_labels == label
                    ax.scatter(
                        umap_embeddings[mask, 0],
                        umap_embeddings[mask, 1],
                        umap_embeddings[mask, 2],
                        c="black",
                        marker="x",
                        s=20,
                        alpha=0.6,
                        label="Noise",
                    )
                else:
                    mask = cluster_labels == label
                    ax.scatter(
                        umap_embeddings[mask, 0],
                        umap_embeddings[mask, 1],
                        umap_embeddings[mask, 2],
                        c=[colors[i]],
                        label=f"Cluster {label}",
                        s=30,
                        alpha=0.7,
                    )

            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_zlabel("UMAP 3")
            ax.set_title("UMAP 3D Visualization of Mammography Embeddings")
            ax.legend()

            # Add research disclaimer
            fig.text(
                0.02,
                0.02,
                RESEARCH_DISCLAIMER_TEXT,
                fontsize=8,
                style="italic",
                alpha=0.7,
            )

            # Save plot
            if output_dir:
                output_path = (
                    Path(output_dir)
                    / f"umap_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    output_path,
                    dpi=self.config["plot_params"]["dpi"],
                    bbox_inches="tight",
                )
                plt.close()
                return output_path
            else:
                plt.show()
                return None

        except Exception as e:
            logger.error(f"Error creating UMAP 3D plot: {e!s}")
            return None

    def _create_pca_2d_plot(
        self,
        embedding_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        Create 2D PCA visualization.

        Educational Note: PCA provides linear dimensionality reduction
        that preserves maximum variance in the data.

        Args:
            embedding_matrix: Input embedding matrix
            cluster_labels: Cluster assignments
            output_dir: Optional output directory

        Returns:
            Path: Path to saved plot, None if failed
        """
        try:
            # Fit PCA model
            pca_params = self.config["pca_params"].copy()
            pca_params["n_components"] = 2
            pca_params["svd_solver"] = resolve_pca_svd_solver(
                embedding_matrix.shape[0],
                embedding_matrix.shape[1],
                pca_params["n_components"],
                pca_params.get("svd_solver"),
            )

            if self.pca_model is None:
                self.pca_model = PCA(**pca_params)

            # Transform embeddings
            with suppress_numpy_matmul_warnings():
                pca_embeddings = self.pca_model.fit_transform(embedding_matrix)

            # Create plot
            plt.figure(figsize=self.config["plot_params"]["figsize"])

            # Plot clusters
            unique_labels = np.unique(cluster_labels)
            colors = sns.color_palette(
                self.config["plot_params"]["palette"], len(unique_labels)
            )

            for i, label in enumerate(unique_labels):
                if label == -1:  # Noise points
                    mask = cluster_labels == label
                    plt.scatter(
                        pca_embeddings[mask, 0],
                        pca_embeddings[mask, 1],
                        c="black",
                        marker="x",
                        s=20,
                        alpha=0.6,
                        label="Noise",
                    )
                else:
                    mask = cluster_labels == label
                    plt.scatter(
                        pca_embeddings[mask, 0],
                        pca_embeddings[mask, 1],
                        c=[colors[i]],
                        label=f"Cluster {label}",
                        s=30,
                        alpha=0.7,
                    )

            # Add explained variance information
            explained_var = self.pca_model.explained_variance_ratio_
            plt.xlabel(f"PC1 ({explained_var[0]:.1%} variance)")
            plt.ylabel(f"PC2 ({explained_var[1]:.1%} variance)")
            plt.title("PCA 2D Visualization of Mammography Embeddings")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add research disclaimer
            plt.figtext(
                0.02,
                0.02,
                RESEARCH_DISCLAIMER_TEXT,
                fontsize=8,
                style="italic",
                alpha=0.7,
            )

            # Save plot
            if output_dir:
                output_path = (
                    Path(output_dir)
                    / f"pca_2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    output_path,
                    dpi=self.config["plot_params"]["dpi"],
                    bbox_inches="tight",
                )
                plt.close()
                return output_path
            else:
                plt.show()
                return None

        except Exception as e:
            logger.error(f"Error creating PCA 2D plot: {e!s}")
            return None

    def _create_cluster_montage(
        self,
        clustering_result: ClusteringResult,
        mammography_images: List[MammographyImage],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        Create cluster montage visualization.

        Educational Note: Cluster montages enable qualitative validation
        of clustering results by examining representative images.

        Args:
            clustering_result: ClusteringResult instance
            mammography_images: List of MammographyImage instances
            output_dir: Optional output directory

        Returns:
            Path: Path to saved montage, None if failed
        """
        try:
            cluster_labels = clustering_result.cluster_labels.numpy()
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise points

            n_samples_per_cluster = self.config["montage_params"][
                "n_samples_per_cluster"
            ]
            grid_size = self.config["montage_params"]["grid_size"]
            image_size = self.config["montage_params"]["image_size"]

            # Create figure with subplots for each cluster
            n_clusters = len(unique_labels)
            fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 4 * n_clusters))
            if n_clusters == 1:
                axes = [axes]

            for i, label in enumerate(unique_labels):
                # Get cluster samples
                cluster_mask = cluster_labels == label
                cluster_indices = np.nonzero(cluster_mask)[0]

                # Select random samples
                n_select = min(n_samples_per_cluster, len(cluster_indices))
                rng = np.random.default_rng(42)
                selected_indices = rng.choice(
                    cluster_indices, size=n_select, replace=False
                )

                # Create montage for this cluster
                montage_images = []
                for _ in selected_indices:
                    # This is a placeholder - in a full implementation, you would load the actual image
                    # For now, create a placeholder image
                    rng = np.random.default_rng(42)
                    placeholder = rng.random(image_size) * 255
                    montage_images.append(placeholder)

                # Create grid of images
                grid_images = []
                for j in range(grid_size[0]):
                    row_images = []
                    for k in range(grid_size[1]):
                        if j * grid_size[1] + k < len(montage_images):
                            row_images.append(montage_images[j * grid_size[1] + k])
                        else:
                            row_images.append(np.zeros(image_size))
                    grid_images.append(np.hstack(row_images))

                montage = np.vstack(grid_images)

                # Plot montage
                axes[i].imshow(montage, cmap="gray")
                axes[i].set_title(f"Cluster {label} ({len(cluster_indices)} samples)")
                axes[i].axis("off")

            plt.suptitle("Cluster Montages - Representative Samples")

            # Add research disclaimer
            fig.text(
                0.02,
                0.02,
                RESEARCH_DISCLAIMER_TEXT,
                fontsize=8,
                style="italic",
                alpha=0.7,
            )

            # Save montage
            if output_dir:
                output_path = (
                    Path(output_dir)
                    / f"cluster_montage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    output_path,
                    dpi=self.config["plot_params"]["dpi"],
                    bbox_inches="tight",
                )
                plt.close()
                return output_path
            else:
                plt.show()
                return None

        except Exception as e:
            logger.error(f"Error creating cluster montage: {e!s}")
            return None

    def _create_metrics_plot(
        self,
        clustering_result: ClusteringResult,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        Create metrics visualization plot.

        Educational Note: Metrics plots help assess clustering quality
        and compare different algorithms.

        Args:
            clustering_result: ClusteringResult instance
            output_dir: Optional output directory

        Returns:
            Path: Path to saved plot, None if failed
        """
        try:
            metrics = clustering_result.metrics

            # Create subplots for different metrics
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            # Plot silhouette score
            if "silhouette" in metrics:
                axes[0].bar(
                    ["Silhouette Score"], [metrics["silhouette"]], color="skyblue"
                )
                axes[0].set_title("Silhouette Score")
                axes[0].set_ylabel("Score")
                axes[0].set_ylim(0, 1)

            # Plot Davies-Bouldin score
            if "davies_bouldin" in metrics:
                axes[1].bar(
                    ["Davies-Bouldin Score"],
                    [metrics["davies_bouldin"]],
                    color="lightcoral",
                )
                axes[1].set_title("Davies-Bouldin Score")
                axes[1].set_ylabel("Score")
                axes[1].set_ylim(0, max(metrics["davies_bouldin"], 1))

            # Plot Calinski-Harabasz score
            if "calinski_harabasz" in metrics:
                axes[2].bar(
                    ["Calinski-Harabasz Score"],
                    [metrics["calinski_harabasz"]],
                    color="lightgreen",
                )
                axes[2].set_title("Calinski-Harabasz Score")
                axes[2].set_ylabel("Score")

            # Plot cluster sizes
            cluster_labels = clustering_result.cluster_labels.numpy()
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise points
            counts = counts[unique_labels != -1]

            axes[3].bar(
                [f"Cluster {label}" for label in unique_labels], counts, color="gold"
            )
            axes[3].set_title("Cluster Sizes")
            axes[3].set_ylabel("Number of Samples")
            axes[3].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.suptitle("Clustering Metrics Summary", y=1.02)

            # Add research disclaimer
            fig.text(
                0.02,
                0.02,
                RESEARCH_DISCLAIMER_TEXT,
                fontsize=8,
                style="italic",
                alpha=0.7,
            )

            # Save plot
            if output_dir:
                output_path = (
                    Path(output_dir)
                    / f"metrics_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    output_path,
                    dpi=self.config["plot_params"]["dpi"],
                    bbox_inches="tight",
                )
                plt.close()
                return output_path
            else:
                plt.show()
                return None

        except Exception as e:
            logger.error(f"Error creating metrics plot: {e!s}")
            return None

    def _create_cluster_size_plot(
        self, cluster_labels: np.ndarray, output_dir: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """
        Create cluster size distribution plot.

        Educational Note: Cluster size analysis helps identify
        potential issues like imbalanced clusters.

        Args:
            cluster_labels: Cluster assignments
            output_dir: Optional output directory

        Returns:
            Path: Path to saved plot, None if failed
        """
        try:
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise points
            counts = counts[unique_labels != -1]

            plt.figure(figsize=self.config["plot_params"]["figsize"])

            # Create bar plot
            bars = plt.bar(
                [f"Cluster {label}" for label in unique_labels],
                counts,
                color=sns.color_palette(
                    self.config["plot_params"]["palette"], len(unique_labels)
                ),
            )

            # Add value labels on bars
            for bar, count in zip(bars, counts, strict=False):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(count),
                    ha="center",
                    va="bottom",
                )

            plt.title("Cluster Size Distribution")
            plt.xlabel("Cluster")
            plt.ylabel("Number of Samples")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis="y")

            # Add research disclaimer
            plt.figtext(
                0.02,
                0.02,
                RESEARCH_DISCLAIMER_TEXT,
                fontsize=8,
                style="italic",
                alpha=0.7,
            )

            # Save plot
            if output_dir:
                output_path = (
                    Path(output_dir)
                    / f"cluster_size_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    output_path,
                    dpi=self.config["plot_params"]["dpi"],
                    bbox_inches="tight",
                )
                plt.close()
                return output_path
            else:
                plt.show()
                return None

        except Exception as e:
            logger.error(f"Error creating cluster size plot: {e!s}")
            return None

    def _create_embedding_heatmap(
        self,
        embedding_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        Create embedding heatmap visualization.

        Educational Note: Embedding heatmaps show the distribution
        of embedding values across samples and features.

        Args:
            embedding_matrix: Input embedding matrix
            cluster_labels: Cluster assignments
            output_dir: Optional output directory

        Returns:
            Path: Path to saved plot, None if failed
        """
        try:
            # Sample a subset of embeddings for visualization
            n_samples = min(100, embedding_matrix.shape[0])
            n_features = min(50, embedding_matrix.shape[1])

            # Randomly sample
            rng = np.random.default_rng(42)
            sample_indices = rng.choice(
                embedding_matrix.shape[0], size=n_samples, replace=False
            )
            feature_indices = rng.choice(
                embedding_matrix.shape[1], size=n_features, replace=False
            )

            sample_matrix = embedding_matrix[sample_indices][:, feature_indices]
            sample_labels = cluster_labels[sample_indices]

            # Sort by cluster
            sort_indices = np.argsort(sample_labels)
            sample_matrix = sample_matrix[sort_indices]
            sample_labels = sample_labels[sort_indices]

            plt.figure(figsize=(12, 8))

            # Create heatmap
            sns.heatmap(
                sample_matrix,
                cmap="viridis",
                cbar=True,
                xticklabels=False,
                yticklabels=False,
            )

            plt.title("Embedding Heatmap (Sample of Features)")
            plt.xlabel("Embedding Features (Sample)")
            plt.ylabel("Samples (Sorted by Cluster)")

            # Add cluster boundaries
            unique_labels = np.unique(sample_labels)
            unique_labels = unique_labels[unique_labels != -1]

            for label in unique_labels:
                mask = sample_labels == label
                if np.any(mask):
                    start_idx = np.nonzero(mask)[0][0]
                    end_idx = np.nonzero(mask)[0][-1]
                    plt.axhline(y=start_idx, color="red", linestyle="--", alpha=0.7)
                    plt.axhline(y=end_idx, color="red", linestyle="--", alpha=0.7)

            # Add research disclaimer
            plt.figtext(
                0.02,
                0.02,
                RESEARCH_DISCLAIMER_TEXT,
                fontsize=8,
                style="italic",
                alpha=0.7,
            )

            # Save plot
            if output_dir:
                output_path = (
                    Path(output_dir)
                    / f"embedding_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    output_path,
                    dpi=self.config["plot_params"]["dpi"],
                    bbox_inches="tight",
                )
                plt.close()
                return output_path
            else:
                plt.show()
                return None

        except Exception as e:
            logger.error(f"Error creating embedding heatmap: {e!s}")
            return None


def create_cluster_visualizer(config: Dict[str, Any]) -> ClusterVisualizer:
    """
    Factory function to create a ClusterVisualizer instance.

    Educational Note: This factory function provides a convenient way
    to create ClusterVisualizer instances with validated configurations.

    Args:
        config: Visualization configuration dictionary

    Returns:
        ClusterVisualizer: Configured ClusterVisualizer instance
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
    Convenience function to visualize clustering results.

    Educational Note: This function provides a simple interface for
    clustering visualization without creating a ClusterVisualizer instance.

    Args:
        clustering_result: ClusteringResult instance to visualize
        embedding_vectors: List of EmbeddingVector instances
        config: Visualization configuration dictionary
        mammography_images: Optional list of MammographyImage instances
        output_dir: Optional output directory for saving plots

    Returns:
        Dict[str, Any]: Visualization results
    """
    visualizer = create_cluster_visualizer(config)
    return visualizer.create_visualizations(
        clustering_result, embedding_vectors, mammography_images, output_dir
    )
