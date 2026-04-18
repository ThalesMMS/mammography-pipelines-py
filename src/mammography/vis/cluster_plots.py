# ruff: noqa
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

from ..clustering.clustering_result import ClusteringResult
from .dimensionality import project_pca, project_umap

# Configure logging for educational purposes
logger = logging.getLogger(__name__)

# Constants
RESEARCH_DISCLAIMER_TEXT = "Research purposes only - NOT for clinical use"


class ClusterPlotMixin:
    def _save_or_show_plot(
        self,
        output_dir: Optional[Union[str, Path]],
        filename_prefix: str,
    ) -> Optional[Path]:
        """Save the current Matplotlib figure to output_dir or display it."""
        if output_dir:
            output_path = (
                Path(output_dir)
                / f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                output_path,
                dpi=self.config["plot_params"]["dpi"],
                bbox_inches="tight",
            )
            plt.close()
            return output_path
        plt.show()
        plt.close()
        return None

    def _create_umap_2d_plot(
        self,
        embedding_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        Create and optionally save a 2D UMAP scatter plot of the provided embeddings colored by cluster labels.

        Parameters:
            embedding_matrix (np.ndarray): 2D array of embedding vectors (samples x features).
            cluster_labels (np.ndarray): Array of cluster assignments for each sample (noise should be labeled `-1`).
            output_dir (Optional[Union[str, Path]]): Directory to save a timestamped PNG; if not provided the plot is shown.

        Returns:
            Path | None: Path to the saved PNG when `output_dir` is provided, `None` otherwise.
        """
        try:
            # Fit UMAP model
            umap_params = self.config["umap_params"].copy()
            umap_params["n_components"] = 2

            umap_model_2d = getattr(self, "umap_model_2d", None)
            umap_embeddings, self.umap_model_2d = project_umap(
                embedding_matrix,
                n_components=2,
                seed=self.config["seed"],
                umap_params=umap_params,
                model=umap_model_2d,
            )
            self.umap_model = self.umap_model_2d

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

            return self._save_or_show_plot(output_dir, "umap_2d")

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
        Create a 3D UMAP scatter plot of embedding vectors colored by cluster labels.

        Points with label -1 are treated as noise and plotted with black "x" markers; other labels are shown as distinct clusters. When `output_dir` is provided the figure is saved as a timestamped PNG and its Path is returned, otherwise the plot is displayed and `None` is returned.

        Parameters:
            embedding_matrix (np.ndarray): Array of shape (n_samples, n_features) to project with UMAP.
            cluster_labels (np.ndarray): Integer cluster labels for each row in `embedding_matrix`; noise is expected as -1.
            output_dir (Optional[Union[str, Path]]): Directory to save the PNG file. If omitted, the plot is shown instead of saved.

        Returns:
            Optional[Path]: Path to the saved PNG when `output_dir` is provided, `None` otherwise.
        """
        try:
            # Fit UMAP model
            umap_params = self.config["umap_params"].copy()
            umap_params["n_components"] = 3

            umap_model_3d = getattr(self, "umap_model_3d", None)
            umap_embeddings, self.umap_model_3d = project_umap(
                embedding_matrix,
                n_components=3,
                seed=self.config["seed"],
                umap_params=umap_params,
                model=umap_model_3d,
            )
            self.umap_model = self.umap_model_3d

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

            return self._save_or_show_plot(output_dir, "umap_3d")

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
        Create a 2D scatter plot of PCA-transformed embeddings colored by cluster labels.

        The plot labels PC axes with explained-variance percentages, marks noise points labeled `-1` as black crosses, and stores the fitted PCA model on `self.pca_model`. When `output_dir` is provided the figure is saved as a timestamped PNG and the saved Path is returned.

        Parameters:
            embedding_matrix (np.ndarray): Row-wise embedding vectors to project with PCA.
            cluster_labels (np.ndarray): Cluster assignment per embedding; noise is indicated by `-1`.
            output_dir (Optional[Union[str, Path]]): Directory to save the PNG; if omitted the plot is shown and `None` is returned.

        Returns:
            Path: Path to the saved PNG when `output_dir` is provided, `None` otherwise.
        """
        try:
            # Fit PCA model
            pca_params = None
            if self.pca_model is None:
                pca_params = self.config["pca_params"].copy()
                pca_params["n_components"] = 2
            pca_embeddings, self.pca_model = project_pca(
                embedding_matrix,
                n_components=2,
                seed=self.config["seed"],
                pca_params=pca_params,
                model=self.pca_model,
            )

            # Create plot
            plt.figure(figsize=self.config["plot_params"]["figsize"])
            x_values = pca_embeddings[:, 0]
            if pca_embeddings.shape[1] > 1:
                y_values = pca_embeddings[:, 1]
            else:
                y_values = np.zeros(pca_embeddings.shape[0])

            # Plot clusters
            unique_labels = np.unique(cluster_labels)
            colors = sns.color_palette(
                self.config["plot_params"]["palette"], len(unique_labels)
            )

            for i, label in enumerate(unique_labels):
                if label == -1:  # Noise points
                    mask = cluster_labels == label
                    plt.scatter(
                        x_values[mask],
                        y_values[mask],
                        c="black",
                        marker="x",
                        s=20,
                        alpha=0.6,
                        label="Noise",
                    )
                else:
                    mask = cluster_labels == label
                    plt.scatter(
                        x_values[mask],
                        y_values[mask],
                        c=[colors[i]],
                        label=f"Cluster {label}",
                        s=30,
                        alpha=0.7,
                    )

            # Add explained variance information
            explained_var = self.pca_model.explained_variance_ratio_
            plt.xlabel(f"PC1 ({explained_var[0]:.1%} variance)")
            if len(explained_var) > 1:
                plt.ylabel(f"PC2 ({explained_var[1]:.1%} variance)")
            else:
                plt.ylabel("PC2")
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

            return self._save_or_show_plot(output_dir, "pca_2d")

        except Exception as e:
            logger.error(f"Error creating PCA 2D plot: {e!s}")
            return None

    def _create_metrics_plot(
        self,
        clustering_result: ClusteringResult,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        Create a 2x2 visualization summarizing clustering metrics and cluster size distribution.

        Parameters:
            clustering_result (ClusteringResult): Object providing a `metrics` dict (may contain keys like
                `"silhouette"`, `"davies_bouldin"`, `"calinski_harabasz"`) and a `cluster_labels` tensor
                (with a `.numpy()` method) used to compute cluster sizes.
            output_dir (Optional[Union[str, Path]]): Directory to save a timestamped PNG. If omitted, the plot
                is shown interactively.

        Returns:
            Path or None: Path to the saved PNG when `output_dir` is provided, `None` when displayed or on failure.
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
                axes[0].set_ylim(-1, 1)

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
            valid_mask = unique_labels != -1
            unique_labels = unique_labels[valid_mask]  # Remove noise points
            counts = counts[valid_mask]

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

            return self._save_or_show_plot(output_dir, "metrics_plot")

        except Exception as e:
            logger.error(f"Error creating metrics plot: {e!s}")
            return None

    def _create_cluster_size_plot(
        self, cluster_labels: np.ndarray, output_dir: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """
        Plot the distribution of cluster sizes, excluding noise points labeled -1.

        Parameters:
            cluster_labels (np.ndarray): Array of cluster assignments; entries equal to -1 are treated as noise and omitted from the plot.
            output_dir (Optional[Union[str, Path]]): Directory to write a timestamped PNG. If omitted, the plot is displayed instead of saved.

        Returns:
            Optional[Path]: Path to the saved PNG when `output_dir` is provided, `None` otherwise.
        """
        try:
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            valid_mask = unique_labels != -1
            unique_labels = unique_labels[valid_mask]  # Remove noise points
            counts = counts[valid_mask]

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

            return self._save_or_show_plot(output_dir, "cluster_size_plot")

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
        Create a heatmap of a sampled subset of the embedding matrix with rows sorted by cluster label.

        This samples up to 100 rows and up to 50 columns from `embedding_matrix` (deterministic sampling with a fixed RNG seed), sorts the sampled rows by `cluster_labels`, renders a seaborn heatmap of the sampled values, and draws dashed red horizontal lines to indicate cluster boundaries for labels other than `-1`. The figure is saved as a timestamped PNG to `output_dir` when provided; otherwise the plot is displayed.

        Returns:
            Path: Path to the saved PNG when `output_dir` is provided, `None` otherwise.
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
                    plt.axhline(y=start_idx, color="red", linestyle="--", alpha=0.7)
            non_noise_indices = np.nonzero(sample_labels != -1)[0]
            if non_noise_indices.size:
                plt.axhline(
                    y=non_noise_indices[-1],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                )

            # Add research disclaimer
            plt.figtext(
                0.02,
                0.02,
                RESEARCH_DISCLAIMER_TEXT,
                fontsize=8,
                style="italic",
                alpha=0.7,
            )

            return self._save_or_show_plot(output_dir, "embedding_heatmap")

        except Exception as e:
            logger.error(f"Error creating embedding heatmap: {e!s}")
            return None
