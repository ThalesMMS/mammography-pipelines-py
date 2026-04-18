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
import torch

from ..clustering.clustering_result import ClusteringResult
from ..io.dicom import MammographyImage
from ..models.embeddings.embedding_vector import EmbeddingVector
from ..utils.embeddings import extract_embedding_matrix
from .dimensionality import project_pca, project_umap

# Configure logging for educational purposes
logger = logging.getLogger(__name__)

# Set matplotlib style for publication-ready plots
plt.style.use("default")
sns.set_palette("husl")

# Constants
RESEARCH_DISCLAIMER_TEXT = "Research purposes only - NOT for clinical use"

def _resize_grayscale_nearest(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize a 2D grayscale image to image_size using deterministic nearest-neighbor sampling."""
    target_h, target_w = int(image_size[0]), int(image_size[1])
    if image.shape == (target_h, target_w):
        return image
    row_idx = np.linspace(0, image.shape[0] - 1, target_h).astype(int)
    col_idx = np.linspace(0, image.shape[1] - 1, target_w).astype(int)
    return image[np.ix_(row_idx, col_idx)]

def _montage_array_from_image(
    mammography_image: MammographyImage,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Convert a MammographyImage pixel array into a normalized montage tile."""
    pixels = np.asarray(mammography_image.pixel_array)
    if pixels.ndim == 3:
        if pixels.shape[-1] in {3, 4}:
            pixels = pixels[..., :3].mean(axis=-1)
        else:
            pixels = pixels[0]
    pixels = np.squeeze(pixels)
    if pixels.ndim != 2:
        raise ValueError(f"Expected 2D pixel array, got shape {pixels.shape}.")

    pixels = np.nan_to_num(pixels.astype(np.float32, copy=False), copy=False)
    min_value = float(np.min(pixels))
    max_value = float(np.max(pixels))
    if max_value > min_value:
        pixels = (pixels - min_value) / (max_value - min_value) * 255.0
    else:
        pixels = np.zeros_like(pixels, dtype=np.float32)
    return _resize_grayscale_nearest(pixels, image_size).astype(np.uint8)

class ClusterMontageMixin:

        def _create_cluster_montage(
            self,
            clustering_result: ClusteringResult,
            mammography_images: List[MammographyImage],
            output_dir: Optional[Union[str, Path]] = None,
        ) -> Optional[Path]:
            """
            Create and save (or display) a montage of representative images for each cluster.
            
            Generates a montage subplot per cluster found in `clustering_result` (excluding label -1), selecting up to the configured number of samples per cluster and arranging them in the configured grid. If `output_dir` is provided, the montage is saved to a timestamped PNG file (plot DPI is taken from configuration); otherwise the figure is shown interactively. Errors during creation are logged and result in no file being written.
            
            Parameters:
                clustering_result (ClusteringResult): Clustering result containing cluster labels.
                mammography_images (List[MammographyImage]): Images aligned to `cluster_labels` and used as montage tiles.
                output_dir (Optional[Union[str, Path]]): Directory to write the PNG file; if omitted, the montage is displayed instead.
            
            Returns:
                Path or None: Path to the saved PNG file when `output_dir` is provided and saving succeeds; `None` if displayed interactively or if an error occurred.
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
                if n_clusters == 0:
                    return None

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
                    for idx in selected_indices:
                        try:
                            montage_images.append(
                                _montage_array_from_image(
                                    mammography_images[int(idx)],
                                    tuple(image_size),
                                )
                            )
                        except (IndexError, AttributeError, ValueError) as exc:
                            logger.warning(
                                "Could not load montage image at index %s: %s",
                                idx,
                                exc,
                            )
                            montage_images.append(np.zeros(image_size))

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

            except Exception:
                logger.exception("Error creating cluster montage")
                return None
