# ruff: noqa
#
# advanced_report.py
# mammography-pipelines
#
# Advanced visualization functions: t-SNE plots, heatmaps, scatterplots, and more.
# DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
# It must NOT be used for clinical or medical diagnostic purposes.
# No medical decision should be based on these results.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""
Advanced visualization utilities for mammography pipeline analysis.

Provides:
- t-SNE, UMAP, PCA 2D/3D scatter plots
- Feature correlation heatmaps
- Confusion matrix heatmaps
- Distribution plots (histograms, KDE, violin)
- Interactive embedding exploration
- Pair plots and feature relationships
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .dimensionality import (
    project_pca,
)
from ._seaborn_compat import _SeabornGrid, _SeabornStub
from .advanced_embeddings import (
    plot_class_separation,
    plot_embedding_comparison,
    plot_tsne_2d,
)
from .advanced_heatmaps import (
    plot_distribution,
    plot_feature_heatmap,
    plot_heatmap_correlation,
    plot_scatter_matrix,
)


def generate_visualization_report(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    metadata: Optional[pd.DataFrame] = None,
    output_dir: Union[str, Path] = "vis_report",
    seed: int = 42,
    label_names: Optional[Dict[int, str]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Create a set of visualization PNGs from feature data (and optional labels) and save them to output_dir.
    
    Generates and writes a fixed set of plots (embedding comparison, t-SNE 2D, correlation heatmap, feature heatmap,
    scatter matrix, optional class-separation plot, and PC1 distribution). When labels are provided, computes class
    separation metrics (silhouette, Davies–Bouldin, intra-class variance). All figures are closed after saving.
    
    Parameters:
        features: Array of shape (N, D) containing feature vectors to visualize.
        labels: Optional 1D array of class labels; enables labeled plots and class-separation metrics when provided.
        metadata: Optional DataFrame (accepted for API compatibility; not used by this function).
        output_dir: Directory where PNG files will be written; created if it does not exist.
        seed: Random seed forwarded to projection/embedding helpers for reproducible plots.
        label_names: Optional mapping from label integers to human-readable names, forwarded to plotting helpers.
        prefix: Optional filename prefix added to each output PNG (e.g., "prefix_name.png").
    
    Returns:
        dict: Mapping of plot identifiers to saved file paths with additional report fields:
            - All generated plot paths keyed by name (e.g., "tsne_2d", "correlation_heatmap", ...).
            - "class_separation_metrics": dict with keys "silhouette", "davies_bouldin", "intra_class_variance" or None.
            - "output_files": alias of the plot-path mapping.
            - "num_samples": number of samples (int).
            - "num_features": number of features (int).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_paths = {}
    class_separation_metrics = None

    def report_path(name: str) -> Path:
        """
        Builds the output file path for a named report PNG, applying the optional prefix.
        
        Parameters:
        	name (str): Base name for the report (without extension).
        
        Returns:
        	path (Path): Path to the PNG file inside `output_dir`, using `"{prefix}_{name}.png"` if `prefix` is non-empty, otherwise `"{name}.png"`.
        """
        filename = f"{prefix}_{name}.png" if prefix else f"{name}.png"
        return output_dir / filename
    
    # 1. Embedding comparison (PCA, t-SNE, UMAP)
    print("Generating embedding comparison...")
    _, fig = plot_embedding_comparison(
        features,
        labels,
        methods=["pca", "tsne"],
        seed=seed,
        out_path=str(report_path("embedding_comparison")),
        label_names=label_names,
    )
    plt.close(fig)
    report_paths["embedding_comparison"] = str(report_path("embedding_comparison"))
    
    # 2. t-SNE 2D detailed
    print("Generating t-SNE 2D plot...")
    _, fig = plot_tsne_2d(
        features,
        labels,
        seed=seed,
        out_path=str(report_path("tsne_2d")),
        label_names=label_names,
    )
    plt.close(fig)
    report_paths["tsne_2d"] = str(report_path("tsne_2d"))
    
    # 3. Correlation heatmap
    print("Generating correlation heatmap...")
    fig = plot_heatmap_correlation(
        features,
        out_path=str(report_path("correlation_heatmap")),
    )
    plt.close(fig)
    report_paths["correlation_heatmap"] = str(report_path("correlation_heatmap"))
    
    # 4. Feature heatmap
    print("Generating feature heatmap...")
    fig = plot_feature_heatmap(
        features,
        out_path=str(report_path("feature_heatmap")),
    )
    plt.close(fig)
    report_paths["feature_heatmap"] = str(report_path("feature_heatmap"))
    
    # 5. Scatter matrix
    print("Generating scatter matrix...")
    fig = plot_scatter_matrix(
        features,
        labels,
        out_path=str(report_path("scatter_matrix")),
        label_names=label_names,
    )
    plt.close(fig)
    report_paths["scatter_matrix"] = str(report_path("scatter_matrix"))
    
    # 6. Class separation (if labels provided)
    if labels is not None:
        print("Generating class separation analysis...")
        fig, metrics = plot_class_separation(
            features,
            labels,
            out_path=str(report_path("class_separation")),
            label_names=label_names,
        )
        plt.close(fig)
        report_paths["class_separation"] = str(report_path("class_separation"))
        class_separation_metrics = {
            "silhouette": metrics["silhouette"],
            "davies_bouldin": metrics["davies_bouldin"],
            "intra_class_variance": metrics["intra_class_variance"],
        }
    
    # 7. First PC distribution
    print("Generating PC1 distribution...")
    pc1, _ = project_pca(features, n_components=1, seed=seed)
    pc1 = pc1.ravel()
    fig = plot_distribution(
        pc1,
        labels,
        kind="kde",
        title="First Principal Component Distribution",
        xlabel="PC1",
        out_path=str(report_path("pc1_distribution")),
        label_names=label_names,
    )
    plt.close(fig)
    report_paths["pc1_distribution"] = str(report_path("pc1_distribution"))
    
    print(f"Visualization report saved to {output_dir}")
    return {
        **report_paths,
        "class_separation_metrics": class_separation_metrics,
        "output_files": report_paths,
        "num_samples": int(features.shape[0]),
        "num_features": int(features.shape[1]) if features.ndim > 1 else 1,
    }
