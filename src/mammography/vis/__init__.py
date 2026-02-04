#
# __init__.py
# mammography-pipelines
#
# Collects visualization helpers for embedding plots and clustering metrics.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Visualization helpers (plots for embeddings, metrics, clustering)."""

from mammography.vis.plots import plot_scatter, plot_clustering_metrics
from mammography.vis.cluster_visualizer import ClusterVisualizer
from mammography.vis.advanced import (
    plot_tsne_2d,
    plot_tsne_3d,
    plot_heatmap_correlation,
    plot_confusion_matrix_heatmap,
    plot_feature_heatmap,
    plot_scatter_matrix,
    plot_distribution,
    plot_embedding_comparison,
    plot_class_separation,
    plot_feature_importance,
    plot_learning_curves,
    generate_visualization_report,
)
from mammography.vis.cancer_plots import (
    HP,
    dicom_debug_preprocess,
    get_dicom_plt,
    get_transforms,
    preview_transformed_samples,
    robust_window,
)
from mammography.vis.explainability import (
    GradCAMExplainer,
    ViTAttentionVisualizer,
    generate_explanations_batch,
    export_explanations_report,
)

__all__ = [
    # Basic plots
    "plot_scatter",
    "plot_clustering_metrics",
    "ClusterVisualizer",
    # Advanced visualizations
    "plot_tsne_2d",
    "plot_tsne_3d",
    "plot_heatmap_correlation",
    "plot_confusion_matrix_heatmap",
    "plot_feature_heatmap",
    "plot_scatter_matrix",
    "plot_distribution",
    "plot_embedding_comparison",
    "plot_class_separation",
    "plot_feature_importance",
    "plot_learning_curves",
    "generate_visualization_report",
    # Cancer detection visualization
    "HP",
    "dicom_debug_preprocess",
    "get_dicom_plt",
    "get_transforms",
    "preview_transformed_samples",
    "robust_window",
    # Explainability
    "GradCAMExplainer",
    "ViTAttentionVisualizer",
    "generate_explanations_batch",
    "export_explanations_report",
]
