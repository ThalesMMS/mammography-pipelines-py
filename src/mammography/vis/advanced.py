"""Compatibility facade for advanced visualization helpers."""

from mammography.vis.advanced_embeddings import (
    plot_class_separation,
    plot_embedding_comparison,
    plot_tsne_2d,
    plot_tsne_3d,
)
from mammography.vis.advanced_heatmaps import (
    plot_confusion_matrix_heatmap,
    plot_distribution,
    plot_feature_heatmap,
    plot_feature_importance,
    plot_heatmap_correlation,
    plot_scatter_matrix,
)
from mammography.vis.advanced_learning import (
    plot_learning_curves,
    plot_learning_curves_from_arrays,
)
from mammography.vis.advanced_report import generate_visualization_report

__all__ = [
    "generate_visualization_report",
    "plot_class_separation",
    "plot_confusion_matrix_heatmap",
    "plot_distribution",
    "plot_embedding_comparison",
    "plot_feature_heatmap",
    "plot_feature_importance",
    "plot_heatmap_correlation",
    "plot_learning_curves",
    "plot_learning_curves_from_arrays",
    "plot_scatter_matrix",
    "plot_tsne_2d",
    "plot_tsne_3d",
]
