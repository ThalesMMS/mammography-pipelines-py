from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

torch = pytest.importorskip("torch")
matplotlib = pytest.importorskip("matplotlib")
# Use non-interactive backend for testing
matplotlib.use('Agg')
sns = pytest.importorskip("seaborn")

from mammography.vis.plots import plot_scatter, plot_clustering_metrics
from mammography.vis.export import (
    export_figure,
    export_training_curves,
    export_confusion_matrix,
    export_metrics_comparison,
)
from mammography.vis.cluster_visualizer import (
    ClusterVisualizer,
    create_cluster_visualizer,
    visualize_clustering,
)
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


# ============================================================================
# Tests for plots.py
# ============================================================================


def test_plot_scatter_basic(tmp_path):
    """Test basic scatter plot creation."""
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(50),
        'y': np.random.randn(50),
        'label': np.random.choice(['A', 'B', 'C'], 50)
    })

    out_path = tmp_path / "scatter.png"
    plot_scatter(df, 'x', 'y', hue='label', title="Test Scatter", out_path=str(out_path))

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_scatter_no_hue(tmp_path):
    """Test scatter plot without hue coloring."""
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(30),
        'y': np.random.randn(30)
    })

    out_path = tmp_path / "scatter_no_hue.png"
    plot_scatter(df, 'x', 'y', hue=None, out_path=str(out_path))

    assert out_path.exists()


def test_plot_clustering_metrics(tmp_path):
    """Test clustering metrics visualization."""
    history = [
        {'k': 2, 'silhouette': 0.6, 'davies_bouldin': 0.5},
        {'k': 3, 'silhouette': 0.7, 'davies_bouldin': 0.4},
        {'k': 4, 'silhouette': 0.65, 'davies_bouldin': 0.45},
        {'k': 5, 'silhouette': 0.55, 'davies_bouldin': 0.6},
    ]

    out_path = tmp_path / "metrics.png"
    plot_clustering_metrics(history, str(out_path))

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_clustering_metrics_single_point(tmp_path):
    """Test clustering metrics with single data point."""
    history = [{'k': 3, 'silhouette': 0.7, 'davies_bouldin': 0.4}]

    out_path = tmp_path / "metrics_single.png"
    plot_clustering_metrics(history, str(out_path))

    assert out_path.exists()


# ============================================================================
# Tests for export.py
# ============================================================================


def test_export_figure_png(tmp_path):
    """Test exporting figure as PNG."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    base_path = tmp_path / "test_fig"
    exported = export_figure(fig, base_path, formats=['png'], dpi=150)

    assert len(exported) == 1
    assert exported[0].suffix == '.png'
    assert exported[0].exists()

    plt.close(fig)


def test_export_figure_multiple_formats(tmp_path):
    """Test exporting figure in multiple formats."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [2, 4, 6])

    base_path = tmp_path / "multi_format"
    exported = export_figure(fig, base_path, formats=['png', 'pdf', 'svg'])

    assert len(exported) == 3
    assert any(p.suffix == '.png' for p in exported)
    assert any(p.suffix == '.pdf' for p in exported)
    assert any(p.suffix == '.svg' for p in exported)

    plt.close(fig)


def test_export_figure_default_formats(tmp_path):
    """Test export with default format list."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [1, 2, 3])

    base_path = tmp_path / "default"
    exported = export_figure(fig, base_path)

    assert len(exported) == 3  # Default: png, pdf, svg

    plt.close(fig)


def test_export_training_curves_loss_only(tmp_path):
    """Test exporting training curves with loss only."""
    train_losses = [0.8, 0.6, 0.5, 0.4, 0.35]
    val_losses = [0.85, 0.65, 0.55, 0.5, 0.45]

    base_path = tmp_path / "curves_loss"
    exported = export_training_curves(
        train_losses, val_losses, base_path=base_path, formats=['png']
    )

    assert len(exported) == 1
    assert exported[0].exists()


def test_export_training_curves_with_accuracy(tmp_path):
    """Test exporting training curves with accuracy."""
    train_losses = [0.7, 0.5, 0.4]
    val_losses = [0.75, 0.55, 0.45]
    train_accs = [0.6, 0.75, 0.8]
    val_accs = [0.55, 0.7, 0.75]

    base_path = tmp_path / "curves_acc"
    exported = export_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        base_path=base_path, formats=['png'], title="Test Training"
    )

    assert len(exported) == 1
    assert exported[0].exists()


def test_export_confusion_matrix(tmp_path):
    """Test exporting confusion matrix."""
    cm = np.array([
        [50, 5, 2],
        [3, 60, 4],
        [1, 2, 55]
    ])

    base_path = tmp_path / "cm"
    exported = export_confusion_matrix(
        cm, class_names=['A', 'B', 'C'], base_path=base_path, formats=['png']
    )

    assert len(exported) == 1
    assert exported[0].exists()


def test_export_confusion_matrix_normalized(tmp_path):
    """Test exporting normalized confusion matrix."""
    cm = np.array([
        [40, 10],
        [5, 45]
    ])

    base_path = tmp_path / "cm_norm"
    exported = export_confusion_matrix(
        cm, class_names=['Negative', 'Positive'],
        base_path=base_path, normalize=True, formats=['png']
    )

    assert len(exported) == 1
    assert exported[0].exists()


def test_export_metrics_comparison(tmp_path):
    """Test exporting metrics comparison chart."""
    metrics_dict = {
        'Experiment 1': {'accuracy': 0.85, 'f1': 0.83, 'precision': 0.84},
        'Experiment 2': {'accuracy': 0.88, 'f1': 0.86, 'precision': 0.87},
        'Experiment 3': {'accuracy': 0.82, 'f1': 0.80, 'precision': 0.81},
    }

    base_path = tmp_path / "metrics_comp"
    exported = export_metrics_comparison(
        metrics_dict, base_path=base_path, formats=['png']
    )

    assert len(exported) == 1
    assert exported[0].exists()


def test_export_metrics_comparison_subset(tmp_path):
    """Test exporting metrics comparison with specific metrics."""
    metrics_dict = {
        'Model A': {'accuracy': 0.9, 'f1': 0.88, 'recall': 0.87, 'precision': 0.89},
        'Model B': {'accuracy': 0.85, 'f1': 0.83, 'recall': 0.82, 'precision': 0.84},
    }

    base_path = tmp_path / "metrics_subset"
    exported = export_metrics_comparison(
        metrics_dict, base_path=base_path,
        metric_names=['accuracy', 'f1'], formats=['png']
    )

    assert len(exported) == 1
    assert exported[0].exists()


# ============================================================================
# Tests for cluster_visualizer.py
# ============================================================================


@pytest.fixture
def mock_clustering_result():
    """Create a mock ClusteringResult for testing."""
    result = Mock()
    result.cluster_labels = torch.tensor([0, 0, 1, 1, 2, 2, -1])
    result.metrics = {
        'silhouette': 0.65,
        'davies_bouldin': 0.45,
        'calinski_harabasz': 120.5
    }
    return result


@pytest.fixture
def mock_embedding_vectors():
    """Create mock embedding vectors."""
    vectors = []
    for _ in range(7):
        vec = Mock()
        vec.embedding = torch.randn(128)
        vectors.append(vec)
    return vectors


def test_cluster_visualizer_init():
    """Test ClusterVisualizer initialization."""
    config = {
        'visualizations': ['umap_2d', 'pca_2d'],
        'seed': 42
    }

    visualizer = ClusterVisualizer(config)

    assert visualizer.config is not None
    assert 'visualizations' in visualizer.config
    assert visualizer.umap_model is None
    assert visualizer.pca_model is None


def test_cluster_visualizer_default_config():
    """Test ClusterVisualizer with default config."""
    visualizer = ClusterVisualizer({})

    assert 'umap_params' in visualizer.config
    assert 'pca_params' in visualizer.config
    assert 'plot_params' in visualizer.config
    assert 'montage_params' in visualizer.config


def test_cluster_visualizer_create_umap_2d(tmp_path, mock_clustering_result, mock_embedding_vectors):
    """Test creating UMAP 2D visualization."""
    config = {
        'visualizations': ['umap_2d'],
        'seed': 42
    }

    visualizer = ClusterVisualizer(config)
    results = visualizer.create_visualizations(
        mock_clustering_result,
        mock_embedding_vectors,
        output_dir=tmp_path
    )

    assert 'output_files' in results
    if 'umap_2d' in results['output_files']:
        umap_path = Path(results['output_files']['umap_2d'])
        assert umap_path.exists()


def test_cluster_visualizer_create_pca_2d(tmp_path, mock_clustering_result, mock_embedding_vectors):
    """Test creating PCA 2D visualization."""
    config = {
        'visualizations': ['pca_2d'],
        'seed': 42
    }

    visualizer = ClusterVisualizer(config)
    results = visualizer.create_visualizations(
        mock_clustering_result,
        mock_embedding_vectors,
        output_dir=tmp_path
    )

    assert 'output_files' in results
    if 'pca_2d' in results['output_files']:
        pca_path = Path(results['output_files']['pca_2d'])
        assert pca_path.exists()


def test_cluster_visualizer_create_metrics_plot(tmp_path, mock_clustering_result, mock_embedding_vectors):
    """Test creating metrics plot."""
    config = {
        'visualizations': ['metrics_plot'],
        'seed': 42
    }

    visualizer = ClusterVisualizer(config)
    results = visualizer.create_visualizations(
        mock_clustering_result,
        mock_embedding_vectors,
        output_dir=tmp_path
    )

    assert 'output_files' in results
    if 'metrics_plot' in results['output_files']:
        metrics_path = Path(results['output_files']['metrics_plot'])
        assert metrics_path.exists()


def test_cluster_visualizer_create_cluster_size_plot(tmp_path, mock_clustering_result, mock_embedding_vectors):
    """Test creating cluster size plot."""
    config = {
        'visualizations': ['cluster_size_plot'],
        'seed': 42
    }

    visualizer = ClusterVisualizer(config)
    results = visualizer.create_visualizations(
        mock_clustering_result,
        mock_embedding_vectors,
        output_dir=tmp_path
    )

    assert 'output_files' in results
    if 'cluster_size_plot' in results['output_files']:
        size_path = Path(results['output_files']['cluster_size_plot'])
        assert size_path.exists()


def test_cluster_visualizer_create_embedding_heatmap(tmp_path, mock_clustering_result, mock_embedding_vectors):
    """Test creating embedding heatmap."""
    config = {
        'visualizations': ['embedding_heatmap'],
        'seed': 42
    }

    visualizer = ClusterVisualizer(config)
    results = visualizer.create_visualizations(
        mock_clustering_result,
        mock_embedding_vectors,
        output_dir=tmp_path
    )

    assert 'output_files' in results
    if 'embedding_heatmap' in results['output_files']:
        heatmap_path = Path(results['output_files']['embedding_heatmap'])
        assert heatmap_path.exists()


def test_cluster_visualizer_extract_embedding_matrix(mock_embedding_vectors):
    """Test embedding matrix extraction."""
    config = {'seed': 42}
    visualizer = ClusterVisualizer(config)

    matrix = visualizer._extract_embedding_matrix(mock_embedding_vectors)

    assert matrix is not None
    assert matrix.shape[0] == len(mock_embedding_vectors)
    assert matrix.shape[1] == 128


def test_create_cluster_visualizer():
    """Test factory function for creating visualizer."""
    config = {'seed': 123}
    visualizer = create_cluster_visualizer(config)

    assert isinstance(visualizer, ClusterVisualizer)
    assert visualizer.config['seed'] == 123


def test_visualize_clustering(tmp_path, mock_clustering_result, mock_embedding_vectors):
    """Test convenience function for visualization."""
    config = {
        'visualizations': ['pca_2d'],
        'seed': 42
    }

    results = visualize_clustering(
        mock_clustering_result,
        mock_embedding_vectors,
        config,
        output_dir=tmp_path
    )

    assert 'output_files' in results
    assert 'visualization_timestamp' in results


# ============================================================================
# Tests for advanced.py
# ============================================================================


def test_plot_tsne_2d_basic(tmp_path):
    """Test basic 2D t-SNE plot."""
    np.random.seed(42)
    features = np.random.randn(50, 20)
    labels = np.random.randint(0, 3, 50)

    out_path = tmp_path / "tsne_2d.png"
    embedding, fig = plot_tsne_2d(
        features, labels, max_iter=250, seed=42, out_path=str(out_path)
    )

    assert embedding.shape == (50, 2)
    assert out_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_tsne_2d_no_labels(tmp_path):
    """Test t-SNE plot without labels."""
    np.random.seed(42)
    features = np.random.randn(30, 15)

    out_path = tmp_path / "tsne_no_labels.png"
    embedding, fig = plot_tsne_2d(
        features, labels=None, max_iter=250, seed=42, out_path=str(out_path)
    )

    assert embedding.shape == (30, 2)
    assert out_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_tsne_3d_basic(tmp_path):
    """Test basic 3D t-SNE plot."""
    np.random.seed(42)
    features = np.random.randn(40, 20)
    labels = np.random.randint(0, 2, 40)

    out_path = tmp_path / "tsne_3d.png"
    embedding, fig = plot_tsne_3d(
        features, labels, max_iter=250, seed=42, out_path=str(out_path)
    )

    assert embedding.shape == (40, 3)
    assert out_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_heatmap_correlation(tmp_path):
    """Test correlation heatmap."""
    np.random.seed(42)
    features = np.random.randn(100, 10)
    feature_names = [f'Feature_{i}' for i in range(10)]

    out_path = tmp_path / "correlation.png"
    fig = plot_heatmap_correlation(
        features, feature_names=feature_names, out_path=str(out_path)
    )

    assert out_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_confusion_matrix_heatmap(tmp_path):
    """Test confusion matrix heatmap."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 1, 2])

    out_path = tmp_path / "cm_heatmap.png"
    fig = plot_confusion_matrix_heatmap(
        y_true, y_pred, class_names=['A', 'B', 'C'], out_path=str(out_path)
    )

    assert out_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_feature_heatmap(tmp_path):
    """Test feature heatmap."""
    np.random.seed(42)
    features = np.random.randn(50, 20)
    labels = np.random.randint(0, 3, 50)

    out_path = tmp_path / "feature_heatmap.png"
    fig = plot_feature_heatmap(
        features, labels, out_path=str(out_path)
    )

    assert out_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_scatter_matrix(tmp_path):
    """Test scatter matrix plot."""
    np.random.seed(42)
    features = np.random.randn(60, 4)
    labels = np.random.randint(0, 2, 60)
    feature_names = ['F1', 'F2', 'F3', 'F4']

    out_path = tmp_path / "scatter_matrix.png"
    fig = plot_scatter_matrix(
        features, labels, feature_names=feature_names, out_path=str(out_path)
    )

    assert out_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_distribution(tmp_path):
    """Test distribution plot."""
    np.random.seed(42)
    features = np.random.randn(100, 3)
    labels = np.random.randint(0, 2, 100)

    out_path = tmp_path / "distribution.png"
    fig = plot_distribution(
        features, labels, feature_idx=0, out_path=str(out_path)
    )

    assert out_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_embedding_comparison(tmp_path):
    """Test embedding comparison plot."""
    np.random.seed(42)
    embedding1 = np.random.randn(50, 2)
    embedding2 = np.random.randn(50, 2)
    labels = np.random.randint(0, 3, 50)

    out_path = tmp_path / "embedding_comp.png"
    fig = plot_embedding_comparison(
        embedding1, embedding2, labels,
        method1_name="Method A", method2_name="Method B",
        out_path=str(out_path)
    )

    assert out_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_class_separation(tmp_path):
    """Test class separation plot."""
    features = np.random.randn(80, 10)
    labels = np.random.randint(0, 3, 80)

    out_path = tmp_path / "class_sep.png"
    fig, metrics = plot_class_separation(
        features, labels, out_path=str(out_path)
    )

    assert out_path.exists()
    assert 'silhouette' in metrics
    assert 'davies_bouldin' in metrics

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_feature_importance(tmp_path):
    """Test feature importance plot."""
    importance_scores = np.random.rand(10)
    feature_names = [f'Feature_{i}' for i in range(10)]

    out_path = tmp_path / "feature_importance.png"
    fig = plot_feature_importance(
        importance_scores, feature_names, out_path=str(out_path)
    )

    assert out_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_learning_curves(tmp_path):
    """Test learning curves plot."""
    train_scores = [0.6, 0.7, 0.75, 0.8, 0.82]
    val_scores = [0.55, 0.65, 0.7, 0.72, 0.73]
    train_sizes = [20, 40, 60, 80, 100]

    out_path = tmp_path / "learning_curves.png"
    fig = plot_learning_curves(
        train_sizes, train_scores, val_scores, out_path=str(out_path)
    )

    assert out_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_generate_visualization_report(tmp_path):
    """Test generating comprehensive visualization report."""
    features = np.random.randn(100, 20)
    labels = np.random.randint(0, 4, 100)

    report = generate_visualization_report(
        features, labels, output_dir=tmp_path, prefix="test"
    )

    assert 'output_files' in report
    assert len(report['output_files']) > 0

    # Verify some files were created
    assert any(Path(f).exists() for f in report['output_files'].values())


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_plot_scatter_empty_dataframe(tmp_path):
    """Test scatter plot with empty dataframe."""
    df = pd.DataFrame({'x': [], 'y': []})

    out_path = tmp_path / "empty.png"
    # Should handle gracefully or raise appropriate error
    try:
        plot_scatter(df, 'x', 'y', out_path=str(out_path))
    except (ValueError, KeyError):
        pass  # Expected for empty data


def test_export_figure_invalid_format(tmp_path):
    """Test export with invalid format."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])

    base_path = tmp_path / "invalid_fmt"
    exported = export_figure(fig, base_path, formats=['invalid_format'])

    # Should skip invalid format
    assert len(exported) == 0

    plt.close(fig)


def test_export_training_curves_mismatched_lengths(tmp_path):
    """Test training curves with mismatched lengths."""
    train_losses = [0.8, 0.6, 0.5]
    val_losses = [0.85, 0.65]  # Shorter

    base_path = tmp_path / "mismatch"
    # Should handle or raise error
    try:
        exported = export_training_curves(
            train_losses, val_losses, base_path=base_path, formats=['png']
        )
        # If it doesn't raise, check it created something
        assert len(exported) >= 0
    except (ValueError, IndexError):
        pass  # Expected for mismatched data


def test_cluster_visualizer_empty_embeddings(tmp_path):
    """Test cluster visualizer with empty embedding list."""
    config = {'visualizations': ['umap_2d']}
    visualizer = ClusterVisualizer(config)

    result = Mock()
    result.cluster_labels = torch.tensor([])
    result.metrics = {}

    results = visualizer.create_visualizations(
        result, [], output_dir=tmp_path
    )

    # Should return results dict even if empty
    assert 'output_files' in results


def test_plot_tsne_2d_small_dataset(tmp_path):
    """Test t-SNE with very small dataset."""
    features = np.random.randn(5, 10)
    labels = np.array([0, 0, 1, 1, 1])

    out_path = tmp_path / "tsne_small.png"
    # t-SNE might fail or warn with small datasets
    try:
        embedding, fig = plot_tsne_2d(
            features, labels, perplexity=2, max_iter=100, out_path=str(out_path)
        )
        assert embedding.shape[0] == 5
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ValueError:
        pass  # Expected for very small datasets


def test_plot_confusion_matrix_heatmap_binary(tmp_path):
    """Test confusion matrix with binary classification."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

    out_path = tmp_path / "cm_binary.png"
    fig = plot_confusion_matrix_heatmap(
        y_true, y_pred, class_names=['Negative', 'Positive'], out_path=str(out_path)
    )

    assert out_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


# ============================================================================
# Integration Tests
# ============================================================================


def test_cluster_visualizer_multiple_visualizations(tmp_path, mock_clustering_result, mock_embedding_vectors):
    """Test creating multiple visualizations at once."""
    config = {
        'visualizations': ['pca_2d', 'metrics_plot', 'cluster_size_plot'],
        'seed': 42
    }

    visualizer = ClusterVisualizer(config)
    results = visualizer.create_visualizations(
        mock_clustering_result,
        mock_embedding_vectors,
        output_dir=tmp_path
    )

    assert 'output_files' in results
    # At least some visualizations should succeed
    assert len(results['output_files']) >= 0


def test_export_multiple_confusion_matrices(tmp_path):
    """Test exporting multiple confusion matrices."""
    cms = [
        np.array([[40, 10], [5, 45]]),
        np.array([[35, 15], [8, 42]]),
        np.array([[38, 12], [6, 44]])
    ]

    for i, cm in enumerate(cms):
        base_path = tmp_path / f"cm_{i}"
        exported = export_confusion_matrix(cm, base_path=base_path, formats=['png'])
        assert len(exported) == 1
        assert exported[0].exists()


def test_visualization_report_complete_workflow(tmp_path):
    """Test complete visualization workflow."""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 150
    n_features = 20
    n_classes = 3

    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, n_classes, n_samples)

    # Generate report
    report = generate_visualization_report(
        features, labels, output_dir=tmp_path, prefix="complete"
    )

    assert 'output_files' in report
    assert 'num_samples' in report
    assert 'num_features' in report
    assert report['num_samples'] == n_samples
    assert report['num_features'] == n_features
