#
# export.py
# mammography-pipelines
#
# Publication-ready figure export utilities supporting multiple formats (PNG, PDF, SVG).
#
# Thales Matheus Mendonça Santos - November 2025
#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import logging

logger = logging.getLogger("mammography")

def export_figure(
    fig: plt.Figure,
    base_path: Union[str, Path],
    formats: Optional[List[str]] = None,
    dpi: int = 300,
    tight_layout: bool = True,
) -> List[Path]:
    """Export a matplotlib figure to multiple publication-ready formats.

    Args:
        fig: Matplotlib figure object to export
        base_path: Base path without extension (e.g., "outputs/figure1")
        formats: List of formats to export. Defaults to ['png', 'pdf', 'svg']
        dpi: DPI for raster formats (PNG). Default 300 for publication quality
        tight_layout: Whether to apply tight_layout before saving

    Returns:
        List of Path objects for successfully exported files

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> export_figure(fig, "results/plot1", formats=['png', 'pdf'])
        [Path('results/plot1.png'), Path('results/plot1.pdf')]
    """
    if formats is None:
        formats = ['png', 'pdf', 'svg']

    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    if tight_layout:
        try:
            fig.tight_layout()
        except Exception as e:
            logger.warning(f"tight_layout failed: {e}")

    exported_paths = []
    for fmt in formats:
        fmt_lower = fmt.lower()
        out_path = base_path.with_suffix(f".{fmt_lower}")

        try:
            if fmt_lower == 'png':
                fig.savefig(out_path, dpi=dpi, bbox_inches='tight', format='png')
            elif fmt_lower == 'pdf':
                fig.savefig(out_path, bbox_inches='tight', format='pdf')
            elif fmt_lower == 'svg':
                fig.savefig(out_path, bbox_inches='tight', format='svg')
            else:
                logger.warning(f"Unsupported format '{fmt}', skipping")
                continue

            exported_paths.append(out_path)
            logger.info(f"Exported figure to {out_path}")
        except Exception as e:
            logger.error(f"Failed to export figure as {fmt}: {e}")

    return exported_paths


def export_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    base_path: Union[str, Path] = "training_curves",
    formats: Optional[List[str]] = None,
    title: str = "Training Curves",
    dpi: int = 300,
) -> List[Path]:
    """Export training and validation curves as publication-ready figures.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accs: Optional list of training accuracies per epoch
        val_accs: Optional list of validation accuracies per epoch
        base_path: Base path for output files
        formats: Export formats (default: ['png', 'pdf', 'svg'])
        title: Figure title
        dpi: DPI for raster formats

    Returns:
        List of exported file paths
    """
    epochs = range(1, len(train_losses) + 1)

    # Determine subplot layout
    has_acc = train_accs is not None and val_accs is not None
    nrows = 2 if has_acc else 1

    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 5 * nrows))
    if nrows == 1:
        axes = [axes]

    # Loss subplot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy subplot (if provided)
    if has_acc:
        axes[1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')

    exported = export_figure(fig, base_path, formats=formats, dpi=dpi)
    plt.close(fig)

    return exported


def export_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    base_path: Union[str, Path] = "confusion_matrix",
    formats: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    dpi: int = 300,
    cmap: str = "Blues",
) -> List[Path]:
    """Export confusion matrix as publication-ready heatmap.

    Args:
        cm: Confusion matrix array (n_classes x n_classes)
        class_names: Optional list of class names for labels
        base_path: Base path for output files
        formats: Export formats (default: ['png', 'pdf', 'svg'])
        title: Figure title
        normalize: Whether to normalize the confusion matrix
        dpi: DPI for raster formats
        cmap: Colormap name for heatmap

    Returns:
        List of exported file paths
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto',
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        ax=ax,
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    exported = export_figure(fig, base_path, formats=formats, dpi=dpi)
    plt.close(fig)

    return exported


def export_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    base_path: Union[str, Path] = "metrics_comparison",
    formats: Optional[List[str]] = None,
    title: str = "Metrics Comparison",
    dpi: int = 300,
    metric_names: Optional[List[str]] = None,
) -> List[Path]:
    """Export metrics comparison across experiments as publication-ready bar chart.

    Args:
        metrics_dict: Dictionary mapping experiment names to metric dictionaries
                     e.g., {'exp1': {'accuracy': 0.9, 'f1': 0.85}, 'exp2': {...}}
        base_path: Base path for output files
        formats: Export formats (default: ['png', 'pdf', 'svg'])
        title: Figure title
        dpi: DPI for raster formats
        metric_names: Optional list of metric names to include (None = all)

    Returns:
        List of exported file paths
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(metrics_dict).T

    if metric_names:
        df = df[metric_names]

    n_metrics = len(df.columns)
    fig, axes = plt.subplots(
        nrows=(n_metrics + 1) // 2,
        ncols=2 if n_metrics > 1 else 1,
        figsize=(12, 4 * ((n_metrics + 1) // 2)),
    )

    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, metric in enumerate(df.columns):
        ax = axes[idx]
        df[metric].plot(kind='bar', ax=ax, color='steelblue', alpha=0.8)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Experiment', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Hide extra subplots if n_metrics is odd
    if n_metrics % 2 == 1 and n_metrics > 1:
        axes[-1].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')

    exported = export_figure(fig, base_path, formats=formats, dpi=dpi)
    plt.close(fig)

    return exported
