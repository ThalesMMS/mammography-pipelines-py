#
# results_visualizer.py
# mammography-pipelines
#
# Results visualizer component for displaying confusion matrix and ROC curves in the web UI.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""Results visualizer component for model evaluation metrics display."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency
    sns = None

from sklearn.metrics import confusion_matrix, roc_curve, auc


LOGGER = logging.getLogger("mammography")

# Visualization defaults
FIGSIZE_DEFAULT = (10, 8)
DPI = 150


def _require_streamlit() -> None:
    """Raise ImportError if Streamlit is not available."""
    if st is None:
        raise ImportError(
            "Streamlit is required to run the web UI dashboard."
        ) from _STREAMLIT_IMPORT_ERROR


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple[int, int] = (8, 7),
) -> plt.Figure:
    """Plot confusion matrix as heatmap.

    This function creates a confusion matrix visualization showing the
    relationship between predicted and true labels. Useful for evaluating
    classification model performance.

    DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
    It must NOT be used for clinical or medical diagnostic purposes.
    No medical decision should be based on these results.

    Args:
        y_true: Ground truth labels (1D array)
        y_pred: Predicted labels (1D array)
        class_names: Names for each class (e.g., ["A", "B", "C", "D"] for BI-RADS)
        normalize: If True, normalize by true labels (show proportions)
        title: Plot title
        cmap: Colormap name
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object containing the confusion matrix heatmap

    Example:
        >>> y_true = np.array([0, 1, 2, 1, 0])
        >>> y_pred = np.array([0, 2, 2, 1, 0])
        >>> fig = plot_confusion_matrix(
        ...     y_true,
        ...     y_pred,
        ...     class_names=["A", "B", "C"],
        ...     normalize=True,
        ... )
        >>> # Display in Streamlit
        >>> st.pyplot(fig)
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize if requested
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    fmt = ".2f" if normalize else "d"

    if sns is not None:
        # Use seaborn if available for better styling
        sns.heatmap(
            cm,
            ax=ax,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Proportion" if normalize else "Count"},
        )
    else:
        # Fallback to matplotlib
        im = ax.imshow(cm, cmap=cmap, aspect="equal")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Proportion" if normalize else "Count")

        # Add annotations
        for i in range(n_classes):
            for j in range(n_classes):
                text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )

        # Set ticks
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    # Labels and title
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    fig.tight_layout()

    LOGGER.debug("Confusion matrix plotted: %dx%d classes", n_classes, n_classes)

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "ROC Curves",
    figsize: tuple[int, int] = FIGSIZE_DEFAULT,
    show_micro_avg: bool = True,
    show_macro_avg: bool = True,
) -> plt.Figure:
    """Plot ROC curves for multi-class classification.

    This function creates ROC (Receiver Operating Characteristic) curves
    showing the trade-off between true positive rate and false positive rate
    for each class. Includes micro and macro-averaged curves.

    DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
    It must NOT be used for clinical or medical diagnostic purposes.
    No medical decision should be based on these results.

    Args:
        y_true: Ground truth labels (1D array of class indices)
        y_prob: Predicted probabilities (2D array of shape [n_samples, n_classes])
        class_names: Names for each class (e.g., ["A", "B", "C", "D"] for BI-RADS)
        title: Plot title
        figsize: Figure size (width, height)
        show_micro_avg: Whether to show micro-averaged ROC curve
        show_macro_avg: Whether to show macro-averaged ROC curve

    Returns:
        Matplotlib Figure object containing the ROC curves

    Example:
        >>> y_true = np.array([0, 1, 2, 1, 0])
        >>> y_prob = np.array([
        ...     [0.8, 0.1, 0.1],
        ...     [0.2, 0.7, 0.1],
        ...     [0.1, 0.2, 0.7],
        ...     [0.3, 0.6, 0.1],
        ...     [0.9, 0.05, 0.05],
        ... ])
        >>> fig = plot_roc_curves(
        ...     y_true,
        ...     y_prob,
        ...     class_names=["A", "B", "C"],
        ... )
        >>> # Display in Streamlit
        >>> st.pyplot(fig)
    """
    # Validate inputs
    if y_prob.ndim != 2:
        raise ValueError(
            f"y_prob must be 2D array, got shape {y_prob.shape}"
        )

    n_classes = y_prob.shape[1]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Binarize the labels for multi-class ROC
    y_true_bin = np.zeros((len(y_true), n_classes), dtype=int)
    for i, label in enumerate(y_true):
        y_true_bin[i, int(label)] = 1

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    if show_micro_avg:
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_bin.ravel(), y_prob.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and AUC
    if show_macro_avg:
        # Aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Average and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot ROC curves
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, n_classes))

    # Plot individual class ROC curves
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
        )

    # Plot micro-average ROC curve
    if show_micro_avg:
        ax.plot(
            fpr["micro"],
            tpr["micro"],
            color="deeppink",
            linestyle=":",
            linewidth=3,
            label=f"Micro-avg (AUC = {roc_auc['micro']:.3f})",
        )

    # Plot macro-average ROC curve
    if show_macro_avg:
        ax.plot(
            fpr["macro"],
            tpr["macro"],
            color="navy",
            linestyle=":",
            linewidth=3,
            label=f"Macro-avg (AUC = {roc_auc['macro']:.3f})",
        )

    # Plot diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")

    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    LOGGER.debug("ROC curves plotted for %d classes", n_classes)

    return fig


def render_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """Render confusion matrix in Streamlit UI.

    Convenience function that creates and displays a confusion matrix
    in the Streamlit interface.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for each class
        normalize: If True, normalize by true labels
        title: Plot title
    """
    _require_streamlit()

    fig = plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        normalize=normalize,
        title=title,
    )

    st.pyplot(fig)
    plt.close(fig)


def render_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "ROC Curves",
    show_micro_avg: bool = True,
    show_macro_avg: bool = True,
) -> None:
    """Render ROC curves in Streamlit UI.

    Convenience function that creates and displays ROC curves
    in the Streamlit interface.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        class_names: Names for each class
        title: Plot title
        show_micro_avg: Whether to show micro-averaged ROC curve
        show_macro_avg: Whether to show macro-averaged ROC curve
    """
    _require_streamlit()

    fig = plot_roc_curves(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        title=title,
        show_micro_avg=show_micro_avg,
        show_macro_avg=show_macro_avg,
    )

    st.pyplot(fig)
    plt.close(fig)


def render_results_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    """Render complete results summary with confusion matrix and ROC curves.

    This function displays a comprehensive evaluation of classification results
    including both confusion matrix and ROC curves side by side.

    DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
    It must NOT be used for clinical or medical diagnostic purposes.
    No medical decision should be based on these results.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for ROC curves)
        class_names: Names for each class
    """
    _require_streamlit()

    st.subheader("📊 Model Evaluation Results")

    # Display confusion matrix
    st.markdown("### Confusion Matrix")
    st.markdown(
        "Shows the relationship between predicted and true labels. "
        "Diagonal elements represent correct predictions."
    )
    render_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        normalize=True,
    )

    # Display ROC curves if probabilities are provided
    if y_prob is not None:
        st.markdown("### ROC Curves")
        st.markdown(
            "ROC curves show the trade-off between true positive rate "
            "and false positive rate for each class. Higher AUC (Area Under Curve) "
            "indicates better classification performance."
        )
        render_roc_curves(
            y_true=y_true,
            y_prob=y_prob,
            class_names=class_names,
        )

    # Display medical disclaimer
    st.warning(
        "⚠️ **DISCLAIMER**: This is an EDUCATIONAL RESEARCH project. "
        "It must NOT be used for clinical or medical diagnostic purposes. "
        "No medical decision should be based on these results."
    )
