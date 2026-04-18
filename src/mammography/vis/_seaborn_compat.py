# ruff: noqa
#
# _seaborn_compat.py
# mammography-pipelines
#
# Shared seaborn compatibility fallback for visualization modules.
# DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
# It must NOT be used for clinical or medical diagnostic purposes.
# No medical decision should be based on these results.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Shared seaborn-compatible fallback used by advanced visualization modules."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class _SeabornGrid:
    """Lightweight wrapper exposing the figure attribute used by callers."""

    def __init__(self, fig: plt.Figure) -> None:
        self.fig = fig


class _SeabornStub:
    """Small subset of seaborn APIs backed by Matplotlib.

    The fallback does not implement seaborn's hierarchical clustering or hue
    coloring; methods warn when those unsupported features are requested.
    """

    @staticmethod
    def color_palette(
        palette: str, n_colors: int
    ) -> list[tuple[float, float, float, float]]:
        cmap_name = palette if palette in plt.colormaps() else "viridis"
        cmap = plt.get_cmap(cmap_name)
        if n_colors <= 1:
            return [cmap(0.5)]
        return [cmap(i / (n_colors - 1)) for i in range(n_colors)]

    @staticmethod
    def heatmap(
        data: Any,
        *,
        ax: Optional[plt.Axes] = None,
        cmap: str = "viridis",
        annot: bool = False,
        annot_kws: Optional[dict[str, Any]] = None,
        fmt: str = ".2f",
        xticklabels: Optional[list[str]] = None,
        yticklabels: Optional[list[str]] = None,
        square: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cbar_kws: Optional[dict[str, Any]] = None,
        **_: Any,
    ) -> Any:
        ax = ax or plt.gca()
        values = np.asarray(data)
        im = ax.imshow(
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal" if square else "auto",
        )
        if xticklabels is not None:
            ax.set_xticks(range(len(xticklabels)))
            ax.set_xticklabels(xticklabels)
        if yticklabels is not None:
            ax.set_yticks(range(len(yticklabels)))
            ax.set_yticklabels(yticklabels)
        if annot:
            annot_kws = annot_kws or {}
            size = annot_kws.get("size", 8)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(values[i, j], fmt),
                        ha="center",
                        va="center",
                        fontsize=size,
                    )
        cbar = ax.figure.colorbar(im, ax=ax)
        if cbar_kws and "label" in cbar_kws:
            cbar.set_label(cbar_kws["label"])
        return im

    @staticmethod
    def clustermap(
        data: Any,
        *,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "viridis",
        **kwargs: Any,
    ) -> _SeabornGrid:
        unsupported = {
            "row_cluster",
            "col_cluster",
            "dendrogram_ratio",
            "cbar_pos",
            "linewidths",
        }.intersection(kwargs)
        if unsupported:
            warnings.warn(
                "The lightweight seaborn fallback does not support clustering "
                f"arguments {sorted(unsupported)}; ignoring them.",
                RuntimeWarning,
                stacklevel=2,
            )
        fig, ax = plt.subplots(figsize=figsize)
        values = np.asarray(data)
        im = ax.imshow(values, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax)
        return _SeabornGrid(fig)

    @staticmethod
    def pairplot(
        df: pd.DataFrame,
        *_: Any,
        hue: Optional[str] = None,
        plot_kws: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> _SeabornGrid:
        if hue is not None or kwargs.get("hue") is not None:
            warnings.warn(
                "_SeabornStub.pairplot does not support hue coloring; "
                "rendering numeric columns without class colors.",
                RuntimeWarning,
                stacklevel=2,
            )
        plot_kws = plot_kws or {}
        numeric = df.select_dtypes(include=[np.number])
        axes = pd.plotting.scatter_matrix(
            numeric,
            figsize=(8, 8),
            alpha=plot_kws.get("alpha", 0.6),
            s=plot_kws.get("s", 20),
        )
        fig = axes[0, 0].get_figure()
        return _SeabornGrid(fig)

    @staticmethod
    def kdeplot(
        data: Any,
        *,
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = None,
        fill: bool = False,
        alpha: float = 0.5,
        **__: Any,
    ) -> Any:
        ax = ax or plt.gca()
        values = np.asarray(data, dtype=float)
        values = values[np.isfinite(values)]
        try:
            from scipy.stats import gaussian_kde

            if values.size < 2 or np.all(values == values[0]):
                raise ValueError("kde requires at least two non-identical values")
            kde = gaussian_kde(values)
            xs = np.linspace(values.min(), values.max(), 512)
            ys = kde(xs)
            ax.plot(xs, ys, label=label)
            if fill:
                ax.fill_between(xs, ys, alpha=alpha)
        except (ImportError, ValueError, np.linalg.LinAlgError):
            ax.hist(values, bins=50, density=True, alpha=alpha, label=label)
        return ax

    @staticmethod
    def violinplot(
        *,
        data: Optional[pd.DataFrame] = None,
        x: Optional[str] = None,
        y: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        **__: Any,
    ) -> Any:
        ax = ax or plt.gca()
        if data is not None and x and y:
            grouped = data.groupby(x)[y].apply(list)
            ax.violinplot(grouped.tolist(), showmedians=True)
            ax.set_xticks(range(1, len(grouped) + 1))
            ax.set_xticklabels(grouped.index)
        elif data is not None and y:
            ax.violinplot(data[y], showmedians=True)
        return ax

    @staticmethod
    def boxplot(
        *,
        data: Optional[pd.DataFrame] = None,
        x: Optional[str] = None,
        y: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        **__: Any,
    ) -> Any:
        ax = ax or plt.gca()
        if data is not None and x and y:
            grouped = data.groupby(x)[y].apply(list)
            ax.boxplot(grouped.tolist(), labels=list(grouped.index))
        elif data is not None and y:
            ax.boxplot(data[y])
        return ax


try:  # pragma: no cover - optional dependency
    import seaborn as sns
except ImportError:  # pragma: no cover - fallback exercised when seaborn is unavailable
    sns = _SeabornStub()
