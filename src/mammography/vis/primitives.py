"""Shared plotting primitives for visualization modules."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency
    sns = None

PALETTE = "viridis"
FIGSIZE_DEFAULT = (10, 8)
FIGSIZE_WIDE = (14, 8)
FIGSIZE_SQUARE = (10, 10)
DPI = 150

plt.style.use("seaborn-v0_8-whitegrid")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create the parent directory for a plot output path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def color_palette(palette: str, n_colors: int):
    """Return a seaborn-compatible color palette with a matplotlib fallback."""
    if sns is not None:
        return sns.color_palette(palette, n_colors)
    cmap_name = palette if palette in plt.colormaps() else PALETTE
    cmap = plt.get_cmap(cmap_name)
    if n_colors <= 1:
        return [cmap(0.5)]
    return [cmap(i / (n_colors - 1)) for i in range(n_colors)]
