"""Figure export helpers used by training commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable


def parse_export_formats(raw: str | None) -> list[str]:
    """Parse comma-separated export formats from a CLI argument."""
    if not raw:
        return []
    formats = [fmt.strip().lower() for fmt in raw.split(",")]
    valid = {"png", "pdf", "svg"}
    invalid = [fmt for fmt in formats if fmt not in valid]
    if invalid:
        raise SystemExit(f"Formatos invalidos: {invalid}. Use: png, pdf, svg")
    return formats


def export_figure_multi_format(
    fig_func: Callable[..., Any],
    output_path: Path,
    formats: list[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Export a matplotlib figure to multiple formats for publication."""
    if not formats:
        fig_func(*args, **kwargs)
        return

    figures_dir = output_path.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    base_name = output_path.stem

    for fmt in formats:
        export_path = figures_dir / f"{base_name}.{fmt}"
        args_for_call = args
        if "out_path" not in kwargs:
            args_list = list(args)
            if args_list:
                args_list[-1] = str(export_path)
                args_for_call = tuple(args_list)

        try:
            if fig_func.__name__ == "plot_history":
                plot_history_format(args_for_call[0], export_path.parent, export_path.name, fmt)
            elif fig_func.__name__ == "save_metrics_figure":
                save_metrics_figure_format(args_for_call[0], str(export_path))
        except Exception as exc:
            logger = logging.getLogger("mammography")
            logger.warning("Falha ao exportar figura %s: %s", export_path, exc)


def plot_history_format(
    history: list[dict[str, Any]],
    outdir: Path,
    base_name: str,
    fmt: str,
) -> None:
    """Save a training history plot in the requested format."""
    import matplotlib.pyplot as plt
    import pandas as pd

    if not history:
        return
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)

    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(df["epoch"], df["train_loss"], label="train")
        ax[0].plot(df["epoch"], df["val_loss"], label="val")
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[1].plot(df["epoch"], df["train_acc"], label="train")
        ax[1].plot(df["epoch"], df["val_acc"], label="val")
        ax[1].set_title("Accuracy")
        ax[1].legend()
        fig.tight_layout()

        if base_name.endswith(f".{fmt}"):
            out_path = outdir / base_name
        else:
            out_path = outdir / f"{base_name.replace('.png', '')}.{fmt}"

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def save_metrics_figure_format(metrics: dict[str, Any], out_path: str) -> None:
    """Save a metrics figure in the requested format."""
    from mammography.training.engine import save_metrics_figure

    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    save_metrics_figure(metrics, out_path)
