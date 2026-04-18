#!/usr/bin/env python3
# ruff: noqa
#
# train_artifacts.py
# mammography-pipelines
#
# Artifact, checkpoint, and interruption helpers for density training.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Artifact and checkpoint helpers for mammography density training."""

import os
import signal
import hashlib
import json
import logging
from typing import Any, Optional, Tuple, List, Dict
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from mammography.utils import export_formats as export_format_utils
from mammography.training.engine import (
    save_metrics_figure,
    save_atomic,
)


def get_file_hash(path: str) -> str:
    """
    Compute the MD5 hash of a file's contents.

    Parameters:
        path (str): Filesystem path to the file to hash.

    Returns:
        str: Hexadecimal MD5 digest of the file contents, or "unknown" if `path` is empty, does not exist, or the file cannot be read.
    """
    if not path or not os.path.exists(path):
        return "unknown"
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except (FileNotFoundError, PermissionError, OSError):
        return "unknown"


class GracefulKiller:
    """Track interrupt requests, optionally registering process signal handlers."""

    def __init__(self, register_signals: bool = False) -> None:
        """
        Initialize the GracefulKiller.

        Sets the internal `kill_now` flag to False. Signal handlers for SIGINT
        and SIGTERM are registered only when `register_signals=True`, which is
        reserved for real CLI training runs so tests and library callers do not
        install global handlers by constructing this class.
        """
        self.kill_now = False
        if register_signals:
            signal.signal(signal.SIGINT, self.exit_gracefully)
            signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame) -> None:
        """
        Handle termination signals by marking that training should stop after the current epoch.

        Sets the instance flag `kill_now` to True and prints an informational message indicating an interrupt was received.

        Parameters:
            signum (int): The signal number received.
            frame (types.FrameType | None): The current stack frame (may be None).
        """
        print(
            "\n[INFO] Sinal de interrupcao recebido! Terminando epoch atual antes de sair..."
        )
        self.kill_now = True


def _find_resume_checkpoint(outdir: str) -> Optional[Path]:
    """
    Locate a resume checkpoint file under the given output directory.

    Searches for immediate generic or view-specific checkpoint files, then scans
    subdirectories matching `results*` for the same patterns. Returns the most
    recently modified checkpoint when multiple candidates are available, or
    `None` when no checkpoint file is found.

    Parameters:
        outdir (str): Path to the output directory to search.

    Returns:
        Path | None: A Path to the discovered checkpoint file, or `None` if none was found.
    """
    base = Path(outdir)
    direct = base / "checkpoint.pt"
    if direct.is_file():
        return direct
    direct_view_candidates = sorted(base.glob("checkpoint_*.pt"))
    direct_view_candidates = [path for path in direct_view_candidates if path.is_file()]
    if direct_view_candidates:
        try:
            return max(direct_view_candidates, key=lambda p: p.stat().st_mtime)
        except Exception:
            return direct_view_candidates[0]
    if base.exists():
        candidates = []
        for results_path in base.glob("results*"):
            if not results_path.is_dir():
                continue
            ckpt = results_path / "checkpoint.pt"
            if ckpt.is_file():
                candidates.append(ckpt)
            candidates.extend(
                path for path in results_path.glob("checkpoint_*.pt") if path.is_file()
            )
        if candidates:
            try:
                return max(candidates, key=lambda p: p.stat().st_mtime)
            except Exception:
                return candidates[0]
    return None


def _write_split_artifacts(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    outdir: Path,
    split_mode: str,
    group_column: str | None,
    logger: logging.Logger,
) -> None:
    """
    Write dataset split CSVs and a JSON manifest into an output "splits" directory.

    Creates outdir/splits (if missing), writes train.csv and val.csv and optionally test.csv from the provided DataFrames, and writes split_manifest.json describing:
    - split_mode and group_column,
    - filesystem paths for each written split file,
    - row counts for train/val/test,
    - number of unique groups per split when a valid group_column is provided and present in the DataFrame.

    Parameters:
        train_df (pd.DataFrame): DataFrame for the training split.
        val_df (pd.DataFrame): DataFrame for the validation split.
        test_df (pd.DataFrame | None): DataFrame for the test split, or None to skip writing a test CSV.
        outdir (Path): Base output directory where a "splits" subdirectory will be created.
        split_mode (str): Identifier for the split-generation mode to record in the manifest.
        group_column (str | None): Column name used to group samples; if None or absent in a split, group counts are omitted for that split.
        logger (logging.Logger): Logger used to record the final output location.
    """
    split_dir = outdir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    split_files: dict[str, str] = {}
    split_frames = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    for split_name, split_df in split_frames.items():
        if split_df is None:
            continue
        split_path = split_dir / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        split_files[split_name] = str(split_path)

    manifest = {
        "split_mode": split_mode,
        "group_column": group_column,
        "files": split_files,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)) if test_df is not None else 0,
        "train_groups": (
            int(train_df[group_column].nunique())
            if group_column and group_column in train_df.columns
            else None
        ),
        "val_groups": (
            int(val_df[group_column].nunique())
            if group_column and group_column in val_df.columns
            else None
        ),
        "test_groups": (
            int(test_df[group_column].nunique())
            if test_df is not None and group_column and group_column in test_df.columns
            else None
        ),
    }
    manifest_path = split_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Split artifacts salvos em %s", split_dir)


def _metrics_artifact_base(split_name: str, current_view: str | None) -> str:
    """
    Builds a standardized base name for metric artifacts for a given split and optional view.

    Parameters:
        split_name (str): Split identifier (e.g., "train", "val", "test").
        current_view (str | None): Optional view name (e.g., "CC", "MLO"); when provided it is lowercased and appended.

    Returns:
        base_name (str): Base artifact name, e.g. "train_metrics" or "val_metrics_cc".
    """
    if current_view:
        return f"{split_name}_metrics_{current_view.lower()}"
    return f"{split_name}_metrics"


def _write_metrics_artifacts(
    *,
    metrics: Dict[str, Any],
    metrics_dir: Path,
    outdir: Path,
    split_name: str,
    current_view: str | None,
    export_formats: list[str],
) -> Path:
    """
    Write the provided metrics to a JSON file and save associated metric figures (PNG plus any requested export formats).

    Parameters:
        metrics (Dict[str, Any]): Metrics dictionary to serialize.
        metrics_dir (Path): Directory where the JSON metrics file and primary PNG will be written.
        outdir (Path): Base output directory used for additional exported figure formats (creates outdir/figures).
        split_name (str): Name of the data split used to derive the artifact base name.
        current_view (str | None): Optional view name appended to the artifact base name when present.
        export_formats (list[str]): Additional image formats to export into outdir/figures (e.g., ["svg", "pdf"]). If empty, no extra exports are produced.

    Returns:
        Path: Path to the written JSON metrics file.
    """
    base_name = _metrics_artifact_base(split_name, current_view)
    metrics_path = metrics_dir / f"{base_name}.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    save_metrics_figure(metrics, str(metrics_dir / f"{base_name}.png"))

    if export_formats:
        figures_dir = outdir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        for fmt in export_formats:
            _save_metrics_figure_format(
                metrics, str(figures_dir / f"{base_name}.{fmt}")
            )

    return metrics_path


def _summarize_metrics_for_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a compact numeric summary by extracting selected metric keys with safe numeric values.

    Only the keys 'acc', 'accuracy', 'kappa_quadratic', 'auc_ovr', 'macro_f1', 'bal_acc',
    'bal_acc_adj', 'loss', 'epoch', and 'num_samples' are considered. Integer-like values
    are converted to Python int; floating values are included only if finite and converted
    to Python float. Non-numeric, missing, or non-finite values are omitted.

    Parameters:
        metrics (Dict[str, Any]): Mapping of metric names to values (typically model/eval metrics).

    Returns:
        Dict[str, Any]: A dictionary containing the filtered and normalized numeric metrics.
    """
    summary: Dict[str, Any] = {}
    for key in (
        "acc",
        "accuracy",
        "kappa_quadratic",
        "auc_ovr",
        "macro_f1",
        "bal_acc",
        "bal_acc_adj",
        "loss",
        "epoch",
        "num_samples",
    ):
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, (int, np.integer)):
            summary[key] = int(value)
            continue
        if isinstance(value, (float, np.floating)):
            if np.isfinite(value):
                summary[key] = float(value)
            continue
    return summary


def _resolve_checkpoint_candidate(
    raw_path: object, *, results_dir: Path
) -> Path | None:
    """
    Resolve a raw checkpoint path into an absolute Path, interpreting relative paths under `results_dir`.

    Parameters:
        raw_path (object): Candidate path (string or Path-like). `None` or an empty/whitespace string yields `None`.
        results_dir (Path): Directory to prepend when `raw_path` is not an absolute path.

    Returns:
        Path | None: The resolved Path (absolute or `results_dir`-prefixed) or `None` if no valid path was provided.
    """
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = results_dir / path
    return path


def _resolve_best_model_path(
    *,
    results_dir: Path,
    current_view: str | None,
    top_k: list[dict[str, Any]],
    resume_path: Path | None,
) -> Path | None:
    """
    Resolve the most appropriate model checkpoint path for evaluation.

    Parameters:
        results_dir (Path): Directory containing current run result checkpoints.
        current_view (str | None): Optional view name; when provided, prefers `best_model_{view}.pt`.
        top_k (list[dict[str, Any]]): Prioritized list of checkpoint entries (each should provide a `"path"`); these are checked first in sorted order.
        resume_path (Path | None): Optional checkpoint from a previous run whose directory and the file itself are considered as candidates.

    Returns:
        Path | None: Path to the first existing checkpoint selected according to priority, or `None` if no candidate exists.
    """
    for entry in _sort_top_k(top_k):
        candidate = _resolve_checkpoint_candidate(
            entry.get("path"), results_dir=results_dir
        )
        if candidate is not None and candidate.is_file():
            return candidate

    best_model_name = (
        f"best_model_{current_view.lower()}.pt" if current_view else "best_model.pt"
    )
    candidates = [
        results_dir / best_model_name,
        *sorted(results_dir.glob("best_model*.pt")),
    ]

    if resume_path is not None:
        resume_results_dir = resume_path.parent
        candidates.extend(
            [
                resume_results_dir / best_model_name,
                *sorted(resume_results_dir.glob("best_model*.pt")),
                resume_path,
            ]
        )

    candidates.append(results_dir / "checkpoint.pt")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_checkpoint_for_eval(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """
    Load a saved checkpoint into the provided model for evaluation.

    Loads the checkpoint file at `checkpoint_path` onto `device`. If the loaded object is a dictionary containing the key `"model_state"`, that sub-dictionary is used as the state to load. The resulting state dict is applied to `model` with `strict=False`.
    Parameters:
        model (torch.nn.Module): Model to receive the loaded state.
        checkpoint_path (Path): Path to the checkpoint file to load.
        device (torch.device): Device to map the loaded checkpoint to.
    """
    loaded = torch.load(checkpoint_path, map_location=device)
    state = loaded
    if isinstance(loaded, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            candidate = loaded.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
        else:
            state = loaded
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(key, str) for key in state)
        or not all(torch.is_tensor(value) for value in state.values())
    ):
        raise ValueError(
            f"Unrecognized checkpoint format for model weights: {checkpoint_path}"
        )
    model.load_state_dict(state, strict=False)


def _parse_export_formats(raw: Optional[str]) -> list[str]:
    """
    Parse a comma-separated string of export formats into a normalized list of format identifiers.

    Parameters:
        raw (Optional[str]): A comma- or whitespace-separated string of export formats (e.g., "png,pdf") or None.

    Returns:
        list[str]: A list of normalized export format names; an empty list if `raw` is None or contains no valid formats.
    """
    return export_format_utils.parse_export_formats(raw)


def _export_figure_multi_format(
    fig_func, output_path: Path, formats: list[str], *args, **kwargs
) -> None:
    """
    Export a figure produced by `fig_func` to multiple file formats at `output_path`.

    Parameters:
        fig_func (callable): A callable that creates or returns a Matplotlib figure when invoked with `*args` and `**kwargs`.
        output_path (Path): Destination path (without extension) where exported files will be written.
        formats (list[str]): List of file format extensions (e.g., `["png", "pdf"]`) to export.
        *args: Positional arguments forwarded to `fig_func`.
        **kwargs: Keyword arguments forwarded to `fig_func`.

    Side effects:
        Writes one output file per format to the filesystem using `output_path` with the appropriate extension.
    """
    return export_format_utils.export_figure_multi_format(
        fig_func,
        output_path,
        formats,
        *args,
        **kwargs,
    )


def _plot_history_format(
    history: List[Dict[str, Any]], outdir: Path, base_name: str, fmt: str
) -> None:
    """
    Save training/validation metric history plots to disk using the requested export format.

    Parameters:
        history (List[Dict[str, Any]]): Sequence of per-epoch metric records used to build the plot.
        outdir (Path): Destination directory for the exported figure.
        base_name (str): Base filename (without extension) to use for the saved plot.
        fmt (str): Export format identifier (e.g., 'png', 'pdf') determining the output file type.
    """
    return export_format_utils.plot_history_format(history, outdir, base_name, fmt)


def _save_metrics_figure_format(metrics: Dict[str, Any], out_path: str) -> None:
    """
    Export a metrics figure to a single file in the specified format.

    Parameters:
        metrics (Dict[str, Any]): Metrics dictionary used to render the figure.
        out_path (str): Destination file path for the exported figure; the file extension determines the export format.
    """
    return export_format_utils.save_metrics_figure_format(metrics, out_path)


def _sort_top_k(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Sort a list of top-k checkpoint entries by score and epoch, highest first.

    Parameters:
        entries (list[dict[str, Any]]): Each entry must contain numeric `score` and integer `epoch`.

    Returns:
        list[dict[str, Any]]: The input entries sorted descending by `(score, epoch)`.
    """
    return sorted(entries, key=lambda e: (e["score"], e["epoch"]), reverse=True)


def _clean_top_k(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize and filter a raw top-k entries list into validated, typed entries sorted by score and epoch.

    Parameters:
        entries (list[dict[str, Any]]): Iterable of raw top-k records; each record may contain the keys
            "score", "epoch", and "path". Values will be coerced to `float` (score), `int` (epoch),
            and string path.

    Returns:
        list[dict[str, Any]]: A list of cleaned entries each with keys:
            - "score" (float)
            - "epoch" (int)
            - "path" (str, existing file path)
        The list includes only entries whose path is non-empty and exists on disk and is sorted by
        descending `(score, epoch)`.
    """
    cleaned: list[dict[str, Any]] = []
    for entry in entries:
        try:
            score = float(entry.get("score", 0.0))
            epoch = int(entry.get("epoch", 0))
            raw_path = entry.get("path")
        except Exception:
            continue
        if raw_path is None:
            continue
        raw_path = str(raw_path).strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_file():
            continue
        cleaned.append({"score": score, "epoch": epoch, "path": str(path)})
    return _sort_top_k(cleaned)


def _update_top_k(
    top_k: list[dict[str, Any]],
    score: float,
    epoch: int,
    model_state: dict[str, Any],
    out_dir: Path,
    k: int,
    metric_name: str,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Update and maintain a top-k set of model checkpoints based on a numeric score.

    Appends and atomically saves a new checkpoint when it is strictly better than the current worst entry (by score then epoch), trims the list to the top `k` entries, and removes on-disk checkpoint files for entries dropped beyond `k`.

    Parameters:
        top_k (list[dict[str, Any]]): Current top-k entries, each with keys `"score"`, `"epoch"`, and `"path"`.
        score (float): Score for the candidate checkpoint (higher is better).
        epoch (int): Epoch number associated with the candidate checkpoint.
        model_state (dict[str, Any]): Model state dictionary to be saved if the candidate is kept.
        out_dir (Path): Directory where checkpoint files should be written.
        k (int): Maximum number of entries to retain; if <= 0, no changes are made.
        metric_name (str): Metric name used to build the checkpoint filename (slashes replaced with underscores).

    Returns:
        Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
            - The updated list of kept top-k entries (sorted by score and epoch, descending).
            - The newly inserted entry dictionary `{"score": float, "epoch": int, "path": str}` if a checkpoint was saved and kept; `None` if no insertion occurred.
    """
    if k <= 0:
        return top_k, None
    if len(top_k) >= k:
        lowest = min(top_k, key=lambda e: (e["score"], e["epoch"]))
        if (score, epoch) <= (lowest["score"], lowest["epoch"]):
            return top_k, None

    out_dir.mkdir(parents=True, exist_ok=True)
    metric_tag = metric_name.replace("/", "_")
    filename = f"model_epoch{epoch:03d}_{metric_tag}{score:.4f}.pt"
    path = out_dir / filename
    save_atomic(model_state, path)
    entry = {"score": float(score), "epoch": int(epoch), "path": str(path)}
    top_k.append(entry)

    sorted_entries = _sort_top_k(top_k)
    keep = sorted_entries[:k]
    for item in sorted_entries[k:]:
        item_path = Path(item["path"])
        if item_path.exists():
            try:
                item_path.unlink()
            except Exception:
                pass
    return keep, entry
