#!/usr/bin/env python3
"""Utilities to validate and consolidate the official rerun benchmark."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import platform
from typing import Any

try:  # pragma: no cover - torch is optional for report generation
    import torch
except ImportError:  # pragma: no cover - fallback when torch is unavailable
    torch = None  # type: ignore[assignment]

from mammography.tools.benchmark_constants import (
    ARTICLE_COLUMNS,
    MASTER_COLUMNS,
)
from mammography.tools.benchmark_models import CollectedRun  # noqa: TC001

LOGGER = logging.getLogger("benchmark_report")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _format_float(value: float) -> str:
    """
    Format a floating-point number to four decimal places.

    Returns:
        A string representation of `value` with exactly four digits after the decimal point.
    """
    return f"{value:.4f}"


def _format_cell(value: Any) -> str:
    """
    Format a cell value for table output.

    Formats float values to four decimal places; for any other type returns the result of `str(value)`.

    Returns:
        A string representing the formatted cell value.
    """
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


def _markdown_escape(value: Any) -> str:
    """
    Format a value and escape Markdown table pipe characters.

    Parameters:
        value (Any): Value to format for table output; floats are formatted to four decimal places.

    Returns:
        str: The formatted string representation with each `|` character escaped as `\\|`.
    """
    return _format_cell(value).replace("|", "\\|")


def _tex_escape(value: Any) -> str:
    """
    Format a value for inclusion in a LaTeX table by escaping LaTeX special characters.

    Parameters:
        value (Any): The value to format and escape for safe inclusion in LaTeX table cells.

    Returns:
        str: A string representation of `value` with LaTeX special characters escaped.
    """
    text = _format_cell(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
        "$": "\\$",
        "~": "\\~{}",
        "^": "\\^{}",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _tex_header(columns: list[str]) -> str:
    return "    " + " & ".join(_tex_escape(column) for column in columns) + " \\\\"


def _write_csv_table(
    path: Path, rows: list[dict[str, Any]], columns: list[str]
) -> None:
    """
    Write a list of row dictionaries to a CSV file using the provided column order.

    Ensures the output directory exists, writes a header using `columns`, and emits one CSV row per dict in `rows`. Values are written as provided by the dicts; missing keys produce empty fields.

    Parameters:
        path (Path): Destination CSV file path.
        rows (list[dict[str, Any]]): Sequence of row mappings from column name to cell value.
        columns (list[str]): Ordered list of column names to use as the CSV header and fieldnames.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    Write a list of row dictionaries to a JSON file, creating the parent directory if needed.

    Parameters:
        path (Path): Destination file path to write the JSON.
        rows (list[dict[str, Any]]): Iterable of row dictionaries to serialize.

    Description:
        Serializes `rows` as pretty-printed JSON (2-space indent, UTF-8) with non-ASCII characters preserved and writes it to `path`. Creates `path.parent` if it does not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _markdown_lines(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    """
    Builds Markdown table lines from row dictionaries and a column order.

    Parameters:
        rows (list[dict[str, Any]]): Sequence of row mappings; each dict should contain keys for every name in `columns`.
        columns (list[str]): Ordered list of column names to use for the header and to extract values from each row.

    Returns:
        list[str]: Lines of a Markdown table: header line, divider line, then one body line per row. Cell values are formatted and escaped for Markdown table usage.
    """
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(_markdown_escape(row[column]) for column in columns) + " |"
        for row in rows
    ]
    return [header, divider, *body]


def _write_markdown_table(
    path: Path, rows: list[dict[str, Any]], columns: list[str]
) -> None:
    """
    Write a Markdown table to the given file path.

    Ensures the destination directory exists, renders the table from `rows` using the header/order in `columns`, and writes the content with a trailing newline.

    Parameters:
        path (Path): Destination file path for the Markdown output.
        rows (list[dict[str, Any]]): Sequence of row mappings; each dict should provide values for the keys named in `columns`.
        columns (list[str]): Ordered list of column names to use for the table header and cell ordering.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_markdown_lines(rows, columns)) + "\n", encoding="utf-8")


def _write_master_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    Write a complete LaTeX "master" table to `path` representing the provided rows.

    Ensures the parent directory exists, then writes a full LaTeX table environment (with caption and label) containing one tabular row per entry in `rows`. Cells are emitted in the order defined by `MASTER_COLUMNS` and LaTeX-special characters in cell values are escaped. The output is written as UTF-8 text and ends with a trailing newline.

    Parameters:
        path (Path): Destination file path for the generated .tex file.
        rows (list[dict[str, Any]]): List of row dictionaries matching keys in `MASTER_COLUMNS`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    latex_lines = [
        "% Auto-generated by mammography benchmark-report",
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\scriptsize",
        "  \\caption{Tabela mestre do rerun oficial 2026Q1.}",
        "  \\label{tab:rerun-2026q1-master}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{llllrrrrrrrrrll}",
        "    \\toprule",
        _tex_header(MASTER_COLUMNS),
        "    \\midrule",
    ]
    for row in rows:
        latex_lines.append(
            "    "
            + " & ".join(_tex_escape(row[column]) for column in MASTER_COLUMNS)
            + " \\\\"
        )
    latex_lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}%",
            "  }",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")


def _write_article_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    Write a LaTeX "article" table file that consolidates one result per dataset/task/model.

    Creates parent directories as needed and writes a complete LaTeX table (subsection, caption,
    tabular with columns defined by ARTICLE_COLUMNS) to `path`. Each entry in `rows` becomes one
    table row; cell values are escaped for LaTeX.

    Parameters:
        path (Path): Destination file path for the generated .tex file.
        rows (list[dict[str, Any]]): Sequence of row dictionaries. Each dictionary must contain
            the keys defined in ARTICLE_COLUMNS (the module-level ordering used to emit columns),
            with values convertible to strings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    latex_lines = [
        "% Auto-generated by mammography benchmark-report",
        "\\subsection{Tabela Consolidada do Rerun Oficial 2026Q1}",
        "\\label{sec:rerun-2026q1-table}",
        (
            "A tabela a seguir deriva automaticamente da tabela mestre do rerun oficial "
            "e resume um resultado por combinacao de dataset, tarefa e modelo."
        ),
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\scriptsize",
        "  \\caption{Rerun oficial 2026Q1 consolidado por dataset, tarefa e modelo.}",
        "  \\label{tab:rerun-2026q1-article}",
        "  \\begin{tabular}{llllrrrr}",
        "    \\toprule",
        _tex_header(ARTICLE_COLUMNS),
        "    \\midrule",
    ]
    for row in rows:
        latex_lines.append(
            "    "
            + " & ".join(_tex_escape(row[column]) for column in ARTICLE_COLUMNS)
            + " \\\\"
        )
    latex_lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")


def _current_environment() -> dict[str, str]:
    """
    Gather current runtime environment information used in report headers.

    Returns:
        dict[str, str]: Mapping with keys:
            - `python`: Python version string from platform.python_version().
            - `platform`: OS/platform string from platform.platform().
            - `torch`: PyTorch version string, `"unavailable"` if PyTorch is not importable, or `"unknown"` if the attribute is missing.
            - `cuda`: CUDA version string if available, `"cpu"` when CUDA is not present, or `"unavailable"` if PyTorch is not importable.
            - `gpu`: GPU device name when a CUDA device is available, `"cpu"` when no CUDA device is present, `"unknown"` if the GPU query fails, or `"unavailable"` if PyTorch is not importable.
    """
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": "unavailable",
        "cuda": "unavailable",
        "gpu": "unavailable",
    }
    if torch is None:
        return env

    env["torch"] = getattr(torch, "__version__", "unknown")
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    env["cuda"] = str(cuda_version or "cpu")
    try:
        env["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    except (RuntimeError, OSError):
        LOGGER.debug("Unable to resolve GPU environment details.", exc_info=True)
        env["gpu"] = "unknown"
    return env


def _environment_from_runs(runs: list[CollectedRun]) -> dict[str, str]:
    """
    Aggregate environment and reproducibility fields from a list of collected runs.

    Parameters:
        runs (list[CollectedRun]): Collected run records whose `summary["reproducibility"]` fields
            will be inspected for `python_version`, `platform`, `torch_version`, `cuda_version`,
            and `gpu_name`. Empty or missing values are ignored.

    Returns:
        dict[str, str]: Mapping with keys `"python"`, `"platform"`, `"torch"`, `"cuda"`, and `"gpu"`.
            For each key, returns a single value if all runs agree, a comma-separated list if
            multiple distinct values are found, or the corresponding value from the current
            environment when no run provides a value.
    """
    env = _current_environment()

    def _value_from_summaries(summary_key: str, fallback_key: str) -> str:
        """
        Collects distinct reproducibility values for `summary_key` from all runs and consolidates them into a single string.

        Parameters:
            summary_key (str): Key to read from each run's `summary["reproducibility"]` mapping.
            fallback_key (str): Key in the local `env` mapping to use if no run provides a value.

        Returns:
            str: The consolidated value: the fallback `env[fallback_key]` if no values are found, the sole value if exactly one distinct value exists, or a comma-separated list of distinct values when multiple exist.
        """
        values = sorted(
            {
                str(run.summary.get("reproducibility", {}).get(summary_key)).strip()
                for run in runs
                if str(
                    run.summary.get("reproducibility", {}).get(summary_key) or ""
                ).strip()
            }
        )
        if not values:
            return env[fallback_key]
        if len(values) == 1:
            return values[0]
        return ", ".join(values)

    return {
        "python": _value_from_summaries("python_version", "python"),
        "platform": _value_from_summaries("platform", "platform"),
        "torch": _value_from_summaries("torch_version", "torch"),
        "cuda": _value_from_summaries("cuda_version", "cuda"),
        "gpu": _value_from_summaries("gpu_name", "gpu"),
    }


def _render_docs_report(
    path: Path,
    namespace: Path,
    runs: list[CollectedRun],
    master_rows: list[dict[str, Any]],
) -> None:
    """
    Render a technical Markdown report summarizing validated runs, environment, and exported tables.

    Builds a report at `path` containing header metadata (namespace, number of validated runs, git commits, execution timestamps, and detected environment), a protocol summary, a run matrix derived from `runs`, dataset split limitations, a master table rendered from `master_rows`, and per-run export references. Creates parent directories for `path` if necessary and writes the report as UTF-8 text.

    Parameters:
        path (Path): Destination file path for the generated Markdown report.
        namespace (Path): Namespace or repository path to display in the report header.
        runs (list[CollectedRun]): Validated run objects; each run must expose `summary`, `expected`, `results_dir`, `export_dir`, and `export_manifest_path` used to populate the report.
        master_rows (list[dict[str, Any]]): Rows for the master table; rendered using the module's `MASTER_COLUMNS`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    env = _environment_from_runs(runs)
    commits = sorted(
        {
            str(run.summary.get("reproducibility", {}).get("git_commit", "unknown"))
            for run in runs
        }
    )
    execution_timestamps = sorted(
        {
            str(
                run.summary.get("created_at")
                or run.summary.get("reproducibility", {}).get("timestamp")
                or "unknown"
            )
            for run in runs
        }
    )

    matrix_rows = [
        {
            "dataset": run.expected.dataset,
            "task": run.expected.task,
            "split_mode": run.expected.split_mode,
            "arch": run.expected.arch,
            "run_path": str(run.results_dir),
            "status": "accepted",
        }
        for run in runs
    ]
    matrix_columns = ["dataset", "task", "split_mode", "arch", "run_path", "status"]

    lines = [
        "# Technical Report: rerun_2026q1",
        "",
        "## Header",
        f"- Namespace: `{namespace}`",
        f"- Official runs validated: `{len(runs)}`",
        f"- Git commits seen in summaries: `{', '.join(commits)}`",
        f"- Execution timestamps seen in summaries: `{', '.join(execution_timestamps)}`",
        f"- Python: `{env['python']}`",
        f"- PyTorch: `{env['torch']}`",
        f"- CUDA: `{env['cuda']}`",
        f"- GPU: `{env['gpu']}`",
        "",
        "## Protocol",
        "- Official matrix: 3 datasets x 2 tasks x 3 models, one seed (42).",
        "- Official metrics come from a held-out test split; the best checkpoint is still selected on validation macro-F1.",
        "- Common settings enforced: deterministic=true, amp=true, allow_tf32=true, pretrained=true, train_backbone=true, unfreeze_last_block=true, augment=true, class_weights=auto, sampler_weighted=true, test_frac=0.1, tracker=local.",
        "- CNN profile: img_size=512, batch_size=16, epochs=30, lr=1e-4, backbone_lr=1e-5, warmup_epochs=2, early_stop_patience=5.",
        "- ViT profile: img_size=224, batch_size=8, epochs=30, lr=1e-3, backbone_lr=1e-4, warmup_epochs=3, early_stop_patience=10.",
        "- Explainability artifacts are explicitly out of scope for rerun acceptance.",
        "",
        "## Run Matrix",
        *_markdown_lines(matrix_rows, matrix_columns),
        "",
        "## Split Limitations",
        "- `archive` uses `split_mode=patient`, and leakage validation is checked against saved split manifests when available.",
        "- `mamografias` uses `split_mode=random` because the current loader does not expose a reliable patient grouping key for this benchmark.",
        "- `patches_completo` uses `split_mode=random` for the same limitation: no reliable patient grouping key is exposed in the current data format.",
        "",
        "## Master Table",
        *_markdown_lines(master_rows, MASTER_COLUMNS),
        "",
        "## Export References",
    ]
    for run in runs:
        export_dir_text = (
            str(run.export_dir) if run.export_dir is not None else "not found"
        )
        export_manifest_text = (
            str(run.export_manifest_path)
            if run.export_manifest_path is not None
            else "not found"
        )
        lines.append(
            f"- `{run.expected.run_name}`: export_dir=`{export_dir_text}` | manifest=`{export_manifest_text}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
