#!/usr/bin/env python3
#
# cli_dispatch.py
# mammography-pipelines
#
# Dispatches CLI subcommands for embedding extraction, density training,
# eval export, reporting, and visualization.
#
"""CLI dispatch helpers for the mammography pipelines."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import shlex
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

if TYPE_CHECKING:
    import argparse

from mammography.cli_config import (
    REPO_ROOT,
    _filter_embed_config_args,
    _load_config_args,
)
from mammography.utils.cli_args import serialize_tracking_args

LOGGER = logging.getLogger("projeto")


def _cli_token(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _configure_logging(level: str) -> None:
    """
    Configure the root logger's level and message format used by the CLI.

    Parameters:
        level (str): Logging level name (e.g., "DEBUG", "INFO"); case-insensitive. If the name is not recognized, defaults to `INFO`.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s | %(message)s", force=True)


def _format_command(command: Sequence[str]) -> str:
    """
    Create a shell-safe command string by quoting each token for easy copy/paste.

    Returns:
        str: A single string where each token is shell-quoted and joined by spaces.
    """
    return " ".join(shlex.quote(_cli_token(part)) for part in command)


@contextmanager
def _working_directory(path: Path) -> Iterator[Path]:
    """
    Temporarily change the current working directory to `path` and restore the original directory on exit.

    Parameters:
        path (Path): Directory to switch to for the context.

    Returns:
        original (Path): The working directory that was active before entering the context.
    """
    current = Path.cwd()
    os.chdir(path)
    try:
        yield current
    finally:
        os.chdir(current)


def _resolve_entrypoint(
    module: str, entrypoint: str | None = None
) -> Callable[..., Any]:
    """
    Locate and return a callable entrypoint from an importable module.

    Parameters:
        module (str): Dotted import path of the target module (e.g., "package.module").
        entrypoint (str | None): Optional attribute name to use as the entrypoint. If omitted, the function will prefer an attribute named "main" and then "run".

    Returns:
        Callable[..., Any]: The resolved callable entrypoint.

    Raises:
        AttributeError: If the specified attribute (or the preferred defaults) is not found or is not callable.
    """
    module_obj = importlib.import_module(module)
    if entrypoint:
        handler = getattr(module_obj, entrypoint)
    else:
        handler = getattr(module_obj, "main", None) or getattr(module_obj, "run", None)
    if not callable(handler):
        raise AttributeError(f"Entrypoint não encontrado em {module}.")
    return handler


def _entrypoint_accepts_args(handler: Callable[..., Any]) -> bool:
    """
    Determine whether a callable can accept an argv-like payload (either var-positional/var-keyword or any declared parameters).

    Returns:
        `true` if the handler accepts `*args`/`**kwargs` or defines one or more parameters, `false` otherwise.
    """
    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            return True
    return bool(sig.parameters)


def _invoke_entrypoint(handler: Callable[..., Any], cmd_args: Sequence[str]) -> int:
    """
    Invoke a discovered entrypoint callable, supplying argv-like tokens when the callable accepts parameters.

    Parameters:
        handler (Callable[..., Any]): The entrypoint callable to invoke.
        cmd_args (Sequence[str]): Argument tokens to forward to the handler if it accepts arguments.

    Returns:
        int: The exit code produced by the handler. If the handler returns an integer, that value is returned; if it returns `None` or any non-integer value, `0` is returned.
    """
    result = handler(list(cmd_args)) if _entrypoint_accepts_args(handler) else handler()
    if result is None:
        return 0
    if isinstance(result, int) and not isinstance(result, bool):
        return result
    return 0


def _run_command(
    module: str,
    args: argparse.Namespace,
    forwarded: Sequence[str],
    entrypoint: str | None = None,
    *,
    run_module_passthrough: Callable[..., int] | None = None,
) -> int:
    """
    Run an internal command module after combining configuration-derived arguments with forwarded CLI tokens.

    Parameters:
        module (str): Import path of the target module to execute.
        args (argparse.Namespace): Parsed CLI namespace containing command-specific options.
        forwarded (Sequence[str]): Raw CLI tokens to forward to the target module.
        entrypoint (str | None): Optional callable name in the target module to invoke; if omitted a default entrypoint is resolved.

    Returns:
        int: Exit code returned by the invoked entrypoint; `0` indicates success.
    """
    config_args = _load_config_args(getattr(args, "config", None), args.command)
    if args.command == "embed":
        config_args = _filter_embed_config_args(config_args, forwarded)
    cmd_args = [*config_args, *forwarded]
    runner = run_module_passthrough or _run_module_passthrough
    return runner(module, args, cmd_args, entrypoint=entrypoint)


def _run_module_passthrough(
    module: str,
    args: argparse.Namespace,
    cmd_args: Sequence[str],
    entrypoint: str | None = None,
) -> int:
    """
    Execute an internal module's entrypoint in-process using the provided command tokens.

    Logs the formatted command, changes the working directory to the repository root, and invokes the resolved entrypoint. If `args.dry_run` is true, logs a dry-run message and skips execution.

    Parameters:
        module (str): Import path of the target module (e.g., "mammography.commands.foo").
        args (argparse.Namespace): Parsed CLI namespace; `args.dry_run` is consulted.
        cmd_args (Sequence[str]): Argument tokens to forward to the entrypoint.
        entrypoint (str | None): Optional callable name to resolve inside the module; if omitted a default entrypoint is used.

    Returns:
        int: Exit code produced by the invoked entrypoint, or `0` when execution is skipped due to dry-run or when the handler yields no explicit code.
    """
    command = [module, *cmd_args]
    LOGGER.info("Executando (in-process): %s", _format_command(command))
    if args.dry_run:
        LOGGER.info("Dry-run habilitado; comando não será executado.")
        return 0
    handler = _resolve_entrypoint(module, entrypoint=entrypoint)
    with _working_directory(REPO_ROOT):
        return _invoke_entrypoint(handler, cmd_args)


def _print_eval_guidance(args: argparse.Namespace, forwarded: Sequence[str]) -> int:
    """
    Log an evaluation/export checklist and audit required artifacts for provided run paths.

    This function logs suggested configuration-derived arguments and any additional forwarded CLI tokens, emits a fixed checklist of expected evaluation/export artifacts, and for each path in `args.runs` verifies the presence of a set of required files. If a run's summary.json is present, it parses and logs key validation metrics (`seed`, `accuracy`, `kappa_quadratic`, `macro_f1`, `auc_ovr`).

    Parameters:
        args (argparse.Namespace): Parsed CLI namespace; may include `config`, `command`, and `runs` (an iterable of Path or str run locations).
        forwarded (Sequence[str]): Additional CLI tokens that were forwarded but are ignored by this audit.

    Returns:
        int: `0` on completion.
    """
    config_args = _load_config_args(getattr(args, "config", None), args.command)
    if config_args:
        LOGGER.info("Argumentos sugeridos (config): %s", " ".join(config_args))
    if forwarded:
        LOGGER.info("Argumentos adicionais ignorados: %s", " ".join(forwarded))
    checklist = [
        "Reutilize checkpoints aprovados (outputs/mammo_efficientnetb0_density/results_*).",
        "Exporte val_predictions.csv, embeddings_val.*, metrics/val_metrics.{json,png}.",
        "Gere figuras (ROC, confusion, Grad-CAM) e copie para Article/assets via report-pack.",
        "Versione os arquivos (timestamp + git SHA) na pasta Article/assets/.",
    ]
    LOGGER.info("Checklist de exportação:")
    for item in checklist:
        LOGGER.info(" • %s", item)
    run_list = getattr(args, "runs", None) or []
    if run_list:
        required = [
            "summary.json",
            "train_history.csv",
            "train_history.png",
            "metrics/val_metrics.json",
            "metrics/val_metrics.png",
            "val_predictions.csv",
            "embeddings_val.csv",
            "embeddings_val.npy",
            "run.log",
            "best_model.pt",
        ]
        for run in run_list:
            run_path = (REPO_ROOT / run) if not run.is_absolute() else run
            LOGGER.info("Auditando artefatos em: %s", run_path)
            missing = [rel for rel in required if not (run_path / rel).exists()]
            if missing:
                LOGGER.warning("Itens faltantes: %s", ", ".join(missing))
            else:
                LOGGER.info("Todos os arquivos obrigatórios estão presentes.")
            summary_path = run_path / "summary.json"
            if summary_path.exists():
                try:
                    payload = json.loads(summary_path.read_text(encoding="utf-8"))
                    legacy_metrics = payload.get("val_metrics", {})

                    def _summary_metric(
                        name: str,
                        payload_snapshot: dict[str, Any] = payload,
                        legacy_snapshot: Any = legacy_metrics,
                    ) -> float:
                        value = payload_snapshot.get(name)
                        if value is None and isinstance(legacy_snapshot, dict):
                            value = legacy_snapshot.get(name)
                        return float(value or 0.0)

                    LOGGER.info(
                        "Resumo: seed=%s | acc=%.3f | κ=%.3f | macro-F1=%.3f | AUC=%.3f",
                        payload.get("seed", "?"),
                        _summary_metric("accuracy"),
                        _summary_metric("kappa_quadratic"),
                        _summary_metric("macro_f1"),
                        _summary_metric("auc_ovr"),
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Falha ao ler summary.json em %s: %s", summary_path, exc
                    )
    return 0


def _run_eval_export(
    args: argparse.Namespace,
    forwarded: Sequence[str],
    *,
    run_module_passthrough: Callable[..., int] | None = None,
) -> int:
    """
    Assemble eval-export command-line arguments from the dispatcher namespace and forwarded tokens, then execute the eval-export subcommand in-process.

    Parameters:
        args (argparse.Namespace): Parsed dispatcher arguments (may include `config`, `command`, `runs`, `output_dir`, and tracking options) used to build subcommand flags.
        forwarded (Sequence[str]): Raw CLI tokens to forward directly to the eval-export subcommand.

    Returns:
        int: Exit code returned by the invoked subcommand; `0` if the handler returns `None` or a non-integer value.
    """
    cmd_args: list[str] = []
    cmd_args.extend(_load_config_args(getattr(args, "config", None), args.command))
    if getattr(args, "runs", None):
        for run in args.runs:
            cmd_args.extend(["--run", _cli_token(run)])
    if hasattr(args, "output_dir") and args.output_dir:
        cmd_args.extend(["--output-dir", _cli_token(args.output_dir)])
    cmd_args.extend(serialize_tracking_args(args))
    cmd_args.extend(forwarded)
    runner = run_module_passthrough or _run_module_passthrough
    return runner("mammography.commands.eval_export", args, cmd_args)


def _run_visualize(
    args: argparse.Namespace,
    forwarded: Sequence[str],
    *,
    run_module_passthrough: Callable[..., int] | None = None,
) -> int:
    """
    Assemble command-line arguments for the visualize subcommand and invoke the in-process visualize handler.

    Parameters:
        args (argparse.Namespace): Parsed CLI namespace containing visualization options (e.g., input, labels, tsne, output, seed).
        forwarded (Sequence[str]): Additional raw CLI tokens to forward to the visualization handler.

    Returns:
        int: Exit code produced by the visualize handler (`0` indicates success).
    """
    cmd_args: list[str] = []
    cmd_args.extend(_load_config_args(getattr(args, "config", None), args.command))

    # Build command from parsed arguments
    if input_path := getattr(args, "input", None):
        cmd_args.extend(["--input", _cli_token(input_path)])
    if labels_path := getattr(args, "labels", None):
        cmd_args.extend(["--labels", _cli_token(labels_path)])
        if label_col := getattr(args, "label_col", None):
            cmd_args.extend(["--label-col", label_col])
    if predictions_path := getattr(args, "predictions", None):
        cmd_args.extend(["--predictions", _cli_token(predictions_path)])
    if history_path := getattr(args, "history", None):
        cmd_args.extend(["--history", _cli_token(history_path)])
    if getattr(args, "from_run", False):
        cmd_args.append("--from-run")
    if output_path := getattr(args, "output", None):
        cmd_args.extend(["--output", _cli_token(output_path)])
    if prefix := getattr(args, "prefix", None):
        cmd_args.extend(["--prefix", prefix])
    if getattr(args, "report", False):
        cmd_args.append("--report")
    if getattr(args, "tsne", False):
        cmd_args.append("--tsne")
    if getattr(args, "tsne_3d", False):
        cmd_args.append("--tsne-3d")
    if getattr(args, "pca", False):
        cmd_args.append("--pca")
    if getattr(args, "umap", False):
        cmd_args.append("--umap")
    if getattr(args, "compare_embeddings", False):
        cmd_args.append("--compare-embeddings")
    if getattr(args, "heatmap", False):
        cmd_args.append("--heatmap")
    if getattr(args, "feature_heatmap", False):
        cmd_args.append("--feature-heatmap")
    if getattr(args, "confusion_matrix", False):
        cmd_args.append("--confusion-matrix")
    if getattr(args, "scatter_matrix", False):
        cmd_args.append("--scatter-matrix")
    if getattr(args, "distribution", False):
        cmd_args.append("--distribution")
    if getattr(args, "class_separation", False):
        cmd_args.append("--class-separation")
    if getattr(args, "learning_curves", False):
        cmd_args.append("--learning-curves")
    if getattr(args, "binary", False):
        cmd_args.append("--binary")
    if (perplexity := getattr(args, "perplexity", None)) is not None:
        cmd_args.extend(["--perplexity", str(perplexity)])
    if (tsne_iter := getattr(args, "tsne_iter", None)) is not None:
        cmd_args.extend(["--tsne-iter", str(tsne_iter)])
    if (seed := getattr(args, "seed", None)) is not None:
        cmd_args.extend(["--seed", str(seed)])
    if pca_svd_solver := getattr(args, "pca_svd_solver", None):
        cmd_args.extend(["--pca-svd-solver", pca_svd_solver])
    cmd_args.extend(serialize_tracking_args(args))

    # Add any forwarded arguments
    cmd_args.extend(forwarded)

    runner = run_module_passthrough or _run_module_passthrough
    return runner("mammography.commands.visualize", args, cmd_args)


def _run_benchmark_report(
    args: argparse.Namespace,
    forwarded: Sequence[str],
    *,
    run_module_passthrough: Callable[..., int] | None = None,
) -> int:
    """
    Assemble CLI arguments for the benchmark-report tool and invoke its in-process entrypoint.

    Parameters:
        args (argparse.Namespace): Parsed CLI namespace; may contain `config`, `namespace`, `output_prefix`, `docs_report`, `article_table`, and `exports_search_root` which are converted into forwarded flags when present.
        forwarded (Sequence[str]): Additional raw CLI tokens to append to the constructed command.

    Returns:
        int: Exit code produced by the invoked benchmark-report entrypoint (0 indicates success).
    """
    cmd_args: list[str] = []
    cmd_args.extend(_load_config_args(getattr(args, "config", None), args.command))
    if hasattr(args, "namespace") and args.namespace:
        cmd_args.extend(["--namespace", str(args.namespace)])
    if hasattr(args, "output_prefix") and args.output_prefix:
        cmd_args.extend(["--output-prefix", str(args.output_prefix)])
    if hasattr(args, "docs_report") and args.docs_report:
        cmd_args.extend(["--docs-report", str(args.docs_report)])
    if hasattr(args, "article_table") and args.article_table:
        cmd_args.extend(["--article-table", str(args.article_table)])
    if hasattr(args, "exports_search_root") and args.exports_search_root:
        cmd_args.extend(["--exports-search-root", str(args.exports_search_root)])
    cmd_args.extend(forwarded)
    runner = run_module_passthrough or _run_module_passthrough
    return runner("mammography.commands.benchmark_report", args, cmd_args)


def _run_data_audit(
    args: argparse.Namespace,
    forwarded: Sequence[str],
    *,
    run_module_passthrough: Callable[..., int] | None = None,
) -> int:
    """
    Assemble command-line arguments from the parsed namespace and run the data-audit tool.

    Parameters:
        args (argparse.Namespace): Parsed CLI namespace; used for config-derived arguments and optional flags such as
            `archive`, `csv`, `manifest`, `audit_csv`, and `log`.
        forwarded (Sequence[str]): Additional CLI tokens to append and forward to the underlying tool.

    Returns:
        int: Exit code returned by the invoked data-audit entrypoint.
    """
    cmd_args: list[str] = []
    cmd_args.extend(_load_config_args(getattr(args, "config", None), args.command))
    if hasattr(args, "archive") and args.archive:
        cmd_args.extend(["--archive", _cli_token(args.archive)])
    if hasattr(args, "csv") and args.csv:
        cmd_args.extend(["--csv", _cli_token(args.csv)])
    if hasattr(args, "manifest") and args.manifest:
        cmd_args.extend(["--manifest", _cli_token(args.manifest)])
    if hasattr(args, "audit_csv") and args.audit_csv:
        cmd_args.extend(["--audit-csv", _cli_token(args.audit_csv)])
    if hasattr(args, "log") and args.log:
        cmd_args.extend(["--log", _cli_token(args.log)])
    cmd_args.extend(forwarded)
    runner = run_module_passthrough or _run_module_passthrough
    return runner("mammography.tools.data_audit", args, cmd_args)


def _run_report_pack(args: argparse.Namespace, forwarded: Sequence[str]) -> int:
    """
    Package one or more density run outputs into an assets directory and optionally register the package in the report-pack registry.

    This normalizes any relative `--run`, `--assets-dir`, and `--tex` paths against the repository root, calls mammography.tools.report_pack.package_density_runs to produce packaged assets, and—unless `args.no_registry` is true—registers the resulting package with mammography.tools.report_pack_registry (including a human-readable command, computed run name, and tracked outputs). If extra CLI tokens are provided in `forwarded`, they are logged and ignored.

    Parameters:
        args (argparse.Namespace): Parsed CLI namespace. Expected attributes used by this function include:
            - runs: sequence of run paths to package (required).
            - assets_dir: output assets directory path.
            - tex_path: optional path to a .tex file to include.
            - gradcam_limit: optional int limit passed to packaging.
            - no_registry: if true, skip registry registration.
            - run_name: optional explicit name to register; otherwise a default name is inferred.
            - registry_csv, registry_md: optional registry output paths passed to registration.
            - tracking_uri, experiment, no_mlflow: tracking-related options passed to registration.
        forwarded (Sequence[str]): Extra CLI tokens forwarded to the dispatcher; any entries here are ignored by report-pack and will be logged.

    Returns:
        int: `0` on successful packaging (and optional registration).
    """
    if forwarded:
        LOGGER.warning(
            "Argumentos adicionais ignorados pelo report-pack: %s", " ".join(forwarded)
        )
    if not args.runs:
        raise SystemExit(
            "Informe pelo menos um --run outputs/.../results_* para empacotar."
        )

    from mammography.tools import report_pack, report_pack_registry

    run_paths = []
    for run in args.runs:
        path = (REPO_ROOT / run) if not run.is_absolute() else run
        run_paths.append(path)
    assets_dir = (
        (REPO_ROOT / args.assets_dir)
        if not args.assets_dir.is_absolute()
        else args.assets_dir
    )
    tex_path = None
    if args.tex_path:
        tex_path = (
            (REPO_ROOT / args.tex_path)
            if not args.tex_path.is_absolute()
            else args.tex_path
        )

    LOGGER.info("Empacotando runs: %s", ", ".join(str(p) for p in run_paths))
    summarized = report_pack.package_density_runs(
        run_paths,
        assets_dir,
        tex_path=tex_path,
        gradcam_limit=args.gradcam_limit,
    )
    if args.no_registry:
        return 0
    asset_names = sorted(
        {asset for run in summarized for asset in run.assets.values() if asset}
    )
    output_paths = [
        path for path in (assets_dir / name for name in asset_names) if path.exists()
    ]
    if tex_path and tex_path.exists():
        output_paths.append(tex_path)
    command_parts = ["mammography", "report-pack"]
    for run in args.runs:
        command_parts.extend(["--run", _cli_token(run)])
    if args.assets_dir:
        command_parts.extend(["--assets-dir", _cli_token(args.assets_dir)])
    if args.tex_path:
        command_parts.extend(["--tex", _cli_token(args.tex_path)])
    if args.gradcam_limit is not None:
        command_parts.extend(["--gradcam-limit", str(args.gradcam_limit)])
    command_parts.extend(serialize_tracking_args(args))
    command = _format_command(command_parts)
    run_name = args.run_name or report_pack_registry.default_run_name(
        report_pack_registry.infer_dataset_name(run_paths)
    )
    report_pack_registry.register_report_pack_run(
        run_paths=run_paths,
        assets_dir=assets_dir,
        tex_path=tex_path,
        output_paths=output_paths,
        run_name=run_name,
        command=command,
        registry_csv=args.registry_csv,
        registry_md=args.registry_md,
        tracking_uri=args.tracking_uri or None,
        experiment=args.experiment or None,
        log_mlflow=not args.no_mlflow,
    )
    return 0
