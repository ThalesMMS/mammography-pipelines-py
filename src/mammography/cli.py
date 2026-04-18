#!/usr/bin/env python3
# DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
# It must NOT be used for clinical or medical diagnostic purposes.
# No medical decision should be based on these results.
"""CLI entrypoint for the mammography pipelines."""

from __future__ import annotations

import hashlib
import logging
import sys
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from pathlib import Path

from mammography import cli_config as _cli_config
from mammography import cli_dispatch as _cli_dispatch
from mammography.cli_config import (
    DEFAULT_CONFIGS,
    REPO_ROOT,
    _coerce_cli_args,
    _default_config,
    _dict_to_cli_args,
    _filter_embed_config_args,
    _forwarded_has_flag,
    _strip_flags_with_values,
)
from mammography.cli_dispatch import (
    _configure_logging,
    _entrypoint_accepts_args,
    _format_command,
    _invoke_entrypoint,
    _print_eval_guidance,
    _resolve_entrypoint,
    _run_module_passthrough,
    _working_directory,
)
from mammography.cli_parser import _build_parser

LOGGER = logging.getLogger("projeto")
yaml = _cli_config.yaml

__all__ = [
    "DEFAULT_CONFIGS",
    "LOGGER",
    "REPO_ROOT",
    "_build_parser",
    "_coerce_cli_args",
    "_configure_logging",
    "_default_config",
    "_dict_to_cli_args",
    "_entrypoint_accepts_args",
    "_filter_embed_config_args",
    "_format_command",
    "_forwarded_has_flag",
    "_invoke_entrypoint",
    "_load_config_args",
    "_print_eval_guidance",
    "_read_config",
    "_resolve_entrypoint",
    "_run_benchmark_report",
    "_run_command",
    "_run_data_audit",
    "_run_eval_export",
    "_run_module_passthrough",
    "_run_report_pack",
    "_run_visualize",
    "_strip_flags_with_values",
    "_working_directory",
    "main",
    "yaml",
]


_SENSITIVE_FORWARDED_FLAGS = {
    "config",
    "csv",
    "data-dir",
    "data_dir",
    "dicom-root",
    "input",
    "output",
    "patient-id",
    "patient_id",
    "series-uid",
    "series_uid",
    "study-uid",
    "study_uid",
    "test-csv",
    "test_csv",
    "train-csv",
    "train_csv",
    "val-csv",
    "val_csv",
}


def _hash_forwarded_value(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _normalize_forwarded_flag(flag: str) -> str:
    return flag.lstrip("-").replace("_", "-")


def _sanitize_forwarded_args(forwarded: Sequence[str]) -> dict[str, Any]:
    """Return a log-safe summary of forwarded CLI tokens without raw path/UID values."""
    tokens: list[str] = []
    sensitive_flags: set[str] = set()
    redact_next = False

    for token in forwarded:
        text = str(token)
        if redact_next:
            tokens.append(_hash_forwarded_value(text))
            redact_next = False
            continue

        if text.startswith("--"):
            flag, separator, value = text.partition("=")
            normalized = _normalize_forwarded_flag(flag)
            if normalized in _SENSITIVE_FORWARDED_FLAGS:
                sensitive_flags.add(flag)
                if separator:
                    tokens.append(f"{flag}={_hash_forwarded_value(value)}")
                else:
                    tokens.append(flag)
                    redact_next = True
                continue
            if separator:
                tokens.append(f"{flag}={_hash_forwarded_value(value)}")
            else:
                tokens.append(text)
            continue

        tokens.append(_hash_forwarded_value(text))

    return {
        "count": len(forwarded),
        "sensitive_flags": sorted(sensitive_flags),
        "tokens": tokens,
    }


def _call_config_with_local_yaml(name: str, *args: Any, **kwargs: Any) -> Any:
    original_yaml = _cli_config.yaml
    try:
        _cli_config.yaml = yaml
        return getattr(_cli_config, name)(*args, **kwargs)
    finally:
        _cli_config.yaml = original_yaml


def _read_config(path: Path) -> Any:
    """Read a configuration file via cli_config._read_config.

    Parameters:
        path (Path): pathlib.Path object pointing to the YAML configuration file.

    Returns:
        Parsed configuration payload returned by cli_config._read_config.
    """
    return _call_config_with_local_yaml("_read_config", path)


def _load_config_args(config_arg: Path | None, command: str) -> list[str]:
    """Load config-derived CLI arguments via cli_config._load_config_args.

    Parameters:
        config_arg (Path | None): Configuration pathlib.Path object or None to use defaults.
        command (str): CLI subcommand name used to select config options.

    Returns:
        list[str]: CLI argument tokens produced from the resolved configuration.
    """
    return _call_config_with_local_yaml("_load_config_args", config_arg, command)


def _call_dispatch_with_facade_patches(name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Call a cli_dispatch helper while preserving legacy monkeypatch points.

    Tests or external callers may replace this module's _run_module_passthrough
    before calling _call_dispatch_with_facade_patches. This function temporarily
    passes the current facade-level _run_module_passthrough into dispatch
    helpers that accept it, along with temporarily patched config helpers below.
    """
    original_filter_embed_config_args = _cli_dispatch._filter_embed_config_args
    original_load_config_args = _cli_dispatch._load_config_args
    try:
        _cli_dispatch._filter_embed_config_args = _filter_embed_config_args
        _cli_dispatch._load_config_args = _load_config_args
        return getattr(_cli_dispatch, name)(*args, **kwargs)
    finally:
        _cli_dispatch._filter_embed_config_args = original_filter_embed_config_args
        _cli_dispatch._load_config_args = original_load_config_args


def _run_command(
    module: str,
    args: Any,
    forwarded: Sequence[str],
    entrypoint: str | None = None,
) -> int:
    return _call_dispatch_with_facade_patches(
        "_run_command",
        module,
        args,
        forwarded,
        entrypoint=entrypoint,
        run_module_passthrough=_run_module_passthrough,
    )


def _run_eval_export(args: Any, forwarded: Sequence[str]) -> int:
    return _call_dispatch_with_facade_patches(
        "_run_eval_export",
        args,
        forwarded,
        run_module_passthrough=_run_module_passthrough,
    )


def _run_report_pack(args: Any, forwarded: Sequence[str]) -> int:
    return _call_dispatch_with_facade_patches("_run_report_pack", args, forwarded)


def _run_data_audit(args: Any, forwarded: Sequence[str]) -> int:
    return _call_dispatch_with_facade_patches(
        "_run_data_audit",
        args,
        forwarded,
        run_module_passthrough=_run_module_passthrough,
    )


def _run_visualize(args: Any, forwarded: Sequence[str]) -> int:
    return _call_dispatch_with_facade_patches(
        "_run_visualize",
        args,
        forwarded,
        run_module_passthrough=_run_module_passthrough,
    )


def _run_benchmark_report(args: Any, forwarded: Sequence[str]) -> int:
    return _call_dispatch_with_facade_patches(
        "_run_benchmark_report",
        args,
        forwarded,
        run_module_passthrough=_run_module_passthrough,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments, dispatch the chosen subcommand, and return an exit code.

    Parameters:
        argv: Optional list of command-line arguments; defaults to ``sys.argv``.

    Returns:
        Exit code produced by the dispatched subcommand (0 indicates success).
    """
    parser = _build_parser()
    args, forwarded = parser.parse_known_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    _configure_logging(args.log_level)
    LOGGER.debug("Forwarded args summary: %s", _sanitize_forwarded_args(forwarded))

    try:
        if args.command == "embed":
            return _run_command(
                "mammography.commands.extract_features", args, forwarded
            )
        if args.command == "train-density":
            return _run_command("mammography.commands.train", args, forwarded)
        if args.command == "eval-export":
            return _run_eval_export(args, forwarded)
        if args.command == "report-pack":
            return _run_report_pack(args, forwarded)
        if args.command == "data-audit":
            return _run_data_audit(args, forwarded)
        if args.command == "visualize":
            return _run_visualize(args, forwarded)
        if args.command == "explain":
            return _run_command("mammography.commands.explain", args, forwarded)
        if args.command == "wizard":
            from mammography import wizard

            return wizard.run_wizard(dry_run=args.dry_run)
        if args.command == "inference":
            return _run_command("mammography.commands.inference", args, forwarded)
        if args.command == "augment":
            return _run_command("mammography.commands.augment", args, forwarded)
        if args.command == "preprocess":
            return _run_command("mammography.commands.preprocess", args, forwarded)
        if args.command == "label-density":
            return _run_command("mammography.commands.label_density", args, forwarded)
        if args.command == "label-patches":
            return _run_command("mammography.commands.label_patches", args, forwarded)
        if args.command == "web":
            return _run_command("mammography.commands.web", args, forwarded)
        if args.command == "eda-cancer":
            return _run_command(
                "mammography.commands.eda_cancer",
                args,
                forwarded,
                entrypoint="run_density_classifier_cli",
            )
        if args.command == "embeddings-baselines":
            return _run_command(
                "mammography.commands.embeddings_baselines", args, forwarded
            )
        if args.command == "tune":
            return _run_command("mammography.commands.tune", args, forwarded)
        if args.command == "cross-validate":
            return _run_command("mammography.commands.cross_validate", args, forwarded)
        if args.command == "batch-inference":
            return _run_command("mammography.commands.batch_inference", args, forwarded)
        if args.command == "compare-models":
            return _run_command("mammography.commands.compare_models", args, forwarded)
        if args.command == "benchmark-report":
            return _run_benchmark_report(args, forwarded)
        if args.command == "automl":
            return _run_command("mammography.commands.automl", args, forwarded)
        parser.error(f"Subcomando desconhecido: {args.command}")
    except SystemExit as exc:
        if isinstance(exc.code, str):
            LOGGER.error("%s", exc.code)
            return 1
        return 0 if exc.code is None else int(exc.code)


if __name__ == "__main__":
    sys.exit(main())
