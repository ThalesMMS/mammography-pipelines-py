#!/usr/bin/env python3
"""CLI configuration helpers for the mammography pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from mammography.utils import config_loader as _config_loader
from mammography.utils.config_loader import (
    DEFAULT_CONFIGS,
    coerce_cli_args,
    dict_to_cli_args,
    load_config_args,
    read_config,
)

LOGGER = logging.getLogger("projeto")
REPO_ROOT = Path(__file__).resolve().parents[2]
yaml = _config_loader.yaml


def _read_config(path: Path) -> Any:
    """
    Read and parse a YAML configuration file at the given path using the module's YAML loader.

    Parameters:
        path (Path): Path to the YAML configuration file to read.

    Returns:
        Parsed configuration contents (typically a dict or list depending on the YAML structure).
    """
    original_yaml = _config_loader.yaml
    try:
        _config_loader.yaml = yaml
        return read_config(path)
    finally:
        _config_loader.yaml = original_yaml


def _dict_to_cli_args(payload: dict[str, Any]) -> list[str]:
    """
    Convert a configuration mapping into a list of command-line argument tokens.

    Parameters:
        payload (dict[str, Any]): Mapping of option names to values; keys become flags and values become corresponding flag values.

    Returns:
        list[str]: Sequence of CLI-style tokens representing the mapping (e.g. `['--flag', 'value', '--other=val']`).
    """
    return dict_to_cli_args(payload)


def _coerce_cli_args(payload: Any) -> list[str]:
    """
    Convert a configuration payload into a list of CLI-style argument tokens.

    Parameters:
        payload (Any): Configuration value(s) (mapping, sequence, or scalar) to be coerced into CLI tokens.

    Returns:
        list[str]: CLI-style argument tokens derived from the payload.
    """
    return coerce_cli_args(payload)


def _default_config(command: str) -> Path | None:
    """
    Select the default configuration file path for a given subcommand if one exists on disk.

    Parameters:
        command (str): Subcommand name used to look up a candidate default config.

    Returns:
        Path | None: The path to the default config for `command` if it exists, `None` otherwise.
    """
    candidate = DEFAULT_CONFIGS.get(command)
    if candidate and candidate.exists():
        return candidate
    return None


def _load_config_args(config_arg: Path | None, command: str) -> list[str]:
    """
    Load CLI-style argument tokens derived from a configuration for a specific subcommand.

    Parameters:
        config_arg (Path | None): Path to a configuration file to load, or None to use defaults.
        command (str): Subcommand name used to select and serialize relevant config options.

    Returns:
        list[str]: A list of CLI argument tokens produced from the resolved configuration.
    """
    original_yaml = _config_loader.yaml
    try:
        _config_loader.yaml = yaml
        return load_config_args(config_arg, command, logger=LOGGER)
    finally:
        _config_loader.yaml = original_yaml


def _forwarded_has_flag(forwarded: Sequence[str], flag: str) -> bool:
    """
    Determine whether a sequence of forwarded CLI tokens includes a specific flag either as a standalone token or in `flag=value` form.

    Parameters:
        forwarded (Sequence[str]): Sequence of forwarded CLI tokens.
        flag (str): Flag to search for (e.g., `--dataset`).

    Returns:
        `True` if an exact `flag` token is present or any token starts with `flag=`; `False` otherwise.
    """
    flag_prefix = f"{flag}="
    return any(token == flag or token.startswith(flag_prefix) for token in forwarded)


def _strip_flags_with_values(args: Sequence[str], flags: set[str]) -> list[str]:
    """
    Remove specified flags and their values from a sequence of CLI tokens.

    Parameters:
        args (Sequence[str]): Sequence of command-line tokens (e.g., sys.argv-style).
        flags (set[str]): Flags to remove. Tokens equal to a flag and the immediate next token
            (assumed to be the flag's value) are removed, as are tokens starting with `flag=`.

    Returns:
        list[str]: A new list of tokens with the specified flags and their associated values removed.
    """
    if not flags:
        return list(args)
    cleaned: list[str] = []
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token in flags:
            skip_next = True
            continue
        if any(token.startswith(f"{flag}=") for flag in flags):
            continue
        cleaned.append(token)
    return cleaned


def _filter_embed_config_args(
    config_args: Sequence[str], forwarded: Sequence[str]
) -> list[str]:
    """
    Remove dataset/source override flags from configuration-derived CLI arguments when the user explicitly selected a dataset or data directory.

    If `forwarded` does not include `--dataset` or `--data_dir`, returns `config_args` unchanged. If the user forwarded a dataset or data directory, this function drops config-provided `--dataset`, `--csv`, and/or `--dicom-root` flags as appropriate so forwarded selections take precedence.

    Parameters:
        config_args (Sequence[str]): CLI-style tokens produced from a configuration file.
        forwarded (Sequence[str]): Tokens forwarded from the user's command-line invocation.

    Returns:
        list[str]: A new list of CLI tokens with the relevant dataset/source flags and their values removed.
    """
    if not config_args:
        return list(config_args)
    dataset_flag = _forwarded_has_flag(forwarded, "--dataset")
    data_dir_flag = _forwarded_has_flag(forwarded, "--data_dir")
    csv_flag = _forwarded_has_flag(forwarded, "--csv")
    dicom_root_flag = _forwarded_has_flag(forwarded, "--dicom-root")

    if not (dataset_flag or data_dir_flag):
        return list(config_args)

    flags_to_drop: set[str] = set()
    if dataset_flag:
        flags_to_drop.add("--dataset")
    if data_dir_flag:
        flags_to_drop.add("--dataset")
    if csv_flag:
        flags_to_drop.add("--csv")
    if dicom_root_flag:
        flags_to_drop.add("--dicom-root")

    if not csv_flag:
        flags_to_drop.add("--csv")
        flags_to_drop.add("--dicom-root")

    return _strip_flags_with_values(config_args, flags_to_drop)
