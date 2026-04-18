"""YAML/JSON config loading helpers for CLI forwarding."""

from __future__ import annotations

import json
import logging
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:  # Optional dependency for YAML configs.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML is not guaranteed to exist.
    yaml = None

LOGGER = logging.getLogger("projeto")
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIGS: dict[str, Path | None] = {
    "embed": REPO_ROOT / "configs" / "paths.yaml",
    "train-density": REPO_ROOT / "configs" / "density.yaml",
    "eval-export": None,
    "visualize": None,
    "explain": None,
    "embeddings-baselines": None,
    "data-audit": None,
    "tune": None,
    "preprocess": None,
    "cross-validate": None,
    "batch-inference": None,
    "compare-models": None,
    "benchmark-report": None,
    "automl": None,
}


def read_config(path: Path) -> Any:
    """Read a YAML/JSON config file and return its payload."""
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    if not text.strip():
        return {}
    return json.loads(text)


def dict_to_cli_args(payload: dict[str, Any]) -> list[str]:
    """Convert a dictionary of arguments into flat CLI flags."""
    args: list[str] = []
    for key, value in payload.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                args.extend([flag, str(item)])
            continue
        args.extend([flag, str(value)])
    return args


def coerce_cli_args(payload: Any) -> list[str]:
    """Accept bool/iterable/string inputs and normalize to a list of CLI tokens."""
    if payload is None:
        return []
    if isinstance(payload, str):
        return shlex.split(payload)
    if isinstance(payload, dict):
        return dict_to_cli_args(payload)
    if isinstance(payload, (list, tuple)):
        return [str(item) for item in payload]
    return [str(payload)]


def _default_config(
    command: str,
    default_configs: Mapping[str, Path | None],
) -> Path | None:
    candidate = default_configs.get(command)
    if candidate and candidate.exists():
        return candidate
    return None


def load_config_args(
    config_arg: Path | None,
    command: str,
    *,
    default_configs: Mapping[str, Path | None] = DEFAULT_CONFIGS,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Load YAML/JSON payloads and convert them to CLI arguments for forwarding."""
    active_logger = logger or LOGGER
    config_path = config_arg or _default_config(command, default_configs)
    if not config_path:
        return []
    resolved = config_path.resolve()
    if not resolved.exists():
        active_logger.warning("Config %s nao encontrado; ignorando.", resolved)
        return []
    try:
        data = read_config(resolved) or {}
    except Exception as exc:
        active_logger.warning("Falha ao ler %s: %s", resolved, exc)
        return []

    args: list[str] = []
    if isinstance(data, dict):
        global_payload = data.get("global")
        if global_payload is not None:
            args.extend(coerce_cli_args(global_payload))
        normalized_keys = {command, command.replace("-", "_")}
        for key in normalized_keys:
            if key in data:
                args.extend(coerce_cli_args(data[key]))
    else:
        args.extend(coerce_cli_args(data))

    if args:
        active_logger.debug("Args carregados de %s (%s): %s", resolved, command, args)
    return args
