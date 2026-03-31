"""Shared helpers for density-class task mode normalization."""

from __future__ import annotations

import argparse
import warnings
from typing import Callable

CANONICAL_CLASS_MODES: tuple[str, ...] = ("multiclass", "binary")
CLASS_MODE_ALIASES: dict[str, str] = {"density": "multiclass"}
VISIBLE_CLASS_MODES_METAVAR = "{multiclass,binary}"
CLASS_MODE_HELP = "multiclass = BI-RADS 1..4, binary = A/B vs C/D"


def _deprecated_density_message(source: str) -> str:
    return (
        f"'density' is deprecated for {source}; use 'multiclass' instead. "
        "Support for 'density' will be removed in a future release."
    )


def normalize_classes_mode(
    mode: str | None,
    *,
    default: str = "multiclass",
    warn: bool = False,
    source: str = "classes",
    allow_unknown: bool = False,
) -> str:
    """Return the canonical classes mode while preserving optional passthrough values."""

    raw_mode = default if mode is None else mode
    normalized = str(raw_mode).strip().lower()
    if not normalized:
        normalized = default

    aliased = CLASS_MODE_ALIASES.get(normalized)
    if aliased is not None:
        if warn:
            warnings.warn(_deprecated_density_message(source), FutureWarning, stacklevel=3)
        return aliased

    if normalized in CANONICAL_CLASS_MODES:
        return normalized

    if allow_unknown:
        return normalized

    valid = ", ".join(CANONICAL_CLASS_MODES)
    raise ValueError(f"Invalid classes mode '{mode}'. Expected one of: {valid}.")


def parse_classes_mode_arg(value: str) -> str:
    """Argparse parser that accepts the legacy density alias but stores canonical values."""

    try:
        return normalize_classes_mode(value, warn=True, source="CLI --classes/--task")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def classes_mode_aliases(mode: str | None) -> tuple[str, ...]:
    """Return accepted storage labels for a canonical task name."""

    canonical = normalize_classes_mode(mode)
    if canonical == "multiclass":
        return ("multiclass", "density")
    return (canonical,)


def get_num_classes(mode: str | None) -> int:
    """Return the model output size for a supported task mode."""

    return 2 if normalize_classes_mode(mode) == "binary" else 4


def _binary_label_mapper(y: int) -> int:
    if y in [1, 2]:
        return 0
    if y in [3, 4]:
        return 1
    return y - 1


def get_label_mapper(mode: str | None) -> Callable[[int], int] | None:
    """Return a mapper function to collapse BI-RADS labels for binary experiments."""

    if normalize_classes_mode(mode) != "binary":
        return None

    return _binary_label_mapper
