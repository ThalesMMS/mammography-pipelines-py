"""Helpers for argparse boolean flag compatibility."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_BOOL_TRUE = {"1", "true", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "no", "n", "off"}

DEFAULT_BOOL_FLAGS: dict[str, tuple[str, str]] = {
    "--sampler-weighted": ("--sampler-weighted", "--no-sampler-weighted"),
    "--unfreeze-last-block": ("--unfreeze-last-block", "--no-unfreeze-last-block"),
    "--augment": ("--augment", "--no-augment"),
}


def parse_bool_literal(value: str | None) -> bool | None:
    """Parse common textual boolean literals."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in _BOOL_TRUE:
        return True
    if normalized in _BOOL_FALSE:
        return False
    return None


def normalize_bool_flags(
    argv: Sequence[str] | None,
    bool_flags: Mapping[str, tuple[str, str]] | None = None,
) -> list[str] | None:
    """Normalize ``--flag true/false`` tokens to argparse-compatible flags."""
    if argv is None:
        return None
    normalized: list[str] = []
    mappings = bool_flags or DEFAULT_BOOL_FLAGS
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        mapping = mappings.get(token)
        if mapping and idx + 1 < len(argv):
            literal = parse_bool_literal(argv[idx + 1])
            if literal is not None:
                normalized.append(mapping[0] if literal else mapping[1])
                idx += 2
                continue
        normalized.append(token)
        idx += 1
    return normalized
