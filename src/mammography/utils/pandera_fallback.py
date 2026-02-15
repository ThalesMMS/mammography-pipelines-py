"""Minimal pandera fallback used when the dependency is unavailable."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Iterable


class SchemaError(Exception):
    """Placeholder for pandera.errors.SchemaError."""


class SchemaErrors(Exception):
    """Placeholder for pandera.errors.SchemaErrors."""


class Check:
    """Lightweight Check container (no-op validation)."""

    def __init__(
        self,
        fn: Callable[..., Any] | None = None,
        name: str | None = None,
        **_: Any,
    ) -> None:
        self.fn = fn
        self.name = name

    @classmethod
    def str_length(
        cls,
        *,
        min_value: int | None = None,
        max_value: int | None = None,
        **_: Any,
    ) -> "Check":
        return cls(name="str_length")

    @classmethod
    def isin(cls, values: Any, **_: Any) -> "Check":
        return cls(name="isin")


@dataclass(frozen=True)
class Column:
    dtype: Any
    nullable: bool = True
    required: bool = False
    coerce: bool = True
    checks: Iterable[Check] | None = None


class DataFrameSchema:
    """Schema stub that returns data unchanged."""

    def __init__(self, columns: dict[str, Column], strict: bool = False) -> None:
        self.columns = columns
        self.strict = strict

    def validate(self, df: Any, lazy: bool = True) -> Any:
        return df


errors = SimpleNamespace(SchemaError=SchemaError, SchemaErrors=SchemaErrors)


__all__ = ["Check", "Column", "DataFrameSchema", "errors", "SchemaError", "SchemaErrors"]
