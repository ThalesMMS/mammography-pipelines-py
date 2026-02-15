"""Minimal fallback helpers when pydantic is unavailable.

This is intentionally small: it supports the subset of behavior used by the
CLI configs (Field defaults, field/model validators, and model_validate).
It is only imported when pydantic is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, get_args, get_origin


class ValidationError(ValueError):
    """Lightweight stand-in for pydantic.ValidationError."""


class ConfigDict(dict):
    """Minimal placeholder for pydantic.ConfigDict."""


_MISSING = object()


@dataclass(frozen=True)
class FieldInfo:
    default: Any = _MISSING
    metadata: dict[str, Any] | None = None


def Field(default: Any = _MISSING, **metadata: Any) -> FieldInfo:
    """Return a lightweight FieldInfo container."""
    return FieldInfo(default=default, metadata=metadata or None)


def field_validator(*fields: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach field validation metadata to a method."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "_field_validator_fields", tuple(fields))
        return func

    return decorator


def model_validator(*, mode: str = "after") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach model validation metadata to a method."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "_model_validator_mode", mode)
        return func

    return decorator


class BaseModel:
    """Minimal BaseModel with validators and model_validate support."""

    model_config: ConfigDict | None = None
    model_fields: dict[str, FieldInfo] = {}
    _field_validators: list[tuple[str, tuple[str, ...]]] = []
    _model_validators: list[tuple[str, str]] = []

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        fields: dict[str, FieldInfo] = {}
        for base in reversed(cls.__mro__):
            annotations = getattr(base, "__annotations__", {})
            for name in annotations:
                if name.startswith("_"):
                    continue
                value = getattr(base, name, _MISSING)
                if isinstance(value, FieldInfo):
                    fields[name] = value
                elif value is _MISSING:
                    fields.setdefault(name, FieldInfo(default=_MISSING))
                else:
                    fields[name] = FieldInfo(default=value)
        cls.model_fields = fields

        field_validators: list[tuple[str, tuple[str, ...]]] = []
        model_validators: list[tuple[str, str]] = []
        for base in cls.__mro__:
            for attr_name, attr in base.__dict__.items():
                fields_attr = getattr(attr, "_field_validator_fields", None)
                if fields_attr:
                    field_validators.append((attr_name, tuple(fields_attr)))
                mode = getattr(attr, "_model_validator_mode", None)
                if mode:
                    model_validators.append((attr_name, str(mode)))
        cls._field_validators = field_validators
        cls._model_validators = model_validators

    def __init__(self, **kwargs: Any) -> None:
        for name in self.model_fields:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    @classmethod
    def model_validate(cls, payload: dict[str, Any]) -> "BaseModel":
        values: dict[str, Any] = {}
        annotations = getattr(cls, "__annotations__", {})
        for name, field in cls.model_fields.items():
            if name in payload:
                values[name] = payload[name]
            elif field.default is not _MISSING:
                values[name] = field.default
            else:
                raise ValidationError(f"Missing required field: {name}")

            annotation = annotations.get(name)
            if annotation is not None:
                origin = get_origin(annotation)
                args = get_args(annotation)
                if annotation is Path:
                    values[name] = Path(values[name]) if values[name] is not None else None
                elif origin is list and args and args[0] is Path:
                    values[name] = [Path(v) for v in values[name]] if values[name] else []
                elif origin is tuple and Path in args:
                    values[name] = tuple(Path(v) for v in values[name]) if values[name] else ()
                elif origin is type(None):
                    pass
                elif origin is not None and Path in args:
                    if values[name] is not None:
                        values[name] = Path(values[name])

        for validator_name, fields in cls._field_validators:
            validator = getattr(cls, validator_name)
            for field_name in fields:
                if field_name in values:
                    try:
                        values[field_name] = validator(values[field_name])
                    except Exception as exc:  # pragma: no cover - best effort fallback
                        raise ValidationError(str(exc)) from exc

        instance = cls.__new__(cls)
        for name, value in values.items():
            setattr(instance, name, value)

        for validator_name, mode in cls._model_validators:
            if mode != "after":
                continue
            validator = getattr(cls, validator_name)
            try:
                result = validator(instance)
            except Exception as exc:  # pragma: no cover - best effort fallback
                raise ValidationError(str(exc)) from exc
            if isinstance(result, cls):
                instance = result

        return instance


__all__ = [
    "BaseModel",
    "ConfigDict",
    "Field",
    "ValidationError",
    "field_validator",
    "model_validator",
]
