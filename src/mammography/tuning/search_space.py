#
# search_space.py
# mammography-pipelines
#
# YAML loader and validator for hyperparameter search space configuration.
#
# Thales Matheus Mendonça Santos - January 2026
#
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CategoricalParam(BaseModel):
    """Categorical hyperparameter with discrete choices."""

    type: Literal["categorical"] = "categorical"
    choices: List[Union[str, int, float, bool]] = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


class IntParam(BaseModel):
    """Integer hyperparameter with min/max bounds."""

    type: Literal["int"] = "int"
    low: int
    high: int
    step: int = Field(default=1, ge=1)
    log: bool = False

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_bounds(self) -> "IntParam":
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be < high ({self.high})")
        return self


class FloatParam(BaseModel):
    """Float hyperparameter with min/max bounds."""

    type: Literal["float"] = "float"
    low: float
    high: float
    step: Optional[float] = Field(default=None, gt=0)
    log: bool = False

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_bounds(self) -> "FloatParam":
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be < high ({self.high})")
        if self.step is not None and self.step > (self.high - self.low):
            raise ValueError(
                f"step ({self.step}) must be <= range ({self.high - self.low})"
            )
        return self


class SearchSpace(BaseModel):
    """
    Validated hyperparameter search space configuration.

    Supports categorical, int, and float parameter types compatible with Optuna.
    Load from YAML using SearchSpace.from_yaml(path).
    """

    parameters: Dict[str, Union[CategoricalParam, IntParam, FloatParam]]
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("parameters")
    @classmethod
    def _validate_nonempty(
        cls, value: Dict[str, Union[CategoricalParam, IntParam, FloatParam]]
    ) -> Dict[str, Union[CategoricalParam, IntParam, FloatParam]]:
        if not value:
            raise ValueError("Search space must contain at least one parameter")
        return value

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SearchSpace":
        """Load and validate search space from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Search space config not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML format in {path}: expected dict")

        # Parse parameters with type discrimination
        parameters = {}
        raw_params = data.get("parameters", {})
        if not isinstance(raw_params, dict):
            raise ValueError("'parameters' must be a dict")

        for name, config in raw_params.items():
            if not isinstance(config, dict):
                raise ValueError(f"Parameter '{name}' config must be a dict")

            param_type = config.get("type")
            if param_type == "categorical":
                parameters[name] = CategoricalParam(**config)
            elif param_type == "int":
                parameters[name] = IntParam(**config)
            elif param_type == "float":
                parameters[name] = FloatParam(**config)
            else:
                raise ValueError(
                    f"Parameter '{name}' has invalid type: {param_type}. "
                    "Must be 'categorical', 'int', or 'float'."
                )

        return cls(parameters=parameters, description=data.get("description"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert search space to dictionary representation."""
        return {
            "description": self.description,
            "parameters": {
                name: param.model_dump() for name, param in self.parameters.items()
            },
        }
