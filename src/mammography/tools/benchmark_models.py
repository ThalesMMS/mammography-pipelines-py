# ruff: noqa
"""Shared benchmark report model objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mammography.utils.class_modes import classes_mode_aliases


class BenchmarkValidationError(RuntimeError):
    """Raised when the official benchmark namespace is incomplete or invalid."""


@dataclass(frozen=True)
class ExpectedRun:
    """Expected benchmark combination for the official rerun battery."""

    dataset: str
    task: str
    arch: str
    split_mode: str
    seed: int = 42

    @property
    def seed_dir_name(self) -> str:
        return f"seed{self.seed}"

    @property
    def run_name(self) -> str:
        return f"{self.dataset}_{self.task}_{self.arch}_seed{self.seed}"

    @property
    def relative_seed_dir(self) -> Path:
        return Path(self.dataset) / self.task / self.arch / self.seed_dir_name

    @property
    def relative_seed_dirs(self) -> tuple[Path, ...]:
        return tuple(
            Path(self.dataset) / task / self.arch / self.seed_dir_name
            for task in classes_mode_aliases(self.task)
        )

    @property
    def run_name_aliases(self) -> tuple[str, ...]:
        return tuple(
            f"{self.dataset}_{task}_{self.arch}_seed{self.seed}"
            for task in classes_mode_aliases(self.task)
        )


@dataclass(frozen=True)
class CollectedRun:
    """Validated run plus resolved artifacts and normalized metrics."""

    expected: ExpectedRun
    results_dir: Path
    summary: dict[str, Any]
    metrics: dict[str, Any]
    accuracy: float
    kappa: float
    macro_f1: float
    auc: float
    best_epoch: int
    export_dir: Path | None
    export_manifest_path: Path | None

    def master_row(self) -> dict[str, Any]:
        return {
            "dataset": self.expected.dataset,
            "task": self.expected.task,
            "split_mode": self.expected.split_mode,
            "arch": self.expected.arch,
            "seed": self.expected.seed,
            "img_size": int(self.summary["img_size"]),
            "batch_size": int(self.summary["batch_size"]),
            "epochs": int(self.summary["epochs"]),
            "accuracy": self.accuracy,
            "kappa": self.kappa,
            "macro_f1": self.macro_f1,
            "auc": self.auc,
            "best_epoch": self.best_epoch,
            "run_path": str(self.results_dir),
            "status": "accepted",
        }

    def article_row(self) -> dict[str, Any]:
        return {
            "dataset": self.expected.dataset,
            "task": self.expected.task,
            "split": self.expected.split_mode,
            "modelo": self.expected.arch,
            "accuracy": self.accuracy,
            "kappa": self.kappa,
            "macro-F1": self.macro_f1,
            "AUC": self.auc,
        }
