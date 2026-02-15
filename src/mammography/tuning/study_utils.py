"""Helpers for loading Optuna study metadata without running new trials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    import optuna
    from optuna.trial import TrialState
except ModuleNotFoundError:  # pragma: no cover - optuna optional in minimal envs
    optuna = None
    TrialState = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import optuna as optuna_typing


@dataclass(frozen=True)
class StudySummary:
    study: "optuna_typing.Study"
    n_trials: int
    completed_trials: int
    pruned_trials: int
    best_trial: int | None
    best_value: float | None
    best_params: dict[str, object]


def load_study_summary(storage: str, study_name: str) -> StudySummary | None:
    if optuna is None:
        return None
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        return None

    trials = study.trials
    n_trials = len(trials)
    completed_trials = len([t for t in trials if t.state == TrialState.COMPLETE])
    pruned_trials = len([t for t in trials if t.state == TrialState.PRUNED])

    best_trial = None
    best_value = None
    best_params: dict[str, object] = {}
    try:
        best_trial = study.best_trial.number
        best_value = float(study.best_value)
        best_params = dict(study.best_params)
    except Exception:
        pass

    return StudySummary(
        study=study,
        n_trials=n_trials,
        completed_trials=completed_trials,
        pruned_trials=pruned_trials,
        best_trial=best_trial,
        best_value=best_value,
        best_params=best_params,
    )


def should_skip_optimization(target_trials: int, summary: StudySummary | None) -> bool:
    if summary is None:
        return False
    if summary.best_trial is None or summary.best_value is None:
        return False
    return summary.n_trials >= target_trials
