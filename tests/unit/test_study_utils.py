from pathlib import Path

import pytest

from mammography.tuning.study_utils import (
    StudySummary,
    load_study_summary,
    should_skip_optimization,
)


def test_should_skip_optimization_without_optuna() -> None:
    summary = StudySummary(
        study=object(),
        n_trials=50,
        completed_trials=48,
        pruned_trials=2,
        best_trial=7,
        best_value=0.91,
        best_params={"lr": 0.001},
    )

    assert should_skip_optimization(50, summary) is True
    assert should_skip_optimization(60, summary) is False
    assert should_skip_optimization(50, None) is False


def test_load_study_summary_and_skip(tmp_path: Path) -> None:
    optuna = pytest.importorskip("optuna")
    db_path = tmp_path / "optuna.db"
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name="demo_study",
        storage=storage,
        direction="maximize",
    )

    def objective(trial: optuna.Trial) -> float:
        return trial.suggest_float("lr", 0.0001, 0.01)

    study.optimize(objective, n_trials=2)

    summary = load_study_summary(storage, "demo_study")
    assert summary is not None
    assert summary.n_trials == 2
    assert summary.completed_trials == 2
    assert summary.pruned_trials == 0
    assert summary.best_trial is not None
    assert summary.best_value is not None
    assert "lr" in summary.best_params

    assert should_skip_optimization(2, summary) is True
    assert should_skip_optimization(3, summary) is False
