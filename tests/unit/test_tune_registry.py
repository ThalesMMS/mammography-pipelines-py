import csv
import json
from pathlib import Path

from mammography.tools import tune_registry as registry


def test_register_tune_run_writes_registry_and_mlflow_fallback(tmp_path: Path) -> None:
    outdir = tmp_path / "outputs" / "archive_tune"
    outdir.mkdir(parents=True, exist_ok=True)

    best_params_payload = {
        "best_trial": 12,
        "best_value": 0.91,
        "best_params": {"batch_size": 16, "lr": 0.001},
        "n_trials": 50,
    }
    best_params_path = outdir / "best_params.json"
    best_params_path.write_text(
        json.dumps(best_params_payload), encoding="utf-8"
    )

    stats_path = outdir / "archive_effnet_tune_stats.json"
    stats_path.write_text("{}", encoding="utf-8")

    optuna_db_path = outdir / "optuna.db"
    optuna_db_path.write_text("dummy", encoding="utf-8")

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"
    tracking_root = tmp_path / "mlruns"

    run_id = registry.register_tune_run(
        outdir=outdir,
        dataset="archive",
        arch="efficientnet_b0",
        classes="density",
        img_size=512,
        run_name="archive_effnet_tune",
        command="mammography tune --dataset archive",
        study_name="archive_effnet_tune",
        n_trials=50,
        completed_trials=45,
        pruned_trials=5,
        best_trial=12,
        best_value=0.91,
        best_params=best_params_payload["best_params"],
        storage="sqlite:///outputs/archive_tune/optuna.db",
        best_params_path=best_params_path,
        stats_path=stats_path,
        optuna_db_path=optuna_db_path,
        registry_csv=registry_csv,
        registry_md=registry_md,
        tracking_uri=str(tracking_root),
    )

    assert run_id
    assert registry_csv.exists()
    assert registry_md.exists()

    with registry_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["dataset"] == "archive"
    assert rows[0]["workflow"] == "tune"
    assert rows[0]["run_name"] == "archive_effnet_tune"
    assert rows[0]["tune_n_trials"] == "50"
    assert rows[0]["tune_completed_trials"] == "45"
    assert rows[0]["tune_pruned_trials"] == "5"
    assert rows[0]["tune_best_trial"] == "12"
    assert "batch_size" in rows[0]["hyperparameters"]

    md_text = registry_md.read_text(encoding="utf-8")
    assert "archive_effnet_tune" in md_text
    assert "Best trial" in md_text
    assert "Best params" in md_text

    artifacts_dir = tracking_root / "0" / run_id / "artifacts"
    assert (artifacts_dir / best_params_path.name).exists()
    assert (artifacts_dir / stats_path.name).exists()
    assert (artifacts_dir / optuna_db_path.name).exists()

    run_dir = tracking_root / "0" / run_id
    meta_text = (run_dir / "meta.yaml").read_text(encoding="utf-8")
    assert "name: 'archive_effnet_tune'" in meta_text

    params_dir = run_dir / "params"
    assert (params_dir / "best_batch_size").read_text(encoding="utf-8") == "16"
    assert (params_dir / "best_lr").read_text(encoding="utf-8") == "0.001"

    metrics_dir = run_dir / "metrics"
    best_value_line = (metrics_dir / "best_value").read_text(encoding="utf-8").strip()
    assert best_value_line.split()[1] == "0.91"
