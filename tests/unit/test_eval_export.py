from __future__ import annotations

import json
from pathlib import Path

from mammography.commands import eval_export


def _write_text(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_export_eval_run_copies_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "archive_density_effnet" / "results_1"
    _write_text(run_dir / "summary.json", json.dumps({"dataset": "archive"}))
    _write_text(run_dir / "train_history.csv", "epoch,loss\n")
    _write_text(run_dir / "train_history.png", "png")
    _write_text(run_dir / "best_model.pt", "weights")

    metrics_dir = run_dir / "metrics"
    _write_text(metrics_dir / "val_metrics.json", json.dumps({"accuracy": 0.9}))
    _write_text(metrics_dir / "val_metrics.png", "png")

    _write_text(run_dir / "val_predictions.csv", "id,pred\n")
    _write_text(run_dir / "embeddings_val.csv", "id,emb\n")
    _write_text(run_dir / "embeddings_val.npy", "npy")

    export_root = tmp_path / "exports"
    result = eval_export.export_eval_run(run_dir, export_root)

    assert result.export_dir.exists()
    assert (result.export_dir / "summary.json").exists()
    assert (result.export_dir / "metrics" / "val_metrics.json").exists()
    assert (result.export_dir / "eval_export_manifest.json").exists()
    assert "run.log" in result.missing

    manifest = json.loads(
        (result.export_dir / "eval_export_manifest.json").read_text(encoding="utf-8")
    )
    exported_files = set(manifest["exported_files"])
    assert "summary.json" in exported_files
    assert "metrics/val_metrics.json" in exported_files


def test_export_eval_run_with_npz_embeddings(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "mammo_density_resnet" / "results_1"
    _write_text(run_dir / "summary.json", json.dumps({"dataset": "mamografias"}))
    _write_text(run_dir / "train_history.csv", "epoch,loss\n")
    _write_text(run_dir / "best_model.pt", "weights")

    metrics_dir = run_dir / "metrics"
    _write_text(metrics_dir / "val_metrics.json", json.dumps({"f1": 0.85}))

    _write_text(run_dir / "embeddings_val.npz", "npz")

    export_root = tmp_path / "exports"
    result = eval_export.export_eval_run(run_dir, export_root)

    assert result.export_dir.exists()
    assert (result.export_dir / "embeddings_val.npz").exists()
    assert "embeddings_val.npy" not in result.missing

    manifest = json.loads(
        (result.export_dir / "eval_export_manifest.json").read_text(encoding="utf-8")
    )
    exported_files = set(manifest["exported_files"])
    assert "embeddings_val.npz" in exported_files


def test_export_eval_run_with_all_optional_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "complete_run" / "results_1"
    _write_text(run_dir / "summary.json", json.dumps({"dataset": "archive"}))
    _write_text(run_dir / "train_history.csv", "epoch,loss\n")
    _write_text(run_dir / "train_history.png", "png")
    _write_text(run_dir / "best_model.pt", "weights")

    metrics_dir = run_dir / "metrics"
    _write_text(metrics_dir / "val_metrics.json", json.dumps({"accuracy": 0.92}))
    _write_text(metrics_dir / "val_metrics.png", "png")
    _write_text(metrics_dir / "confusion_matrix.png", "png")
    _write_text(metrics_dir / "test_metrics.json", json.dumps({"accuracy": 0.89}))
    _write_text(metrics_dir / "test_metrics.png", "png")

    _write_text(run_dir / "val_predictions.csv", "id,pred\n")
    _write_text(run_dir / "test_predictions.csv", "id,pred\n")
    _write_text(run_dir / "embeddings_val.csv", "id,emb\n")
    _write_text(run_dir / "embeddings_val.npy", "npy")
    _write_text(run_dir / "run.log", "logs\n")

    export_root = tmp_path / "exports"
    result = eval_export.export_eval_run(run_dir, export_root)

    assert result.export_dir.exists()
    assert (result.export_dir / "train_history.png").exists()
    assert (result.export_dir / "metrics" / "confusion_matrix.png").exists()
    assert (result.export_dir / "metrics" / "test_metrics.json").exists()
    assert (result.export_dir / "metrics" / "test_metrics.png").exists()
    assert (result.export_dir / "run.log").exists()
    assert (result.export_dir / "test_predictions.csv").exists()
    assert len(result.missing) == 0

    manifest = json.loads(
        (result.export_dir / "eval_export_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["missing_files"]) == 0
    exported_files = set(manifest["exported_files"])
    assert "run.log" in exported_files
    assert "train_history.png" in exported_files
    assert "metrics/confusion_matrix.png" in exported_files
    assert "metrics/test_metrics.json" in exported_files
    assert "metrics/test_metrics.png" in exported_files
    assert "test_predictions.csv" in exported_files


def test_export_eval_run_with_minimal_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "minimal_run" / "results_1"
    _write_text(run_dir / "summary.json", json.dumps({"dataset": "patches"}))
    _write_text(run_dir / "train_history.csv", "epoch,loss\n")
    _write_text(run_dir / "best_model.pt", "weights")

    metrics_dir = run_dir / "metrics"
    _write_text(metrics_dir / "val_metrics.json", json.dumps({"accuracy": 0.75}))

    export_root = tmp_path / "exports"
    result = eval_export.export_eval_run(run_dir, export_root)

    assert result.export_dir.exists()
    assert (result.export_dir / "summary.json").exists()
    assert (result.export_dir / "train_history.csv").exists()
    assert (result.export_dir / "best_model.pt").exists()
    assert (result.export_dir / "metrics" / "val_metrics.json").exists()

    expected_missing = [
        "train_history.png",
        "confusion_matrix.png",
        "val_predictions.csv",
        "embeddings_val.csv",
        "embeddings_val.npy",
        "run.log",
    ]
    for missing_file in expected_missing:
        assert missing_file in result.missing


def test_export_eval_run_uses_resumed_top_k_checkpoint(tmp_path: Path) -> None:
    previous_run_dir = (
        tmp_path / "outputs" / "archive" / "density" / "resnet50" / "seed42" / "results"
    )
    previous_top_k = previous_run_dir / "top_k" / "model_epoch028_macro_f10.6413.pt"
    _write_text(previous_top_k, "weights")
    _write_text(previous_run_dir / "checkpoint.pt", "resume")

    run_dir = (
        tmp_path / "outputs" / "archive" / "density" / "resnet50" / "seed42_2" / "results"
    )
    _write_text(
        run_dir / "summary.json",
        json.dumps(
            {
                "dataset": "archive",
                "resume_from": str(previous_run_dir / "checkpoint.pt"),
                "top_k": [
                    {
                        "score": 0.6413,
                        "epoch": 28,
                        "path": str(previous_top_k),
                    }
                ],
            }
        ),
    )
    _write_text(run_dir / "train_history.csv", "epoch,loss\n")
    _write_text(run_dir / "checkpoint.pt", "latest")

    metrics_dir = run_dir / "metrics"
    _write_text(metrics_dir / "val_metrics.json", json.dumps({"accuracy": 0.91}))
    _write_text(metrics_dir / "val_metrics.png", "png")

    export_root = tmp_path / "exports"
    result = eval_export.export_eval_run(run_dir, export_root)

    assert result.artifacts.checkpoint_path == previous_top_k
    assert (result.export_dir / previous_top_k.name).exists()


def test_main_with_registry_disabled(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "test_run" / "results_1"
    _write_text(run_dir / "summary.json", json.dumps({"dataset": "archive"}))
    _write_text(run_dir / "train_history.csv", "epoch,loss\n")
    _write_text(run_dir / "best_model.pt", "weights")

    metrics_dir = run_dir / "metrics"
    _write_text(metrics_dir / "val_metrics.json", json.dumps({"accuracy": 0.88}))

    export_root = tmp_path / "exports"

    monkeypatch.setattr(eval_export, "REPO_ROOT", tmp_path)

    registry_calls: list[dict] = []

    def _fake_register(**kwargs):
        registry_calls.append(kwargs)

    monkeypatch.setattr(
        eval_export.eval_export_registry,
        "register_eval_export_run",
        _fake_register,
    )

    exit_code = eval_export.main(
        [
            "--run",
            str(run_dir),
            "--output-dir",
            str(export_root),
            "--no-registry",
        ]
    )

    assert exit_code == 0
    assert (export_root / "test_run" / "results_1" / "summary.json").exists()
    assert len(registry_calls) == 0


def test_main_with_mlflow_disabled(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "test_run" / "results_1"
    _write_text(run_dir / "summary.json", json.dumps({"dataset": "mamografias"}))
    _write_text(run_dir / "train_history.csv", "epoch,loss\n")
    _write_text(run_dir / "best_model.pt", "weights")

    metrics_dir = run_dir / "metrics"
    _write_text(metrics_dir / "val_metrics.json", json.dumps({"f1": 0.82}))

    export_root = tmp_path / "exports"
    registry_csv = tmp_path / "registry.csv"
    registry_md = tmp_path / "registry.md"

    monkeypatch.setattr(eval_export, "REPO_ROOT", tmp_path)

    captured_log_mlflow: list[bool] = []

    def _fake_register(**kwargs):
        captured_log_mlflow.append(kwargs.get("log_mlflow", True))

    monkeypatch.setattr(
        eval_export.eval_export_registry,
        "register_eval_export_run",
        _fake_register,
    )

    exit_code = eval_export.main(
        [
            "--run",
            str(run_dir),
            "--output-dir",
            str(export_root),
            "--registry-csv",
            str(registry_csv),
            "--registry-md",
            str(registry_md),
            "--no-mlflow",
        ]
    )

    assert exit_code == 0
    assert len(captured_log_mlflow) == 1
    assert captured_log_mlflow[0] is False


def test_main_with_custom_run_name(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "exp_run" / "results_1"
    _write_text(run_dir / "summary.json", json.dumps({"dataset": "archive"}))
    _write_text(run_dir / "train_history.csv", "epoch,loss\n")
    _write_text(run_dir / "best_model.pt", "weights")

    metrics_dir = run_dir / "metrics"
    _write_text(metrics_dir / "val_metrics.json", json.dumps({"accuracy": 0.91}))

    export_root = tmp_path / "exports"
    registry_csv = tmp_path / "registry.csv"
    registry_md = tmp_path / "registry.md"

    monkeypatch.setattr(eval_export, "REPO_ROOT", tmp_path)

    captured_run_name: list[str] = []

    def _fake_register(**kwargs):
        captured_run_name.append(kwargs.get("run_name", ""))

    monkeypatch.setattr(
        eval_export.eval_export_registry,
        "register_eval_export_run",
        _fake_register,
    )

    exit_code = eval_export.main(
        [
            "--run",
            str(run_dir),
            "--output-dir",
            str(export_root),
            "--registry-csv",
            str(registry_csv),
            "--registry-md",
            str(registry_md),
            "--run-name",
            "custom_experiment",
            "--no-mlflow",
        ]
    )

    assert exit_code == 0
    assert len(captured_run_name) == 1
    assert captured_run_name[0] == "custom_experiment"


def test_main_with_multiple_runs(tmp_path: Path, monkeypatch) -> None:
    run_dir_1 = tmp_path / "outputs" / "run_a" / "results_1"
    _write_text(run_dir_1 / "summary.json", json.dumps({"dataset": "archive"}))
    _write_text(run_dir_1 / "train_history.csv", "epoch,loss\n")
    _write_text(run_dir_1 / "best_model.pt", "weights")
    _write_text(run_dir_1 / "metrics" / "val_metrics.json", json.dumps({"acc": 0.9}))

    run_dir_2 = tmp_path / "outputs" / "run_b" / "results_2"
    _write_text(run_dir_2 / "summary.json", json.dumps({"dataset": "mamografias"}))
    _write_text(run_dir_2 / "train_history.csv", "epoch,loss\n")
    _write_text(run_dir_2 / "best_model.pt", "weights")
    _write_text(run_dir_2 / "metrics" / "val_metrics.json", json.dumps({"acc": 0.85}))

    export_root = tmp_path / "exports"
    registry_csv = tmp_path / "registry.csv"
    registry_md = tmp_path / "registry.md"

    monkeypatch.setattr(eval_export, "REPO_ROOT", tmp_path)

    captured_runs: list[str] = []

    def _fake_register(**kwargs):
        captured_runs.append(kwargs.get("run_name", ""))

    monkeypatch.setattr(
        eval_export.eval_export_registry,
        "register_eval_export_run",
        _fake_register,
    )

    exit_code = eval_export.main(
        [
            "--run",
            str(run_dir_1),
            "--run",
            str(run_dir_2),
            "--output-dir",
            str(export_root),
            "--registry-csv",
            str(registry_csv),
            "--registry-md",
            str(registry_md),
            "--run-name",
            "multi_run",
            "--no-mlflow",
        ]
    )

    assert exit_code == 0
    assert (export_root / "run_a" / "results_1" / "summary.json").exists()
    assert (export_root / "run_b" / "results_2" / "summary.json").exists()
    assert len(captured_runs) == 2
    assert "multi_run_results_1" in captured_runs
    assert "multi_run_results_2" in captured_runs
