from __future__ import annotations

import csv
from pathlib import Path

from mammography.tools import eval_export_registry as registry


def _write_text(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_default_run_name_from_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "archive_density_effnet" / "results_1"
    run_dir.mkdir(parents=True)
    dataset = registry.infer_dataset_name({"dataset": "archive"}, run_dir)
    assert dataset == "archive"
    assert registry.default_run_name(dataset) == "archive_eval_export"


def test_register_eval_export_run_writes_registry(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports" / "run_1"
    export_dir.mkdir(parents=True)

    summary_path = export_dir / "summary.json"
    train_history_path = export_dir / "train_history.csv"
    val_metrics_path = export_dir / "metrics" / "val_metrics.json"
    confusion_matrix_path = export_dir / "metrics" / "val_metrics.png"
    checkpoint_path = export_dir / "best_model.pt"
    val_predictions_path = export_dir / "val_predictions.csv"

    _write_text(summary_path, "{}")
    _write_text(train_history_path, "epoch,loss\n")
    _write_text(val_metrics_path, "{}")
    _write_text(confusion_matrix_path, "png")
    _write_text(checkpoint_path, "weights")
    _write_text(val_predictions_path, "id,pred\n")

    export_paths = [
        summary_path,
        train_history_path,
        val_metrics_path,
        confusion_matrix_path,
        checkpoint_path,
        val_predictions_path,
    ]

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"

    registry.register_eval_export_run(
        dataset="archive",
        run_name="archive_eval_export",
        command="mammography eval-export --run outputs/archive_density_effnet/results_1",
        export_dir=export_dir,
        export_paths=export_paths,
        summary_path=summary_path,
        train_history_path=train_history_path,
        val_metrics_path=val_metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        checkpoint_path=checkpoint_path,
        val_predictions_path=val_predictions_path,
        registry_csv=registry_csv,
        registry_md=registry_md,
        log_mlflow=False,
    )

    with registry_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["workflow"] == "eval-export"
    assert row["summary_path"] == str(summary_path)
    assert row["val_metrics_path"] == str(val_metrics_path)
    assert row["checkpoint_path"] == str(checkpoint_path)
    assert str(summary_path) in row["visualization_output_paths"]

    content = registry_md.read_text(encoding="utf-8")
    assert "archive_eval_export" in content
    assert str(export_dir) in content
