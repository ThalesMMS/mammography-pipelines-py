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
