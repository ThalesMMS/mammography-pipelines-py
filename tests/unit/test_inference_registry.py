from __future__ import annotations

import csv
from pathlib import Path

import mammography.tools.inference_registry as inference_registry
from mammography.tools.inference_registry import (
    InferenceMetrics,
    default_run_name,
    infer_dataset_name,
    register_inference_run,
)


def test_default_run_name_effnet(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "archive"
    dataset_dir.mkdir()
    dataset = infer_dataset_name(dataset_dir)
    assert dataset == "archive"
    assert default_run_name(dataset, "efficientnet_b0") == "archive_inference_effnet"


def test_register_inference_run_writes_registry(tmp_path: Path) -> None:
    input_dir = tmp_path / "archive"
    input_dir.mkdir()
    output_path = tmp_path / "preds.csv"
    output_path.write_text("file,pred_class\n", encoding="utf-8")
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_text("weights", encoding="utf-8")

    metrics = InferenceMetrics(total_images=8, images_per_sec=4.0, duration_sec=2.0)
    registry_csv = tmp_path / "registry.csv"
    registry_md = tmp_path / "registry.md"

    register_inference_run(
        input_path=input_dir,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        arch="efficientnet_b0",
        classes="density",
        img_size=512,
        batch_size=16,
        metrics=metrics,
        run_name="archive_inference_effnet",
        command="mammography inference --output preds.csv",
        registry_csv=registry_csv,
        registry_md=registry_md,
        log_mlflow=False,
    )

    assert registry_csv.exists()
    with registry_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    assert row["checkpoint_path"] == str(checkpoint_path)
    assert row["inference_total_images"] == "8"
    assert row["inference_images_per_sec"] == "4.0"
    assert row["inference_duration_sec"] == "2.0"

    assert registry_md.exists()
    content = registry_md.read_text(encoding="utf-8")
    assert "archive_inference_effnet" in content


def test_register_inference_run_logs_mlflow_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "mamografias"
    input_dir.mkdir()
    output_path = tmp_path / "preds.csv"
    output_path.write_text("file,pred_class\n", encoding="utf-8")
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_text("weights", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_log_mlflow_run(
        *,
        run_name: str,
        params: dict[str, str],
        metrics: dict[str, float],
        artifacts,
        tracking_uri: str | None,
        experiment: str | None,
    ) -> str:
        captured["run_name"] = run_name
        captured["params"] = params
        captured["metrics"] = metrics
        captured["artifacts"] = list(artifacts)
        captured["tracking_uri"] = tracking_uri
        captured["experiment"] = experiment
        return "run-123"

    monkeypatch.setattr(inference_registry, "log_mlflow_run", _fake_log_mlflow_run)

    metrics = InferenceMetrics(total_images=12, images_per_sec=3.0, duration_sec=4.0)
    registry_csv = tmp_path / "registry.csv"
    registry_md = tmp_path / "registry.md"

    run_id = register_inference_run(
        input_path=input_dir,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        arch="efficientnet_b0",
        classes="density",
        img_size=512,
        batch_size=16,
        metrics=metrics,
        run_name="mamografias_inference_effnet",
        command="mammography inference --output preds.csv",
        registry_csv=registry_csv,
        registry_md=registry_md,
        log_mlflow=True,
    )

    assert run_id == "run-123"
    assert captured["run_name"] == "mamografias_inference_effnet"
    assert captured["params"]["checkpoint_path"] == str(checkpoint_path)
    assert captured["metrics"]["total_images"] == 12.0
    assert captured["metrics"]["images_per_sec"] == 3.0
    assert captured["metrics"]["inference_seconds"] == 4.0
