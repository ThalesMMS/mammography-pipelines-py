from __future__ import annotations

import csv
from pathlib import Path

from mammography.tools.explain_registry import (
    ExplainMetrics,
    default_run_name,
    default_run_name_from_output,
    infer_dataset_name,
    register_explain_run,
)


def test_default_run_name_gradcam(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "archive"
    dataset_dir.mkdir()
    dataset = infer_dataset_name(dataset_dir)
    assert dataset == "archive"
    assert default_run_name(dataset, "gradcam") == "archive_explain_gradcam"


def test_default_run_name_from_output_dir() -> None:
    output_dir = Path("outputs/patches_explain")
    assert default_run_name_from_output(output_dir, "gradcam") == "patches_explain_gradcam"


def test_register_explain_run_writes_registry(tmp_path: Path) -> None:
    input_dir = tmp_path / "archive"
    input_dir.mkdir()
    output_dir = tmp_path / "outputs" / "archive_explain_gradcam"
    gradcam_dir = output_dir / "gradcam"
    gradcam_dir.mkdir(parents=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_text("weights", encoding="utf-8")

    metrics = ExplainMetrics(
        total_images=10,
        loaded_images=8,
        saved_images=8,
        failed_images=2,
    )
    registry_csv = tmp_path / "registry.csv"
    registry_md = tmp_path / "registry.md"

    register_explain_run(
        input_path=input_dir,
        output_dir=output_dir,
        output_path=gradcam_dir,
        checkpoint_path=checkpoint_path,
        arch="efficientnet_b0",
        method="gradcam",
        layer="features.7",
        img_size=512,
        batch_size=8,
        metrics=metrics,
        run_name="archive_explain_gradcam",
        command="mammography explain --model-path best_model.pt",
        registry_csv=registry_csv,
        registry_md=registry_md,
        artifacts=[summary_path],
        artifact_dirs=[gradcam_dir],
        log_mlflow=False,
    )

    assert registry_csv.exists()
    with registry_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    assert row["checkpoint_path"] == str(checkpoint_path)
    assert row["explain_output_path"] == str(gradcam_dir)
    assert row["explain_total_images"] == "10"
    assert row["explain_loaded_images"] == "8"
    assert row["explain_saved_images"] == "8"
    assert row["explain_failed_images"] == "2"
    assert row["explain_method"] == "gradcam"

    assert registry_md.exists()
    content = registry_md.read_text(encoding="utf-8")
    assert "archive_explain_gradcam" in content
    assert str(gradcam_dir) in content


def test_register_explain_run_logs_mlflow_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "mamografias"
    input_dir.mkdir()
    output_dir = tmp_path / "outputs" / "mamografias_explain"
    gradcam_dir = output_dir / "gradcam"
    gradcam_dir.mkdir(parents=True)
    (gradcam_dir / "gradcam_0_sample_0.png").write_text("fake", encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_text("weights", encoding="utf-8")

    metrics = ExplainMetrics(
        total_images=4,
        loaded_images=4,
        saved_images=4,
        failed_images=0,
    )
    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"
    tracking_root = tmp_path / "mlruns"

    run_id = register_explain_run(
        input_path=input_dir,
        output_dir=output_dir,
        output_path=gradcam_dir,
        checkpoint_path=checkpoint_path,
        arch="efficientnet_b0",
        method="gradcam",
        layer="features.7",
        img_size=512,
        batch_size=8,
        metrics=metrics,
        run_name="mamografias_explain_gradcam",
        command="mammography explain --model-path best_model.pt",
        registry_csv=registry_csv,
        registry_md=registry_md,
        artifacts=[summary_path],
        artifact_dirs=[gradcam_dir],
        tracking_uri=str(tracking_root),
    )

    assert run_id
    artifacts_dir = tracking_root / "0" / run_id / "artifacts"
    assert (artifacts_dir / "summary.json").exists()
    assert (artifacts_dir / "gradcam" / "gradcam_0_sample_0.png").exists()
