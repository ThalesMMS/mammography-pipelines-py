import csv
from pathlib import Path

from mammography.tools import visualization_registry as registry


def _write_input_paths(base: Path) -> tuple[Path, Path]:
    input_path = base / "outputs" / "run" / "features.npy"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("fake", encoding="utf-8")
    labels_path = base / "outputs" / "run" / "metadata.csv"
    labels_path.write_text("raw_label\n0\n", encoding="utf-8")
    return input_path, labels_path


def _write_outputs(base: Path) -> Path:
    output_dir = base / "outputs" / "archive_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tsne_2d.png").write_text("fake", encoding="utf-8")
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "embedding_comparison.png").write_text(
        "fake",
        encoding="utf-8",
    )
    return output_dir


def test_register_visualization_run_writes_registry_and_mlflow_fallback(
    tmp_path: Path,
) -> None:
    input_path, labels_path = _write_input_paths(tmp_path)
    output_dir = _write_outputs(tmp_path)
    artifacts = registry.collect_visualization_outputs(output_dir)

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"
    tracking_root = tmp_path / "mlruns"

    run_id = registry.register_visualization_run(
        input_path=input_path,
        labels_path=labels_path,
        output_dir=output_dir,
        output_paths=artifacts.output_paths,
        run_name="archive_visualizations",
        command="mammography visualize --report",
        registry_csv=registry_csv,
        registry_md=registry_md,
        tracking_uri=str(tracking_root),
    )

    assert run_id
    with registry_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["workflow"] == "visualize"
    assert rows[0]["run_name"] == "archive_visualizations"
    assert rows[0]["features_path"] == str(input_path)
    assert "tsne_2d.png" in rows[0]["visualization_output_paths"]
    assert "embedding_comparison.png" in rows[0]["visualization_output_paths"]

    md_text = registry_md.read_text(encoding="utf-8")
    assert "archive_visualizations" in md_text
    assert str(output_dir) in md_text

    artifacts_dir = tracking_root / "0" / run_id / "artifacts" / output_dir.name
    assert (artifacts_dir / "tsne_2d.png").exists()
    assert (artifacts_dir / "report" / "embedding_comparison.png").exists()
