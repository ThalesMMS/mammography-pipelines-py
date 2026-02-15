import csv
import json
from pathlib import Path

from mammography.tools import report_pack_registry as registry


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "outputs" / "archive_density_effnet" / "results_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"dataset": "archive"}, indent=2),
        encoding="utf-8",
    )
    return run_dir


def _write_assets(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    assets_dir = tmp_path / "Article" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    asset_names = [
        "density_train_seed42.png",
        "density_confusion_seed42.png",
    ]
    for name in asset_names:
        (assets_dir / name).write_text("fake", encoding="utf-8")
    tex_path = tmp_path / "Article" / "sections" / "density_model.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("% fake tex", encoding="utf-8")
    return assets_dir, tex_path, asset_names


def test_register_report_pack_run_writes_registry_and_mlflow_fallback(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path)
    assets_dir, tex_path, asset_names = _write_assets(tmp_path)
    artifacts = registry.collect_report_pack_outputs(
        assets_dir, tex_path, asset_names=asset_names
    )

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"
    tracking_root = tmp_path / "mlruns"

    run_id = registry.register_report_pack_run(
        run_paths=[run_dir],
        assets_dir=assets_dir,
        tex_path=tex_path,
        output_paths=artifacts.output_paths,
        run_name="archive_report_pack",
        command="mammography report-pack --run outputs/archive_density_effnet/results_1",
        registry_csv=registry_csv,
        registry_md=registry_md,
        tracking_uri=str(tracking_root),
    )

    assert run_id
    with registry_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["workflow"] == "report-pack"
    assert rows[0]["run_name"] == "archive_report_pack"
    assert rows[0]["dataset"] == "archive"
    assert rows[0]["outdir"] == str(assets_dir)
    assert "density_train_seed42.png" in rows[0]["visualization_output_paths"]
    assert str(tex_path) in rows[0]["visualization_output_paths"]

    md_text = registry_md.read_text(encoding="utf-8")
    assert "archive_report_pack" in md_text
    assert str(assets_dir) in md_text
    assert str(tex_path) in md_text

    artifacts_dir = tracking_root / "0" / run_id / "artifacts"
    assert (artifacts_dir / "density_train_seed42.png").exists()
    assert (artifacts_dir / "density_confusion_seed42.png").exists()
    assert (artifacts_dir / tex_path.name).exists()
