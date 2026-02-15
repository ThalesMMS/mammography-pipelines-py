import argparse
import csv
import json
from pathlib import Path

from mammography.commands import embeddings_baselines


def _write_baselines_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2026-02-05T00:00:00Z",
        "embeddings_dir": "outputs/mamografias_embeddings_resnet50",
        "outdir": "outputs/mamografias_baselines_resnet50",
        "feature_sets": {
            "embeddings": {
                "logreg": {
                    "accuracy_mean": 0.81,
                    "macro_f1_mean": 0.77,
                    "auc_mean": 0.85,
                }
            },
            "handcrafted": {
                "rf": {
                    "accuracy_mean": 0.71,
                    "macro_f1_mean": 0.69,
                    "auc_mean": 0.7,
                }
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_register_baselines_run_defaults_run_name(tmp_path: Path) -> None:
    report_path = (
        tmp_path / "outputs" / "mamografias_baselines_resnet50" / "baselines_report.json"
    )
    _write_baselines_report(report_path)

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"
    args = argparse.Namespace(
        dataset="",
        run_name="",
        registry_csv=registry_csv,
        registry_md=registry_md,
        tracking_uri="",
        experiment="",
        no_mlflow=True,
    )

    run_name, run_id = embeddings_baselines._register_baselines_run(
        report_path=report_path,
        embeddings_dir=tmp_path / "outputs" / "mamografias_embeddings_resnet50",
        outdir=report_path.parent,
        args=args,
        command=(
            "mammography embeddings-baselines --embeddings-dir "
            "outputs/mamografias_embeddings_resnet50"
        ),
    )

    assert run_name == "mamografias_baselines_resnet50"
    assert run_id is None
    assert registry_csv.exists()
    assert registry_md.exists()

    with registry_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    rows_by_model = {row["baseline_model"]: row for row in rows}
    assert rows_by_model["logreg"]["baseline_feature_set"] == "embeddings"
    assert rows_by_model["logreg"]["dataset"] == "mamografias"
    assert rows_by_model["rf"]["baseline_feature_set"] == "handcrafted"

    md_text = registry_md.read_text(encoding="utf-8")
    assert "mamografias_baselines_resnet50" in md_text
