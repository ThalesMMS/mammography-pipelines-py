import csv
import json
from pathlib import Path

from mammography.tools import baselines_registry as registry


def _write_baselines_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2026-02-05T00:00:00Z",
        "embeddings_dir": "outputs/archive_embeddings_resnet50",
        "outdir": "outputs/archive_baselines_resnet50",
        "feature_sets": {
            "embeddings": {
                "logreg": {
                    "accuracy_mean": 0.91,
                    "macro_f1_mean": 0.87,
                    "auc_mean": 0.93,
                }
            },
            "handcrafted": {
                "rf": {
                    "accuracy_mean": 0.71,
                    "macro_f1_mean": 0.65,
                    "auc_mean": 0.7,
                }
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_multi_model_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2026-02-05T00:00:00Z",
        "embeddings_dir": "outputs/mamografias_embeddings_effnet",
        "outdir": "outputs/mamografias_baselines_effnet",
        "feature_sets": {
            "embeddings": {
                "logreg": {
                    "accuracy_mean": 0.52,
                    "macro_f1_mean": 0.48,
                    "auc_mean": 0.55,
                },
                "svm-rbf": {
                    "accuracy_mean": 0.57,
                    "macro_f1_mean": 0.5,
                    "auc_mean": 0.58,
                },
            },
            "handcrafted": {
                "rf": {
                    "accuracy_mean": 0.61,
                    "macro_f1_mean": 0.59,
                    "auc_mean": 0.6,
                },
                "svm-linear": {
                    "accuracy_mean": 0.63,
                    "macro_f1_mean": 0.6,
                    "auc_mean": 0.62,
                },
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_register_baselines_run_writes_registry_and_mlflow_fallback(
    tmp_path: Path,
) -> None:
    report_path = (
        tmp_path / "outputs" / "archive_baselines_resnet50" / "baselines_report.json"
    )
    _write_baselines_report(report_path)

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"
    tracking_root = tmp_path / "mlruns"

    run_id = registry.register_baselines_run(
        report_path=report_path,
        dataset="archive",
        run_name="archive_baselines_resnet50",
        command="mammography embeddings-baselines --embeddings-dir outputs/archive_embeddings_resnet50",
        registry_csv=registry_csv,
        registry_md=registry_md,
        tracking_uri=str(tracking_root),
    )

    assert run_id is not None
    with registry_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    rows_by_model = {row["baseline_model"]: row for row in rows}
    assert rows_by_model["logreg"]["baseline_feature_set"] == "embeddings"
    assert float(rows_by_model["logreg"]["baseline_accuracy"]) == 0.91
    assert float(rows_by_model["logreg"]["baseline_macro_f1"]) == 0.87
    assert float(rows_by_model["logreg"]["baseline_auc"]) == 0.93
    assert rows_by_model["logreg"]["baseline_report_path"] == str(report_path)

    assert rows_by_model["rf"]["baseline_feature_set"] == "handcrafted"
    assert float(rows_by_model["rf"]["baseline_accuracy"]) == 0.71
    assert float(rows_by_model["rf"]["baseline_macro_f1"]) == 0.65
    assert float(rows_by_model["rf"]["baseline_auc"]) == 0.7

    md_text = registry_md.read_text(encoding="utf-8")
    assert "archive_baselines_resnet50" in md_text
    assert "Accuracy: 0.910" in md_text
    assert "Macro-F1: 0.870" in md_text
    assert "AUC: 0.930" in md_text

    artifacts_dir = tracking_root / "0" / run_id / "artifacts"
    assert (artifacts_dir / "baselines_report.json").exists()
    meta_yaml = (tracking_root / "0" / run_id / "meta.yaml").read_text(encoding="utf-8")
    assert "name: 'archive_baselines_resnet50'" in meta_yaml
    metrics_dir = tracking_root / "0" / run_id / "metrics"
    assert (metrics_dir / "embeddings__logreg__accuracy").exists()


def test_register_baselines_run_appends_all_models(tmp_path: Path) -> None:
    report_path = (
        tmp_path / "outputs" / "mamografias_baselines_effnet" / "baselines_report.json"
    )
    _write_multi_model_report(report_path)

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"

    registry.register_baselines_run(
        report_path=report_path,
        dataset="mamografias",
        run_name="mamografias_baselines_effnet",
        command="mammography embeddings-baselines --embeddings-dir outputs/mamografias_embeddings_effnet",
        registry_csv=registry_csv,
        registry_md=registry_md,
        log_mlflow=False,
    )

    with registry_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    models = {(row["baseline_feature_set"], row["baseline_model"]) for row in rows}
    assert models == {
        ("embeddings", "logreg"),
        ("embeddings", "svm-rbf"),
        ("handcrafted", "rf"),
        ("handcrafted", "svm-linear"),
    }
    for row in rows:
        assert row["run_name"] == "mamografias_baselines_effnet"
