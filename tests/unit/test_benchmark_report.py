from __future__ import annotations

import json
from pathlib import Path

import pytest

from mammography.tools.benchmark_report import (
    BenchmarkValidationError,
    ARTICLE_COLUMNS,
    MASTER_COLUMNS,
    ARCH_CONFIG,
    EXPECTED_SPLITS,
    expected_runs,
    generate_benchmark_report,
)


def _write_official_run(namespace: Path, dataset: str, task: str, arch: str) -> Path:
    seed_dir = namespace / dataset / task / arch / "seed42"
    results_dir = seed_dir / "results"
    metrics_dir = results_dir / "metrics"
    splits_dir = results_dir / "splits"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    config = ARCH_CONFIG[arch]
    summary = {
        "run_id": "results",
        "seed": 42,
        "created_at": "2026-03-12T12:00:00+00:00",
        "arch": arch,
        "classes": task,
        "dataset": dataset,
        "subset": 0,
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "img_size": config["img_size"],
        "pretrained": True,
        "lr": config["lr"],
        "backbone_lr": config["backbone_lr"],
        "augment": True,
        "train_backbone": True,
        "unfreeze_last_block": True,
        "warmup_epochs": config["warmup_epochs"],
        "deterministic": True,
        "allow_tf32": True,
        "amp": True,
        "split_mode": EXPECTED_SPLITS[dataset],
        "split_group_column": "patient_id" if dataset == "archive" else None,
        "test_frac": 0.1,
        "class_weights": "auto",
        "sampler_weighted": True,
        "early_stop_patience": config["early_stop_patience"],
        "resume_from": None,
        "tracker": "local",
        "tracker_run_name": f"{dataset}_{task}_{arch}_seed42",
        "view_specific_training": False,
        "save_val_preds": True,
        "export_val_embeddings": True,
        "best_acc": 0.81,
        "best_metric": 0.79,
        "best_epoch": 7,
        "reproducibility": {
            "git_commit": "deadbeef",
            "timestamp": "2026-03-12T12:00:00",
            "python_version": "3.13.3",
            "platform": "Windows-11",
            "torch_version": "2.8.0",
            "cuda_version": "12.6",
            "gpu_name": "Test GPU",
        },
    }
    (results_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    metrics = {
        "acc": 0.81,
        "kappa_quadratic": 0.74,
        "macro_f1": 0.79,
        "auc_ovr": 0.88,
        "epoch": 7,
        "classification_report": {
            "macro avg": {"f1-score": 0.79},
        },
    }
    (metrics_dir / "best_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    test_metrics = {
        "acc": 0.83,
        "kappa_quadratic": 0.76,
        "macro_f1": 0.81,
        "auc_ovr": 0.9,
        "epoch": 7,
        "classification_report": {
            "macro avg": {"f1-score": 0.81},
        },
    }
    (metrics_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics),
        encoding="utf-8",
    )
    (results_dir / "val_predictions.csv").write_text(
        "path,accession,raw_label,y_true,y_pred,probs\n"
        "img.png,ACC001,1,0,0,\"[0.9,0.1]\"\n",
        encoding="utf-8",
    )
    (results_dir / "embeddings_val.csv").write_text(
        "path,accession,raw_label\nimg.png,ACC001,1\n",
        encoding="utf-8",
    )
    (results_dir / "embeddings_val.npy").write_bytes(b"npy")
    (results_dir / "run.log").write_text(
        "Leakage check: using patient ids\nSplit criado: ok\nPatientID\n",
        encoding="utf-8",
    )

    if dataset == "archive":
        splits_dir.mkdir(parents=True, exist_ok=True)
        (splits_dir / "train.csv").write_text(
            "image_path,patient_id\ntrain_1.dcm,PAT001\ntrain_2.dcm,PAT002\n",
            encoding="utf-8",
        )
        (splits_dir / "val.csv").write_text(
            "image_path,patient_id\nval_1.dcm,PAT003\n",
            encoding="utf-8",
        )
        split_manifest = {
            "split_mode": "patient",
            "group_column": "patient_id",
        }
        (splits_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest),
            encoding="utf-8",
        )

    return results_dir


def _build_namespace(tmp_path: Path, *, legacy_density_alias: bool = False) -> Path:
    namespace = tmp_path / "outputs" / "rerun_2026q1"
    for run in expected_runs():
        task = "density" if legacy_density_alias and run.task == "multiclass" else run.task
        _write_official_run(namespace, run.dataset, task, run.arch)
    return namespace


def test_generate_benchmark_report_writes_expected_outputs(tmp_path: Path) -> None:
    namespace = _build_namespace(tmp_path)
    output_prefix = tmp_path / "results" / "rerun_2026q1_master"
    docs_report = tmp_path / "docs" / "reports" / "rerun_2026q1_technical_report.md"
    article_table = tmp_path / "Article" / "sections" / "rerun_2026q1_benchmark_table.tex"

    collected = generate_benchmark_report(
        namespace=namespace,
        output_prefix=output_prefix,
        docs_report_path=docs_report,
        article_table_path=article_table,
        exports_search_root=tmp_path / "outputs",
    )

    assert len(collected) == 18
    assert output_prefix.with_suffix(".csv").exists()
    assert output_prefix.with_suffix(".md").exists()
    assert output_prefix.with_suffix(".json").exists()
    assert output_prefix.with_suffix(".tex").exists()
    assert docs_report.exists()
    assert article_table.exists()

    json_rows = json.loads(output_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert len(json_rows) == 18
    assert list(json_rows[0].keys()) == MASTER_COLUMNS
    assert all(row["status"] == "accepted" for row in json_rows)

    article_text = article_table.read_text(encoding="utf-8")
    for column in ARTICLE_COLUMNS:
        assert column in article_text
    assert "archive & multiclass & patient & efficientnet\\_b0 & 0.8300 & 0.7600 & 0.8100 & 0.9000 \\\\" in article_text

    docs_text = docs_report.read_text(encoding="utf-8")
    assert "Execution timestamps seen in summaries" in docs_text
    assert "2026-03-12T12:00:00+00:00" in docs_text
    assert "Python: `3.13.3`" in docs_text
    assert "PyTorch: `2.8.0`" in docs_text
    assert "CUDA: `12.6`" in docs_text
    assert "GPU: `Test GPU`" in docs_text
    assert "held-out test split" in docs_text
    assert "test_frac=0.1" in docs_text


def test_generate_benchmark_report_rejects_incomplete_run(tmp_path: Path) -> None:
    namespace = _build_namespace(tmp_path)
    broken_run = namespace / "archive" / "multiclass" / "efficientnet_b0" / "seed42" / "results"
    (broken_run / "val_predictions.csv").unlink()

    with pytest.raises(BenchmarkValidationError, match="val_predictions.csv ausente"):
        generate_benchmark_report(
            namespace=namespace,
            output_prefix=tmp_path / "results" / "rerun_2026q1_master",
            docs_report_path=tmp_path / "docs" / "reports" / "rerun_2026q1_technical_report.md",
            article_table_path=tmp_path / "Article" / "sections" / "rerun_2026q1_benchmark_table.tex",
            exports_search_root=tmp_path / "outputs",
        )


def test_generate_benchmark_report_requires_test_metrics_when_test_split_is_enabled(
    tmp_path: Path,
) -> None:
    namespace = _build_namespace(tmp_path)
    broken_run = namespace / "archive" / "multiclass" / "efficientnet_b0" / "seed42" / "results"
    (broken_run / "metrics" / "test_metrics.json").unlink()

    with pytest.raises(BenchmarkValidationError, match="metrica de teste"):
        generate_benchmark_report(
            namespace=namespace,
            output_prefix=tmp_path / "results" / "rerun_2026q1_master",
            docs_report_path=tmp_path / "docs" / "reports" / "rerun_2026q1_technical_report.md",
            article_table_path=tmp_path / "Article" / "sections" / "rerun_2026q1_benchmark_table.tex",
            exports_search_root=tmp_path / "outputs",
        )


def test_generate_benchmark_report_rejects_archive_leakage(tmp_path: Path) -> None:
    namespace = _build_namespace(tmp_path)
    broken_run = namespace / "archive" / "multiclass" / "efficientnet_b0" / "seed42" / "results"
    (broken_run / "splits" / "train.csv").write_text(
        "image_path,patient_id\ntrain_1.dcm,PAT001\ntrain_2.dcm,PAT003\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkValidationError, match="vazamento detectado entre train/val"):
        generate_benchmark_report(
            namespace=namespace,
            output_prefix=tmp_path / "results" / "rerun_2026q1_master",
            docs_report_path=tmp_path / "docs" / "reports" / "rerun_2026q1_technical_report.md",
            article_table_path=tmp_path / "Article" / "sections" / "rerun_2026q1_benchmark_table.tex",
            exports_search_root=tmp_path / "outputs",
        )


def test_generate_benchmark_report_accepts_legacy_density_namespace(tmp_path: Path) -> None:
    namespace = _build_namespace(tmp_path, legacy_density_alias=True)

    collected = generate_benchmark_report(
        namespace=namespace,
        output_prefix=tmp_path / "results" / "rerun_2026q1_master",
        docs_report_path=tmp_path / "docs" / "reports" / "rerun_2026q1_technical_report.md",
        article_table_path=tmp_path / "Article" / "sections" / "rerun_2026q1_benchmark_table.tex",
        exports_search_root=tmp_path / "outputs",
    )

    assert len(collected) == 18
    assert all(run.expected.task in {"multiclass", "binary"} for run in collected)
