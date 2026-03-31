import argparse
import csv
import json
import logging
from pathlib import Path

import pytest

from mammography.tools import train_registry as registry


def _write_training_artifacts(
    outdir: Path,
    *,
    val_acc: float = 0.6,
    val_macro_f1: float = 0.65,
    val_auc: float = 0.75,
    val_loss: float = 0.8,
) -> Path:
    results_dir = outdir / "results_1"
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": "archive",
        "arch": "efficientnet_b0",
        "batch_size": 16,
        "img_size": 512,
        "epochs": 3,
        "lr": 0.0001,
        "backbone_lr": 1e-05,
        "warmup_epochs": 2,
        "early_stop_patience": 5,
        "unfreeze_last_block": True,
        "classes": "density",
    }
    (results_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    train_history = (
        "epoch,train_loss,train_acc,val_loss,val_acc,val_auc,val_kappa,"
        "val_macro_f1,val_bal_acc,val_bal_acc_adj\n"
        f"1,0.9,0.7,{val_loss},{val_acc},{val_auc},0.5,{val_macro_f1},0.6,0.0\n"
    )
    (results_dir / "train_history.csv").write_text(train_history, encoding="utf-8")

    val_metrics = {
        "acc": val_acc,
        "macro_f1": val_macro_f1,
        "auc_ovr": val_auc,
        "loss": val_loss,
        "confusion_matrix": [[1, 2], [3, 4]],
        "classification_report": {
            "0": {"precision": 0.5, "recall": 0.6, "f1-score": 0.55, "support": 2},
            "1": {"precision": 0.7, "recall": 0.5, "f1-score": 0.58, "support": 2},
            "accuracy": 0.6,
            "macro avg": {
                "precision": 0.6,
                "recall": 0.55,
                "f1-score": 0.565,
                "support": 4,
            },
            "weighted avg": {
                "precision": 0.6,
                "recall": 0.6,
                "f1-score": 0.59,
                "support": 4,
            },
        },
    }
    (metrics_dir / "val_metrics.json").write_text(
        json.dumps(val_metrics), encoding="utf-8"
    )
    (metrics_dir / "val_metrics.png").write_bytes(b"dummy")

    (results_dir / "best_model.pt").write_bytes(b"checkpoint")
    (results_dir / "val_predictions.csv").write_text("id,pred\n1,0\n", encoding="utf-8")

    return results_dir


def _write_minimal_results(results_dir: Path, *, arch: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "archive",
        "arch": arch,
        "batch_size": 16,
        "img_size": 512,
        "epochs": 1,
        "classes": "density",
    }
    (results_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _write_view_specific_artifacts(outdir: Path, view: str) -> Path:
    metrics_dir = outdir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": "archive",
        "arch": "efficientnet_b0",
        "batch_size": 16,
        "img_size": 512,
        "epochs": 3,
        "lr": 0.0001,
        "classes": "density",
        "view": view,
        "view_specific_training": True,
    }
    (outdir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    train_history = (
        "epoch,train_loss,train_acc,val_loss,val_acc,val_auc,val_kappa,"
        "val_macro_f1,val_bal_acc,val_bal_acc_adj\n"
        "1,0.9,0.7,0.8,0.6,0.75,0.5,0.65,0.6,0.0\n"
    )
    (outdir / "train_history.csv").write_text(train_history, encoding="utf-8")

    val_metrics = {
        "acc": 0.6,
        "macro_f1": 0.65,
        "auc_ovr": 0.75,
        "loss": 0.8,
        "confusion_matrix": [[1, 2], [3, 4]],
        "classification_report": {
            "0": {"precision": 0.5, "recall": 0.6, "f1-score": 0.55, "support": 2},
            "1": {"precision": 0.7, "recall": 0.5, "f1-score": 0.58, "support": 2},
            "accuracy": 0.6,
            "macro avg": {
                "precision": 0.6,
                "recall": 0.55,
                "f1-score": 0.565,
                "support": 4,
            },
            "weighted avg": {
                "precision": 0.6,
                "recall": 0.6,
                "f1-score": 0.59,
                "support": 4,
            },
        },
    }
    view_suffix = view.lower()
    (metrics_dir / f"val_metrics_{view_suffix}.json").write_text(
        json.dumps(val_metrics), encoding="utf-8"
    )
    (metrics_dir / f"val_metrics_{view_suffix}.png").write_bytes(b"dummy")

    (outdir / f"best_model_{view_suffix}.pt").write_bytes(b"checkpoint")
    (outdir / "val_predictions.csv").write_text("id,pred\n1,0\n", encoding="utf-8")

    return outdir


def _write_ensemble_artifacts(outdir: Path) -> Path:
    metrics_dir = outdir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ensemble_metrics = {
        "acc": 0.8,
        "macro_f1": 0.75,
        "auc_ovr": 0.7,
        "kappa_quadratic": 0.6,
        "num_samples": 10,
        "confusion_matrix": [[5, 1], [2, 6]],
        "classification_report": {
            "0": {"precision": 0.7, "recall": 0.83, "f1-score": 0.76, "support": 6},
            "1": {"precision": 0.86, "recall": 0.75, "f1-score": 0.8, "support": 4},
            "accuracy": 0.8,
            "macro avg": {
                "precision": 0.78,
                "recall": 0.79,
                "f1-score": 0.78,
                "support": 10,
            },
            "weighted avg": {
                "precision": 0.78,
                "recall": 0.8,
                "f1-score": 0.78,
                "support": 10,
            },
        },
    }
    metrics_path = metrics_dir / "ensemble_metrics.json"
    metrics_path.write_text(json.dumps(ensemble_metrics), encoding="utf-8")
    (metrics_dir / "ensemble_metrics.png").write_bytes(b"dummy")
    return metrics_path


def test_register_training_run_writes_registry_and_mlflow_fallback(
    tmp_path: Path,
) -> None:
    baseline_outdir = tmp_path / "outputs" / "archive_density_baseline"
    _write_training_artifacts(
        baseline_outdir,
        val_acc=0.5,
        val_macro_f1=0.375,
        val_auc=0.75,
        val_loss=1.0,
    )
    outdir = tmp_path / "outputs" / "archive_density_effnet"
    _write_training_artifacts(
        outdir,
        val_acc=0.625,
        val_macro_f1=0.5,
        val_auc=0.875,
        val_loss=0.875,
    )

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"
    tracking_root = tmp_path / "mlruns"

    run_id = registry.register_training_run(
        outdir=outdir,
        dataset="archive",
        workflow="train-density",
        run_name="archive_effnet_density_v1",
        command="mammography train-density --dataset archive",
        registry_csv=registry_csv,
        registry_md=registry_md,
        tracking_uri=str(tracking_root),
        baseline_outdir=baseline_outdir,
        baseline_run_name="archive_effnet_density_v1",
        tuned_keys=[
            "batch_size",
            "lr",
            "backbone_lr",
            "warmup_epochs",
            "early_stop_patience",
            "unfreeze_last_block",
        ],
    )

    assert run_id
    with registry_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[0]
    assert row["workflow"] == "train-density"
    assert row["model"] == "efficientnet_b0"
    assert row["batch_size"] == "16"
    assert row["train_loss"] == "0.9"
    assert row["val_auc"] == "0.875"
    assert row["val_sensitivity"] == str(4 / 7)
    assert row["val_specificity"] == str(1 / 3)
    assert row["confusion_matrix_path"].endswith("val_metrics.png")
    assert "epochs" in row["hyperparameters"]
    assert row["baseline_run_name"] == "archive_effnet_density_v1"
    assert row["baseline_val_acc"] == "0.5"
    assert row["baseline_val_f1"] == "0.375"
    assert row["baseline_val_auc"] == "0.75"
    assert row["baseline_val_loss"] == "1.0"
    assert row["delta_val_acc"] == "0.125"
    assert row["delta_val_f1"] == "0.125"
    assert row["delta_val_auc"] == "0.125"
    assert row["delta_val_loss"] == "-0.125"
    assert "batch_size" in row["tuned_hyperparameters"]

    md_text = registry_md.read_text(encoding="utf-8")
    assert "archive_effnet_density_v1" in md_text
    assert "Hyperparameters" in md_text
    assert "Confusion matrix" in md_text
    assert "Tuned hyperparameters" in md_text
    assert "Baseline run" in md_text
    assert "Train loss" in md_text
    assert "Val sensitivity" in md_text
    assert "Val specificity" in md_text

    artifacts_dir = tracking_root / "0" / run_id / "artifacts"
    assert (artifacts_dir / "best_model.pt").exists()
    assert (artifacts_dir / "val_metrics.png").exists()


def test_collect_artifacts_uses_top_k_checkpoint_for_resumed_run(
    tmp_path: Path,
) -> None:
    previous_results = _write_training_artifacts(tmp_path / "outputs" / "previous_run")
    previous_checkpoint = previous_results / "checkpoint.pt"
    previous_checkpoint.write_bytes(b"resume-checkpoint")
    previous_top_k_dir = previous_results / "top_k"
    previous_top_k_dir.mkdir(parents=True, exist_ok=True)
    previous_top_k = previous_top_k_dir / "model_epoch028_macro_f10.6413.pt"
    previous_top_k.write_bytes(b"best-top-k")

    resumed_results = _write_training_artifacts(tmp_path / "outputs" / "resumed_run")
    (resumed_results / "best_model.pt").unlink()
    (resumed_results / "checkpoint.pt").write_bytes(b"latest-checkpoint")

    summary = json.loads((resumed_results / "summary.json").read_text(encoding="utf-8"))
    summary["resume_from"] = str(previous_checkpoint)
    summary["top_k"] = [
        {
            "score": 0.6413,
            "epoch": 28,
            "path": str(previous_top_k),
        }
    ]
    (resumed_results / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    artifacts = registry._collect_artifacts(resumed_results)

    assert artifacts.checkpoint_path == previous_top_k


def test_compute_sensitivity_specificity_from_report() -> None:
    val_metrics = {
        "classification_report": {
            "0": {"recall": 0.75},
            "1": {"recall": 0.25},
        }
    }

    sensitivity, specificity = registry._compute_sensitivity_specificity(val_metrics)

    assert sensitivity == pytest.approx(0.25)
    assert specificity == pytest.approx(0.75)


def test_collect_metrics_fallbacks_from_report() -> None:
    history_metrics = {
        "train_loss": None,
        "train_acc": None,
        "val_loss": None,
        "val_acc": None,
        "val_f1": None,
        "val_auc": None,
    }
    val_metrics = {
        "loss": 0.8,
        "auc": 0.9,
        "classification_report": {
            "0": {"recall": 0.82},
            "1": {"recall": 0.64},
            "accuracy": 0.72,
            "macro avg": {"f1-score": 0.68},
        },
    }

    metrics = registry._collect_metrics(history_metrics, val_metrics)

    assert metrics["val_loss"] == pytest.approx(0.8)
    assert metrics["val_acc"] == pytest.approx(0.72)
    assert metrics["val_f1"] == pytest.approx(0.68)
    assert metrics["val_auc"] == pytest.approx(0.9)
    assert metrics["val_sensitivity"] == pytest.approx(0.64)
    assert metrics["val_specificity"] == pytest.approx(0.82)


def test_find_results_dir_prefers_latest_suffix(tmp_path: Path) -> None:
    outdir = tmp_path / "outputs" / "archive_density_resnet"
    results_dir = outdir / "results"
    results_dir_1 = outdir / "results_1"

    _write_minimal_results(results_dir, arch="resnet50")
    _write_minimal_results(results_dir_1, arch="efficientnet_b0")

    picked = registry._find_results_dir(outdir)
    assert picked == results_dir_1


def test_register_training_run_handles_view_specific_metrics(tmp_path: Path) -> None:
    outdir = tmp_path / "outputs" / "archive_viewspecific" / "results_CC"
    _write_view_specific_artifacts(outdir, "CC")

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"
    tracking_root = tmp_path / "mlruns"

    run_id = registry.register_training_run(
        outdir=outdir,
        dataset="archive",
        workflow="train-density",
        run_name="archive_viewspecific_v1",
        command="mammography train-density --dataset archive",
        registry_csv=registry_csv,
        registry_md=registry_md,
        tracking_uri=str(tracking_root),
    )

    assert run_id
    with registry_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[0]
    assert row["val_metrics_path"].endswith("val_metrics_cc.json")
    assert row["confusion_matrix_path"].endswith("val_metrics_cc.png")
    assert row["checkpoint_path"].endswith("best_model_cc.pt")

    artifacts_dir = tracking_root / "0" / run_id / "artifacts"
    assert (artifacts_dir / "best_model_cc.pt").exists()
    assert (artifacts_dir / "val_metrics_cc.png").exists()


def test_register_ensemble_run_writes_registry_and_mlflow_fallback(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "outputs" / "archive_viewspecific" / "results"
    metrics_path = _write_ensemble_artifacts(outdir)

    summary_path = tmp_path / "summary.json"
    summary = {
        "dataset": "archive",
        "arch": "efficientnet_b0",
        "batch_size": 16,
        "img_size": 512,
        "epochs": 3,
        "lr": 0.0001,
        "classes": "density",
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"
    tracking_root = tmp_path / "mlruns"

    run_id = registry.register_ensemble_run(
        outdir=outdir,
        metrics_path=metrics_path,
        summary_path=summary_path,
        dataset="archive",
        workflow="train-density-ensemble",
        run_name="archive_viewspecific_v1",
        command="mammography train-density --dataset archive",
        registry_csv=registry_csv,
        registry_md=registry_md,
        tracking_uri=str(tracking_root),
    )

    assert run_id
    with registry_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[0]
    assert row["workflow"] == "train-density-ensemble"
    assert row["val_acc"] == "0.8"
    assert row["val_f1"] == "0.75"
    assert row["confusion_matrix_path"].endswith("ensemble_metrics.png")

    md_text = registry_md.read_text(encoding="utf-8")
    assert "ensemble" in md_text
    assert "Val acc" in md_text

    artifacts_dir = tracking_root / "0" / run_id / "artifacts"
    assert (artifacts_dir / "ensemble_metrics.json").exists()
    assert (artifacts_dir / "ensemble_metrics.png").exists()


def test_train_registry_hook_appends_registry_entry(tmp_path: Path) -> None:
    train_command = pytest.importorskip("mammography.commands.train")
    outdir = tmp_path / "outputs" / "archive_density_effnet"
    _write_training_artifacts(outdir)

    registry_csv = tmp_path / "results" / "registry.csv"
    registry_md = tmp_path / "results" / "registry.md"
    tracking_root = tmp_path / "mlruns"

    args = argparse.Namespace(
        tracker_run_name="archive_effnet_density_v1",
        tracker_uri=str(tracking_root),
        tracker_project="mammography",
        dataset="archive",
        view_specific_training=False,
        no_registry=False,
        registry_csv=registry_csv,
        registry_md=registry_md,
    )

    run_id = train_command._maybe_register_training_run(
        args=args,
        outdir_root=outdir,
        command="mammography train-density --dataset archive",
        logger=logging.getLogger("test"),
    )

    assert run_id
    with registry_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[0]
    assert row["run_name"] == "archive_effnet_density_v1"
    assert row["train_loss"] == "0.9"
    assert row["val_acc"] == "0.6"
    assert row["confusion_matrix_path"].endswith("val_metrics.png")
    assert "epochs" in row["hyperparameters"]

    md_text = registry_md.read_text(encoding="utf-8")
    assert "archive_effnet_density_v1" in md_text
