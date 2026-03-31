from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mammography.commands import visualize as visualize_cmd


def test_parse_args_minimal() -> None:
    """Test parsing minimal required arguments."""
    args = visualize_cmd.parse_args(["--input", "features.npy"])
    assert args.input == "features.npy"
    assert args.output == "outputs/visualizations"
    assert args.seed == 42
    assert not args.report
    assert not args.tsne


def test_parse_args_report_mode() -> None:
    """Test parsing with report mode enabled."""
    args = visualize_cmd.parse_args([
        "--input", "features.npy",
        "--labels", "metadata.csv",
        "--report",
        "--output", "custom_output",
    ])
    assert args.input == "features.npy"
    assert args.labels == "metadata.csv"
    assert args.output == "custom_output"
    assert args.report is True


def test_parse_args_specific_visualizations() -> None:
    """Test parsing with specific visualization types."""
    args = visualize_cmd.parse_args([
        "--input", "features.npy",
        "--tsne",
        "--tsne-3d",
        "--heatmap",
        "--scatter-matrix",
    ])
    assert args.tsne is True
    assert args.tsne_3d is True
    assert args.heatmap is True
    assert args.scatter_matrix is True
    assert args.pca is False


def test_parse_args_from_run_mode() -> None:
    """Test parsing with --from-run mode."""
    args = visualize_cmd.parse_args([
        "--input", "outputs/run_001",
        "--from-run",
        "--report",
    ])
    assert args.input == "outputs/run_001"
    assert args.from_run is True
    assert args.report is True


def test_parse_args_tsne_parameters() -> None:
    """Test parsing t-SNE specific parameters."""
    args = visualize_cmd.parse_args([
        "--input", "features.npy",
        "--tsne",
        "--perplexity", "50",
        "--tsne-iter", "2000",
    ])
    assert args.perplexity == 50.0
    assert args.tsne_iter == 2000


def test_parse_args_binary_mode() -> None:
    """Test parsing with binary class names."""
    args = visualize_cmd.parse_args([
        "--input", "features.npy",
        "--binary",
        "--tsne",
    ])
    assert args.binary is True


def test_parse_args_registry_options() -> None:
    """Test parsing registry-related arguments."""
    args = visualize_cmd.parse_args([
        "--input", "features.npy",
        "--run-name", "test_run",
        "--registry-csv", "custom_registry.csv",
        "--no-mlflow",
        "--no-registry",
    ])
    assert args.run_name == "test_run"
    assert args.registry_csv == Path("custom_registry.csv")
    assert args.no_mlflow is True
    assert args.no_registry is True


def test_load_features_npy(tmp_path: Path) -> None:
    """Test loading features from .npy file."""
    features_path = tmp_path / "features.npy"
    features = np.random.rand(10, 128)
    np.save(features_path, features)

    loaded = visualize_cmd.load_features(str(features_path))
    assert loaded.shape == (10, 128)
    np.testing.assert_array_almost_equal(loaded, features)


def test_load_features_npz(tmp_path: Path) -> None:
    """Test loading features from .npz file with known key."""
    features_path = tmp_path / "features.npz"
    features = np.random.rand(10, 128)
    np.savez(features_path, features=features)

    loaded = visualize_cmd.load_features(str(features_path))
    assert loaded.shape == (10, 128)
    np.testing.assert_array_almost_equal(loaded, features)


def test_load_features_npz_first_array(tmp_path: Path) -> None:
    """Test loading features from .npz file with unknown key."""
    features_path = tmp_path / "features.npz"
    features = np.random.rand(10, 128)
    np.savez(features_path, custom_key=features)

    loaded = visualize_cmd.load_features(str(features_path))
    assert loaded.shape == (10, 128)


def test_load_labels_with_raw_label_column(tmp_path: Path) -> None:
    """Test loading labels from CSV with raw_label column."""
    labels_path = tmp_path / "metadata.csv"
    df = pd.DataFrame({"raw_label": [0, 1, 2, 3]})
    df.to_csv(labels_path, index=False)

    labels = visualize_cmd.load_labels(str(labels_path))
    assert labels is not None
    np.testing.assert_array_equal(labels, [0, 1, 2, 3])


def test_load_labels_with_alternative_column(tmp_path: Path) -> None:
    """Test loading labels from CSV with alternative column name."""
    labels_path = tmp_path / "metadata.csv"
    df = pd.DataFrame({"label": [0, 1, 2]})
    df.to_csv(labels_path, index=False)

    labels = visualize_cmd.load_labels(str(labels_path))
    assert labels is not None
    np.testing.assert_array_equal(labels, [0, 1, 2])


def test_load_labels_missing_file() -> None:
    """Test loading labels from non-existent file."""
    labels = visualize_cmd.load_labels("nonexistent.csv")
    assert labels is None


def test_load_labels_no_valid_column(tmp_path: Path) -> None:
    """Test loading labels from CSV without valid label column."""
    labels_path = tmp_path / "metadata.csv"
    df = pd.DataFrame({"other_col": [0, 1, 2]})
    df.to_csv(labels_path, index=False)

    labels = visualize_cmd.load_labels(str(labels_path))
    assert labels is None


def test_load_predictions(tmp_path: Path) -> None:
    """Test loading predictions CSV."""
    pred_path = tmp_path / "predictions.csv"
    df = pd.DataFrame({
        "true_label": [0, 1, 2],
        "pred_label": [0, 1, 1],
    })
    df.to_csv(pred_path, index=False)

    loaded = visualize_cmd.load_predictions(str(pred_path))
    assert loaded is not None
    assert "true_label" in loaded.columns
    assert len(loaded) == 3


def test_load_predictions_missing_file() -> None:
    """Test loading predictions from non-existent file."""
    loaded = visualize_cmd.load_predictions("nonexistent.csv")
    assert loaded is None


def test_load_history_json(tmp_path: Path) -> None:
    """Test loading training history from JSON."""
    history_path = tmp_path / "history.json"
    history = [
        {"epoch": 1, "loss": 0.5, "acc": 0.8},
        {"epoch": 2, "loss": 0.3, "acc": 0.9},
    ]
    with open(history_path, "w") as f:
        import json
        json.dump(history, f)

    loaded = visualize_cmd.load_history(str(history_path))
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0]["epoch"] == 1


def test_load_history_csv(tmp_path: Path) -> None:
    """Test loading training history from CSV."""
    history_path = tmp_path / "history.csv"
    df = pd.DataFrame({
        "epoch": [1, 2],
        "loss": [0.5, 0.3],
        "acc": [0.8, 0.9],
    })
    df.to_csv(history_path, index=False)

    loaded = visualize_cmd.load_history(str(history_path))
    assert loaded is not None
    assert len(loaded) == 2


def test_load_history_missing_file() -> None:
    """Test loading history from non-existent file."""
    loaded = visualize_cmd.load_history("nonexistent.json")
    assert loaded is None


def test_discover_run_artifacts(tmp_path: Path) -> None:
    """Test discovering artifacts in a run directory."""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    # Create various artifact files
    (run_dir / "features.npy").write_text("fake", encoding="utf-8")
    (run_dir / "metadata.csv").write_text("fake", encoding="utf-8")
    (run_dir / "val_predictions.csv").write_text("fake", encoding="utf-8")
    (run_dir / "train_history.csv").write_text("fake", encoding="utf-8")
    (run_dir / "val_metrics.json").write_text("fake", encoding="utf-8")

    artifacts = visualize_cmd.discover_run_artifacts(run_dir)

    assert artifacts["features"] == run_dir / "features.npy"
    assert artifacts["metadata"] == run_dir / "metadata.csv"
    assert artifacts["predictions"] == run_dir / "val_predictions.csv"
    assert artifacts["history"] == run_dir / "train_history.csv"
    assert artifacts["metrics"] == run_dir / "val_metrics.json"


def test_discover_run_artifacts_partial(tmp_path: Path) -> None:
    """Test discovering artifacts when only some files exist."""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    (run_dir / "embeddings.npy").write_text("fake", encoding="utf-8")

    artifacts = visualize_cmd.discover_run_artifacts(run_dir)

    assert artifacts["features"] == run_dir / "embeddings.npy"
    assert artifacts["metadata"] is None
    assert artifacts["predictions"] is None


def test_main_no_input_error(capsys) -> None:
    """Test that missing --input returns error."""
    result = visualize_cmd.main([])
    assert result == 1
    captured = capsys.readouterr()
    assert "required" in captured.err.lower() or "error" in captured.err.lower()


def test_main_missing_input_file_error(tmp_path: Path, capsys) -> None:
    """Test that non-existent input file returns error."""
    result = visualize_cmd.main(["--input", "nonexistent.npy"])
    assert result == 1


def test_main_no_visualization_type_warning(tmp_path: Path, capsys) -> None:
    """Test warning when no visualization type is specified."""
    features_path = tmp_path / "features.npy"
    features = np.random.rand(10, 128)
    np.save(features_path, features)

    result = visualize_cmd.main(["--input", str(features_path)])
    assert result == 0
    captured = capsys.readouterr()
    # Should warn but not error
    assert "No visualization type" in captured.out or "No visualization type" in captured.err


def test_main_tsne_plot_generation(tmp_path: Path, monkeypatch) -> None:
    """Test generating t-SNE plot."""
    features_path = tmp_path / "features.npy"
    features = np.random.rand(10, 128)
    np.save(features_path, features)

    output_dir = tmp_path / "output"

    # Mock plot_tsne_2d to avoid actual computation
    plot_called = {"called": False}

    def _fake_plot_tsne_2d(features, labels, **kwargs):
        plot_called["called"] = True
        out_path = kwargs.get("out_path")
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text("fake", encoding="utf-8")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        return None, fig

    monkeypatch.setattr(visualize_cmd, "plot_tsne_2d", _fake_plot_tsne_2d)
    monkeypatch.setattr(visualize_cmd.visualization_registry, "collect_visualization_outputs", lambda x: type('obj', (), {"output_paths": []})())
    monkeypatch.setattr(visualize_cmd.visualization_registry, "register_visualization_run", lambda **k: "run-123")
    monkeypatch.setattr(visualize_cmd.visualization_registry, "default_run_name", lambda x: "test_run")

    result = visualize_cmd.main([
        "--input", str(features_path),
        "--tsne",
        "--output", str(output_dir),
        "--no-registry",
    ])

    assert result == 0
    assert plot_called["called"]
    assert (output_dir / "tsne_2d.png").exists()


def test_main_report_generation(tmp_path: Path, monkeypatch) -> None:
    """Test generating visualization report."""
    features_path = tmp_path / "features.npy"
    features = np.random.rand(10, 128)
    np.save(features_path, features)

    labels_path = tmp_path / "metadata.csv"
    df = pd.DataFrame({"raw_label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    df.to_csv(labels_path, index=False)

    output_dir = tmp_path / "output"

    report_called = {"called": False}

    def _fake_generate_report(features, labels, **kwargs):
        report_called["called"] = True
        output_dir = kwargs.get("output_dir")
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "summary.png").write_text("fake", encoding="utf-8")
        return [output_dir / "summary.png"]

    monkeypatch.setattr(visualize_cmd, "generate_visualization_report", _fake_generate_report)
    monkeypatch.setattr(visualize_cmd.visualization_registry, "collect_visualization_outputs", lambda x: type('obj', (), {"output_paths": []})())
    monkeypatch.setattr(visualize_cmd.visualization_registry, "register_visualization_run", lambda **k: "run-123")
    monkeypatch.setattr(visualize_cmd.visualization_registry, "default_run_name", lambda x: "test_run")

    result = visualize_cmd.main([
        "--input", str(features_path),
        "--labels", str(labels_path),
        "--report",
        "--output", str(output_dir),
        "--no-registry",
    ])

    assert result == 0
    assert report_called["called"]


def test_main_from_run_mode(tmp_path: Path, monkeypatch) -> None:
    """Test --from-run mode discovers and uses artifacts."""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    features_path = run_dir / "features.npy"
    features = np.random.rand(10, 128)
    np.save(features_path, features)

    labels_path = run_dir / "metadata.csv"
    df = pd.DataFrame({"raw_label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    df.to_csv(labels_path, index=False)

    output_dir = tmp_path / "output"

    plot_called = {"called": False}

    def _fake_plot_tsne_2d(features, labels, **kwargs):
        plot_called["called"] = True
        out_path = kwargs.get("out_path")
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text("fake", encoding="utf-8")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        return None, fig

    monkeypatch.setattr(visualize_cmd, "plot_tsne_2d", _fake_plot_tsne_2d)
    monkeypatch.setattr(visualize_cmd.visualization_registry, "collect_visualization_outputs", lambda x: type('obj', (), {"output_paths": []})())
    monkeypatch.setattr(visualize_cmd.visualization_registry, "register_visualization_run", lambda **k: "run-123")
    monkeypatch.setattr(visualize_cmd.visualization_registry, "default_run_name", lambda x: "test_run")

    result = visualize_cmd.main([
        "--input", str(run_dir),
        "--from-run",
        "--tsne",
        "--output", str(output_dir),
        "--no-registry",
    ])

    assert result == 0
    assert plot_called["called"]


def test_main_from_run_missing_features_error(tmp_path: Path) -> None:
    """Test --from-run mode errors when no features found."""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    result = visualize_cmd.main([
        "--input", str(run_dir),
        "--from-run",
        "--tsne",
    ])

    assert result == 1


def test_main_from_run_not_directory_error(tmp_path: Path) -> None:
    """Test --from-run mode errors when input is not a directory."""
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("fake", encoding="utf-8")

    result = visualize_cmd.main([
        "--input", str(file_path),
        "--from-run",
        "--tsne",
    ])

    assert result == 1


def test_main_registers_visualization_run(tmp_path: Path, monkeypatch) -> None:
    """Test that visualization run is registered with correct metadata."""
    features_path = tmp_path / "features.npy"
    features = np.random.rand(10, 128)
    np.save(features_path, features)

    labels_path = tmp_path / "metadata.csv"
    df = pd.DataFrame({"raw_label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    df.to_csv(labels_path, index=False)

    output_dir = tmp_path / "output"
    registry_csv = tmp_path / "registry.csv"
    registry_md = tmp_path / "registry.md"

    def _fake_plot_tsne_2d(features, labels, **kwargs):
        out_path = kwargs.get("out_path")
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text("fake", encoding="utf-8")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        return None, fig

    captured_registry: dict[str, Any] = {}

    def _fake_register(**kwargs):
        captured_registry.update(kwargs)
        return "run-123"

    def _fake_collect(output_dir):
        return type('obj', (), {"output_paths": [output_dir / "tsne_2d.png"]})()

    monkeypatch.setattr(visualize_cmd, "plot_tsne_2d", _fake_plot_tsne_2d)
    monkeypatch.setattr(visualize_cmd.visualization_registry, "collect_visualization_outputs", _fake_collect)
    monkeypatch.setattr(visualize_cmd.visualization_registry, "register_visualization_run", _fake_register)
    monkeypatch.setattr(visualize_cmd.visualization_registry, "default_run_name", lambda x: "test_run")

    result = visualize_cmd.main([
        "--input", str(features_path),
        "--labels", str(labels_path),
        "--tsne",
        "--output", str(output_dir),
        "--run-name", "custom_viz_run",
        "--registry-csv", str(registry_csv),
        "--registry-md", str(registry_md),
        "--no-mlflow",
    ])

    assert result == 0
    assert captured_registry["run_name"] == "custom_viz_run"
    assert captured_registry["input_path"] == features_path
    assert captured_registry["labels_path"] == labels_path
    assert captured_registry["output_dir"] == output_dir


def test_main_confusion_matrix_without_predictions_warning(tmp_path: Path, monkeypatch, capsys) -> None:
    """Test that --confusion-matrix without --predictions shows warning."""
    features_path = tmp_path / "features.npy"
    features = np.random.rand(10, 128)
    np.save(features_path, features)

    output_dir = tmp_path / "output"

    monkeypatch.setattr(visualize_cmd.visualization_registry, "collect_visualization_outputs", lambda x: type('obj', (), {"output_paths": []})())
    monkeypatch.setattr(visualize_cmd.visualization_registry, "register_visualization_run", lambda **k: "run-123")
    monkeypatch.setattr(visualize_cmd.visualization_registry, "default_run_name", lambda x: "test_run")

    result = visualize_cmd.main([
        "--input", str(features_path),
        "--confusion-matrix",
        "--output", str(output_dir),
        "--no-registry",
    ])

    assert result == 0
    captured = capsys.readouterr()
    assert "requires --predictions" in captured.out or "requires --predictions" in captured.err


def test_main_exception_handling(tmp_path: Path, monkeypatch, capsys) -> None:
    """Test that exceptions during visualization are caught and logged."""
    features_path = tmp_path / "features.npy"
    features = np.random.rand(10, 128)
    np.save(features_path, features)

    def _raise_error(*args, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(visualize_cmd, "plot_tsne_2d", _raise_error)

    result = visualize_cmd.main([
        "--input", str(features_path),
        "--tsne",
        "--no-registry",
    ])

    assert result == 1
    captured = capsys.readouterr()
    assert "Error" in captured.out or "Error" in captured.err
