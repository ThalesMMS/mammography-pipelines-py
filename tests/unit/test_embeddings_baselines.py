from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mammography.commands.embeddings_baselines import (
    _coerce_label,
    _compute_basic_stats,
    _limit_samples,
    _normalize_path,
    _resolve_label_column,
    _resolve_path_column,
    build_models,
    compute_handcrafted_features,
    load_embeddings,
    paired_t_test,
    render_report,
)


def _write_image(path: Path, value: int) -> None:
    data = np.full((8, 8), value, dtype=np.uint8)
    Image.fromarray(data, mode="L").save(path)


def test_compute_handcrafted_features_caps_components(tmp_path: Path) -> None:
    image_paths = []
    for idx in range(3):
        img_path = tmp_path / f"img_{idx}.png"
        _write_image(img_path, value=idx * 20)
        image_paths.append(img_path)

    metadata = pd.DataFrame(
        {
            "path": [str(path) for path in image_paths],
            "raw_label": [1, 2, 3],
        }
    )
    cache_path = tmp_path / "classic_features.npy"

    features = compute_handcrafted_features(
        metadata,
        cache_path,
        img_size=8,
        embeddings_dir=tmp_path,
        pca_svd_solver="full",
    )

    assert features.shape == (3, 15)
    assert cache_path.exists()

    cached = compute_handcrafted_features(
        metadata,
        cache_path,
        img_size=8,
        embeddings_dir=tmp_path,
        pca_svd_solver="full",
    )
    assert np.allclose(features, cached, equal_nan=True)


def test_load_embeddings_sanitizes_non_finite(tmp_path: Path) -> None:
    features = np.array([[1.0, np.nan], [np.inf, -np.inf]], dtype=np.float32)
    np.save(tmp_path / "features.npy", features)
    pd.DataFrame({"raw_label": [1, 2]}).to_csv(tmp_path / "metadata.csv", index=False)

    loaded, labels, meta = load_embeddings(tmp_path)

    assert loaded.shape == (2, 2)
    assert np.isfinite(loaded).all()
    assert labels.tolist() == [2, 3]
    assert len(meta) == 2


def test_coerce_label_handles_letter_grades() -> None:
    assert _coerce_label("A") == 1
    assert _coerce_label("B") == 2
    assert _coerce_label("C") == 3
    assert _coerce_label("D") == 4
    assert _coerce_label("a") == 1
    assert _coerce_label("  C  ") == 3


def test_coerce_label_handles_zero_indexed() -> None:
    assert _coerce_label(0) == 1
    assert _coerce_label(1) == 2
    assert _coerce_label(2) == 3
    assert _coerce_label(3) == 4


def test_coerce_label_handles_one_indexed() -> None:
    assert _coerce_label(1) == 2
    assert _coerce_label(2) == 3
    assert _coerce_label(3) == 4
    assert _coerce_label(4) == 4


def test_coerce_label_handles_invalid() -> None:
    assert _coerce_label(None) is None
    assert _coerce_label(np.nan) is None
    assert _coerce_label("invalid") is None
    assert _coerce_label("") is None
    assert _coerce_label("E") is None


def test_coerce_label_handles_string_numbers() -> None:
    assert _coerce_label("1") == 2
    assert _coerce_label("2") == 3
    assert _coerce_label("0") == 1


def test_resolve_label_column_finds_raw_label() -> None:
    meta = pd.DataFrame({"raw_label": [1, 2], "other": [3, 4]})
    assert _resolve_label_column(meta) == "raw_label"


def test_resolve_label_column_finds_professional_label() -> None:
    meta = pd.DataFrame({"professional_label": [1, 2], "other": [3, 4]})
    assert _resolve_label_column(meta) == "professional_label"


def test_resolve_label_column_finds_density_label() -> None:
    meta = pd.DataFrame({"density_label": [1, 2], "other": [3, 4]})
    assert _resolve_label_column(meta) == "density_label"


def test_resolve_label_column_finds_classification() -> None:
    meta = pd.DataFrame({"Classification": [1, 2], "other": [3, 4]})
    assert _resolve_label_column(meta) == "Classification"


def test_resolve_label_column_raises_on_missing() -> None:
    meta = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    with pytest.raises(ValueError, match="metadata.csv nao possui coluna de labels"):
        _resolve_label_column(meta)


def test_resolve_path_column_finds_path() -> None:
    meta = pd.DataFrame({"path": ["a.png", "b.png"], "other": [1, 2]})
    assert _resolve_path_column(meta) == "path"


def test_resolve_path_column_finds_image_path() -> None:
    meta = pd.DataFrame({"image_path": ["a.png", "b.png"], "other": [1, 2]})
    assert _resolve_path_column(meta) == "image_path"


def test_resolve_path_column_finds_dicom_path() -> None:
    meta = pd.DataFrame({"dicom_path": ["a.dcm", "b.dcm"], "other": [1, 2]})
    assert _resolve_path_column(meta) == "dicom_path"


def test_resolve_path_column_raises_on_missing() -> None:
    meta = pd.DataFrame({"foo": ["a", "b"], "bar": [1, 2]})
    with pytest.raises(ValueError, match="metadata.csv nao possui coluna de caminho"):
        _resolve_path_column(meta)


def test_normalize_path_handles_absolute(tmp_path: Path) -> None:
    file_path = tmp_path / "test.png"
    file_path.touch()
    result = _normalize_path(str(file_path), [tmp_path])
    assert result == file_path


def test_normalize_path_resolves_relative(tmp_path: Path) -> None:
    file_path = tmp_path / "test.png"
    file_path.touch()
    result = _normalize_path("test.png", [tmp_path])
    assert result.exists()
    assert result.name == "test.png"


def test_normalize_path_tries_multiple_roots(tmp_path: Path) -> None:
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    root1.mkdir()
    root2.mkdir()
    file_path = root2 / "test.png"
    file_path.touch()
    result = _normalize_path("test.png", [root1, root2])
    assert result == file_path


def test_normalize_path_handles_backslashes(tmp_path: Path) -> None:
    file_path = tmp_path / "sub" / "test.png"
    file_path.parent.mkdir(parents=True)
    file_path.touch()
    result = _normalize_path("sub\\test.png", [tmp_path])
    assert result.exists()


def test_compute_basic_stats_returns_expected_shape() -> None:
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    stats = _compute_basic_stats(arr)
    assert len(stats) == 12
    assert all(isinstance(v, float) for v in stats)


def test_compute_basic_stats_handles_uniform_array() -> None:
    arr = np.full((8, 8), 128, dtype=np.uint8)
    stats = _compute_basic_stats(arr)
    assert stats[0] == 128.0
    assert stats[2] == 128.0
    assert stats[3] == 128.0


def test_limit_samples_returns_all_when_max_exceeds_count() -> None:
    features = np.random.rand(10, 5).astype(np.float32)
    labels = np.array([1, 1, 2, 2, 3, 3, 4, 4, 1, 2], dtype=np.int64)
    meta = pd.DataFrame({"idx": range(10)})

    result_feat, result_labels, result_meta = _limit_samples(features, labels, meta, max_samples=100)

    assert result_feat.shape == (10, 5)
    assert len(result_labels) == 10
    assert len(result_meta) == 10


def test_limit_samples_stratifies_correctly() -> None:
    features = np.random.rand(100, 5).astype(np.float32)
    labels = np.array([1, 2, 3, 4] * 25, dtype=np.int64)
    meta = pd.DataFrame({"idx": range(100)})

    result_feat, result_labels, result_meta = _limit_samples(features, labels, meta, max_samples=20)

    assert result_feat.shape == (20, 5)
    assert len(result_labels) == 20
    assert len(result_meta) == 20
    unique, counts = np.unique(result_labels, return_counts=True)
    assert len(unique) == 4
    assert all(count >= 1 for count in counts)


def test_limit_samples_raises_when_max_less_than_classes() -> None:
    features = np.random.rand(10, 5).astype(np.float32)
    labels = np.array([1, 1, 2, 2, 3, 3, 4, 4, 1, 2], dtype=np.int64)
    meta = pd.DataFrame({"idx": range(10)})

    with pytest.raises(ValueError, match="max_samples precisa ser maior ou igual ao numero de classes"):
        _limit_samples(features, labels, meta, max_samples=2)


def test_limit_samples_handles_none_or_zero() -> None:
    features = np.random.rand(10, 5).astype(np.float32)
    labels = np.array([1, 1, 2, 2, 3, 3, 4, 4, 1, 2], dtype=np.int64)
    meta = pd.DataFrame({"idx": range(10)})

    result_feat, result_labels, result_meta = _limit_samples(features, labels, meta, max_samples=None)
    assert result_feat.shape == (10, 5)

    result_feat, result_labels, result_meta = _limit_samples(features, labels, meta, max_samples=0)
    assert result_feat.shape == (10, 5)


def test_load_embeddings_handles_1d_features(tmp_path: Path) -> None:
    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.save(tmp_path / "features.npy", features)
    pd.DataFrame({"raw_label": [1]}).to_csv(tmp_path / "metadata.csv", index=False)

    loaded, labels, meta = load_embeddings(tmp_path)

    assert loaded.shape == (1, 3)
    assert labels.tolist() == [2]
    assert len(meta) == 1


def test_load_embeddings_filters_invalid_labels(tmp_path: Path) -> None:
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    np.save(tmp_path / "features.npy", features)
    pd.DataFrame({"raw_label": [1, 99, 3]}).to_csv(tmp_path / "metadata.csv", index=False)

    loaded, labels, meta = load_embeddings(tmp_path)

    assert loaded.shape == (2, 2)
    assert labels.tolist() == [2, 4]
    assert len(meta) == 2


def test_load_embeddings_respects_max_samples(tmp_path: Path) -> None:
    features = np.random.rand(100, 5).astype(np.float32)
    np.save(tmp_path / "features.npy", features)
    pd.DataFrame({"raw_label": [1, 2, 3, 4] * 25}).to_csv(tmp_path / "metadata.csv", index=False)

    loaded, labels, meta = load_embeddings(tmp_path, max_samples=20)

    assert loaded.shape == (20, 5)
    assert len(labels) == 20
    assert len(meta) == 20


def test_build_models_returns_all_classifiers() -> None:
    models = build_models()
    assert "logreg" in models
    assert "svm-linear" in models
    assert "svm-rbf" in models
    assert "rf" in models
    assert all(callable(builder) for builder in models.values())


def test_build_models_pipelines_are_instantiable() -> None:
    models = build_models()
    for model_name, builder in models.items():
        pipeline = builder()
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict")


def test_paired_t_test_writes_results(tmp_path: Path) -> None:
    deep_scores = [0.8, 0.82, 0.81, 0.79, 0.83]
    classical_scores = [0.75, 0.76, 0.74, 0.77, 0.75]
    out_path = tmp_path / "ttest.json"

    paired_t_test(deep_scores, classical_scores, out_path)

    assert out_path.exists()
    import json
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert "t_statistic" in result
    assert "p_value" in result
    assert isinstance(result["t_statistic"], float)
    assert isinstance(result["p_value"], float)


def test_render_report_generates_markdown(tmp_path: Path) -> None:
    embeddings_summary = {
        "logreg": {
            "accuracy_mean": 0.85,
            "accuracy_std": 0.02,
            "macro_f1_mean": 0.83,
            "auc_mean": 0.90,
            "balanced_accuracy_mean": 0.84,
            "kappa_mean": 0.78,
        }
    }
    classical_summary = {
        "rf": {
            "accuracy_mean": 0.80,
            "accuracy_std": 0.03,
            "macro_f1_mean": 0.78,
            "auc_mean": 0.85,
            "balanced_accuracy_mean": 0.79,
            "kappa_mean": 0.72,
        }
    }
    report_path = tmp_path / "report.md"

    render_report(embeddings_summary, classical_summary, report_path)

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "# Baselines de Embeddings" in content
    assert "Embeddings (ResNet50)" in content
    assert "Classicos (PCA + textura)" in content
    assert "logreg" in content
    assert "rf" in content
    assert "0.850" in content
    assert "0.800" in content


def test_compute_handcrafted_features_uses_cache_when_size_matches(tmp_path: Path) -> None:
    image_paths = []
    for idx in range(2):
        img_path = tmp_path / f"img_{idx}.png"
        _write_image(img_path, value=idx * 50)
        image_paths.append(img_path)

    metadata = pd.DataFrame(
        {
            "path": [str(path) for path in image_paths],
            "raw_label": [1, 2],
        }
    )
    cache_path = tmp_path / "classic_features.npy"

    cached_data = np.random.rand(2, 15).astype(np.float32)
    np.save(cache_path, cached_data)

    features = compute_handcrafted_features(
        metadata,
        cache_path,
        img_size=8,
        embeddings_dir=tmp_path,
        pca_svd_solver="full",
    )

    assert np.array_equal(features, cached_data)


def test_compute_handcrafted_features_ignores_cache_when_size_mismatch(tmp_path: Path) -> None:
    image_paths = []
    for idx in range(3):
        img_path = tmp_path / f"img_{idx}.png"
        _write_image(img_path, value=idx * 50)
        image_paths.append(img_path)

    metadata = pd.DataFrame(
        {
            "path": [str(path) for path in image_paths],
            "raw_label": [1, 2, 3],
        }
    )
    cache_path = tmp_path / "classic_features.npy"

    cached_data = np.random.rand(2, 15).astype(np.float32)
    np.save(cache_path, cached_data)

    features = compute_handcrafted_features(
        metadata,
        cache_path,
        img_size=8,
        embeddings_dir=tmp_path,
        pca_svd_solver="full",
    )

    assert features.shape == (3, 15)
    assert not np.array_equal(features[:2], cached_data)
