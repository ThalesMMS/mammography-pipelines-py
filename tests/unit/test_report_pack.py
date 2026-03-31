#!/usr/bin/env python3
#
# test_report_pack.py
# mammography-pipelines
#
# Unit tests for report_pack.py helper functions.
#
# Medical Disclaimer: This is an educational research project.
# Not intended for clinical use or medical diagnosis.
#
"""Unit tests for the report_pack module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mammography.tools.report_pack import (
    DensityRun,
    _build_explanation_grid,
    _build_gradcam_grid,
    _copy_asset,
    _format_metric_table,
    _iter_explanation_images,
    _iter_gradcam_images,
    _load_json,
    _summarize_run,
    package_density_runs,
)


class TestLoadJson:
    """Test _load_json helper function."""

    def test_load_json_valid_file(self, tmp_path: Path):
        """Test loading a valid JSON file."""
        json_file = tmp_path / "test.json"
        test_data = {"key": "value", "number": 42}
        json_file.write_text(json.dumps(test_data), encoding="utf-8")

        result = _load_json(json_file)
        assert result == test_data

    def test_load_json_empty_file(self, tmp_path: Path):
        """Test loading an empty JSON file."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("{}", encoding="utf-8")

        result = _load_json(json_file)
        assert result == {}


class TestCopyAsset:
    """Test _copy_asset helper function."""

    def test_copy_asset_creates_dest_dir(self, tmp_path: Path):
        """Test that _copy_asset creates destination directory if needed."""
        src = tmp_path / "source.png"
        src.write_text("fake image", encoding="utf-8")

        dest_dir = tmp_path / "nested" / "assets"
        dest = dest_dir / "copied.png"

        result = _copy_asset(src, dest)

        assert dest.exists()
        assert result == "copied.png"
        assert dest.read_text(encoding="utf-8") == "fake image"

    def test_copy_asset_preserves_content(self, tmp_path: Path):
        """Test that _copy_asset preserves file content."""
        src = tmp_path / "source.png"
        content = "test content"
        src.write_text(content, encoding="utf-8")

        dest = tmp_path / "dest.png"
        _copy_asset(src, dest)

        assert dest.read_text(encoding="utf-8") == content


class TestIterGradcamImages:
    """Test _iter_gradcam_images helper function."""

    def test_iter_gradcam_images_finds_pngs(self, tmp_path: Path):
        """Test that _iter_gradcam_images finds gradcam PNG files."""
        gradcam_dir = tmp_path / "gradcam"
        gradcam_dir.mkdir()

        # Create gradcam files
        for i in range(3):
            (gradcam_dir / f"gradcam_{i:03d}.png").write_text("fake", encoding="utf-8")

        # Create non-gradcam file (should be ignored)
        (gradcam_dir / "other.png").write_text("fake", encoding="utf-8")

        result = list(_iter_gradcam_images(gradcam_dir))
        assert len(result) == 3
        assert all(p.name.startswith("gradcam_") for p in result)

    def test_iter_gradcam_images_sorted_order(self, tmp_path: Path):
        """Test that _iter_gradcam_images returns files in sorted order."""
        gradcam_dir = tmp_path / "gradcam"
        gradcam_dir.mkdir()

        # Create files in non-sorted order
        for i in [2, 0, 1]:
            (gradcam_dir / f"gradcam_{i:03d}.png").write_text("fake", encoding="utf-8")

        result = list(_iter_gradcam_images(gradcam_dir))
        names = [p.name for p in result]
        assert names == sorted(names)

    def test_iter_gradcam_images_empty_dir(self, tmp_path: Path):
        """Test _iter_gradcam_images with empty directory."""
        gradcam_dir = tmp_path / "gradcam"
        gradcam_dir.mkdir()

        result = list(_iter_gradcam_images(gradcam_dir))
        assert len(result) == 0


class TestIterExplanationImages:
    """Test _iter_explanation_images helper function."""

    def test_iter_explanation_images_all_patterns(self, tmp_path: Path):
        """Test _iter_explanation_images finds both gradcam and attention files."""
        explanations_dir = tmp_path / "explanations"
        explanations_dir.mkdir()

        # Create gradcam files
        for i in range(2):
            (explanations_dir / f"gradcam_{i:03d}.png").write_text("fake", encoding="utf-8")

        # Create attention files
        for i in range(2):
            (explanations_dir / f"attention_{i:03d}.png").write_text("fake", encoding="utf-8")

        result = list(_iter_explanation_images(explanations_dir, pattern="*"))
        assert len(result) == 4

    def test_iter_explanation_images_gradcam_only(self, tmp_path: Path):
        """Test _iter_explanation_images with gradcam pattern."""
        explanations_dir = tmp_path / "explanations"
        explanations_dir.mkdir()

        (explanations_dir / "gradcam_001.png").write_text("fake", encoding="utf-8")
        (explanations_dir / "attention_001.png").write_text("fake", encoding="utf-8")

        result = list(_iter_explanation_images(explanations_dir, pattern="gradcam"))
        assert len(result) == 1
        assert result[0].name.startswith("gradcam_")

    def test_iter_explanation_images_attention_only(self, tmp_path: Path):
        """Test _iter_explanation_images with attention pattern."""
        explanations_dir = tmp_path / "explanations"
        explanations_dir.mkdir()

        (explanations_dir / "gradcam_001.png").write_text("fake", encoding="utf-8")
        (explanations_dir / "attention_001.png").write_text("fake", encoding="utf-8")

        result = list(_iter_explanation_images(explanations_dir, pattern="attention"))
        assert len(result) == 1
        assert result[0].name.startswith("attention_")


class TestBuildGradcamGrid:
    """Test _build_gradcam_grid helper function."""

    @pytest.mark.skipif(
        not pytest.importorskip("PIL", reason="Pillow not available"),
        reason="Pillow required for grid building",
    )
    def test_build_gradcam_grid_creates_image(self, tmp_path: Path):
        """Test that _build_gradcam_grid creates a grid image."""
        from PIL import Image

        # Create sample images
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        image_paths = []
        for i in range(4):
            img = Image.new("RGB", (100, 100), color=(i * 50, 100, 150))
            img_path = image_dir / f"gradcam_{i:03d}.png"
            img.save(img_path)
            image_paths.append(img_path)

        dest = tmp_path / "grid.png"
        result = _build_gradcam_grid(image_paths, dest, max_tiles=4)

        assert result == "grid.png"
        assert dest.exists()

        # Verify grid dimensions (2x2 grid)
        grid_img = Image.open(dest)
        assert grid_img.size == (200, 200)  # 2 cols x 2 rows of 100x100 images

    @pytest.mark.skipif(
        not pytest.importorskip("PIL", reason="Pillow not available"),
        reason="Pillow required for grid building",
    )
    def test_build_gradcam_grid_respects_max_tiles(self, tmp_path: Path):
        """Test that _build_gradcam_grid respects max_tiles parameter."""
        from PIL import Image

        # Create 6 sample images
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        image_paths = []
        for i in range(6):
            img = Image.new("RGB", (100, 100), color=(i * 40, 100, 150))
            img_path = image_dir / f"gradcam_{i:03d}.png"
            img.save(img_path)
            image_paths.append(img_path)

        dest = tmp_path / "grid.png"
        result = _build_gradcam_grid(image_paths, dest, max_tiles=4)

        assert result == "grid.png"
        assert dest.exists()

        # Should only use first 4 images
        grid_img = Image.open(dest)
        assert grid_img.size == (200, 200)  # 2x2 grid

    def test_build_gradcam_grid_empty_list(self, tmp_path: Path):
        """Test _build_gradcam_grid with empty image list."""
        dest = tmp_path / "grid.png"
        result = _build_gradcam_grid([], dest, max_tiles=4)

        assert result is None
        assert not dest.exists()


class TestBuildExplanationGrid:
    """Test _build_explanation_grid helper function."""

    @pytest.mark.skipif(
        not pytest.importorskip("PIL", reason="Pillow not available"),
        reason="Pillow required for grid building",
    )
    def test_build_explanation_grid_creates_image(self, tmp_path: Path):
        """Test that _build_explanation_grid creates a grid image."""
        from PIL import Image

        # Create sample images
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        image_paths = []
        for i in range(4):
            img = Image.new("RGB", (100, 100), color=(i * 50, 100, 150))
            img_path = image_dir / f"explanation_{i:03d}.png"
            img.save(img_path)
            image_paths.append(img_path)

        dest = tmp_path / "grid.png"
        result = _build_explanation_grid(image_paths, dest, max_tiles=4)

        assert result == "grid.png"
        assert dest.exists()

        # Verify grid dimensions (2x2 grid)
        grid_img = Image.open(dest)
        assert grid_img.size == (200, 200)

    def test_build_explanation_grid_empty_list(self, tmp_path: Path):
        """Test _build_explanation_grid with empty image list."""
        dest = tmp_path / "grid.png"
        result = _build_explanation_grid([], dest, max_tiles=4)

        assert result is None
        assert not dest.exists()


class TestFormatMetricTable:
    """Test _format_metric_table helper function."""

    def test_format_metric_table_single_run(self, tmp_path: Path):
        """Test _format_metric_table with a single run."""
        run = DensityRun(
            path=tmp_path / "run1",
            run_id="test_run_1",
            seed=42,
            summary_path=tmp_path / "summary.json",
            metrics_path=tmp_path / "metrics.json",
            metrics={},
            assets={},
            stats={
                "accuracy": 0.85,
                "kappa": 0.75,
                "macro_f1": 0.80,
                "auc": 0.90,
            },
        )

        table_rows, mean_std = _format_metric_table([run])

        assert len(table_rows) == 1
        assert table_rows[0]["seed"] == "42"
        assert table_rows[0]["run_id"] == "test_run_1"
        assert table_rows[0]["accuracy"] == "0.850"
        assert table_rows[0]["kappa"] == "0.750"

        # Mean/std for single run
        assert mean_std["accuracy"] == (0.85, 0.0)
        assert mean_std["kappa"] == (0.75, 0.0)
        assert mean_std["macro_f1"] == (0.80, 0.0)
        assert mean_std["auc"] == (0.90, 0.0)

    def test_format_metric_table_multiple_runs(self, tmp_path: Path):
        """Test _format_metric_table with multiple runs."""
        runs = [
            DensityRun(
                path=tmp_path / f"run{i}",
                run_id=f"test_run_{i}",
                seed=42 + i,
                summary_path=tmp_path / f"summary{i}.json",
                metrics_path=tmp_path / f"metrics{i}.json",
                metrics={},
                assets={},
                stats={
                    "accuracy": 0.80 + i * 0.02,
                    "kappa": 0.70 + i * 0.02,
                    "macro_f1": 0.75 + i * 0.02,
                    "auc": 0.85 + i * 0.02,
                },
            )
            for i in range(3)
        ]

        table_rows, mean_std = _format_metric_table(runs)

        assert len(table_rows) == 3

        # Check mean calculations
        assert abs(mean_std["accuracy"][0] - 0.82) < 0.01
        assert abs(mean_std["kappa"][0] - 0.72) < 0.01


class TestSummarizeRun:
    """Test _summarize_run helper function."""

    def test_summarize_run_basic(self, tmp_path: Path):
        """Test _summarize_run with basic run structure."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Create summary.json
        summary = {"run_id": "test_run", "seed": 42, "dataset": "mamografias"}
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        # Create metrics/val_metrics.json
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir()
        metrics = {
            "acc": 0.85,
            "kappa_quadratic": 0.75,
            "auc_ovr": 0.90,
            "classification_report": {
                "macro avg": {"f1-score": 0.80, "recall": 0.78},
                "weighted avg": {"f1-score": 0.82},
            },
        }
        metrics_path = metrics_dir / "val_metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        result = _summarize_run(run_dir, assets_dir, gradcam_limit=4)

        assert result.run_id == "test_run"
        assert result.seed == 42
        assert result.stats["accuracy"] == 0.85
        assert result.stats["kappa"] == 0.75
        assert result.stats["macro_f1"] == 0.80
        assert result.stats["auc"] == 0.90

    def test_summarize_run_with_train_curve(self, tmp_path: Path):
        """Test _summarize_run with train curve asset."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Create required files
        summary = {"run_id": "test_run", "seed": 42}
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir()
        metrics = {
            "acc": 0.85,
            "kappa_quadratic": 0.75,
            "auc_ovr": 0.90,
            "classification_report": {
                "macro avg": {"f1-score": 0.80, "recall": 0.78},
                "weighted avg": {"f1-score": 0.82},
            },
        }
        (metrics_dir / "val_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

        # Create train_history.png
        train_curve = run_dir / "train_history.png"
        train_curve.write_text("fake image", encoding="utf-8")

        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        result = _summarize_run(run_dir, assets_dir, gradcam_limit=4)

        assert result.assets["train_curve"] is not None
        assert result.assets["train_curve"] == "density_train_seed42.png"
        assert (assets_dir / "density_train_seed42.png").exists()

    def test_summarize_run_missing_summary_raises(self, tmp_path: Path):
        """Test _summarize_run raises error when summary.json is missing."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="summary.json ausente"):
            _summarize_run(run_dir, assets_dir, gradcam_limit=4)

    def test_summarize_run_missing_metrics_raises(self, tmp_path: Path):
        """Test _summarize_run raises error when metrics are missing."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        summary = {"run_id": "test_run", "seed": 42}
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="val_metrics.json ausente"):
            _summarize_run(run_dir, assets_dir, gradcam_limit=4)


class TestPackageDensityRuns:
    """Test package_density_runs integration function."""

    def test_package_density_runs_creates_assets_dir(self, tmp_path: Path):
        """Test that package_density_runs creates assets directory."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Create required files
        summary = {"run_id": "test_run", "seed": 42}
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir()
        metrics = {
            "acc": 0.85,
            "kappa_quadratic": 0.75,
            "auc_ovr": 0.90,
            "classification_report": {
                "macro avg": {"f1-score": 0.80, "recall": 0.78},
                "weighted avg": {"f1-score": 0.82},
            },
        }
        (metrics_dir / "val_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

        assets_dir = tmp_path / "assets"

        result = package_density_runs([run_dir], assets_dir, tex_path=None, gradcam_limit=4)

        assert assets_dir.exists()
        assert len(result) == 1
        assert result[0].seed == 42

    def test_package_density_runs_sorts_by_seed(self, tmp_path: Path):
        """Test that package_density_runs sorts results by seed."""
        run_dirs = []
        for seed in [44, 42, 43]:
            run_dir = tmp_path / f"run_{seed}"
            run_dir.mkdir()

            summary = {"run_id": f"run_{seed}", "seed": seed}
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir()
            metrics = {
                "acc": 0.85,
                "kappa_quadratic": 0.75,
                "auc_ovr": 0.90,
                "classification_report": {
                    "macro avg": {"f1-score": 0.80, "recall": 0.78},
                    "weighted avg": {"f1-score": 0.82},
                },
            }
            (metrics_dir / "val_metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )

            run_dirs.append(run_dir)

        assets_dir = tmp_path / "assets"

        result = package_density_runs(run_dirs, assets_dir, tex_path=None, gradcam_limit=4)

        assert len(result) == 3
        assert result[0].seed == 42
        assert result[1].seed == 43
        assert result[2].seed == 44


if __name__ == "__main__":
    pytest.main([__file__, "-v"])