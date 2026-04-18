#!/usr/bin/env python3
#
# test_pipeline_reporting.py
# mammography-pipelines
#
# Unit tests for PipelineReportingMixin in pipeline/reporting.py.
#
"""Unit tests for mammography.pipeline.reporting module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


def _make_reporting():
    """Create a minimal concrete object with PipelineReportingMixin methods."""
    from mammography.pipeline.reporting import PipelineReportingMixin

    class ConcreteReporting(PipelineReportingMixin):
        pass

    return ConcreteReporting()


def _make_pipeline_results(**kwargs) -> Dict[str, Any]:
    """Create a minimal pipeline_results dict suitable for testing."""
    defaults = {
        "pipeline_timestamp": "2026-01-01T12:00:00",
        "input_dir": "/data/archive",
        "output_dir": "/data/output",
        "device": "cpu",
        "success": True,
        "total_processing_time": 10.5,
        "errors": [],
        "step_results": {},
    }
    defaults.update(kwargs)
    return defaults


class TestGenerateReportContent:
    """Tests for PipelineReportingMixin._generate_report_content."""

    def test_returns_dict_with_pipeline_summary(self):
        """Result contains a 'pipeline_summary' key."""
        reporting = _make_reporting()
        results = _make_pipeline_results()
        content = reporting._generate_report_content(results)
        assert "pipeline_summary" in content

    def test_pipeline_summary_has_expected_fields(self):
        """pipeline_summary contains timestamp, input/output dirs, device, success."""
        reporting = _make_reporting()
        results = _make_pipeline_results()
        content = reporting._generate_report_content(results)
        summary = content["pipeline_summary"]
        assert "timestamp" in summary
        assert "input_dir" in summary
        assert "output_dir" in summary
        assert "device" in summary
        assert "success" in summary
        assert "total_processing_time" in summary

    def test_pipeline_summary_values_match_input(self):
        """pipeline_summary values mirror the input pipeline_results."""
        reporting = _make_reporting()
        results = _make_pipeline_results(
            pipeline_timestamp="2026-03-15T10:00:00",
            input_dir="/custom/input",
            output_dir="/custom/output",
            device="cuda",
            success=False,
            total_processing_time=42.0,
        )
        content = reporting._generate_report_content(results)
        summary = content["pipeline_summary"]
        assert summary["timestamp"] == "2026-03-15T10:00:00"
        assert summary["input_dir"] == "/custom/input"
        assert summary["device"] == "cuda"
        assert summary["success"] is False
        assert summary["total_processing_time"] == 42.0

    def test_returns_dict_with_errors(self):
        """Result contains an 'errors' key mirroring pipeline-level errors."""
        reporting = _make_reporting()
        results = _make_pipeline_results(errors=["some error"])
        content = reporting._generate_report_content(results)
        assert "errors" in content
        assert content["errors"] == ["some error"]

    def test_returns_dict_with_step_results(self):
        """Result contains a 'step_results' key."""
        reporting = _make_reporting()
        results = _make_pipeline_results()
        content = reporting._generate_report_content(results)
        assert "step_results" in content

    def test_preprocessing_step_includes_file_counts(self):
        """preprocessing step result includes total_files, processed_files, failed_files."""
        reporting = _make_reporting()
        results = _make_pipeline_results(
            step_results={
                "preprocessing": {
                    "success": True,
                    "processing_time": 3.0,
                    "errors": [],
                    "total_files": 100,
                    "processed_files": 95,
                    "failed_files": 5,
                }
            }
        )
        content = reporting._generate_report_content(results)
        step = content["step_results"]["preprocessing"]
        assert step["total_files"] == 100
        assert step["processed_files"] == 95
        assert step["failed_files"] == 5

    def test_embedding_step_includes_tensor_counts(self):
        """embedding step result includes total_tensors, processed_tensors, failed_tensors."""
        reporting = _make_reporting()
        results = _make_pipeline_results(
            step_results={
                "embedding": {
                    "success": True,
                    "processing_time": 5.0,
                    "errors": [],
                    "total_tensors": 50,
                    "processed_tensors": 48,
                    "failed_tensors": 2,
                }
            }
        )
        content = reporting._generate_report_content(results)
        step = content["step_results"]["embedding"]
        assert step["total_tensors"] == 50
        assert step["processed_tensors"] == 48
        assert step["failed_tensors"] == 2

    def test_clustering_step_without_result_excluded(self):
        """clustering step with None clustering_result does not include algorithm/clusters."""
        reporting = _make_reporting()
        results = _make_pipeline_results(
            step_results={
                "clustering": {
                    "success": False,
                    "processing_time": 0.1,
                    "errors": ["failed"],
                    "clustering_result": None,
                }
            }
        )
        content = reporting._generate_report_content(results)
        step = content["step_results"]["clustering"]
        assert "algorithm" not in step

    def test_empty_step_results(self):
        """Empty step_results produces empty step_results in content."""
        reporting = _make_reporting()
        results = _make_pipeline_results(step_results={})
        content = reporting._generate_report_content(results)
        assert content["step_results"] == {}


class TestGenerateFinalReport:
    """Tests for PipelineReportingMixin._generate_final_report."""

    def test_returns_dict_with_step_key(self, tmp_path):
        """Returned dict has 'step' key set to 'report'."""
        reporting = _make_reporting()
        results = _make_pipeline_results()
        with patch("mammography.pipeline.reporting.yaml.dump"):
            with patch("builtins.open", create=True) as mock_open:
                import io
                mock_open.return_value.__enter__ = lambda s: io.StringIO()
                mock_open.return_value.__exit__ = MagicMock(return_value=False)
                report = reporting._generate_final_report(results, tmp_path)
        assert report["step"] == "report"

    def test_returns_success_true_on_normal_path(self, tmp_path):
        """Returns success=True when YAML is written without error."""
        reporting = _make_reporting()
        results = _make_pipeline_results()
        with patch("mammography.pipeline.reporting.yaml.dump"):
            with patch("builtins.open", create=True) as mock_open:
                import io
                mock_open.return_value.__enter__ = lambda s: io.StringIO()
                mock_open.return_value.__exit__ = MagicMock(return_value=False)
                report = reporting._generate_final_report(results, tmp_path)
        assert report["success"] is True

    def test_returns_success_false_on_error(self, tmp_path):
        """Returns success=False when an error occurs during report generation."""
        reporting = _make_reporting()
        results = _make_pipeline_results()
        with patch.object(reporting, "_generate_report_content", side_effect=RuntimeError("boom")):
            report = reporting._generate_final_report(results, tmp_path)
        assert report["success"] is False

    def test_errors_list_populated_on_failure(self, tmp_path):
        """errors list is non-empty when report generation fails."""
        reporting = _make_reporting()
        results = _make_pipeline_results()
        with patch.object(reporting, "_generate_report_content", side_effect=RuntimeError("test_error")):
            report = reporting._generate_final_report(results, tmp_path)
        assert len(report["errors"]) > 0

    def test_processing_time_is_positive(self, tmp_path):
        """processing_time is >= 0."""
        reporting = _make_reporting()
        results = _make_pipeline_results()
        with patch("mammography.pipeline.reporting.yaml.dump"):
            with patch("builtins.open", create=True) as mock_open:
                import io
                mock_open.return_value.__enter__ = lambda s: io.StringIO()
                mock_open.return_value.__exit__ = MagicMock(return_value=False)
                report = reporting._generate_final_report(results, tmp_path)
        assert report["processing_time"] >= 0

    def test_writes_yaml_file_to_output_dir(self, tmp_path):
        """A YAML file is created inside output_dir."""
        reporting = _make_reporting()
        results = _make_pipeline_results()
        tmp_path.mkdir(parents=True, exist_ok=True)
        report = reporting._generate_final_report(results, tmp_path)
        if report["success"]:
            report_path = Path(report["report_path"])
            assert report_path.parent == tmp_path
            assert report_path.suffix == ".yaml"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])