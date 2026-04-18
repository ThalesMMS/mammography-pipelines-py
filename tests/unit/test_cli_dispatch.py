#!/usr/bin/env python3
#
# test_cli_dispatch.py
# mammography-pipelines
#
# Unit tests for CLI dispatch helpers extracted to cli_dispatch.py.
#
"""Unit tests for mammography.cli_dispatch module."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from mammography.cli_dispatch import (
    _configure_logging,
    _entrypoint_accepts_args,
    _format_command,
    _invoke_entrypoint,
    _print_eval_guidance,
    _resolve_entrypoint,
    _run_benchmark_report,
    _run_data_audit,
    _run_eval_export,
    _run_module_passthrough,
    _working_directory,
)


class TestConfigureLogging:
    """Tests for _configure_logging."""

    def test_sets_info_level_by_default(self):
        """Configures root logger at INFO level."""
        with patch("mammography.cli_dispatch.logging.basicConfig") as mock_basic:
            _configure_logging("INFO")
            mock_basic.assert_called_once()
            call_kwargs = mock_basic.call_args
            assert call_kwargs.kwargs["level"] == logging.INFO

    def test_sets_debug_level(self):
        """Configures root logger at DEBUG level."""
        with patch("mammography.cli_dispatch.logging.basicConfig") as mock_basic:
            _configure_logging("DEBUG")
            assert mock_basic.call_args.kwargs["level"] == logging.DEBUG

    def test_sets_warning_level(self):
        """Configures root logger at WARNING level."""
        with patch("mammography.cli_dispatch.logging.basicConfig") as mock_basic:
            _configure_logging("WARNING")
            assert mock_basic.call_args.kwargs["level"] == logging.WARNING

    def test_unrecognized_level_defaults_to_info(self):
        """Falls back to INFO when level name is unrecognized."""
        with patch("mammography.cli_dispatch.logging.basicConfig") as mock_basic:
            _configure_logging("NOTREAL")
            assert mock_basic.call_args.kwargs["level"] == logging.INFO

    def test_case_insensitive_level(self):
        """Accepts lowercase level names."""
        with patch("mammography.cli_dispatch.logging.basicConfig") as mock_basic:
            _configure_logging("debug")
            assert mock_basic.call_args.kwargs["level"] == logging.DEBUG


class TestFormatCommand:
    """Tests for _format_command."""

    def test_single_token(self):
        """Formats a single token correctly."""
        result = _format_command(["mammography"])
        assert result == "mammography"

    def test_multiple_tokens_joined_with_spaces(self):
        """Joins tokens with spaces."""
        result = _format_command(["mammography", "embed", "--csv", "labels.csv"])
        assert result == "mammography embed --csv labels.csv"

    def test_tokens_with_spaces_are_quoted(self):
        """Tokens containing spaces are shell-quoted."""
        result = _format_command(["python", "my script.py"])
        assert "my script.py" not in result or "'my script.py'" in result

    def test_path_object_converted_to_string(self):
        """Path objects are converted to strings."""
        result = _format_command([Path("/usr/bin/python"), "script.py"])
        assert "/usr/bin/python" in result

    def test_empty_command(self):
        """Empty command produces empty string."""
        result = _format_command([])
        assert result == ""


class TestWorkingDirectory:
    """Tests for _working_directory context manager."""

    def test_changes_to_target_directory(self, tmp_path):
        """Changes working directory to specified path inside context."""
        with _working_directory(tmp_path) as original:
            assert Path.cwd() == tmp_path
            assert original != tmp_path or original == tmp_path  # just that it yielded

    def test_restores_original_directory_on_exit(self, tmp_path):
        """Restores original working directory after context exits."""
        original_cwd = Path.cwd()
        with _working_directory(tmp_path):
            pass
        assert Path.cwd() == original_cwd

    def test_restores_directory_on_exception(self, tmp_path):
        """Restores original working directory even when exception is raised."""
        original_cwd = Path.cwd()
        with pytest.raises(ValueError):
            with _working_directory(tmp_path):
                raise ValueError("test error")
        assert Path.cwd() == original_cwd

    def test_yields_original_directory(self, tmp_path):
        """Yields the directory that was current before entering context."""
        expected_original = Path.cwd()
        with _working_directory(tmp_path) as yielded:
            assert yielded == expected_original


class TestResolveEntrypoint:
    """Tests for _resolve_entrypoint."""

    def test_resolves_main_function(self):
        """Finds 'main' as default entrypoint when no explicit name given."""
        mock_module = MagicMock()
        mock_module.main = lambda: 0
        with patch("mammography.cli_dispatch.importlib.import_module", return_value=mock_module):
            handler = _resolve_entrypoint("fake.module")
            assert callable(handler)

    def test_resolves_run_function_as_fallback(self):
        """Falls back to 'run' when 'main' is absent."""
        mock_module = MagicMock(spec=[])
        mock_module.run = lambda: 0
        with patch("mammography.cli_dispatch.importlib.import_module", return_value=mock_module):
            # Make getattr return None for 'main' but return run
            with patch.object(type(mock_module), "__getattr__", side_effect=AttributeError):
                pass
        # Use a more direct approach
        mock_module2 = type("FakeMod", (), {"run": lambda: 0})()
        with patch("mammography.cli_dispatch.importlib.import_module", return_value=mock_module2):
            handler = _resolve_entrypoint("fake.module")
            assert callable(handler)

    def test_resolves_explicit_entrypoint_name(self):
        """Uses the explicitly named entrypoint attribute."""
        mock_fn = MagicMock(return_value=0)
        mock_module = type("FakeMod", (), {"custom_entry": mock_fn})()
        with patch("mammography.cli_dispatch.importlib.import_module", return_value=mock_module):
            handler = _resolve_entrypoint("fake.module", entrypoint="custom_entry")
            assert handler is mock_fn

    def test_raises_attribute_error_when_not_callable(self):
        """Raises AttributeError when resolved attribute is not callable."""
        mock_module = type("FakeMod", (), {"main": "not_a_callable"})()
        with patch("mammography.cli_dispatch.importlib.import_module", return_value=mock_module):
            with pytest.raises(AttributeError):
                _resolve_entrypoint("fake.module")


class TestEntrypointAcceptsArgs:
    """Tests for _entrypoint_accepts_args."""

    def test_function_with_positional_param_returns_true(self):
        """Returns True for function with a named parameter."""
        def handler(argv): pass
        assert _entrypoint_accepts_args(handler) is True

    def test_function_with_var_positional_returns_true(self):
        """Returns True for function with *args."""
        def handler(*args): pass
        assert _entrypoint_accepts_args(handler) is True

    def test_function_with_var_keyword_returns_true(self):
        """Returns True for function with **kwargs."""
        def handler(**kwargs): pass
        assert _entrypoint_accepts_args(handler) is True

    def test_function_with_no_params_returns_false(self):
        """Returns False for a no-argument function."""
        def handler(): pass
        assert _entrypoint_accepts_args(handler) is False

    def test_non_inspectable_callable_returns_false(self):
        """Returns False when inspect.signature raises TypeError."""
        class NonInspectable:
            def __call__(self): pass
        obj = NonInspectable()
        with patch("mammography.cli_dispatch.inspect.signature", side_effect=TypeError):
            assert _entrypoint_accepts_args(obj) is False


class TestInvokeEntrypoint:
    """Tests for _invoke_entrypoint."""

    def test_forwards_args_to_accepting_handler(self):
        """Passes cmd_args list to a handler that accepts args."""
        received = []
        def handler(args): received.extend(args)
        _invoke_entrypoint(handler, ["--flag", "value"])
        assert received == ["--flag", "value"]

    def test_calls_no_arg_handler_without_args(self):
        """Calls a zero-argument handler with no arguments."""
        called = []
        def handler(): called.append(True)
        _invoke_entrypoint(handler, ["--ignored"])
        assert called == [True]

    def test_returns_zero_when_handler_returns_none(self):
        """Returns 0 when handler returns None."""
        def handler(): return None
        assert _invoke_entrypoint(handler, []) == 0

    def test_returns_integer_exit_code(self):
        """Returns the integer value returned by the handler."""
        def handler(): return 42
        assert _invoke_entrypoint(handler, []) == 42

    def test_returns_zero_for_bool_true(self):
        """Returns 0 when handler returns True (bool is not int for our purposes)."""
        def handler(): return True
        assert _invoke_entrypoint(handler, []) == 0

    def test_returns_zero_for_non_int_return(self):
        """Returns 0 when handler returns a non-integer, non-None value."""
        def handler(): return "done"
        assert _invoke_entrypoint(handler, []) == 0

    def test_returns_nonzero_exit_code(self):
        """Returns non-zero exit code when handler returns non-zero integer."""
        def handler(args): return 1
        assert _invoke_entrypoint(handler, []) == 1


class TestRunModulePassthrough:
    """Tests for _run_module_passthrough."""

    def _make_args(self, dry_run: bool = False) -> argparse.Namespace:
        args = argparse.Namespace()
        args.dry_run = dry_run
        return args

    def test_dry_run_returns_zero_without_executing(self):
        """Returns 0 and does not call _resolve_entrypoint in dry-run mode."""
        args = self._make_args(dry_run=True)
        with patch("mammography.cli_dispatch._resolve_entrypoint") as mock_resolve:
            result = _run_module_passthrough("fake.module", args, ["--arg"])
            mock_resolve.assert_not_called()
            assert result == 0

    def test_non_dry_run_invokes_entrypoint(self):
        """Resolves and invokes the entrypoint when not in dry-run mode."""
        args = self._make_args(dry_run=False)
        mock_handler = MagicMock(return_value=0)
        mock_handler.__code__ = MagicMock()  # make it inspectable
        with patch("mammography.cli_dispatch._resolve_entrypoint", return_value=mock_handler):
            with patch("mammography.cli_dispatch._working_directory"):
                with patch("mammography.cli_dispatch._invoke_entrypoint", return_value=0) as mock_invoke:
                    result = _run_module_passthrough("fake.module", args, ["--flag"])
                    mock_invoke.assert_called_once()
                    assert result == 0

    def test_module_name_included_in_log(self, caplog):
        """Logs the module name in the execution message."""
        args = self._make_args(dry_run=True)
        with caplog.at_level(logging.INFO, logger="projeto"):
            _run_module_passthrough("mammography.commands.train", args, [])
        assert "mammography.commands.train" in caplog.text


class TestPrintEvalGuidance:
    """Tests for _print_eval_guidance."""

    def _make_args(self, runs=None) -> argparse.Namespace:
        args = argparse.Namespace()
        args.command = "eval-export"
        args.config = None
        args.runs = runs
        return args

    def test_returns_zero(self):
        """Always returns 0."""
        args = self._make_args()
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            result = _print_eval_guidance(args, [])
        assert result == 0

    def test_logs_checklist_items(self, caplog):
        """Logs the standard evaluation checklist items."""
        args = self._make_args()
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with caplog.at_level(logging.INFO, logger="projeto"):
                _print_eval_guidance(args, [])
        assert "Checklist" in caplog.text

    def test_logs_forwarded_args_when_present(self, caplog):
        """Logs forwarded arguments when provided."""
        args = self._make_args()
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with caplog.at_level(logging.INFO, logger="projeto"):
                _print_eval_guidance(args, ["--extra-flag"])
        assert "--extra-flag" in caplog.text

    def test_audits_missing_files_in_run_paths(self, tmp_path, caplog):
        """Warns about missing required artifacts for each run path."""
        run_dir = tmp_path / "results_seed42"
        run_dir.mkdir()
        args = self._make_args(runs=[run_dir])
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with patch("mammography.cli_dispatch.REPO_ROOT", tmp_path):
                with caplog.at_level(logging.WARNING, logger="projeto"):
                    _print_eval_guidance(args, [])
        assert "faltantes" in caplog.text.lower() or "Itens" in caplog.text

    def test_reads_summary_json_when_present(self, tmp_path, caplog):
        """Parses and logs metrics from summary.json when it exists."""
        import json
        run_dir = tmp_path / "results_seed42"
        run_dir.mkdir()
        summary = {
            "seed": 42,
            "val_metrics": {
                "accuracy": 0.85,
                "kappa_quadratic": 0.75,
                "macro_f1": 0.82,
                "auc_ovr": 0.91,
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        args = self._make_args(runs=[run_dir])
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with patch("mammography.cli_dispatch.REPO_ROOT", tmp_path):
                with caplog.at_level(logging.INFO, logger="projeto"):
                    _print_eval_guidance(args, [])
        assert "0.850" in caplog.text or "acc=0.850" in caplog.text


class TestRunBenchmarkReport:
    """Tests for _run_benchmark_report."""

    def _make_args(self, **kwargs) -> argparse.Namespace:
        defaults = {
            "command": "benchmark-report",
            "config": None,
            "namespace": Path("outputs/rerun_2026q1"),
            "output_prefix": Path("results/master"),
            "docs_report": Path("docs/reports/report.md"),
            "article_table": Path("Article/sections/table.tex"),
            "exports_search_root": Path("outputs"),
            "dry_run": True,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_dry_run_returns_zero(self):
        """Returns 0 in dry-run mode without executing."""
        args = self._make_args()
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with patch("mammography.cli_dispatch._run_module_passthrough", return_value=0) as mock_run:
                result = _run_benchmark_report(args, [])
                assert result == 0
                mock_run.assert_called_once()

    def test_includes_namespace_in_cmd_args(self):
        """Includes --namespace flag in assembled command args."""
        args = self._make_args()
        captured_cmd_args = []
        def capture_passthrough(module, a, cmd_args, entrypoint=None):
            captured_cmd_args.extend(cmd_args)
            return 0
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with patch("mammography.cli_dispatch._run_module_passthrough", side_effect=capture_passthrough):
                _run_benchmark_report(args, [])
        assert "--namespace" in captured_cmd_args
        assert str(args.namespace) in captured_cmd_args

    def test_includes_output_prefix_in_cmd_args(self):
        """Includes --output-prefix flag in assembled command args."""
        args = self._make_args()
        captured_cmd_args = []
        def capture_passthrough(module, a, cmd_args, entrypoint=None):
            captured_cmd_args.extend(cmd_args)
            return 0
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with patch("mammography.cli_dispatch._run_module_passthrough", side_effect=capture_passthrough):
                _run_benchmark_report(args, [])
        assert "--output-prefix" in captured_cmd_args

    def test_appends_forwarded_args(self):
        """Appends extra forwarded args to the command."""
        args = self._make_args()
        captured_cmd_args = []
        def capture_passthrough(module, a, cmd_args, entrypoint=None):
            captured_cmd_args.extend(cmd_args)
            return 0
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with patch("mammography.cli_dispatch._run_module_passthrough", side_effect=capture_passthrough):
                _run_benchmark_report(args, ["--extra-flag"])
        assert "--extra-flag" in captured_cmd_args

    def test_routes_to_benchmark_report_module(self):
        """Routes command to mammography.commands.benchmark_report module."""
        args = self._make_args()
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with patch("mammography.cli_dispatch._run_module_passthrough", return_value=0) as mock_run:
                _run_benchmark_report(args, [])
                called_module = mock_run.call_args[0][0]
                assert called_module == "mammography.commands.benchmark_report"


class TestRunDataAudit:
    """Tests for _run_data_audit."""

    def _make_args(self, **kwargs) -> argparse.Namespace:
        defaults = {
            "command": "data-audit",
            "config": None,
            "archive": Path("archive"),
            "csv": Path("classificacao.csv"),
            "manifest": Path("data_manifest.json"),
            "audit_csv": Path("outputs/audit.csv"),
            "log": Path("Article/assets/data_qc.log"),
            "dry_run": True,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_includes_archive_in_cmd_args(self):
        """Includes --archive flag in assembled command args."""
        args = self._make_args()
        captured_cmd_args = []
        def capture_passthrough(module, a, cmd_args, entrypoint=None):
            captured_cmd_args.extend(cmd_args)
            return 0
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with patch("mammography.cli_dispatch._run_module_passthrough", side_effect=capture_passthrough):
                _run_data_audit(args, [])
        assert "--archive" in captured_cmd_args
        assert "archive" in captured_cmd_args

    def test_includes_csv_in_cmd_args(self):
        """Includes --csv flag in assembled command args."""
        args = self._make_args()
        captured_cmd_args = []
        def capture_passthrough(module, a, cmd_args, entrypoint=None):
            captured_cmd_args.extend(cmd_args)
            return 0
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with patch("mammography.cli_dispatch._run_module_passthrough", side_effect=capture_passthrough):
                _run_data_audit(args, [])
        assert "--csv" in captured_cmd_args

    def test_routes_to_data_audit_module(self):
        """Routes command to mammography.tools.data_audit module."""
        args = self._make_args()
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with patch("mammography.cli_dispatch._run_module_passthrough", return_value=0) as mock_run:
                _run_data_audit(args, [])
                called_module = mock_run.call_args[0][0]
                assert called_module == "mammography.tools.data_audit"

    def test_appends_forwarded_args(self):
        """Appends extra forwarded args to the command."""
        args = self._make_args()
        captured_cmd_args = []
        def capture_passthrough(module, a, cmd_args, entrypoint=None):
            captured_cmd_args.extend(cmd_args)
            return 0
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with patch("mammography.cli_dispatch._run_module_passthrough", side_effect=capture_passthrough):
                _run_data_audit(args, ["--verbose"])
        assert "--verbose" in captured_cmd_args


class TestRunEvalExport:
    """Tests for _run_eval_export."""

    def _make_args(self, **kwargs) -> argparse.Namespace:
        from mammography.utils.cli_args import add_tracking_args
        import argparse as _argparse
        p = _argparse.ArgumentParser()
        add_tracking_args(p)
        ns, _ = p.parse_known_args([])
        defaults = {
            "command": "eval-export",
            "config": None,
            "runs": None,
            "output_dir": Path("outputs/exports"),
            "dry_run": True,
        }
        for k, v in defaults.items():
            setattr(ns, k, v)
        for k, v in kwargs.items():
            setattr(ns, k, v)
        return ns

    def test_routes_to_eval_export_module(self):
        """Routes command to mammography.commands.eval_export module."""
        args = self._make_args()
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with patch("mammography.cli_dispatch._run_module_passthrough", return_value=0) as mock_run:
                _run_eval_export(args, [])
                called_module = mock_run.call_args[0][0]
                assert called_module == "mammography.commands.eval_export"

    def test_includes_run_paths_in_cmd_args(self):
        """Includes --run flags for each run path."""
        args = self._make_args(runs=[Path("outputs/run1"), Path("outputs/run2")])
        captured_cmd_args = []
        def capture_passthrough(module, a, cmd_args, entrypoint=None):
            captured_cmd_args.extend(cmd_args)
            return 0
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with patch("mammography.cli_dispatch._run_module_passthrough", side_effect=capture_passthrough):
                _run_eval_export(args, [])
        assert captured_cmd_args.count("--run") == 2

    def test_includes_output_dir_in_cmd_args(self):
        """Includes --output-dir flag when output_dir is set."""
        args = self._make_args(output_dir=Path("custom/exports"))
        captured_cmd_args = []
        def capture_passthrough(module, a, cmd_args, entrypoint=None):
            captured_cmd_args.extend(cmd_args)
            return 0
        with patch("mammography.cli_dispatch._load_config_args", return_value=[]):
            with patch("mammography.cli_dispatch._run_module_passthrough", side_effect=capture_passthrough):
                _run_eval_export(args, [])
        assert "--output-dir" in captured_cmd_args
        assert "custom/exports" in captured_cmd_args


if __name__ == "__main__":
    pytest.main([__file__, "-v"])