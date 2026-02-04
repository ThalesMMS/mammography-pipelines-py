"""Integration tests for mammography tune workflow."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.mark.integration
def test_tune_command_dry_run(tmp_path):
    """Test tune command with --dry-run validates configuration."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mammography.cli",
            "tune",
            "--dataset",
            "patches_completo",
            "--n-trials",
            "2",
            "--subset",
            "50",
            "--epochs",
            "1",
            "--dry-run",
            "--outdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "Dry-run complete" in result.stdout
    assert "Configuration Validated" in result.stdout


@pytest.mark.integration
@pytest.mark.slow
def test_tune_minimal_workflow(tmp_path):
    """Test actual tuning with minimal configuration."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mammography.cli",
            "tune",
            "--dataset",
            "patches_completo",
            "--n-trials",
            "1",  # Only 1 trial for speed
            "--subset",
            "20",  # Very small subset
            "--epochs",
            "1",  # Single epoch
            "--outdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"

    # Verify output files were created
    output_files = list(tmp_path.glob("**/tune_config.json"))
    assert len(output_files) > 0, "No tune_config.json created"

    # Verify best params were saved
    best_params_files = list(tmp_path.glob("**/*_best_params.json"))
    assert len(best_params_files) > 0, "No best_params.json created"

    # Verify stats were saved
    stats_files = list(tmp_path.glob("**/*_stats.json"))
    assert len(stats_files) > 0, "No stats.json created"


@pytest.mark.integration
def test_search_space_loading():
    """Test that search space can be loaded from default config."""
    from mammography.tuning.search_space import SearchSpace

    space = SearchSpace.from_yaml("configs/tune.yaml")
    assert len(space.parameters) == 6
    assert space.description is not None
