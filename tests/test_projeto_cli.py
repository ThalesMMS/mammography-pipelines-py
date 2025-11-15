from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "Projeto.py"


def _run_cli(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    assert CLI.is_file(), "Projeto.py precisa existir na raiz do repositório"
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), pythonpath]))
    cmd = [sys.executable, str(CLI), *args]
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"Comando {' '.join(cmd)} falhou (exit={completed.returncode})\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def test_cli_help_runs() -> None:
    _run_cli(["--help"])


@pytest.mark.parametrize("subcommand", ["embed", "train-density"])
def test_cli_train_like_commands_support_dry_run(subcommand: str) -> None:
    _run_cli(["--dry-run", subcommand])


def test_cli_eval_export_prints_guidance() -> None:
    result = _run_cli(["eval-export"])
    assert "Checklist" in (result.stdout + result.stderr)


def test_cli_rl_refine_stub_handles_dry_run() -> None:
    _run_cli(["--dry-run", "rl-refine"])
