from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], timeout: int = 600) -> None:
    env = os.environ.copy()
    env.setdefault("TORCH_HOME", str(REPO_ROOT / ".torch_cache"))
    subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.stage1
def test_extract_mammo_resnet50_limit(tmp_path: Path) -> None:
    out_dir = REPO_ROOT / "outputs" / "test_stage1_smoke"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "extract_mammo_resnet50.py"),
        "--data_dir",
        "archive",
        "--csv_path",
        "classificacao.csv",
        "--out_dir",
        str(out_dir),
        "--batch_size",
        "1",
        "--num_workers",
        "0",
        "--device",
        "cpu",
        "--limit",
        "1",
        "--preview_max",
        "1",
    ]
    _run(cmd, timeout=900)
    features = out_dir / "features.npy"
    assert features.exists(), "features.npy não foi gerado no smoke test."
    shutil.rmtree(out_dir)
