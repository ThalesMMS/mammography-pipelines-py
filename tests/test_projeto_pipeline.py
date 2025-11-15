from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("TORCH_HOME", str(REPO_ROOT / ".torch_cache"))
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.stage2
def test_projeto_train_density_subset(tmp_path: Path) -> None:
    out_root = REPO_ROOT / "outputs" / "test_stage2_smoke"
    if out_root.exists():
        shutil.rmtree(out_root)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "Projeto.py"),
        "train-density",
        "--csv",
        "classificacao.csv",
        "--dicom-root",
        "archive",
        "--outdir",
        str(out_root),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--img-size",
        "224",
        "--cache-mode",
        "none",
        "--subset",
        "16",
        "--class-weights",
        "auto",
        "--use-embeddings",
        "--seed",
        "123",
        "--device",
        "cpu",
        "--log-level",
        "warning",
    ]
    result = _run(cmd, timeout=1200)
    metrics_dir_candidates = sorted(out_root.glob("results*/metrics"))
    assert metrics_dir_candidates, f"Nenhum diretório de métricas em {out_root}\n{result.stdout}\n{result.stderr}"
    metrics_file = metrics_dir_candidates[-1] / "val_metrics.json"
    assert metrics_file.exists(), "val_metrics.json ausente após smoke test da Etapa 2."
    shutil.rmtree(out_root)
