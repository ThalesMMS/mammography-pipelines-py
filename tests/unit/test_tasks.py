from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography.cli import _coerce_cli_args, _filter_embed_config_args, _load_config_args


def test_coerce_cli_args_variants() -> None:
    assert _coerce_cli_args("--seed 42") == ["--seed", "42"]

    args = _coerce_cli_args({"seed": 42, "amp": True, "opt": None})
    assert "--seed" in args
    assert "42" in args
    assert "--amp" in args
    assert "--opt" not in args

    args = _coerce_cli_args(["--epochs", 2])
    assert args == ["--epochs", "2"]


def test_load_config_args_merges_global_and_command(tmp_path: Path) -> None:
    payload = {
        "global": {"seed": 123, "log_level": "warning"},
        "embed": {"outdir": "outputs/embed", "pca": True},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    args = _load_config_args(config_path, "embed")
    assert "--seed" in args
    assert "123" in args
    assert "--log-level" in args
    assert "warning" in args
    assert "--outdir" in args
    assert "outputs/embed" in args
    assert "--pca" in args


def test_filter_embed_config_args_drops_csv_for_dataset_override() -> None:
    config_args = [
        "--dataset",
        "archive",
        "--csv",
        "./classificacao.csv",
        "--dicom-root",
        "./archive",
        "--outdir",
        "./outputs/embeddings_resnet50",
    ]
    forwarded = [
        "--dataset",
        "patches_completo",
        "--outdir",
        "outputs/patches_embeddings_resnet50",
    ]

    filtered = _filter_embed_config_args(config_args, forwarded)

    assert "--csv" not in filtered
    assert "--dicom-root" not in filtered
    assert "--dataset" not in filtered
