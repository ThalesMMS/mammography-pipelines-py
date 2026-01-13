from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography import cli


def test_cli_embed_routes_to_extract_features() -> None:
    with patch.object(cli, "_run_module_passthrough") as mock_run:
        mock_run.return_value = 0
        exit_code = cli.main(["--dry-run", "embed"])
    assert exit_code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0] == "mammography.commands.extract_features"


def test_cli_train_routes_to_train_script() -> None:
    with patch.object(cli, "_run_module_passthrough") as mock_run:
        mock_run.return_value = 0
        exit_code = cli.main(["--dry-run", "train-density"])
    assert exit_code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0] == "mammography.commands.train"


def test_cli_visualize_routes_to_visualize_script() -> None:
    with patch.object(cli, "_run_module_passthrough") as mock_run:
        mock_run.return_value = 0
        exit_code = cli.main(["--dry-run", "visualize", "--input", "x.npy", "--outdir", "outputs/vis"])
    assert exit_code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0] == "mammography.commands.visualize"


def test_cli_embeddings_baselines_routes_to_script() -> None:
    with patch.object(cli, "_run_module_passthrough") as mock_run:
        mock_run.return_value = 0
        exit_code = cli.main(["--dry-run", "embeddings-baselines"])
    assert exit_code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0] == "mammography.commands.embeddings_baselines"


def test_cli_data_audit_routes_to_tool() -> None:
    with patch.object(cli, "_run_data_audit") as mock_run:
        mock_run.return_value = 0
        exit_code = cli.main(["--dry-run", "data-audit"])
    assert exit_code == 0
    mock_run.assert_called_once()
