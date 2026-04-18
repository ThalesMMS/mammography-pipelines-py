from __future__ import annotations

from pathlib import Path

from mammography.utils.config_loader import (
    coerce_cli_args,
    dict_to_cli_args,
    load_config_args,
    read_config,
)


def test_dict_to_cli_args_converts_supported_values() -> None:
    assert dict_to_cli_args(
        {
            "csv_path": "labels.csv",
            "dry_run": True,
            "skip": False,
            "none_value": None,
            "tag": ["a", "b"],
        }
    ) == [
        "--csv-path",
        "labels.csv",
        "--dry-run",
        "--tag",
        "a",
        "--tag",
        "b",
    ]


def test_coerce_cli_args_handles_strings_dicts_and_lists() -> None:
    assert coerce_cli_args("--epochs 2 --augment") == ["--epochs", "2", "--augment"]
    assert coerce_cli_args({"epochs": 2}) == ["--epochs", "2"]
    assert coerce_cli_args(["--epochs", 2]) == ["--epochs", "2"]


def test_read_config_and_load_config_args(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        """
        {
          "global": {"log_level": "DEBUG"},
          "train-density": {"epochs": 2},
          "train_density": "--batch-size 4"
        }
        """,
        encoding="utf-8",
    )

    assert read_config(config)["global"]["log_level"] == "DEBUG"
    args = load_config_args(config, "train-density")

    assert args[:2] == ["--log-level", "DEBUG"]
    assert ["--epochs", "2"] == args[2:4] or ["--epochs", "2"] == args[4:6]
    assert ["--batch-size", "4"] == args[2:4] or ["--batch-size", "4"] == args[4:6]


def test_load_config_args_returns_empty_for_missing_default(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert load_config_args(missing, "train-density") == []
