from __future__ import annotations

import argparse
from pathlib import Path

from mammography.utils.cli_args import add_tracking_args, serialize_tracking_args


def test_add_tracking_args_parses_shared_options() -> None:
    parser = argparse.ArgumentParser()
    add_tracking_args(parser)

    args = parser.parse_args(
        [
            "--run-name",
            "run-1",
            "--tracking-uri",
            "file:mlruns",
            "--experiment",
            "density",
            "--registry-csv",
            "custom.csv",
            "--registry-md",
            "custom.md",
            "--no-mlflow",
            "--no-registry",
        ]
    )

    assert args.run_name == "run-1"
    assert args.tracking_uri == "file:mlruns"
    assert args.experiment == "density"
    assert args.registry_csv == Path("custom.csv")
    assert args.registry_md == Path("custom.md")
    assert args.no_mlflow is True
    assert args.no_registry is True


def test_serialize_tracking_args_skips_empty_values() -> None:
    registry_csv = Path("results/registry.csv")
    args = argparse.Namespace(
        run_name="run-1",
        tracking_uri="",
        experiment="density",
        registry_csv=registry_csv,
        registry_md=None,
        no_mlflow=True,
        no_registry=False,
    )

    assert serialize_tracking_args(args) == [
        "--run-name",
        "run-1",
        "--experiment",
        "density",
        "--registry-csv",
        str(registry_csv),
        "--no-mlflow",
    ]
