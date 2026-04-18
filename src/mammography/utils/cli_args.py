"""Shared CLI argument helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def add_tracking_args(parser: argparse.ArgumentParser) -> None:
    """Add MLflow/local registry tracking arguments to a subparser."""
    parser.add_argument("--run-name", default="", help="Nome do run no MLflow")
    parser.add_argument("--tracking-uri", default="", help="Tracking URI para MLflow")
    parser.add_argument("--experiment", default="", help="Experimento MLflow")
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("results/registry.csv"),
        help="Arquivo CSV do registry local",
    )
    parser.add_argument(
        "--registry-md",
        type=Path,
        default=Path("results/registry.md"),
        help="Arquivo Markdown do registry local",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Nao registrar no MLflow",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Nao atualizar registry local",
    )


def serialize_tracking_args(args: Any) -> list[str]:
    """Serialize parsed tracking arguments back into CLI tokens."""
    tokens: list[str] = []
    if getattr(args, "run_name", ""):
        tokens.extend(["--run-name", args.run_name])
    if getattr(args, "tracking_uri", ""):
        tokens.extend(["--tracking-uri", args.tracking_uri])
    if getattr(args, "experiment", ""):
        tokens.extend(["--experiment", args.experiment])
    if getattr(args, "registry_csv", None):
        tokens.extend(["--registry-csv", str(args.registry_csv)])
    if getattr(args, "registry_md", None):
        tokens.extend(["--registry-md", str(args.registry_md)])
    if getattr(args, "no_mlflow", False):
        tokens.append("--no-mlflow")
    if getattr(args, "no_registry", False):
        tokens.append("--no-registry")
    return tokens
