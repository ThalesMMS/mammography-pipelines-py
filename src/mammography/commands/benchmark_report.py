#!/usr/bin/env python3
"""CLI entrypoint for the official rerun benchmark aggregation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from mammography.tools.benchmark_report import (
    BenchmarkValidationError,
    generate_benchmark_report,
)

LOGGER = logging.getLogger("benchmark_report")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse benchmark-report CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the official rerun namespace and generate the master table, "
            "technical report, and article-ready LaTeX table."
        )
    )
    parser.add_argument(
        "--namespace",
        type=Path,
        default=Path("outputs/rerun_2026q1"),
        help="Official rerun namespace (default: outputs/rerun_2026q1).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/rerun_2026q1_master"),
        help=(
            "Output prefix for the master table files; .csv/.md/.json/.tex are "
            "added automatically (default: results/rerun_2026q1_master)."
        ),
    )
    parser.add_argument(
        "--docs-report",
        type=Path,
        default=Path("docs/reports/rerun_2026q1_technical_report.md"),
        help="Consolidated technical report path (default: docs/reports/rerun_2026q1_technical_report.md).",
    )
    parser.add_argument(
        "--article-table",
        type=Path,
        default=Path("Article/sections/rerun_2026q1_benchmark_table.tex"),
        help="Article-ready LaTeX table path (default: Article/sections/rerun_2026q1_benchmark_table.tex).",
    )
    parser.add_argument(
        "--exports-search-root",
        type=Path,
        default=Path("outputs"),
        help="Root used to discover eval_export manifests (default: outputs).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark-report workflow and return an exit code."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
    )

    try:
        generate_benchmark_report(
            namespace=args.namespace,
            output_prefix=args.output_prefix,
            docs_report_path=args.docs_report,
            article_table_path=args.article_table,
            exports_search_root=args.exports_search_root,
        )
    except BenchmarkValidationError as exc:
        LOGGER.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - CLI safety net
        LOGGER.error("benchmark-report falhou: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
