#!/usr/bin/env python3
#
# compare_models.py
# mammography-pipelines
#
# CLI command for comparing multiple trained models across metrics, datasets, and configurations.
# DISCLAIMER: Educational project only - NOT for clinical or medical diagnostic purposes.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""
Compare Models CLI — Compare multiple trained models with side-by-side metrics and statistical tests.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.

Usage:
  # Compare two model runs
  py -m mammography.commands.compare_models --run outputs/run_001 --run outputs/run_002 --outdir comparison_report

  # Compare with statistical tests
  py -m mammography.commands.compare_models --run outputs/run_001 --run outputs/run_002 --statistical-tests

  # Export comparison tables
  py -m mammography.commands.compare_models --run outputs/run_001 --run outputs/run_002 --export csv,xlsx,md

  # Generate full HTML report with visualizations
  py -m mammography.commands.compare_models --run outputs/run_001 --run outputs/run_002 --html-report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from mammography.vis.model_comparison import (
    ModelComparisonEngine,
    ModelMetrics,
    create_metrics_comparison_table,
    create_metrics_comparison_chart,
    create_per_class_comparison,
    create_confusion_matrix_comparison,
    export_comparison_table,
    format_statistical_summary,
)

LOGGER = logging.getLogger("mammography")


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure the module logger with a basic timestamped formatter and the specified level.
    
    Parameters:
        level (str): Logging level name (e.g., "DEBUG", "INFO", "WARNING"). Case-insensitive.
    
    Returns:
        logging.Logger: The configured logger for this module.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def discover_run_directories(paths: List[Path]) -> List[Path]:
    """
    Identify and return model run directories that contain run summaries.
    
    Searches each provided path and:
    - treats a path as a run directory if it contains `summary.json` or `cv_summary.json`;
    - if a path is a parent directory, includes any `results_*` subdirectories that contain a summary file.
    Non-existent paths are skipped (and logged).
    
    Parameters:
        paths (List[Path]): Paths to inspect; each may be a run directory or a parent containing `results_*` subdirectories.
    
    Returns:
        List[Path]: Discovered run directories that contain `summary.json` or `cv_summary.json`.
    """
    discovered: List[Path] = []

    for path in paths:
        if not path.exists():
            LOGGER.warning("Path does not exist: %s", path)
            continue

        # Check if it's a direct results directory
        if (path / "summary.json").exists() or (path / "cv_summary.json").exists():
            discovered.append(path)
            continue

        # Check if it contains a results_* subdirectory
        results_dirs = list(path.glob("results_*"))
        for results_dir in results_dirs:
            if (results_dir / "summary.json").exists() or (results_dir / "cv_summary.json").exists():
                discovered.append(results_dir)

    return discovered


def generate_html_report(
    engine: ModelComparisonEngine,
    metrics_list: List[ModelMetrics],
    outdir: Path,
    include_statistical_tests: bool = False,
) -> Path:
    """
    Generate an interactive HTML report that embeds comparison visualizations for the provided ModelComparisonEngine.

    Builds a single HTML file in outdir containing a summary of compared models and any available visualizations (side-by-side metrics table and chart, per-class F1 comparison, confusion matrices). If a statistical_tests.csv file exists in outdir and include_statistical_tests is True, includes a McNemar test results table.

    Parameters:
        engine (ModelComparisonEngine): Engine with loaded model comparison data and model_names.
        metrics_list (List[ModelMetrics]): List of loaded model metrics objects.
        outdir (Path): Directory where the report and related assets will be written.
        include_statistical_tests (bool): If True, include the statistical tests section when a statistical_tests.csv file is present in outdir.

    Returns:
        Path: Path to the written HTML report file (outdir / "model_comparison_report.html").
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        LOGGER.error("Plotly is required for HTML reports. Install with: pip install plotly")
        raise

    LOGGER.info("Generating HTML report...")

    # Create visualizations
    figures: Dict[str, Any] = {}

    # Side-by-side metrics comparison
    try:
        figures["metrics_table"] = create_metrics_comparison_table(
            metrics_list,
            title="Model Comparison - Key Metrics",
        )
        figures["metrics_chart"] = create_metrics_comparison_chart(
            metrics_list,
            title="Model Performance Comparison",
            chart_type="grouped",
        )
    except Exception as e:
        LOGGER.warning("Failed to create metrics comparison: %s", e)

    # Per-class performance breakdown
    try:
        figures["per_class"] = create_per_class_comparison(
            metrics_list,
            metric="f1-score",
            title="Per-Class F1-Score Comparison",
        )
    except Exception as e:
        LOGGER.warning("Failed to create per-class comparison: %s", e)

    # Confusion matrices comparison
    try:
        figures["confusion_matrices"] = create_confusion_matrix_comparison(
            metrics_list,
            normalize=True,
            title="Confusion Matrices Comparison (Normalized)",
        )
    except Exception as e:
        LOGGER.warning("Failed to create confusion matrix comparison: %s", e)

    # Build HTML content
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Model Comparison Report</title>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { color: #333; }",
        "h2 { color: #555; margin-top: 30px; }",
        ".section { margin-bottom: 40px; }",
        ".disclaimer { background: #fff3cd; border: 1px solid #ffc107; padding: 15px; margin: 20px 0; }",
        ".disclaimer strong { color: #856404; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Model Comparison Report</h1>",
        "<div class='disclaimer'>",
        "<strong>⚠️ DISCLAIMER:</strong> This is an EDUCATIONAL RESEARCH project. ",
        "It must NOT be used for clinical or medical diagnostic purposes.",
        "</div>",
    ]

    # Summary section
    html_parts.append("<div class='section'>")
    html_parts.append("<h2>Summary</h2>")
    html_parts.append(f"<p><strong>Number of models compared:</strong> {len(engine.model_names)}</p>")
    html_parts.append("<ul>")
    for name in engine.model_names:
        html_parts.append(f"<li>{name}</li>")
    html_parts.append("</ul>")
    html_parts.append("</div>")

    # Add figures
    for section_name, fig in figures.items():
        if fig is not None:
            html_parts.append(f"<div class='section' id='{section_name}'>")
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
            html_parts.append("</div>")

    # Statistical tests section (if available)
    test_csv_path = outdir / "statistical_tests.csv"
    if test_csv_path.exists() and include_statistical_tests:
        try:
            test_df = pd.read_csv(test_csv_path)

            html_parts.append("<h2>Statistical Significance Tests</h2>")
            html_parts.append("<p>Pairwise McNemar's test comparing model performance. "
                           "A p-value < 0.05 indicates statistically significant difference.</p>")

            # Create Plotly table for statistical tests
            test_table = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=[
                                "Model A",
                                "Model B",
                                "χ² Statistic",
                                "p-value",
                                "α",
                                "Significant",
                            ],
                            fill_color="lightgray",
                            align="left",
                            font=dict(size=12, color="black"),
                        ),
                        cells=dict(
                            values=[
                                test_df["model_a"],
                                test_df["model_b"],
                                test_df["statistic"].map("{:.4f}".format),
                                test_df["p_value"].map("{:.4f}".format),
                                test_df["alpha"].map("{:.4f}".format),
                                test_df["significant"].map(
                                    lambda x: "✓ Yes" if x else "✗ No"
                                ),
                            ],
                            fill_color=[
                                "white",
                                "white",
                                "white",
                                "white",
                                "white",
                                [
                                    "lightgreen" if sig else "lightpink"
                                    for sig in test_df["significant"]
                                ],
                            ],
                            align="left",
                            font=dict(size=11),
                        ),
                    )
                ]
            )
            test_table.update_layout(
                title="McNemar's Test Results",
                height=300 + len(test_df) * 30,
            )
            html_parts.append(test_table.to_html(include_plotlyjs=False))

            LOGGER.debug("Added statistical tests section to HTML report")

        except Exception as e:
            LOGGER.warning("Failed to add statistical tests to HTML report: %s", e)

    html_parts.extend([
        "</body>",
        "</html>",
    ])

    # Write HTML file
    report_path = outdir / "model_comparison_report.html"
    report_path.write_text("\n".join(html_parts), encoding="utf-8")

    LOGGER.info("HTML report saved to: %s", report_path)
    return report_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the compare-models CLI.
    
    Parameters:
        argv (Optional[Sequence[str]]): Sequence of argument strings to parse. If None, uses the process's command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes such as `runs`, `outdir`, `html_report`, `no_visualizations`, `export`, `statistical_tests`, `confidence_level`, `metrics`, `rank_by`, and `log_level`.
    """
    parser = argparse.ArgumentParser(
        description="Compare multiple trained models across metrics, datasets, and configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two model runs
  py -m mammography.commands.compare_models --run outputs/run_001 --run outputs/run_002

  # Compare with custom output directory
  py -m mammography.commands.compare_models --run outputs/run_001 --run outputs/run_002 --outdir comparison_results

  # Generate HTML report with visualizations
  py -m mammography.commands.compare_models --run outputs/run_001 --run outputs/run_002 --html-report

  # Export comparison tables to multiple formats
  py -m mammography.commands.compare_models --run outputs/run_001 --run outputs/run_002 --export csv,xlsx,md

  # Include statistical significance tests
  py -m mammography.commands.compare_models --run outputs/run_001 --run outputs/run_002 --statistical-tests
        """,
    )

    # Required arguments
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        type=Path,
        required=True,
        help="Path to model run directory (results_* or directory containing results_*). Can be specified multiple times.",
    )

    # Output options
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/model_comparison"),
        help="Output directory for comparison results (default: outputs/model_comparison)",
    )

    # Visualization options
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML report with interactive visualizations",
    )

    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip visualization generation (only export tables)",
    )

    # Export options
    parser.add_argument(
        "--export",
        type=str,
        default="csv",
        help="Export formats (comma-separated): csv,xlsx,json,md,tex (default: csv)",
    )

    # Statistical testing
    parser.add_argument(
        "--statistical-tests",
        action="store_true",
        help="Perform statistical significance tests (requires prediction data)",
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for statistical tests (default: 0.95)",
    )

    # Comparison options
    parser.add_argument(
        "--metrics",
        type=str,
        default="accuracy,kappa,macro_f1,auc",
        help="Metrics to compare (comma-separated, default: accuracy,kappa,macro_f1,auc)",
    )

    parser.add_argument(
        "--rank-by",
        type=str,
        default="accuracy",
        help="Metric to use for ranking models (default: accuracy)",
    )

    # General options
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Logging level (default: info)",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the compare-models command-line workflow to compare multiple trained model runs and produce tables, visualizations, reports, and optional statistical tests.
    
    This function parses CLI arguments, discovers valid model run directories, loads metrics into a ModelComparisonEngine, generates side-by-side comparison tables (and exports them), ranks models by a chosen metric, optionally creates visualizations and a consolidated HTML report, and optionally runs pairwise McNemar statistical tests when prediction files are available. Progress and errors are logged; outputs are written to the specified output directory.
    
    Parameters:
        argv (Optional[Sequence[str]]): Command-line arguments to parse (e.g., sys.argv[1:]). If None, the program's default argument source is used.
    
    Returns:
        int: Exit code — 0 on success, 1 on failure.
    """
    args = parse_args(argv)
    setup_logging(args.log_level)

    # Discover run directories
    LOGGER.info("Discovering model run directories...")
    run_dirs = discover_run_directories(args.runs)

    if not run_dirs:
        LOGGER.error("No valid run directories found. Each run must contain summary.json or cv_summary.json")
        return 1

    if len(run_dirs) < 2:
        LOGGER.error("At least 2 model runs are required for comparison. Found: %d", len(run_dirs))
        return 1

    LOGGER.info("Found %d valid run directories:", len(run_dirs))
    for run_dir in run_dirs:
        LOGGER.info("  - %s", run_dir)

    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Initialize comparison engine
    LOGGER.info("Loading model metrics...")
    try:
        engine = ModelComparisonEngine(run_dirs)
    except Exception as e:
        LOGGER.error("Failed to load model metrics: %s", e)
        return 1

    LOGGER.info("Loaded metrics for %d models:", len(engine.model_names))
    for name in engine.model_names:
        LOGGER.info("  - %s", name)

    # Load per-model metrics
    try:
        metrics_list = engine.load_all_metrics()
    except Exception as e:
        LOGGER.error("Failed to load per-model metrics: %s", e)
        return 1

    # Generate side-by-side comparison
    LOGGER.info("Generating comparison tables...")
    metrics_to_compare = [m.strip() for m in args.metrics.split(",")]

    try:
        comparison_df = engine.get_side_by_side_comparison(metrics_list, metrics=metrics_to_compare)

        # Save comparison table to JSON
        comparison_json = args.outdir / "comparison_metrics.json"
        comparison_df.to_json(comparison_json, orient="records", indent=2)
        LOGGER.info("Comparison metrics saved to: %s", comparison_json)

        # Export to requested formats
        export_formats = [fmt.strip() for fmt in args.export.split(",")]
        exported_paths = export_comparison_table(
            comparison_df,
            args.outdir / "comparison_table",
            formats=export_formats,
        )
        for path in exported_paths:
            LOGGER.info("Exported comparison table to: %s", path)

    except Exception as e:
        LOGGER.error("Failed to generate comparison tables: %s", e)
        return 1

    # Rank models
    LOGGER.info("Ranking models by %s...", args.rank_by)
    try:
        ranked_df = engine.rank_models_by_metric(metrics_list, metric=args.rank_by)
        LOGGER.info("Model rankings:")
        for idx, row in ranked_df.iterrows():
            LOGGER.info("  %d. %s: %.4f", idx + 1, row["model_name"], row[args.rank_by])
    except Exception as e:
        LOGGER.warning("Failed to rank models: %s", e)

    # Generate visualizations
    if not args.no_visualizations:
        LOGGER.info("Generating visualizations...")

        # Save individual visualizations
        try:
            # Metrics comparison chart
            fig_chart = create_metrics_comparison_chart(
                metrics_list,
                title="Model Performance Comparison",
                chart_type="grouped",
            )
            chart_path = args.outdir / "metrics_comparison_chart.html"
            fig_chart.write_html(str(chart_path))
            LOGGER.info("Metrics chart saved to: %s", chart_path)

            # Per-class comparison
            fig_per_class = create_per_class_comparison(
                metrics_list,
                metric="f1-score",
                title="Per-Class F1-Score Comparison",
            )
            per_class_path = args.outdir / "per_class_comparison.html"
            fig_per_class.write_html(str(per_class_path))
            LOGGER.info("Per-class comparison saved to: %s", per_class_path)

        except Exception as e:
            LOGGER.warning("Some visualizations failed: %s", e)

    # Generate HTML report
    if args.html_report:
        try:
            report_path = generate_html_report(
                engine,
                metrics_list,
                args.outdir,
                include_statistical_tests=args.statistical_tests,
            )
            LOGGER.info("Full HTML report generated: %s", report_path)
        except Exception as e:
            LOGGER.error("Failed to generate HTML report: %s", e)
            return 1

    # Statistical tests
    if args.statistical_tests:
        LOGGER.info("Running statistical significance tests...")

        try:
            # Load predictions for all models
            predictions = {}
            y_true = None

            for metrics in metrics_list:
                try:
                    true_labels, preds = engine.load_predictions(metrics.run_path)
                    predictions[metrics.model_name] = preds

                    # Verify all models use same ground truth
                    if y_true is None:
                        y_true = true_labels
                    elif not np.array_equal(y_true, true_labels):
                        LOGGER.error(
                            "Ground truth mismatch between models. "
                            "All models must be evaluated on same validation set."
                        )
                        return 1

                    LOGGER.info(
                        "Loaded predictions for %s: %d samples",
                        metrics.model_name,
                        len(preds),
                    )

                except FileNotFoundError as e:
                    LOGGER.error(
                        "Cannot run statistical tests: %s\n"
                        "Update training pipeline to save val_predictions.csv",
                        e,
                    )
                    return 1

            # Pairwise McNemar tests
            from mammography.vis.model_comparison import compute_mcnemar_test

            LOGGER.info("Running pairwise McNemar tests...")
            test_results = []

            for i, model_a in enumerate(metrics_list):
                for model_b in metrics_list[i + 1 :]:
                    name_a = model_a.model_name
                    name_b = model_b.model_name

                    stat, p_val = compute_mcnemar_test(
                        y_true,
                        predictions[name_a],
                        predictions[name_b],
                        continuity_correction=True,
                    )

                    # Determine significance using user-specified confidence level
                    alpha = 1.0 - args.confidence_level
                    significant = p_val < alpha

                    test_results.append(
                        {
                            "model_a": name_a,
                            "model_b": name_b,
                            "statistic": stat,
                            "p_value": p_val,
                            "alpha": alpha,
                            "significant": significant,
                        }
                    )

                    # Log result
                    sig_str = "SIGNIFICANT" if significant else "not significant"
                    LOGGER.info(
                        "%s vs %s: χ²=%.4f, p=%.4f (%s)",
                        name_a,
                        name_b,
                        stat,
                        p_val,
                        sig_str,
                    )

            # Save results to CSV
            test_df = pd.DataFrame(test_results)
            test_csv_path = args.outdir / "statistical_tests.csv"
            test_df.to_csv(test_csv_path, index=False)
            LOGGER.info("Statistical test results saved to: %s", test_csv_path)

            # Export to LaTeX for papers
            test_tex_path = args.outdir / "statistical_tests.tex"
            test_df.to_latex(
                test_tex_path,
                index=False,
                float_format="%.4f",
                caption="Pairwise McNemar's Test Results for Model Comparison",
                label="tab:mcnemar_tests",
            )
            LOGGER.info("LaTeX table saved to: %s", test_tex_path)

        except Exception as e:
            LOGGER.error("Statistical test workflow failed: %s", e, exc_info=True)
            return 1

    LOGGER.info("Model comparison complete! Results saved to: %s", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())