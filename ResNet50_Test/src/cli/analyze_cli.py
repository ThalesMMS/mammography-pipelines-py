"""
Analysis CLI for mammography clustering results.

This module provides command-line interface for analyzing clustering
results including visualization generation, sanity checks, and clinical
analysis for the breast density exploration pipeline.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- CLI provides user-friendly interface for analysis operations
- Visualization generation enables qualitative validation
- Sanity checks ensure clinical relevance and catch obvious failures
- Clinical analysis provides insights into clustering patterns

Author: Research Team
Version: 1.0.0
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import yaml

from ..clustering.clustering_result import ClusteringResult
from ..eval.clustering_evaluator import ClusteringEvaluator
from ..io_dicom.mammography_image import MammographyImage
from ..models.embeddings.embedding_vector import EmbeddingVector
from ..viz.cluster_visualizer import ClusterVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Typer app
app = typer.Typer(
    name="mmg-analyze",
    help="Analyze clustering results with visualizations and sanity checks",
    add_completion=False,
)

# Research disclaimer
RESEARCH_DISCLAIMER = """
⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.
"""


@app.command()
def visualize(
    clustering_result_file: Path = typer.Option(
        ..., "--result-file", "-r", help="Path to clustering result file"
    ),
    embedding_dir: Path = typer.Option(
        ..., "--embedding-dir", "-e", help="Directory containing embedding vectors"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to save visualizations"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to visualization configuration file"
    ),
    mammography_dir: Optional[Path] = typer.Option(
        None,
        "--mammography-dir",
        "-m",
        help="Directory containing mammography images for montages",
    ),
    visualizations: str = typer.Option(
        "umap_2d,cluster_montage,metrics_plot",
        "--visualizations",
        "-v",
        help="Comma-separated list of visualizations to generate",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Generate visualizations for clustering results.

    This command creates comprehensive visualizations including UMAP plots,
    cluster montages, and metrics plots for clustering analysis.

    Educational Note: Visualizations enable qualitative validation
    of clustering results and help identify patterns.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print research disclaimer
    typer.echo(RESEARCH_DISCLAIMER)
    typer.echo()

    try:
        # Load configuration
        config = _load_config(config_file, visualizations)

        # Validate input files
        if not clustering_result_file.exists():
            typer.echo(
                f"Error: Clustering result file {clustering_result_file} does not exist",
                err=True,
            )
            raise typer.Exit(1)

        if not embedding_dir.exists():
            typer.echo(
                f"Error: Embedding directory {embedding_dir} does not exist", err=True
            )
            raise typer.Exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load clustering result
        clustering_result = _load_clustering_result(clustering_result_file)
        if clustering_result is None:
            typer.echo("Failed to load clustering result!", err=True)
            raise typer.Exit(1)

        # Load embeddings
        embedding_vectors = _load_embeddings(embedding_dir)
        if not embedding_vectors:
            typer.echo("No embedding vectors found!", err=True)
            raise typer.Exit(1)

        # Load mammography images if available
        mammography_images = None
        if mammography_dir and mammography_dir.exists():
            mammography_images = _load_mammography_images(mammography_dir)

        typer.echo(
            f"Generating visualizations for {clustering_result.algorithm} clustering"
        )
        typer.echo(f"Loaded {len(embedding_vectors)} embedding vectors")
        if mammography_images:
            typer.echo(f"Loaded {len(mammography_images)} mammography images")

        # Initialize visualizer
        visualizer = ClusterVisualizer(config)

        # Generate visualizations
        visualization_results = visualizer.create_visualizations(
            clustering_result, embedding_vectors, mammography_images, output_dir
        )

        # Print results
        _print_visualization_results(visualization_results)

        typer.echo("Visualization generation completed successfully!")
        typer.echo(f"Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error in visualization generation: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


@app.command()
def evaluate(
    clustering_result_file: Path = typer.Option(
        ..., "--result-file", "-r", help="Path to clustering result file"
    ),
    embedding_dir: Path = typer.Option(
        ..., "--embedding-dir", "-e", help="Directory containing embedding vectors"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to save evaluation results"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to evaluation configuration file"
    ),
    mammography_dir: Optional[Path] = typer.Option(
        None,
        "--mammography-dir",
        "-m",
        help="Directory containing mammography images for sanity checks",
    ),
    sanity_checks: str = typer.Option(
        "cluster_size_analysis,embedding_statistics,projection_distribution",
        "--sanity-checks",
        "-s",
        help="Comma-separated list of sanity checks to perform",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Evaluate clustering results with comprehensive analysis.

    This command performs comprehensive evaluation including quality metrics,
    sanity checks, and visual prototype selection.

    Educational Note: Evaluation ensures clustering results are clinically
    relevant and catch obvious failures or biases.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print research disclaimer
    typer.echo(RESEARCH_DISCLAIMER)
    typer.echo()

    try:
        # Load configuration
        config = _load_config(config_file, sanity_checks)

        # Validate input files
        if not clustering_result_file.exists():
            typer.echo(
                f"Error: Clustering result file {clustering_result_file} does not exist",
                err=True,
            )
            raise typer.Exit(1)

        if not embedding_dir.exists():
            typer.echo(
                f"Error: Embedding directory {embedding_dir} does not exist", err=True
            )
            raise typer.Exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load clustering result
        clustering_result = _load_clustering_result(clustering_result_file)
        if clustering_result is None:
            typer.echo("Failed to load clustering result!", err=True)
            raise typer.Exit(1)

        # Load embeddings
        embedding_vectors = _load_embeddings(embedding_dir)
        if not embedding_vectors:
            typer.echo("No embedding vectors found!", err=True)
            raise typer.Exit(1)

        # Load mammography images if available
        mammography_images = None
        if mammography_dir and mammography_dir.exists():
            mammography_images = _load_mammography_images(mammography_dir)

        typer.echo(f"Evaluating {clustering_result.algorithm} clustering results")
        typer.echo(f"Loaded {len(embedding_vectors)} embedding vectors")
        if mammography_images:
            typer.echo(f"Loaded {len(mammography_images)} mammography images")

        # Initialize evaluator
        evaluator = ClusteringEvaluator(config)

        # Perform evaluation
        evaluation_results = evaluator.evaluate_clustering(
            clustering_result, embedding_vectors, mammography_images
        )

        # Save evaluation results
        _save_evaluation_results(evaluation_results, output_dir)

        # Print results
        _print_evaluation_results(evaluation_results)

        typer.echo("Clustering evaluation completed successfully!")
        typer.echo(f"Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error in clustering evaluation: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


@app.command()
def report(
    clustering_result_file: Path = typer.Option(
        ..., "--result-file", "-r", help="Path to clustering result file"
    ),
    embedding_dir: Path = typer.Option(
        ..., "--embedding-dir", "-e", help="Directory containing embedding vectors"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to save analysis report"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to analysis configuration file"
    ),
    mammography_dir: Optional[Path] = typer.Option(
        None, "--mammography-dir", "-m", help="Directory containing mammography images"
    ),
    report_format: str = typer.Option(
        "markdown", "--format", "-f", help="Report format (markdown, html, pdf)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Generate comprehensive analysis report.

    This command creates a comprehensive analysis report including
    visualizations, evaluation results, and clinical insights.

    Educational Note: Analysis reports provide comprehensive insights
    into clustering results and their clinical relevance.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print research disclaimer
    typer.echo(RESEARCH_DISCLAIMER)
    typer.echo()

    try:
        # Load configuration
        config = _load_config(config_file)

        # Validate input files
        if not clustering_result_file.exists():
            typer.echo(
                f"Error: Clustering result file {clustering_result_file} does not exist",
                err=True,
            )
            raise typer.Exit(1)

        if not embedding_dir.exists():
            typer.echo(
                f"Error: Embedding directory {embedding_dir} does not exist", err=True
            )
            raise typer.Exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load clustering result
        clustering_result = _load_clustering_result(clustering_result_file)
        if clustering_result is None:
            typer.echo("Failed to load clustering result!", err=True)
            raise typer.Exit(1)

        # Load embeddings
        embedding_vectors = _load_embeddings(embedding_dir)
        if not embedding_vectors:
            typer.echo("No embedding vectors found!", err=True)
            raise typer.Exit(1)

        # Load mammography images if available
        mammography_images = None
        if mammography_dir and mammography_dir.exists():
            mammography_images = _load_mammography_images(mammography_dir)

        typer.echo("Generating comprehensive analysis report")
        typer.echo(f"Algorithm: {clustering_result.algorithm}")
        typer.echo(f"Loaded {len(embedding_vectors)} embedding vectors")
        if mammography_images:
            typer.echo(f"Loaded {len(mammography_images)} mammography images")

        # Generate report
        report_results = _generate_analysis_report(
            clustering_result,
            embedding_vectors,
            mammography_images,
            output_dir,
            report_format,
            config,
        )

        # Print results
        _print_report_results(report_results)

        typer.echo("Analysis report generation completed successfully!")
        typer.echo(f"Report saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error in report generation: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


@app.command()
def config_template(
    output_file: Path = typer.Option(
        "analysis_config.yaml",
        "--output",
        "-o",
        help="Output file for configuration template",
    )
):
    """
    Generate an analysis configuration template.

    This command creates a template configuration file with
    default values for analysis parameters.

    Educational Note: Configuration templates help users understand
    available parameters and their effects on analysis.
    """
    # Print research disclaimer
    typer.echo(RESEARCH_DISCLAIMER)
    typer.echo()

    try:
        # Generate template configuration
        template_config = _generate_config_template()

        # Save template
        with open(output_file, "w") as f:
            yaml.dump(template_config, f, default_flow_style=False, indent=2)

        typer.echo(f"Configuration template saved to {output_file}")
        typer.echo("Edit the configuration file to customize analysis parameters")

    except Exception as e:
        logger.error(f"Error generating config template: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


def _load_config(
    config_file: Optional[Path], visualizations: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load analysis configuration.

    Educational Note: Configuration loading enables reproducible
    experiments and parameter management.

    Args:
        config_file: Optional path to configuration file
        visualizations: Optional comma-separated list of visualizations

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if config_file and config_file.exists():
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_file}")
    else:
        # Use default configuration
        config = _get_default_config()
        logger.info("Using default configuration")

    # Override visualizations if specified
    if visualizations:
        config["visualizations"] = [viz.strip() for viz in visualizations.split(",")]

    return config


def _get_default_config() -> Dict[str, Any]:
    """
    Get default analysis configuration.

    Educational Note: Default configurations provide sensible
    starting points for analysis parameters.

    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        "visualizations": ["umap_2d", "cluster_montage", "metrics_plot"],
        "sanity_checks": ["cluster_size_analysis", "embedding_statistics"],
        "umap_params": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 2,
            "metric": "euclidean",
            "random_state": 42,
        },
        "plot_params": {
            "figsize": (10, 8),
            "dpi": 300,
            "style": "whitegrid",
            "palette": "husl",
        },
        "montage_params": {
            "n_samples_per_cluster": 4,
            "image_size": (224, 224),
            "grid_size": (2, 2),
        },
        "seed": 42,
    }


def _load_clustering_result(result_file: Path) -> Optional[ClusteringResult]:
    """
    Load clustering result from file.

    Educational Note: Result loading enables analysis
    of previously computed clustering results.

    Args:
        result_file: Path to clustering result file

    Returns:
        ClusteringResult: Loaded clustering result, None if failed
    """
    try:
        clustering_result = torch.load(result_file, map_location="cpu")
        return clustering_result
    except Exception as e:
        logger.error(f"Error loading clustering result from {result_file}: {e!s}")
        return None


def _load_embeddings(embedding_dir: Path) -> List[EmbeddingVector]:
    """
    Load embedding vectors from directory.

    Educational Note: Efficient embedding loading enables
    fast processing for analysis operations.

    Args:
        embedding_dir: Embedding directory path

    Returns:
        List[EmbeddingVector]: List of loaded embedding vectors
    """
    embedding_vectors = []

    try:
        # Find embedding files
        embedding_files = list(embedding_dir.rglob("*.embedding.pt"))

        for file_path in embedding_files:
            try:
                # Load embedding data
                embedding_data = torch.load(file_path, map_location="cpu")

                # Load metadata
                metadata_path = file_path.with_suffix(".metadata.yaml")
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = yaml.safe_load(f)
                else:
                    metadata = {}

                # Create EmbeddingVector instance
                embedding_vector = EmbeddingVector(
                    image_id=metadata.get("image_id", file_path.stem),
                    patient_id=metadata.get("patient_id", "unknown"),
                    embedding=embedding_data,
                    extraction_config=metadata.get("extraction_config", {}),
                    timestamp=datetime.fromisoformat(
                        metadata.get("timestamp", datetime.now().isoformat())
                    ),
                )

                embedding_vectors.append(embedding_vector)

            except Exception as e:
                logger.error(f"Error loading embedding from {file_path}: {e!s}")
                continue

        logger.info(f"Loaded {len(embedding_vectors)} embedding vectors")

    except Exception as e:
        logger.error(f"Error loading embeddings: {e!s}")

    return embedding_vectors


def _load_mammography_images(mammography_dir: Path) -> List[MammographyImage]:
    """
    Load mammography images from directory.

    Educational Note: Mammography image loading enables
    sanity checks and clinical analysis.

    Args:
        mammography_dir: Mammography directory path

    Returns:
        List[MammographyImage]: List of loaded mammography images
    """
    mammography_images = []

    try:
        # Find DICOM files
        dicom_files = list(mammography_dir.rglob("*.dcm"))

        for file_path in dicom_files:
            try:
                # Load DICOM file
                from ..io_dicom.dicom_reader import DicomReader

                dicom_reader = DicomReader({})
                mammography_image = dicom_reader.read_dicom_file(file_path)

                if mammography_image is not None:
                    mammography_images.append(mammography_image)

            except Exception as e:
                logger.error(f"Error loading mammography image from {file_path}: {e!s}")
                continue

        logger.info(f"Loaded {len(mammography_images)} mammography images")

    except Exception as e:
        logger.error(f"Error loading mammography images: {e!s}")

    return mammography_images


def _generate_analysis_report(
    clustering_result: ClusteringResult,
    embedding_vectors: List[EmbeddingVector],
    mammography_images: Optional[List[MammographyImage]],
    output_dir: Path,
    report_format: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate comprehensive analysis report.

    Educational Note: Analysis reports provide comprehensive insights
    into clustering results and their clinical relevance.

    Args:
        clustering_result: ClusteringResult instance
        embedding_vectors: List of EmbeddingVector instances
        mammography_images: Optional list of MammographyImage instances
        output_dir: Output directory path
        report_format: Report format (markdown, html, pdf)
        config: Configuration dictionary

    Returns:
        Dict[str, Any]: Report generation results
    """
    report_results = {
        "clustering_result": clustering_result,
        "report_timestamp": datetime.now().isoformat(),
        "output_files": {},
        "sections": [],
    }

    try:
        # Generate visualizations
        visualizer = ClusterVisualizer(config)
        visualization_results = visualizer.create_visualizations(
            clustering_result, embedding_vectors, mammography_images, output_dir
        )
        report_results["output_files"].update(
            visualization_results.get("output_files", {})
        )

        # Perform evaluation
        evaluator = ClusteringEvaluator(config)
        evaluation_results = evaluator.evaluate_clustering(
            clustering_result, embedding_vectors, mammography_images
        )

        # Generate report content
        report_content = _generate_report_content(
            clustering_result, evaluation_results, visualization_results, config
        )

        # Save report
        report_path = _save_report(report_content, output_dir, report_format)
        if report_path:
            report_results["output_files"]["report"] = str(report_path)

        report_results["sections"] = list(report_content.keys())

    except Exception as e:
        logger.error(f"Error generating analysis report: {e!s}")
        report_results["error"] = str(e)

    return report_results


def _generate_report_content(
    clustering_result: ClusteringResult,
    evaluation_results: Dict[str, Any],
    visualization_results: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, str]:
    """
    Generate report content sections.

    Educational Note: Report content provides comprehensive
    analysis of clustering results and their implications.

    Args:
        clustering_result: ClusteringResult instance
        evaluation_results: Evaluation results dictionary
        visualization_results: Visualization results dictionary
        config: Configuration dictionary

    Returns:
        Dict[str, str]: Report content sections
    """
    report_content = {}

    # Title and disclaimer
    report_content["title"] = (
        f"Clustering Analysis Report - {clustering_result.algorithm}"
    )
    report_content["disclaimer"] = RESEARCH_DISCLAIMER

    # Executive summary
    report_content["executive_summary"] = _generate_executive_summary(
        clustering_result, evaluation_results
    )

    # Methodology
    report_content["methodology"] = _generate_methodology_section(
        clustering_result, config
    )

    # Results
    report_content["results"] = _generate_results_section(
        clustering_result, evaluation_results
    )

    # Visualizations
    report_content["visualizations"] = _generate_visualizations_section(
        visualization_results
    )

    # Clinical analysis
    report_content["clinical_analysis"] = _generate_clinical_analysis_section(
        evaluation_results
    )

    # Limitations
    report_content["limitations"] = _generate_limitations_section()

    # Conclusions
    report_content["conclusions"] = _generate_conclusions_section(
        clustering_result, evaluation_results
    )

    return report_content


def _generate_executive_summary(
    clustering_result: ClusteringResult, evaluation_results: Dict[str, Any]
) -> str:
    """Generate executive summary section."""
    n_clusters = len(torch.unique(clustering_result.cluster_labels))
    n_samples = len(clustering_result.cluster_labels)

    summary = f"""
## Executive Summary

This report presents the analysis of mammography embedding clustering using the {clustering_result.algorithm} algorithm.

**Key Findings:**
- Algorithm: {clustering_result.algorithm}
- Number of clusters: {n_clusters}
- Total samples: {n_samples}
- Processing time: {clustering_result.processing_time:.3f} seconds

**Clustering Quality:**
"""

    if "metrics" in evaluation_results:
        metrics = evaluation_results["metrics"]
        if "silhouette" in metrics:
            summary += f"- Silhouette score: {metrics['silhouette']:.3f}\n"
        if "davies_bouldin" in metrics:
            summary += f"- Davies-Bouldin score: {metrics['davies_bouldin']:.3f}\n"
        if "calinski_harabasz" in metrics:
            summary += (
                f"- Calinski-Harabasz score: {metrics['calinski_harabasz']:.3f}\n"
            )

    summary += "\n**Research Disclaimer:**\n"
    summary += "This analysis is for research purposes only and should not be used for clinical decision-making.\n"

    return summary


def _generate_methodology_section(
    clustering_result: ClusteringResult, config: Dict[str, Any]
) -> str:
    """Generate methodology section."""
    methodology = f"""
## Methodology

**Clustering Algorithm:** {clustering_result.algorithm}

**Preprocessing:**
- DICOM files were preprocessed to remove borders and artifacts
- Images were normalized and resized to standard dimensions
- ResNet-50 embeddings were extracted from preprocessed images

**Clustering Parameters:**
- Algorithm: {clustering_result.algorithm}
- Hyperparameters: {clustering_result.hyperparameters}

**Evaluation Metrics:**
- Silhouette score: Measures cluster separation and cohesion
- Davies-Bouldin score: Measures cluster separation (lower is better)
- Calinski-Harabasz score: Measures cluster separation (higher is better)

**Visualization:**
- UMAP dimensionality reduction for 2D visualization
- Cluster montages for qualitative validation
- Metrics plots for quantitative assessment
"""

    return methodology


def _generate_results_section(
    clustering_result: ClusteringResult, evaluation_results: Dict[str, Any]
) -> str:
    """Generate results section."""
    results = f"""
## Results

**Clustering Performance:**
- Algorithm: {clustering_result.algorithm}
- Number of clusters: {len(torch.unique(clustering_result.cluster_labels))}
- Total samples: {len(clustering_result.cluster_labels)}
- Processing time: {clustering_result.processing_time:.3f} seconds

**Quality Metrics:**
"""

    if "metrics" in evaluation_results:
        metrics = evaluation_results["metrics"]
        for metric, value in metrics.items():
            results += f"- {metric}: {value:.3f}\n"

    # Cluster size distribution
    cluster_labels = clustering_result.cluster_labels.numpy()
    unique_labels, counts = torch.unique(
        clustering_result.cluster_labels, return_counts=True
    )

    results += "\n**Cluster Size Distribution:**\n"
    for label, count in zip(unique_labels, counts, strict=False):
        results += f"- Cluster {label.item()}: {count.item()} samples\n"

    return results


def _generate_visualizations_section(visualization_results: Dict[str, Any]) -> str:
    """Generate visualizations section."""
    visualizations = """
## Visualizations

The following visualizations were generated to analyze the clustering results:

"""

    if "output_files" in visualization_results:
        for viz_type, file_path in visualization_results["output_files"].items():
            visualizations += f"- **{viz_type}**: {file_path}\n"

    visualizations += """
**Visualization Descriptions:**
- UMAP plots: Show the 2D projection of embeddings with cluster assignments
- Cluster montages: Display representative images from each cluster
- Metrics plots: Show clustering quality metrics and cluster size distribution
"""

    return visualizations


def _generate_clinical_analysis_section(evaluation_results: Dict[str, Any]) -> str:
    """Generate clinical analysis section."""
    clinical_analysis = """
## Clinical Analysis

**Important Disclaimer:**
This analysis is for research purposes only and should not be used for clinical decision-making.

**Cluster Characteristics:**
"""

    if "sanity_checks" in evaluation_results:
        sanity_checks = evaluation_results["sanity_checks"]

        if "cluster_size_analysis" in sanity_checks:
            size_analysis = sanity_checks["cluster_size_analysis"]
            clinical_analysis += (
                f"- Number of clusters: {size_analysis.get('n_clusters', 'N/A')}\n"
            )

            if "size_statistics" in size_analysis:
                stats = size_analysis["size_statistics"]
                clinical_analysis += (
                    f"- Mean cluster size: {stats.get('mean_size', 'N/A'):.1f}\n"
                )
                clinical_analysis += f"- Cluster size range: {stats.get('min_size', 'N/A')} - {stats.get('max_size', 'N/A')}\n"

        if "potential_issues" in sanity_checks:
            issues = sanity_checks.get("potential_issues", [])
            if issues:
                clinical_analysis += "\n**Potential Issues Identified:**\n"
                for issue in issues:
                    clinical_analysis += f"- {issue}\n"

    clinical_analysis += """
**Clinical Relevance:**
- Clusters may represent different breast density patterns
- Further validation with clinical annotations is needed
- Results should be interpreted with caution due to unsupervised nature
"""

    return clinical_analysis


def _generate_limitations_section() -> str:
    """Generate limitations section."""
    limitations = """
## Limitations

**Methodological Limitations:**
- Unsupervised clustering without ground truth labels
- Limited to ResNet-50 feature representations
- No clinical validation of cluster meanings
- Potential bias from preprocessing choices

**Data Limitations:**
- Dataset may not be representative of general population
- Limited diversity in imaging conditions
- No clinical annotations for validation

**Technical Limitations:**
- Clustering quality depends on embedding quality
- UMAP visualization may not preserve all relationships
- Metrics may not capture clinically relevant patterns
"""

    return limitations


def _generate_conclusions_section(
    clustering_result: ClusteringResult, evaluation_results: Dict[str, Any]
) -> str:
    """Generate conclusions section."""
    conclusions = f"""
## Conclusions

**Summary:**
The {clustering_result.algorithm} clustering algorithm was applied to mammography embeddings, resulting in {len(torch.unique(clustering_result.cluster_labels))} clusters from {len(clustering_result.cluster_labels)} samples.

**Key Findings:**
"""

    if "metrics" in evaluation_results:
        metrics = evaluation_results["metrics"]
        if "silhouette" in metrics:
            silhouette = metrics["silhouette"]
            if silhouette > 0.5:
                conclusions += "- Good cluster separation (Silhouette > 0.5)\n"
            elif silhouette > 0.25:
                conclusions += "- Moderate cluster separation (Silhouette > 0.25)\n"
            else:
                conclusions += "- Poor cluster separation (Silhouette < 0.25)\n"

    conclusions += """
**Recommendations:**
- Further validation with clinical annotations is needed
- Consider different clustering algorithms for comparison
- Investigate cluster characteristics with domain experts
- Expand dataset for more robust analysis

**Research Disclaimer:**
This analysis is for research purposes only and should not be used for clinical decision-making.
"""

    return conclusions


def _save_report(
    report_content: Dict[str, str], output_dir: Path, report_format: str
) -> Optional[Path]:
    """
    Save analysis report to file.

    Educational Note: Report saving enables documentation
    and sharing of analysis results.

    Args:
        report_content: Report content dictionary
        output_dir: Output directory path
        report_format: Report format (markdown, html, pdf)

    Returns:
        Path: Path to saved report, None if failed
    """
    try:
        if report_format == "markdown":
            return _save_markdown_report(report_content, output_dir)
        elif report_format == "html":
            return _save_html_report(report_content, output_dir)
        else:
            logger.warning(
                f"Unsupported report format: {report_format}, using markdown"
            )
            return _save_markdown_report(report_content, output_dir)

    except Exception as e:
        logger.error(f"Error saving report: {e!s}")
        return None


def _save_markdown_report(
    report_content: Dict[str, str], output_dir: Path
) -> Optional[Path]:
    """Save markdown report."""
    try:
        report_path = (
            output_dir
            / f"clustering_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )

        with open(report_path, "w") as f:
            f.write(
                f"# {report_content.get('title', 'Clustering Analysis Report')}\n\n"
            )
            f.write(f"{report_content.get('disclaimer', '')}\n\n")
            f.write(f"{report_content.get('executive_summary', '')}\n\n")
            f.write(f"{report_content.get('methodology', '')}\n\n")
            f.write(f"{report_content.get('results', '')}\n\n")
            f.write(f"{report_content.get('visualizations', '')}\n\n")
            f.write(f"{report_content.get('clinical_analysis', '')}\n\n")
            f.write(f"{report_content.get('limitations', '')}\n\n")
            f.write(f"{report_content.get('conclusions', '')}\n\n")

        return report_path

    except Exception as e:
        logger.error(f"Error saving markdown report: {e!s}")
        return None


def _save_html_report(
    report_content: Dict[str, str], output_dir: Path
) -> Optional[Path]:
    """Save HTML report."""
    try:
        report_path = (
            output_dir
            / f"clustering_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_content.get('title', 'Clustering Analysis Report')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2 {{ color: #333; }}
        .disclaimer {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; margin: 20px 0; }}
        .metrics {{ background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 15px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>{report_content.get('title', 'Clustering Analysis Report')}</h1>
    <div class="disclaimer">{report_content.get('disclaimer', '')}</div>
    {report_content.get('executive_summary', '').replace('##', '<h2>').replace('**', '<strong>').replace('**', '</strong>')}
    {report_content.get('methodology', '').replace('##', '<h2>').replace('**', '<strong>').replace('**', '</strong>')}
    {report_content.get('results', '').replace('##', '<h2>').replace('**', '<strong>').replace('**', '</strong>')}
    {report_content.get('visualizations', '').replace('##', '<h2>').replace('**', '<strong>').replace('**', '</strong>')}
    {report_content.get('clinical_analysis', '').replace('##', '<h2>').replace('**', '<strong>').replace('**', '</strong>')}
    {report_content.get('limitations', '').replace('##', '<h2>').replace('**', '<strong>').replace('**', '</strong>')}
    {report_content.get('conclusions', '').replace('##', '<h2>').replace('**', '<strong>').replace('**', '</strong>')}
</body>
</html>
"""

        with open(report_path, "w") as f:
            f.write(html_content)

        return report_path

    except Exception as e:
        logger.error(f"Error saving HTML report: {e!s}")
        return None


def _save_evaluation_results(
    evaluation_results: Dict[str, Any], output_dir: Path
) -> None:
    """
    Save evaluation results to files.

    Educational Note: Results saving enables reproducibility
    and further analysis of evaluation outcomes.

    Args:
        evaluation_results: Evaluation results dictionary
        output_dir: Output directory path
    """
    try:
        # Save evaluation results
        evaluation_path = output_dir / "clustering_evaluation.yaml"
        with open(evaluation_path, "w") as f:
            yaml.dump(evaluation_results, f, default_flow_style=False, indent=2)

        logger.info(f"Evaluation results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error saving evaluation results: {e!s}")


def _print_visualization_results(visualization_results: Dict[str, Any]) -> None:
    """Print visualization generation results."""
    typer.echo("Visualization Results:")

    if "output_files" in visualization_results:
        for viz_type, file_path in visualization_results["output_files"].items():
            typer.echo(f"  {viz_type}: {file_path}")

    if "error" in visualization_results:
        typer.echo(f"  Error: {visualization_results['error']}")


def _print_evaluation_results(evaluation_results: Dict[str, Any]) -> None:
    """Print evaluation results."""
    typer.echo("Evaluation Results:")

    if "summary" in evaluation_results:
        summary = evaluation_results["summary"]
        typer.echo(
            f"  Clustering algorithm: {summary.get('clustering_algorithm', 'N/A')}"
        )
        typer.echo(f"  Number of clusters: {summary.get('n_clusters', 'N/A')}")
        typer.echo(f"  Number of samples: {summary.get('n_samples', 'N/A')}")

    if "sanity_checks" in evaluation_results:
        sanity_checks = evaluation_results["sanity_checks"]
        typer.echo(f"  Sanity checks performed: {len(sanity_checks)}")

        for check_name, check_results in sanity_checks.items():
            if isinstance(check_results, dict) and "potential_issues" in check_results:
                issues = check_results["potential_issues"]
                if issues:
                    typer.echo(f"    {check_name}: {len(issues)} issues found")
                else:
                    typer.echo(f"    {check_name}: No issues found")


def _print_report_results(report_results: Dict[str, Any]) -> None:
    """Print report generation results."""
    typer.echo("Report Generation Results:")

    if "output_files" in report_results:
        for file_type, file_path in report_results["output_files"].items():
            typer.echo(f"  {file_type}: {file_path}")

    if "sections" in report_results:
        typer.echo(f"  Report sections: {', '.join(report_results['sections'])}")

    if "error" in report_results:
        typer.echo(f"  Error: {report_results['error']}")


def _generate_config_template() -> Dict[str, Any]:
    """
    Generate analysis configuration template.

    Educational Note: Configuration templates help users
    understand available parameters and their effects.

    Returns:
        Dict[str, Any]: Configuration template
    """
    return {
        "visualizations": ["umap_2d", "cluster_montage", "metrics_plot"],
        "sanity_checks": [
            "cluster_size_analysis",
            "embedding_statistics",
            "projection_distribution",
        ],
        "umap_params": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 2,
            "metric": "euclidean",
            "random_state": 42,
        },
        "plot_params": {
            "figsize": (10, 8),
            "dpi": 300,
            "style": "whitegrid",
            "palette": "husl",
        },
        "montage_params": {
            "n_samples_per_cluster": 4,
            "image_size": (224, 224),
            "grid_size": (2, 2),
        },
        "seed": 42,
    }


if __name__ == "__main__":
    app()
