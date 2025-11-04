"""
Main CLI entry point for mammography analysis pipeline.

This module provides the main command-line interface that integrates
all pipeline components including preprocessing, embedding extraction,
clustering, and analysis for the breast density exploration project.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- Main CLI provides unified interface for all pipeline operations
- Pipeline integration enables end-to-end processing
- Configuration management enables reproducible experiments
- Error handling provides robust processing capabilities

Author: Research Team
Version: 1.0.0
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import typer
import yaml

from ..pipeline.mammography_pipeline import MammographyPipeline
from .analyze_cli import app as analyze_app
from .cluster_cli import app as cluster_app
from .embed_cli import app as embed_app
from .preprocess_cli import app as preprocess_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize main Typer app
app = typer.Typer(
    name="mmg",
    help="Mammography Analysis Pipeline - Research purposes only",
    add_completion=False,
)

# Add subcommands
app.add_typer(preprocess_app, name="preprocess", help="Preprocess DICOM files")
app.add_typer(embed_app, name="embed", help="Extract ResNet-50 embeddings")
app.add_typer(cluster_app, name="cluster", help="Cluster embeddings")
app.add_typer(analyze_app, name="analyze", help="Analyze clustering results")

# Research disclaimer
RESEARCH_DISCLAIMER = """
⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

This project is intended exclusively for research and education purposes
in medical imaging processing and machine learning.
"""


@app.command()
def pipeline(
    input_dir: Path = typer.Option(
        ..., "--input-dir", "-i", help="Directory containing DICOM files"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to save results"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to pipeline configuration file"
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device to use (auto, cpu, cuda, cuda:0, etc.)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Run the complete mammography analysis pipeline.

    This command runs the complete pipeline from DICOM files to clustering
    analysis and visualization in a single execution.

    Educational Note: The complete pipeline demonstrates the full workflow
    from raw DICOM data to clustering analysis and visualization.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print research disclaimer
    typer.echo(RESEARCH_DISCLAIMER)
    typer.echo()

    try:
        # Load configuration
        config = _load_config(config_file)

        # Validate input directory
        if not input_dir.exists():
            typer.echo(f"Error: Input directory {input_dir} does not exist", err=True)
            raise typer.Exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        typer.echo("Starting complete mammography analysis pipeline")
        typer.echo(f"Input directory: {input_dir}")
        typer.echo(f"Output directory: {output_dir}")
        typer.echo(f"Device: {device}")
        typer.echo()

        # Initialize and run pipeline
        pipeline = MammographyPipeline(config)
        results = pipeline.run_complete_pipeline(input_dir, output_dir, device)

        # Print results summary
        _print_pipeline_summary(results)

        if results["success"]:
            typer.echo("Pipeline completed successfully!")
            typer.echo(f"Results saved to {output_dir}")
        else:
            typer.echo("Pipeline completed with errors", err=True)
            for error in results["errors"]:
                typer.echo(f"  Error: {error}", err=True)
            raise typer.Exit(1)

    except Exception as e:
        logger.error(f"Error in pipeline execution: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


@app.command()
def config_template(
    output_file: Path = typer.Option(
        "pipeline_config.yaml",
        "--output",
        "-o",
        help="Output file for configuration template",
    )
):
    """
    Generate a complete pipeline configuration template.

    This command creates a template configuration file with
    default values for all pipeline parameters.

    Educational Note: Configuration templates help users understand
    available parameters and their effects on the complete pipeline.
    """
    # Print research disclaimer
    typer.echo(RESEARCH_DISCLAIMER)
    typer.echo()

    try:
        # Generate template configuration
        template_config = _generate_pipeline_config_template()

        # Save template
        with open(output_file, "w") as f:
            yaml.dump(template_config, f, default_flow_style=False, indent=2)

        typer.echo(f"Pipeline configuration template saved to {output_file}")
        typer.echo("Edit the configuration file to customize pipeline parameters")

    except Exception as e:
        logger.error(f"Error generating config template: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


@app.command()
def validate(
    input_dir: Path = typer.Option(
        ..., "--input-dir", "-i", help="Directory containing DICOM files to validate"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to pipeline configuration file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Validate input data and configuration for pipeline execution.

    This command checks DICOM files and configuration for compatibility
    with the complete pipeline.

    Educational Note: Validation ensures data quality and prevents
    processing failures during pipeline execution.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print research disclaimer
    typer.echo(RESEARCH_DISCLAIMER)
    typer.echo()

    try:
        # Load configuration
        config = _load_config(config_file)

        # Validate input directory
        if not input_dir.exists():
            typer.echo(f"Error: Input directory {input_dir} does not exist", err=True)
            raise typer.Exit(1)

        typer.echo("Validating pipeline configuration and input data")
        typer.echo(f"Input directory: {input_dir}")
        typer.echo()

        # Validate configuration
        config_validation = _validate_configuration(config)
        _print_config_validation(config_validation)

        # Validate DICOM files
        dicom_validation = _validate_dicom_files(input_dir, config)
        _print_dicom_validation(dicom_validation)

        # Overall validation result
        overall_success = config_validation["success"] and dicom_validation["success"]

        if overall_success:
            typer.echo("Validation completed successfully!")
            typer.echo("Pipeline is ready for execution")
        else:
            typer.echo("Validation completed with errors", err=True)
            typer.echo("Please fix the issues before running the pipeline")
            raise typer.Exit(1)

    except Exception as e:
        logger.error(f"Error in validation: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


def _load_config(config_file: Optional[Path]) -> Dict[str, Any]:
    """
    Load pipeline configuration.

    Educational Note: Configuration loading enables reproducible
    experiments and parameter management.

    Args:
        config_file: Optional path to configuration file

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

    return config


def _get_default_config() -> Dict[str, Any]:
    """
    Get default pipeline configuration.

    Educational Note: Default configurations provide sensible
    starting points for pipeline parameters.

    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        "seed": 42,
        "dicom_reader": {
            "validate_metadata": True,
            "required_tags": ["Pixel Spacing", "Bits Stored", "Manufacturer"],
            "patient_split": True,
        },
        "preprocessing": {
            "border_removal": {"enabled": True, "threshold": 0.1, "margin": 10},
            "normalization": {
                "method": "z_score_per_image",
                "target_mean": 0.0,
                "target_std": 1.0,
            },
            "resizing": {
                "enabled": True,
                "target_size": [224, 224],
                "preserve_aspect_ratio": True,
            },
        },
        "embedding": {
            "model_name": "resnet50",
            "pretrained": True,
            "input_adapter": "replicate_channels",
            "output_layer": "avgpool",
            "embedding_dimension": 2048,
            "normalize_embeddings": True,
        },
        "clustering": {
            "algorithm": "kmeans",
            "n_clusters": 4,
            "pca_dimensions": 50,
            "evaluation_metrics": ["silhouette", "davies_bouldin", "calinski_harabasz"],
            "hyperparameters": {},
        },
        "evaluation": {
            "metrics": ["silhouette", "davies_bouldin", "calinski_harabasz"],
            "sanity_checks": ["cluster_size_analysis", "embedding_statistics"],
            "visual_prototypes": {
                "n_samples_per_cluster": 4,
                "selection_method": "centroid_distance",
            },
        },
        "visualization": {
            "visualizations": ["umap_2d", "cluster_montage", "metrics_plot"],
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
        },
        "output": {
            "save_intermediate_results": True,
            "generate_report": True,
            "format": "yaml",
        },
    }


def _validate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate pipeline configuration.

    Educational Note: Configuration validation ensures all required
    parameters are present and within valid ranges.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Dict[str, Any]: Validation results
    """
    validation_results = {"success": True, "errors": [], "warnings": []}

    # Check required sections
    required_sections = ["dicom_reader", "preprocessing", "embedding", "clustering"]
    for section in required_sections:
        if section not in config:
            validation_results["errors"].append(
                f"Missing required configuration section: {section}"
            )
            validation_results["success"] = False

    # Validate clustering algorithm
    if "clustering" in config:
        clustering_config = config["clustering"]
        if "algorithm" in clustering_config:
            algorithm = clustering_config["algorithm"]
            valid_algorithms = ["kmeans", "gmm", "hdbscan", "agglomerative"]
            if algorithm not in valid_algorithms:
                validation_results["errors"].append(
                    f"Invalid clustering algorithm: {algorithm}"
                )
                validation_results["success"] = False

    # Validate PCA dimensions
    if "clustering" in config and "pca_dimensions" in config["clustering"]:
        pca_dims = config["clustering"]["pca_dimensions"]
        if not isinstance(pca_dims, int) or pca_dims < 2:
            validation_results["errors"].append(
                "pca_dimensions must be an integer >= 2"
            )
            validation_results["success"] = False

    return validation_results


def _validate_dicom_files(input_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate DICOM files for pipeline compatibility.

    Educational Note: DICOM validation ensures data quality and prevents
    processing failures during pipeline execution.

    Args:
        input_dir: Input directory containing DICOM files
        config: Pipeline configuration

    Returns:
        Dict[str, Any]: Validation results
    """
    validation_results = {
        "success": True,
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "errors": [],
    }

    try:
        # Find DICOM files
        dicom_files = list(input_dir.rglob("*.dcm"))
        validation_results["total_files"] = len(dicom_files)

        if not dicom_files:
            validation_results["errors"].append(
                "No DICOM files found in input directory"
            )
            validation_results["success"] = False
            return validation_results

        # Initialize DICOM reader for validation
        from ..io_dicom.dicom_reader import DicomReader

        dicom_reader = DicomReader(config["dicom_reader"])

        # Validate each DICOM file
        for file_path in dicom_files:
            try:
                is_valid = dicom_reader.validate_dicom_file(file_path)
                if is_valid:
                    validation_results["valid_files"] += 1
                else:
                    validation_results["invalid_files"] += 1
                    validation_results["errors"].append(
                        f"Invalid DICOM file: {file_path}"
                    )

            except Exception as e:
                validation_results["invalid_files"] += 1
                validation_results["errors"].append(
                    f"Error validating {file_path}: {e!s}"
                )

        # Check if we have enough valid files
        if validation_results["valid_files"] == 0:
            validation_results["success"] = False
            validation_results["errors"].append("No valid DICOM files found")
        elif validation_results["valid_files"] < 10:
            validation_results["warnings"] = validation_results.get("warnings", [])
            validation_results["warnings"].append(
                "Very few valid DICOM files found - clustering may not work well"
            )

    except Exception as e:
        validation_results["success"] = False
        validation_results["errors"].append(f"Error during DICOM validation: {e!s}")

    return validation_results


def _print_pipeline_summary(results: Dict[str, Any]) -> None:
    """
    Print pipeline execution summary.

    Educational Note: Clear reporting helps users understand
    pipeline execution outcomes and performance.

    Args:
        results: Pipeline execution results
    """
    typer.echo("Pipeline Execution Summary:")
    typer.echo(f"  Success: {results['success']}")
    typer.echo(f"  Total processing time: {results['total_processing_time']:.3f}s")
    typer.echo(f"  Input directory: {results['input_dir']}")
    typer.echo(f"  Output directory: {results['output_dir']}")
    typer.echo(f"  Device: {results['device']}")

    typer.echo("\nStage Results:")
    for stage_name, stage_results in results["stages"].items():
        typer.echo(
            f"  {stage_name}: {'✓' if stage_results['success'] else '✗'} ({stage_results['processing_time']:.3f}s)"
        )

        # Add stage-specific details
        if stage_name == "preprocessing":
            typer.echo(
                f"    Files processed: {stage_results['processed_files']}/{stage_results['total_files']}"
            )
        elif stage_name == "embedding":
            typer.echo(
                f"    Tensors processed: {stage_results['processed_tensors']}/{stage_results['total_tensors']}"
            )
        elif stage_name == "clustering" and stage_results["clustering_result"]:
            clustering_result = stage_results["clustering_result"]
            typer.echo(f"    Algorithm: {clustering_result.algorithm}")
            typer.echo(
                f"    Clusters: {len(torch.unique(clustering_result.cluster_labels))}"
            )
            typer.echo(f"    Samples: {len(clustering_result.cluster_labels)}")

    if results["errors"]:
        typer.echo(f"\nErrors ({len(results['errors'])}):")
        for error in results["errors"]:
            typer.echo(f"  - {error}")


def _print_config_validation(validation_results: Dict[str, Any]) -> None:
    """
    Print configuration validation results.

    Educational Note: Clear reporting helps users understand
    configuration issues and how to fix them.

    Args:
        validation_results: Configuration validation results
    """
    typer.echo("Configuration Validation:")
    typer.echo(f"  Success: {'✓' if validation_results['success'] else '✗'}")

    if validation_results["errors"]:
        typer.echo(f"  Errors ({len(validation_results['errors'])}):")
        for error in validation_results["errors"]:
            typer.echo(f"    - {error}")

    if validation_results.get("warnings"):
        typer.echo(f"  Warnings ({len(validation_results['warnings'])}):")
        for warning in validation_results["warnings"]:
            typer.echo(f"    - {warning}")


def _print_dicom_validation(validation_results: Dict[str, Any]) -> None:
    """
    Print DICOM validation results.

    Educational Note: Clear reporting helps users understand
    data quality issues and how to fix them.

    Args:
        validation_results: DICOM validation results
    """
    typer.echo("DICOM File Validation:")
    typer.echo(f"  Success: {'✓' if validation_results['success'] else '✗'}")
    typer.echo(f"  Total files: {validation_results['total_files']}")
    typer.echo(f"  Valid files: {validation_results['valid_files']}")
    typer.echo(f"  Invalid files: {validation_results['invalid_files']}")

    if validation_results["errors"]:
        typer.echo(f"  Errors ({len(validation_results['errors'])}):")
        for error in validation_results["errors"][:5]:  # Show first 5 errors
            typer.echo(f"    - {error}")
        if len(validation_results["errors"]) > 5:
            typer.echo(
                f"    ... and {len(validation_results['errors']) - 5} more errors"
            )


def _generate_pipeline_config_template() -> Dict[str, Any]:
    """
    Generate complete pipeline configuration template.

    Educational Note: Configuration templates help users
    understand available parameters and their effects.

    Returns:
        Dict[str, Any]: Configuration template
    """
    return {
        "seed": 42,
        "dicom_reader": {
            "validate_metadata": True,
            "required_tags": ["Pixel Spacing", "Bits Stored", "Manufacturer"],
            "patient_split": True,
        },
        "preprocessing": {
            "border_removal": {"enabled": True, "threshold": 0.1, "margin": 10},
            "normalization": {
                "method": "z_score_per_image",  # Options: z_score_per_image, fixed_window
                "target_mean": 0.0,
                "target_std": 1.0,
            },
            "resizing": {
                "enabled": True,
                "target_size": [224, 224],
                "preserve_aspect_ratio": True,
            },
        },
        "embedding": {
            "model_name": "resnet50",  # Options: resnet50, resnet101, resnet152
            "pretrained": True,
            "input_adapter": "replicate_channels",  # Options: replicate_channels, conv1_adaptation
            "output_layer": "avgpool",  # Options: avgpool, fc
            "embedding_dimension": 2048,
            "normalize_embeddings": True,
        },
        "clustering": {
            "algorithm": "kmeans",  # Options: kmeans, gmm, hdbscan, agglomerative
            "n_clusters": 4,
            "pca_dimensions": 50,
            "evaluation_metrics": ["silhouette", "davies_bouldin", "calinski_harabasz"],
            "hyperparameters": {
                # K-means specific
                "n_init": 10,
                "max_iter": 300,
                # GMM specific
                "covariance_type": "full",
                "max_iter": 100,
                # HDBSCAN specific
                "min_cluster_size": 10,
                "min_samples": 5,
                # Agglomerative specific
                "linkage": "ward",
            },
        },
        "evaluation": {
            "metrics": ["silhouette", "davies_bouldin", "calinski_harabasz"],
            "sanity_checks": [
                "cluster_size_analysis",
                "embedding_statistics",
                "projection_distribution",
            ],
            "visual_prototypes": {
                "n_samples_per_cluster": 4,
                "selection_method": "centroid_distance",  # Options: centroid_distance, random
            },
        },
        "visualization": {
            "visualizations": ["umap_2d", "cluster_montage", "metrics_plot"],
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
        },
        "output": {
            "save_intermediate_results": True,
            "generate_report": True,
            "format": "yaml",  # Options: yaml, json
        },
    }


if __name__ == "__main__":
    app()
