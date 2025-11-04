"""
Embedding CLI for mammography feature extraction.

This module provides command-line interface for extracting ResNet-50
embeddings from preprocessed mammography tensors for the breast
density exploration pipeline.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- CLI provides user-friendly interface for embedding extraction
- GPU/CPU device selection enables flexible deployment
- Batch processing and caching improve efficiency
- Configuration management enables reproducible experiments

Author: Research Team
Version: 1.0.0
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import typer
import yaml

from ..models.embeddings.embedding_vector import EmbeddingVector
from ..models.embeddings.resnet50_extractor import ResNet50Extractor
from ..preprocess.preprocessed_tensor import PreprocessedTensor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Typer app
app = typer.Typer(
    name="mmg-embed",
    help="Extract ResNet-50 embeddings from preprocessed mammography tensors",
    add_completion=False,
)

# Research disclaimer
RESEARCH_DISCLAIMER = """
⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.
"""


@app.command()
def extract(
    input_dir: Path = typer.Option(
        ..., "--input-dir", "-i", help="Directory containing preprocessed tensors"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to save extracted embeddings"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to embedding configuration file"
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device to use (auto, cpu, cuda, cuda:0, etc.)"
    ),
    batch_size: int = typer.Option(
        16, "--batch-size", "-b", help="Batch size for embedding extraction"
    ),
    cache_embeddings: bool = typer.Option(
        True, "--cache/--no-cache", help="Enable embedding caching"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Extract ResNet-50 embeddings from preprocessed tensors.

    This command processes preprocessed tensors through the ResNet-50
    feature extraction pipeline to generate 2048-dimensional embeddings.

    Educational Note: Embedding extraction is the core feature extraction
    step that converts images to numerical representations suitable for clustering.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print research disclaimer
    typer.echo(RESEARCH_DISCLAIMER)
    typer.echo()

    try:
        # Load configuration
        config = _load_config(config_file)

        # Determine device
        device = _determine_device(device)
        typer.echo(f"Using device: {device}")

        # Validate input directory
        if not input_dir.exists():
            typer.echo(f"Error: Input directory {input_dir} does not exist", err=True)
            raise typer.Exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ResNet-50 extractor
        extractor = ResNet50Extractor(config["embedding"], device=device)

        typer.echo(f"Starting embedding extraction from {input_dir} to {output_dir}")
        typer.echo(f"Configuration: {config}")

        # Extract embeddings
        results = _extract_embeddings(
            extractor, input_dir, output_dir, batch_size, cache_embeddings, config
        )

        # Save results
        _save_embedding_results(results, output_dir)

        typer.echo("Embedding extraction completed successfully!")
        typer.echo(f"Processed {results['total_tensors']} tensors")
        typer.echo(f"Generated {results['total_embeddings']} embeddings")
        typer.echo(f"Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error in embedding extraction: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


@app.command()
def validate(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        "-i",
        help="Directory containing preprocessed tensors to validate",
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to embedding configuration file"
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device to use for validation"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Validate preprocessed tensors for embedding extraction.

    This command checks preprocessed tensors for compatibility
    with the ResNet-50 embedding extraction pipeline.

    Educational Note: Validation ensures data quality and prevents
    processing failures during embedding extraction.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print research disclaimer
    typer.echo(RESEARCH_DISCLAIMER)
    typer.echo()

    try:
        # Load configuration
        config = _load_config(config_file)

        # Determine device
        device = _determine_device(device)
        typer.echo(f"Using device: {device}")

        # Validate input directory
        if not input_dir.exists():
            typer.echo(f"Error: Input directory {input_dir} does not exist", err=True)
            raise typer.Exit(1)

        # Initialize ResNet-50 extractor
        extractor = ResNet50Extractor(config["embedding"], device=device)

        typer.echo(f"Validating preprocessed tensors in {input_dir}")

        # Validate tensors
        validation_results = _validate_tensors(extractor, input_dir)

        # Print results
        _print_validation_results(validation_results)

        if validation_results["valid_tensors"] == 0:
            typer.echo("No valid tensors found!", err=True)
            raise typer.Exit(1)

        typer.echo(
            f"Validation completed: {validation_results['valid_tensors']}/{validation_results['total_tensors']} tensors valid"
        )

    except Exception as e:
        logger.error(f"Error in validation: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


@app.command()
def config_template(
    output_file: Path = typer.Option(
        "embedding_config.yaml",
        "--output",
        "-o",
        help="Output file for configuration template",
    )
):
    """
    Generate an embedding configuration template.

    This command creates a template configuration file with
    default values for embedding extraction parameters.

    Educational Note: Configuration templates help users understand
    available parameters and their effects on embedding extraction.
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
        typer.echo(
            "Edit the configuration file to customize embedding extraction parameters"
        )

    except Exception as e:
        logger.error(f"Error generating config template: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


def _load_config(config_file: Optional[Path]) -> Dict[str, Any]:
    """
    Load embedding configuration.

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
    Get default embedding configuration.

    Educational Note: Default configurations provide sensible
    starting points for embedding extraction parameters.

    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        "embedding": {
            "model_name": "resnet50",
            "pretrained": True,
            "input_adapter": "replicate_channels",  # Options: replicate_channels, conv1_adaptation
            "output_layer": "avgpool",
            "embedding_dimension": 2048,
            "normalize_embeddings": True,
            "cache_embeddings": True,
        },
        "preprocessing": {
            "resize_to_imagenet": True,
            "normalize_imagenet": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "output": {"format": "tensor", "compression": True, "metadata": True},
    }


def _determine_device(device: str) -> str:
    """
    Determine the device to use for embedding extraction.

    Educational Note: Device selection affects performance and
    memory usage during embedding extraction.

    Args:
        device: Device specification string

    Returns:
        str: Device string for PyTorch
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            typer.echo("CUDA available, using GPU")
        else:
            device = "cpu"
            typer.echo("CUDA not available, using CPU")
    else:
        if device.startswith("cuda") and not torch.cuda.is_available():
            typer.echo(
                "Warning: CUDA requested but not available, falling back to CPU",
                err=True,
            )
            device = "cpu"

    return device


def _extract_embeddings(
    extractor: ResNet50Extractor,
    input_dir: Path,
    output_dir: Path,
    batch_size: int,
    cache_embeddings: bool,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract embeddings from preprocessed tensors.

    Educational Note: Batch processing enables efficient handling
    of large datasets while maintaining memory efficiency.

    Args:
        extractor: ResNet50Extractor instance
        input_dir: Input directory path
        output_dir: Output directory path
        batch_size: Batch size for processing
        cache_embeddings: Whether to cache embeddings
        config: Configuration dictionary

    Returns:
        Dict[str, Any]: Extraction results
    """
    results = {
        "total_tensors": 0,
        "total_embeddings": 0,
        "failed_extractions": 0,
        "processing_time": 0.0,
        "errors": [],
    }

    start_time = datetime.now()

    try:
        # Find tensor files
        tensor_files = list(input_dir.rglob("*.pt"))
        results["total_tensors"] = len(tensor_files)

        if not tensor_files:
            logger.warning(f"No tensor files found in {input_dir}")
            return results

        typer.echo(f"Found {len(tensor_files)} tensor files")

        # Process files in batches
        for i in range(0, len(tensor_files), batch_size):
            batch_files = tensor_files[i : i + batch_size]
            typer.echo(
                f"Processing batch {i//batch_size + 1}/{(len(tensor_files) + batch_size - 1)//batch_size}"
            )

            batch_results = _process_batch(
                extractor, batch_files, output_dir, cache_embeddings, config
            )

            # Update results
            results["total_embeddings"] += batch_results["embeddings_created"]
            results["failed_extractions"] += batch_results["failed_extractions"]
            results["errors"].extend(batch_results["errors"])

        results["processing_time"] = (datetime.now() - start_time).total_seconds()

    except Exception as e:
        logger.error(f"Error extracting embeddings: {e!s}")
        results["errors"].append(str(e))

    return results


def _process_batch(
    extractor: ResNet50Extractor,
    batch_files: list,
    output_dir: Path,
    cache_embeddings: bool,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process a batch of tensor files for embedding extraction.

    Educational Note: Batch processing improves efficiency
    and enables progress tracking for large datasets.

    Args:
        extractor: ResNet50Extractor instance
        batch_files: List of tensor file paths
        output_dir: Output directory path
        cache_embeddings: Whether to cache embeddings
        config: Configuration dictionary

    Returns:
        Dict[str, Any]: Batch processing results
    """
    batch_results = {"embeddings_created": 0, "failed_extractions": 0, "errors": []}

    for file_path in batch_files:
        try:
            # Check if embedding already exists (caching)
            if cache_embeddings:
                embedding_path = _get_embedding_path(file_path, output_dir)
                if embedding_path.exists():
                    batch_results["embeddings_created"] += 1
                    continue

            # Load preprocessed tensor
            preprocessed_tensor = _load_preprocessed_tensor(file_path)
            if preprocessed_tensor is None:
                batch_results["failed_extractions"] += 1
                batch_results["errors"].append(f"Failed to load {file_path}")
                continue

            # Extract embedding
            embedding_vector = extractor.extract_embedding(preprocessed_tensor)
            if embedding_vector is None:
                batch_results["failed_extractions"] += 1
                batch_results["errors"].append(
                    f"Failed to extract embedding from {file_path}"
                )
                continue

            # Save embedding
            embedding_path = _get_embedding_path(file_path, output_dir)
            if _save_embedding_vector(embedding_vector, embedding_path):
                batch_results["embeddings_created"] += 1
            else:
                batch_results["failed_extractions"] += 1
                batch_results["errors"].append(
                    f"Failed to save embedding for {file_path}"
                )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e!s}")
            batch_results["failed_extractions"] += 1
            batch_results["errors"].append(f"Error processing {file_path}: {e!s}")

    return batch_results


def _validate_tensors(extractor: ResNet50Extractor, input_dir: Path) -> Dict[str, Any]:
    """
    Validate preprocessed tensors for embedding extraction.

    Educational Note: Validation ensures data quality and prevents
    processing failures during embedding extraction.

    Args:
        extractor: ResNet50Extractor instance
        input_dir: Input directory path

    Returns:
        Dict[str, Any]: Validation results
    """
    validation_results = {
        "total_tensors": 0,
        "valid_tensors": 0,
        "invalid_tensors": 0,
        "validation_errors": [],
    }

    # Find tensor files
    tensor_files = list(input_dir.rglob("*.pt"))
    validation_results["total_tensors"] = len(tensor_files)

    for file_path in tensor_files:
        try:
            # Load and validate tensor
            preprocessed_tensor = _load_preprocessed_tensor(file_path)
            if preprocessed_tensor is None:
                validation_results["invalid_tensors"] += 1
                validation_results["validation_errors"].append(
                    f"Failed to load {file_path}"
                )
                continue

            # Validate tensor format
            is_valid = _validate_tensor_format(preprocessed_tensor, extractor)
            if is_valid:
                validation_results["valid_tensors"] += 1
            else:
                validation_results["invalid_tensors"] += 1
                validation_results["validation_errors"].append(
                    f"Invalid tensor format: {file_path}"
                )

        except Exception as e:
            logger.error(f"Error validating {file_path}: {e!s}")
            validation_results["invalid_tensors"] += 1
            validation_results["validation_errors"].append(
                f"Error validating {file_path}: {e!s}"
            )

    return validation_results


def _print_validation_results(validation_results: Dict[str, Any]) -> None:
    """
    Print validation results to console.

    Educational Note: Clear reporting helps users understand
    data quality and identify issues.

    Args:
        validation_results: Validation results dictionary
    """
    typer.echo("Validation Results:")
    typer.echo(f"  Total tensors: {validation_results['total_tensors']}")
    typer.echo(f"  Valid tensors: {validation_results['valid_tensors']}")
    typer.echo(f"  Invalid tensors: {validation_results['invalid_tensors']}")

    if validation_results["validation_errors"]:
        typer.echo("  Validation errors:")
        for error in validation_results["validation_errors"][
            :10
        ]:  # Show first 10 errors
            typer.echo(f"    - {error}")
        if len(validation_results["validation_errors"]) > 10:
            typer.echo(
                f"    ... and {len(validation_results['validation_errors']) - 10} more errors"
            )


def _load_preprocessed_tensor(file_path: Path) -> Optional[PreprocessedTensor]:
    """
    Load preprocessed tensor from file.

    Educational Note: Efficient tensor loading enables
    fast processing for embedding extraction.

    Args:
        file_path: Path to tensor file

    Returns:
        PreprocessedTensor: Loaded tensor, None if failed
    """
    try:
        # Load tensor data
        tensor_data = torch.load(file_path, map_location="cpu")

        # Load metadata
        metadata_path = file_path.with_suffix(".metadata.yaml")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)
        else:
            metadata = {}

        # Create PreprocessedTensor instance
        preprocessed_tensor = PreprocessedTensor(
            image_id=metadata.get("image_id", file_path.stem),
            patient_id=metadata.get("patient_id", "unknown"),
            tensor=tensor_data,
            preprocessing_config=metadata.get("preprocessing_config", {}),
            timestamp=datetime.fromisoformat(
                metadata.get("timestamp", datetime.now().isoformat())
            ),
        )

        return preprocessed_tensor

    except Exception as e:
        logger.error(f"Error loading preprocessed tensor from {file_path}: {e!s}")
        return None


def _validate_tensor_format(
    preprocessed_tensor: PreprocessedTensor, extractor: ResNet50Extractor
) -> bool:
    """
    Validate tensor format for embedding extraction.

    Educational Note: Format validation ensures compatibility
    with the ResNet-50 extractor.

    Args:
        preprocessed_tensor: PreprocessedTensor instance
        extractor: ResNet50Extractor instance

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check tensor shape
        if (
            preprocessed_tensor.tensor.dim() != 4
        ):  # Should be [batch, channels, height, width]
            return False

        # Check tensor values
        if torch.isnan(preprocessed_tensor.tensor).any():
            return False

        if torch.isinf(preprocessed_tensor.tensor).any():
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating tensor format: {e!s}")
        return False


def _get_embedding_path(tensor_path: Path, output_dir: Path) -> Path:
    """
    Generate output path for embedding vector.

    Educational Note: Consistent output paths enable easy
    tracking and organization of extracted embeddings.

    Args:
        tensor_path: Input tensor file path
        output_dir: Output directory path

    Returns:
        Path: Output path for embedding vector
    """
    # Create relative path structure
    relative_path = tensor_path.relative_to(tensor_path.parents[1])
    embedding_path = output_dir / relative_path.with_suffix(".embedding.pt")

    # Create parent directories
    embedding_path.parent.mkdir(parents=True, exist_ok=True)

    return embedding_path


def _save_embedding_vector(
    embedding_vector: EmbeddingVector, output_path: Path
) -> bool:
    """
    Save embedding vector to file.

    Educational Note: Efficient embedding storage enables
    fast loading for subsequent pipeline stages.

    Args:
        embedding_vector: EmbeddingVector instance
        output_path: Output file path

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Save embedding data
        torch.save(embedding_vector.embedding, output_path)

        # Save metadata
        metadata_path = output_path.with_suffix(".metadata.yaml")
        metadata = {
            "image_id": embedding_vector.image_id,
            "patient_id": embedding_vector.patient_id,
            "embedding_dimension": embedding_vector.embedding.shape[0],
            "extraction_config": embedding_vector.extraction_config,
            "timestamp": embedding_vector.timestamp.isoformat(),
        }

        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        return True

    except Exception as e:
        logger.error(f"Error saving embedding vector to {output_path}: {e!s}")
        return False


def _save_embedding_results(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Save embedding extraction results summary.

    Educational Note: Results summaries enable tracking
    of extraction outcomes and debugging.

    Args:
        results: Extraction results dictionary
        output_dir: Output directory path
    """
    try:
        # Save results summary
        results_path = output_dir / "embedding_results.yaml"
        with open(results_path, "w") as f:
            yaml.dump(results, f, default_flow_style=False, indent=2)

        logger.info(f"Embedding extraction results saved to {results_path}")

    except Exception as e:
        logger.error(f"Error saving embedding results: {e!s}")


def _generate_config_template() -> Dict[str, Any]:
    """
    Generate embedding configuration template.

    Educational Note: Configuration templates help users
    understand available parameters and their effects.

    Returns:
        Dict[str, Any]: Configuration template
    """
    return {
        "embedding": {
            "model_name": "resnet50",  # Options: resnet50, resnet101, resnet152
            "pretrained": True,
            "input_adapter": "replicate_channels",  # Options: replicate_channels, conv1_adaptation
            "output_layer": "avgpool",  # Options: avgpool, fc
            "embedding_dimension": 2048,
            "normalize_embeddings": True,
            "cache_embeddings": True,
        },
        "preprocessing": {
            "resize_to_imagenet": True,
            "normalize_imagenet": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "output": {"format": "tensor", "compression": True, "metadata": True},
    }


if __name__ == "__main__":
    app()
