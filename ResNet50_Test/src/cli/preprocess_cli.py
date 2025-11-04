"""
Preprocessing CLI for mammography DICOM processing.

This module provides command-line interface for preprocessing mammography
DICOM files including border removal, normalization, and standardization
for the breast density exploration pipeline.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- CLI provides user-friendly interface for preprocessing operations
- Configuration management enables reproducible experiments
- Patient-level splitting prevents data leakage
- Batch processing enables efficient handling of large datasets

Author: Research Team
Version: 1.0.0
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import typer
import yaml

from ..io_dicom.dicom_reader import DicomReader
from ..preprocess.image_preprocessor import ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Typer app
app = typer.Typer(
    name="mmg-preprocess",
    help="Preprocess mammography DICOM files for embedding extraction",
    add_completion=False,
)

# Research disclaimer
RESEARCH_DISCLAIMER = """
⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.
"""


@app.command()
def preprocess(
    input_dir: Path = typer.Option(
        ..., "--input-dir", "-i", help="Directory containing DICOM files to preprocess"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to save preprocessed tensors"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to preprocessing configuration file"
    ),
    patient_split: bool = typer.Option(
        True,
        "--patient-split/--no-patient-split",
        help="Enable patient-level data splitting",
    ),
    batch_size: int = typer.Option(
        32, "--batch-size", "-b", help="Batch size for processing"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Preprocess mammography DICOM files for embedding extraction.

    This command processes DICOM files through the complete preprocessing
    pipeline including border removal, normalization, and standardization.

    Educational Note: Preprocessing is crucial for consistent feature
    extraction and prevents bias from imaging artifacts.
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

        # Initialize components
        dicom_reader = DicomReader(config["dicom_reader"])
        preprocessor = ImagePreprocessor(config["preprocessing"])

        typer.echo(f"Starting preprocessing from {input_dir} to {output_dir}")
        typer.echo(f"Configuration: {config}")

        # Process DICOM files
        results = _process_dicom_files(
            dicom_reader,
            preprocessor,
            input_dir,
            output_dir,
            patient_split,
            batch_size,
            config,
        )

        # Save results
        _save_preprocessing_results(results, output_dir)

        typer.echo("Preprocessing completed successfully!")
        typer.echo(f"Processed {results['total_files']} files")
        typer.echo(f"Generated {results['total_tensors']} preprocessed tensors")
        typer.echo(f"Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error in preprocessing: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


@app.command()
def validate(
    input_dir: Path = typer.Option(
        ..., "--input-dir", "-i", help="Directory containing DICOM files to validate"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to preprocessing configuration file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Validate DICOM files for preprocessing compatibility.

    This command checks DICOM files for required metadata and
    compatibility with the preprocessing pipeline.

    Educational Note: Validation ensures data quality and prevents
    processing failures during the pipeline.
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

        # Initialize DICOM reader
        dicom_reader = DicomReader(config["dicom_reader"])

        typer.echo(f"Validating DICOM files in {input_dir}")

        # Validate files
        validation_results = _validate_dicom_files(dicom_reader, input_dir)

        # Print results
        _print_validation_results(validation_results)

        if validation_results["valid_files"] == 0:
            typer.echo("No valid files found!", err=True)
            raise typer.Exit(1)

        typer.echo(
            f"Validation completed: {validation_results['valid_files']}/{validation_results['total_files']} files valid"
        )

    except Exception as e:
        logger.error(f"Error in validation: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


@app.command()
def config_template(
    output_file: Path = typer.Option(
        "preprocessing_config.yaml",
        "--output",
        "-o",
        help="Output file for configuration template",
    )
):
    """
    Generate a preprocessing configuration template.

    This command creates a template configuration file with
    default values for preprocessing parameters.

    Educational Note: Configuration templates help users understand
    available parameters and their effects on preprocessing.
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
        typer.echo("Edit the configuration file to customize preprocessing parameters")

    except Exception as e:
        logger.error(f"Error generating config template: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


def _load_config(config_file: Optional[Path]) -> Dict[str, Any]:
    """
    Load preprocessing configuration.

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
    Get default preprocessing configuration.

    Educational Note: Default configurations provide sensible
    starting points for preprocessing parameters.

    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
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
            "augmentation": {"enabled": False},
        },
        "output": {"format": "tensor", "compression": True, "metadata": True},
    }


def _process_dicom_files(
    dicom_reader: DicomReader,
    preprocessor: ImagePreprocessor,
    input_dir: Path,
    output_dir: Path,
    patient_split: bool,
    batch_size: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process DICOM files through preprocessing pipeline.

    Educational Note: Batch processing enables efficient handling
    of large datasets while maintaining memory efficiency.

    Args:
        dicom_reader: DicomReader instance
        preprocessor: ImagePreprocessor instance
        input_dir: Input directory path
        output_dir: Output directory path
        patient_split: Whether to split by patient
        batch_size: Batch size for processing
        config: Configuration dictionary

    Returns:
        Dict[str, Any]: Processing results
    """
    results = {
        "total_files": 0,
        "total_tensors": 0,
        "failed_files": 0,
        "processing_time": 0.0,
        "patient_splits": {},
        "errors": [],
    }

    start_time = datetime.now()

    try:
        # Find DICOM files
        dicom_files = list(input_dir.rglob("*.dcm"))
        results["total_files"] = len(dicom_files)

        if not dicom_files:
            logger.warning(f"No DICOM files found in {input_dir}")
            return results

        typer.echo(f"Found {len(dicom_files)} DICOM files")

        # Process files in batches
        for i in range(0, len(dicom_files), batch_size):
            batch_files = dicom_files[i : i + batch_size]
            typer.echo(
                f"Processing batch {i//batch_size + 1}/{(len(dicom_files) + batch_size - 1)//batch_size}"
            )

            batch_results = _process_batch(
                dicom_reader,
                preprocessor,
                batch_files,
                output_dir,
                patient_split,
                config,
            )

            # Update results
            results["total_tensors"] += batch_results["tensors_created"]
            results["failed_files"] += batch_results["failed_files"]
            results["errors"].extend(batch_results["errors"])

            if patient_split:
                for patient_id, split in batch_results["patient_splits"].items():
                    if patient_id not in results["patient_splits"]:
                        results["patient_splits"][patient_id] = split
                    else:
                        results["patient_splits"][patient_id].extend(split)

        results["processing_time"] = (datetime.now() - start_time).total_seconds()

    except Exception as e:
        logger.error(f"Error processing DICOM files: {e!s}")
        results["errors"].append(str(e))

    return results


def _process_batch(
    dicom_reader: DicomReader,
    preprocessor: ImagePreprocessor,
    batch_files: list,
    output_dir: Path,
    patient_split: bool,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process a batch of DICOM files.

    Educational Note: Batch processing improves efficiency
    and enables progress tracking for large datasets.

    Args:
        dicom_reader: DicomReader instance
        preprocessor: ImagePreprocessor instance
        batch_files: List of DICOM file paths
        output_dir: Output directory path
        patient_split: Whether to split by patient
        config: Configuration dictionary

    Returns:
        Dict[str, Any]: Batch processing results
    """
    batch_results = {
        "tensors_created": 0,
        "failed_files": 0,
        "patient_splits": {},
        "errors": [],
    }

    for file_path in batch_files:
        try:
            # Read DICOM file
            mammography_image = dicom_reader.read_dicom_file(file_path)
            if mammography_image is None:
                batch_results["failed_files"] += 1
                batch_results["errors"].append(f"Failed to read {file_path}")
                continue

            # Preprocess image
            preprocessed_tensor = preprocessor.preprocess_image(mammography_image)
            if preprocessed_tensor is None:
                batch_results["failed_files"] += 1
                batch_results["errors"].append(f"Failed to preprocess {file_path}")
                continue

            # Save preprocessed tensor
            output_path = _get_output_path(file_path, output_dir)
            if _save_preprocessed_tensor(preprocessed_tensor, output_path):
                batch_results["tensors_created"] += 1

                # Track patient split
                if patient_split:
                    patient_id = mammography_image.patient_id
                    if patient_id not in batch_results["patient_splits"]:
                        batch_results["patient_splits"][patient_id] = []
                    batch_results["patient_splits"][patient_id].append(str(output_path))
            else:
                batch_results["failed_files"] += 1
                batch_results["errors"].append(f"Failed to save {file_path}")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e!s}")
            batch_results["failed_files"] += 1
            batch_results["errors"].append(f"Error processing {file_path}: {e!s}")

    return batch_results


def _validate_dicom_files(dicom_reader: DicomReader, input_dir: Path) -> Dict[str, Any]:
    """
    Validate DICOM files for preprocessing compatibility.

    Educational Note: Validation ensures data quality and prevents
    processing failures during the pipeline.

    Args:
        dicom_reader: DicomReader instance
        input_dir: Input directory path

    Returns:
        Dict[str, Any]: Validation results
    """
    validation_results = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "validation_errors": [],
    }

    # Find DICOM files
    dicom_files = list(input_dir.rglob("*.dcm"))
    validation_results["total_files"] = len(dicom_files)

    for file_path in dicom_files:
        try:
            # Validate DICOM file
            is_valid = dicom_reader.validate_dicom_file(file_path)
            if is_valid:
                validation_results["valid_files"] += 1
            else:
                validation_results["invalid_files"] += 1
                validation_results["validation_errors"].append(
                    f"Invalid file: {file_path}"
                )

        except Exception as e:
            logger.error(f"Error validating {file_path}: {e!s}")
            validation_results["invalid_files"] += 1
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
    typer.echo(f"  Total files: {validation_results['total_files']}")
    typer.echo(f"  Valid files: {validation_results['valid_files']}")
    typer.echo(f"  Invalid files: {validation_results['invalid_files']}")

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


def _get_output_path(input_path: Path, output_dir: Path) -> Path:
    """
    Generate output path for preprocessed tensor.

    Educational Note: Consistent output paths enable easy
    tracking and organization of processed data.

    Args:
        input_path: Input DICOM file path
        output_dir: Output directory path

    Returns:
        Path: Output path for preprocessed tensor
    """
    # Create relative path structure
    relative_path = input_path.relative_to(input_path.parents[1])
    output_path = output_dir / relative_path.with_suffix(".pt")

    # Create parent directories
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return output_path


def _save_preprocessed_tensor(preprocessed_tensor, output_path: Path) -> bool:
    """
    Save preprocessed tensor to file.

    Educational Note: Efficient tensor storage enables
    fast loading for subsequent pipeline stages.

    Args:
        preprocessed_tensor: PreprocessedTensor instance
        output_path: Output file path

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Save tensor data
        torch.save(preprocessed_tensor.tensor, output_path)

        # Save metadata
        metadata_path = output_path.with_suffix(".metadata.yaml")
        metadata = {
            "image_id": preprocessed_tensor.image_id,
            "patient_id": preprocessed_tensor.patient_id,
            "preprocessing_config": preprocessed_tensor.preprocessing_config,
            "timestamp": preprocessed_tensor.timestamp.isoformat(),
        }

        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        return True

    except Exception as e:
        logger.error(f"Error saving preprocessed tensor to {output_path}: {e!s}")
        return False


def _save_preprocessing_results(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Save preprocessing results summary.

    Educational Note: Results summaries enable tracking
    of processing outcomes and debugging.

    Args:
        results: Processing results dictionary
        output_dir: Output directory path
    """
    try:
        # Save results summary
        results_path = output_dir / "preprocessing_results.yaml"
        with open(results_path, "w") as f:
            yaml.dump(results, f, default_flow_style=False, indent=2)

        logger.info(f"Preprocessing results saved to {results_path}")

    except Exception as e:
        logger.error(f"Error saving preprocessing results: {e!s}")


def _generate_config_template() -> Dict[str, Any]:
    """
    Generate preprocessing configuration template.

    Educational Note: Configuration templates help users
    understand available parameters and their effects.

    Returns:
        Dict[str, Any]: Configuration template
    """
    return {
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
            "augmentation": {"enabled": False},
        },
        "output": {"format": "tensor", "compression": True, "metadata": True},
    }


if __name__ == "__main__":
    app()
