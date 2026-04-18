# ruff: noqa
"""
Mammography analysis pipeline for end-to-end processing.

This module provides a complete pipeline for mammography analysis including
DICOM preprocessing, embedding extraction, clustering, and visualization
for the breast density exploration project.

DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- Pipeline integration connects all processing steps
- Data flow validation ensures consistency between steps
- Error handling provides robust processing capabilities
- Configuration management enables reproducible experiments
- Parallel processing support for improved throughput (configurable via
  max_workers parameter, default: 4 workers)

Author: Research Team
Version: 1.0.0
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import typer
import yaml

from ..clustering.clustering_algorithms import ClusteringAlgorithms
from ..clustering.clustering_result import ClusteringResult
from ..eval.clustering_evaluator import ClusteringEvaluator
from ..io.dicom import (
    DicomReader,
    MammographyImage,
    create_mammography_image_from_dicom,
)
from ..models.embeddings.embedding_vector import EmbeddingVector
from ..models.embeddings.resnet50_extractor import ResNet50Extractor
from ..preprocess.image_preprocessor import ImagePreprocessor
from ..preprocess.preprocessed_tensor import PreprocessedTensor
from ..vis.cluster_visualizer import ClusterVisualizer

# Configure logging
logger = logging.getLogger(__name__)

# Constants
METADATA_SUFFIX = ".metadata.yaml"

from .reporting import PipelineReportingMixin
from .steps import PipelineStepsMixin
from .storage import PipelineStorageMixin


class MammographyPipeline(
    PipelineStepsMixin, PipelineReportingMixin, PipelineStorageMixin
):
    """Complete mammography analysis pipeline.

    This class provides end-to-end processing capabilities for mammography
    analysis including preprocessing, embedding extraction, clustering,
    evaluation, and visualization.

    Educational Notes:
    - Pipeline integration connects all processing steps
    - Data flow validation ensures consistency between steps
    - Error handling provides robust processing capabilities
    - Configuration management enables reproducible experiments

    Attributes:
        config: Pipeline configuration dictionary
        max_workers: Maximum number of worker threads for parallel processing
        dicom_reader: DicomReader instance
        preprocessor: ImagePreprocessor instance
        extractor: ResNet50Extractor instance
        clusterer: ClusteringAlgorithms instance
        evaluator: ClusteringEvaluator instance
        visualizer: ClusterVisualizer instance
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize mammography pipeline with configuration.

        Args:
            config: Pipeline configuration dictionary. Supports the following
                optional parameters:
                - max_workers (int): Maximum number of worker threads for
                  parallel processing. Defaults to 4.
                - seed (int): Random seed for reproducibility. Defaults to 42.

        Educational Note: Configuration validation ensures all required
        parameters are present and valid for pipeline execution.

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = self._validate_config(config)

        # Set random seed for reproducibility
        if "seed" in self.config:
            torch.manual_seed(self.config["seed"])
            np.random.seed(self.config["seed"])

        # Set parallel processing configuration
        self.max_workers = self.config.get("max_workers", 4)

        # Initialize components
        self.dicom_reader = None
        self.preprocessor = None
        self.extractor = None
        self.clusterer = None
        self.evaluator = None
        self.visualizer = None

        # Initialize components based on configuration
        self._initialize_components()

        logger.info(f"Initialized MammographyPipeline with config: {self.config}")

    def run_complete_pipeline(
        self, input_dir: Path, output_dir: Path, device: str = "auto"
    ) -> Dict[str, Any]:
        """
        Run the complete mammography analysis pipeline.

        Educational Note: This method demonstrates the complete pipeline
        from DICOM files to clustering analysis and visualization.

        Args:
            input_dir: Directory containing DICOM files
            output_dir: Directory to save results
            device: Device to use for processing (auto, cpu, cuda)

        Returns:
            Dict[str, Any]: Pipeline execution results
        """
        pipeline_results = {
            "pipeline_timestamp": datetime.now().isoformat(),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "device": device,
            "success": False,
            "step_results": {},
            "errors": [],
            "total_processing_time": 0.0,
        }

        start_time = time.time()

        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # DICOM reading and preprocessing
            typer.echo("Leitura DICOM e pre-processamento")
            preprocessing_results = self._run_preprocessing_step(input_dir, output_dir)
            pipeline_results["step_results"]["preprocessing"] = preprocessing_results

            if preprocessing_results["success"]:
                # Embedding extraction
                typer.echo("Extracao de embeddings")
                embedding_results = self._run_embedding_step(
                    preprocessing_results["output_dir"], output_dir, device
                )
                pipeline_results["step_results"]["embedding"] = embedding_results

                if embedding_results["success"]:
                    # Clustering
                    typer.echo("Clusterizacao")
                    clustering_results = self._run_clustering_step(
                        embedding_results["output_dir"], output_dir
                    )
                    pipeline_results["step_results"]["clustering"] = clustering_results

                    if clustering_results["success"]:
                        # Evaluation
                        typer.echo("Avaliacao")
                        evaluation_results = self._run_evaluation_step(
                            clustering_results["clustering_result"],
                            embedding_results["embedding_vectors"],
                            output_dir,
                        )
                        pipeline_results["step_results"]["evaluation"] = (
                            evaluation_results
                        )

                        # Visualization
                        typer.echo("Visualizacao")
                        visualization_results = self._run_visualization_step(
                            clustering_results["clustering_result"],
                            embedding_results["embedding_vectors"],
                            preprocessing_results["mammography_images"],
                            output_dir,
                        )
                        pipeline_results["step_results"]["visualization"] = (
                            visualization_results
                        )

                        pipeline_results["success"] = all(
                            step.get("success", False)
                            for step in pipeline_results["step_results"].values()
                        )

                        generate_report = bool(
                            self.config.get("output", {}).get("generate_report", True)
                        )
                        if generate_report:
                            # Generate final report
                            typer.echo("Generating Final Report")
                            report_results = self._generate_final_report(
                                pipeline_results, output_dir
                            )
                            pipeline_results["step_results"]["report"] = report_results
                            pipeline_results["success"] = all(
                                step.get("success", False)
                                for step in pipeline_results["step_results"].values()
                            )

                        if pipeline_results["success"]:
                            typer.echo("Pipeline completed successfully!")
                        else:
                            typer.echo("Pipeline completed with errors.")
                    else:
                        pipeline_results["success"] = False
                        pipeline_results["errors"].append("Falha na clusterizacao")
                else:
                    pipeline_results["success"] = False
                    pipeline_results["errors"].append("Falha na extracao de embeddings")
            else:
                pipeline_results["success"] = False
                pipeline_results["errors"].append("Falha no pre-processamento")

            pipeline_results["total_processing_time"] = time.time() - start_time

        except Exception as e:
            logger.error(f"Error in pipeline execution: {e!s}")
            pipeline_results["success"] = False
            pipeline_results["errors"].append(str(e))
            pipeline_results["total_processing_time"] = time.time() - start_time

        return pipeline_results

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize the pipeline configuration and apply required defaults.

        Ensures the configuration contains the required sections: "dicom_reader",
        "preprocessing", "embedding", and "clustering". Shallow-copies the provided
        mapping and any nested section dictionaries, then applies defaults for:
        - seed (default 42)
        - max_workers (default 4)
        - output (defaults: save_intermediate_results=True, generate_report=True, format="yaml")
        - preprocessing.input_adapter ("1to3_replication")
        - embedding.input_adapter ("1to3_replication")
        - clustering.pca_dimensions (2)

        Parameters:
            config (Dict[str, Any]): Configuration mapping to validate and normalize.

        Returns:
            Dict[str, Any]: The validated and augmented configuration dictionary.

        Raises:
            ValueError: If any required configuration section is missing.
        """
        config = dict(config)
        for section, value in list(config.items()):
            if isinstance(value, dict):
                config[section] = dict(value)

        # Check required sections
        required_sections = ["dicom_reader", "preprocessing", "embedding", "clustering"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Set default values
        config.setdefault("seed", 42)
        config.setdefault("max_workers", 4)
        config.setdefault(
            "output",
            {
                "save_intermediate_results": True,
                "generate_report": True,
                "format": "yaml",
            },
        )
        config["preprocessing"].setdefault("input_adapter", "1to3_replication")
        config["embedding"].setdefault("input_adapter", "1to3_replication")
        config["clustering"].setdefault("pca_dimensions", 2)

        return config


def create_mammography_pipeline(config: Dict[str, Any]) -> MammographyPipeline:
    """
    Create a configured MammographyPipeline from the provided configuration.

    The provided `config` will be validated and normalized by the pipeline constructor (defaults applied where missing).

    Parameters:
        config (Dict[str, Any]): Pipeline configuration dictionary containing sections like "dicom_reader", "preprocessing", "embedding", and "clustering".

    Returns:
        MammographyPipeline: An initialized MammographyPipeline instance configured according to `config`.
    """
    return MammographyPipeline(config)


def run_mammography_pipeline(
    input_dir: Path, output_dir: Path, config: Dict[str, Any], device: str = "auto"
) -> Dict[str, Any]:
    """
    Run the full mammography processing pipeline for the given input and output directories.

    Parameters:
        input_dir (Path): Directory containing input DICOM files.
        output_dir (Path): Directory where pipeline outputs and reports will be written.
        config (Dict[str, Any]): Pipeline configuration; it will be validated and augmented by the pipeline (e.g., seed, max_workers, component settings).
        device (str): Compute device to use (e.g., 'cpu', 'cuda', or 'auto' to let the pipeline choose).

    Returns:
        pipeline_results (Dict[str, Any]): A dictionary summarizing execution, including keys such as 'timestamp', 'input_dir', 'output_dir', 'device', 'step_results' (per-step output and success flags), 'errors', 'success', and 'total_processing_time'.
    """
    pipeline = create_mammography_pipeline(config)
    return pipeline.run_complete_pipeline(input_dir, output_dir, device)
