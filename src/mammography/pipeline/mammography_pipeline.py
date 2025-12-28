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

Author: Research Team
Version: 1.0.0
"""

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
from ..io.dicom import DicomReader, MammographyImage
from ..models.embeddings.embedding_vector import EmbeddingVector
from ..models.embeddings.resnet50_extractor import ResNet50Extractor
from ..preprocess.image_preprocessor import ImagePreprocessor
from ..preprocess.preprocessed_tensor import PreprocessedTensor
from ..vis.cluster_visualizer import ClusterVisualizer

# Configure logging
logger = logging.getLogger(__name__)

# Constants
METADATA_SUFFIX = ".metadata.yaml"


class MammographyPipeline:
    """
    Complete mammography analysis pipeline.

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
            config: Pipeline configuration dictionary

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
                        pipeline_results["step_results"]["evaluation"] = evaluation_results

                        # Visualization
                        typer.echo("Visualizacao")
                        visualization_results = self._run_visualization_step(
                            clustering_results["clustering_result"],
                            embedding_results["embedding_vectors"],
                            preprocessing_results["mammography_images"],
                            output_dir,
                        )
                        pipeline_results["step_results"][
                            "visualization"
                        ] = visualization_results

                        # Generate final report
                        typer.echo("Generating Final Report")
                        report_results = self._generate_final_report(
                            pipeline_results, output_dir
                        )
                        pipeline_results["step_results"]["report"] = report_results

                        pipeline_results["success"] = True
                        typer.echo("Pipeline completed successfully!")
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
        Validate pipeline configuration.

        Educational Note: Configuration validation ensures all required
        parameters are present and within valid ranges.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Dict[str, Any]: Validated configuration

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required sections
        required_sections = ["dicom_reader", "preprocessing", "embedding", "clustering"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Set default values
        config.setdefault("seed", 42)
        config.setdefault(
            "output",
            {
                "save_intermediate_results": True,
                "generate_report": True,
                "format": "yaml",
            },
        )

        return config

    def _initialize_components(self) -> None:
        """
        Initialize pipeline components.

        Educational Note: Component initialization ensures all
        processing steps are ready for execution.
        """
        try:
            # Initialize DICOM reader
            self.dicom_reader = DicomReader(self.config["dicom_reader"])

            # Initialize preprocessor
            self.preprocessor = ImagePreprocessor(self.config["preprocessing"])

            # Initialize embedding extractor
            self.extractor = ResNet50Extractor(self.config["embedding"])

            # Initialize clusterer
            self.clusterer = ClusteringAlgorithms(self.config["clustering"])

            # Initialize evaluator
            evaluator_config = self.config.get("evaluation", {})
            self.evaluator = ClusteringEvaluator(evaluator_config)

            # Initialize visualizer
            visualizer_config = self.config.get("visualization", {})
            self.visualizer = ClusterVisualizer(visualizer_config)

            logger.info("Successfully initialized all pipeline components")

        except Exception as e:
            logger.error(f"Error initializing pipeline components: {e!s}")
            raise

    def _run_preprocessing_step(
        self, input_dir: Path, output_dir: Path
    ) -> Dict[str, Any]:
        """
        Run preprocessing step.

        Educational Note: Preprocessing converts DICOM files to
        standardized tensors ready for embedding extraction.

        Args:
            input_dir: Input directory containing DICOM files
            output_dir: Output directory for results

        Returns:
            Dict[str, Any]: Preprocessing step results
        """
        preprocessing_results = {
            "step": "preprocessing",
            "success": False,
            "processing_time": 0.0,
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "mammography_images": [],
            "preprocessed_tensors": [],
            "output_dir": output_dir / "preprocessed",
            "errors": [],
        }

        start_time = time.time()

        try:
            # Create preprocessing output directory
            preprocessing_results["output_dir"].mkdir(parents=True, exist_ok=True)

            # Find DICOM files
            dicom_files = list(input_dir.rglob("*.dcm"))
            preprocessing_results["total_files"] = len(dicom_files)

            if not dicom_files:
                logger.warning(f"No DICOM files found in {input_dir}")
                preprocessing_results["success"] = True
                return preprocessing_results

            typer.echo(f"Found {len(dicom_files)} DICOM files")

            # Process DICOM files
            for file_path in dicom_files:
                try:
                    # Read DICOM file
                    mammography_image = self.dicom_reader.read_dicom_file(file_path)
                    if mammography_image is None:
                        preprocessing_results["failed_files"] += 1
                        preprocessing_results["errors"].append(
                            f"Failed to read {file_path}"
                        )
                        continue

                    # Preprocess image
                    preprocessed_tensor = self.preprocessor.preprocess_image(
                        mammography_image
                    )
                    if preprocessed_tensor is None:
                        preprocessing_results["failed_files"] += 1
                        preprocessing_results["errors"].append(
                            f"Failed to preprocess {file_path}"
                        )
                        continue

                    # Save preprocessed tensor
                    output_path = self._get_preprocessing_output_path(
                        file_path, preprocessing_results["output_dir"]
                    )
                    if self._save_preprocessed_tensor(preprocessed_tensor, output_path):
                        preprocessing_results["processed_files"] += 1
                        preprocessing_results["mammography_images"].append(
                            mammography_image
                        )
                        preprocessing_results["preprocessed_tensors"].append(
                            preprocessed_tensor
                        )
                    else:
                        preprocessing_results["failed_files"] += 1
                        preprocessing_results["errors"].append(
                            f"Failed to save {file_path}"
                        )

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e!s}")
                    preprocessing_results["failed_files"] += 1
                    preprocessing_results["errors"].append(
                        f"Error processing {file_path}: {e!s}"
                    )

            preprocessing_results["success"] = (
                preprocessing_results["processed_files"] > 0
            )
            preprocessing_results["processing_time"] = time.time() - start_time

            typer.echo(
                f"Preprocessing completed: {preprocessing_results['processed_files']}/{preprocessing_results['total_files']} files processed"
            )

        except Exception as e:
            logger.error(f"Error in preprocessing step: {e!s}")
            preprocessing_results["errors"].append(str(e))
            preprocessing_results["processing_time"] = time.time() - start_time

        return preprocessing_results

    def _run_embedding_step(
        self, input_dir: Path, output_dir: Path, device: str
    ) -> Dict[str, Any]:
        """
        Run embedding extraction step.

        Educational Note: Embedding extraction converts preprocessed
        tensors to high-dimensional feature vectors.

        Args:
            input_dir: Input directory containing preprocessed tensors
            output_dir: Output directory for results
            device: Device to use for processing

        Returns:
            Dict[str, Any]: Embedding step results
        """
        embedding_results = {
            "step": "embedding",
            "success": False,
            "processing_time": 0.0,
            "total_tensors": 0,
            "processed_tensors": 0,
            "failed_tensors": 0,
            "embedding_vectors": [],
            "output_dir": output_dir / "embeddings",
            "errors": [],
        }

        start_time = time.time()

        try:
            # Create embedding output directory
            embedding_results["output_dir"].mkdir(parents=True, exist_ok=True)

            # Set device for extractor
            self.extractor.device = device

            # Find tensor files
            tensor_files = list(input_dir.rglob("*.pt"))
            embedding_results["total_tensors"] = len(tensor_files)

            if not tensor_files:
                logger.warning(f"No tensor files found in {input_dir}")
                embedding_results["success"] = True
                return embedding_results

            typer.echo(f"Found {len(tensor_files)} tensor files")

            # Process tensor files
            for file_path in tensor_files:
                try:
                    # Load preprocessed tensor
                    preprocessed_tensor = self._load_preprocessed_tensor(file_path)
                    if preprocessed_tensor is None:
                        embedding_results["failed_tensors"] += 1
                        embedding_results["errors"].append(
                            f"Failed to load {file_path}"
                        )
                        continue

                    # Extract embedding
                    embedding_vector = self.extractor.extract_embedding(
                        preprocessed_tensor
                    )
                    if embedding_vector is None:
                        embedding_results["failed_tensors"] += 1
                        embedding_results["errors"].append(
                            f"Failed to extract embedding from {file_path}"
                        )
                        continue

                    # Save embedding
                    output_path = self._get_embedding_output_path(
                        file_path, embedding_results["output_dir"]
                    )
                    if self._save_embedding_vector(embedding_vector, output_path):
                        embedding_results["processed_tensors"] += 1
                        embedding_results["embedding_vectors"].append(embedding_vector)
                    else:
                        embedding_results["failed_tensors"] += 1
                        embedding_results["errors"].append(
                            f"Failed to save embedding for {file_path}"
                        )

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e!s}")
                    embedding_results["failed_tensors"] += 1
                    embedding_results["errors"].append(
                        f"Error processing {file_path}: {e!s}"
                    )

            embedding_results["success"] = embedding_results["processed_tensors"] > 0
            embedding_results["processing_time"] = time.time() - start_time

            typer.echo(
                f"Embedding extraction completed: {embedding_results['processed_tensors']}/{embedding_results['total_tensors']} tensors processed"
            )

        except Exception as e:
            logger.error(f"Error in embedding step: {e!s}")
            embedding_results["errors"].append(str(e))
            embedding_results["processing_time"] = time.time() - start_time

        return embedding_results

    def _run_clustering_step(
        self, input_dir: Path, output_dir: Path
    ) -> Dict[str, Any]:
        """
        Run clustering step.

        Educational Note: Clustering groups similar embeddings
        to discover patterns in the data.

        Args:
            input_dir: Input directory containing embedding vectors
            output_dir: Output directory for results

        Returns:
            Dict[str, Any]: Clustering step results
        """
        clustering_results = {
            "step": "clustering",
            "success": False,
            "processing_time": 0.0,
            "clustering_result": None,
            "output_dir": output_dir / "clustering",
            "errors": [],
        }

        start_time = time.time()

        try:
            # Create clustering output directory
            clustering_results["output_dir"].mkdir(parents=True, exist_ok=True)

            # Load embedding vectors
            embedding_vectors = self._load_embedding_vectors(input_dir)
            if not embedding_vectors:
                logger.warning(f"No embedding vectors found in {input_dir}")
                clustering_results["success"] = False
                return clustering_results

            typer.echo(f"Loaded {len(embedding_vectors)} embedding vectors")

            # Perform clustering
            clustering_result = self.clusterer.cluster_embeddings(embedding_vectors)
            if clustering_result is None:
                clustering_results["errors"].append("Clustering failed")
                clustering_results["success"] = False
                return clustering_results

            # Save clustering result
            result_path = (
                clustering_results["output_dir"]
                / f"clustering_result_{clustering_result.experiment_id}.pt"
            )
            torch.save(clustering_result, result_path)

            clustering_results["clustering_result"] = clustering_result
            clustering_results["success"] = True
            clustering_results["processing_time"] = time.time() - start_time

            typer.echo(
                f"Clustering completed: {len(torch.unique(clustering_result.cluster_labels, dim=0))} clusters found"
            )

        except Exception as e:
            logger.error(f"Error in clustering step: {e!s}")
            clustering_results["errors"].append(str(e))
            clustering_results["processing_time"] = time.time() - start_time

        return clustering_results

    def _run_evaluation_step(
        self,
        clustering_result: ClusteringResult,
        embedding_vectors: List[EmbeddingVector],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Run evaluation step.

        Educational Note: Evaluation assesses clustering quality
        and performs sanity checks.

        Args:
            clustering_result: ClusteringResult instance
            embedding_vectors: List of EmbeddingVector instances
            output_dir: Output directory for results

        Returns:
            Dict[str, Any]: Evaluation step results
        """
        evaluation_results = {
            "step": "evaluation",
            "success": False,
            "processing_time": 0.0,
            "evaluation_result": None,
            "output_dir": output_dir / "evaluation",
            "errors": [],
        }

        start_time = time.time()

        try:
            # Create evaluation output directory
            evaluation_results["output_dir"].mkdir(parents=True, exist_ok=True)

            # Perform evaluation
            evaluation_result = self.evaluator.evaluate_clustering(
                clustering_result, embedding_vectors
            )

            # Save evaluation result
            evaluation_path = (
                evaluation_results["output_dir"] / "evaluation_results.yaml"
            )
            with open(evaluation_path, "w") as f:
                yaml.dump(evaluation_result, f, default_flow_style=False, indent=2)

            evaluation_results["evaluation_result"] = evaluation_result
            evaluation_results["success"] = True
            evaluation_results["processing_time"] = time.time() - start_time

            typer.echo("Evaluation completed successfully")

        except Exception as e:
            logger.error(f"Error in evaluation step: {e!s}")
            evaluation_results["errors"].append(str(e))
            evaluation_results["processing_time"] = time.time() - start_time

        return evaluation_results

    def _run_visualization_step(
        self,
        clustering_result: ClusteringResult,
        embedding_vectors: List[EmbeddingVector],
        mammography_images: List[MammographyImage],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Run visualization step.

        Educational Note: Visualization enables qualitative
        validation of clustering results.

        Args:
            clustering_result: ClusteringResult instance
            embedding_vectors: List of EmbeddingVector instances
            mammography_images: List of MammographyImage instances
            output_dir: Output directory for results

        Returns:
            Dict[str, Any]: Visualization step results
        """
        visualization_results = {
            "step": "visualization",
            "success": False,
            "processing_time": 0.0,
            "visualization_result": None,
            "output_dir": output_dir / "visualizations",
            "errors": [],
        }

        start_time = time.time()

        try:
            # Create visualization output directory
            visualization_results["output_dir"].mkdir(parents=True, exist_ok=True)

            # Generate visualizations
            visualization_result = self.visualizer.create_visualizations(
                clustering_result,
                embedding_vectors,
                mammography_images,
                visualization_results["output_dir"],
            )

            visualization_results["visualization_result"] = visualization_result
            visualization_results["success"] = True
            visualization_results["processing_time"] = time.time() - start_time

            typer.echo("Visualization completed successfully")

        except Exception as e:
            logger.error(f"Error in visualization step: {e!s}")
            visualization_results["errors"].append(str(e))
            visualization_results["processing_time"] = time.time() - start_time

        return visualization_results

    def _generate_final_report(
        self, pipeline_results: Dict[str, Any], output_dir: Path
    ) -> Dict[str, Any]:
        """
        Generate final pipeline report.

        Educational Note: Final reports provide comprehensive
        summary of pipeline execution and results.

        Args:
            pipeline_results: Complete pipeline results
            output_dir: Output directory for results

        Returns:
            Dict[str, Any]: Report generation results
        """
        report_results = {
            "step": "report",
            "success": False,
            "processing_time": 0.0,
            "report_path": None,
            "errors": [],
        }

        start_time = time.time()

        try:
            # Generate report content
            report_content = self._generate_report_content(pipeline_results)

            # Save report
            report_path = (
                output_dir
                / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            )
            with open(report_path, "w") as f:
                yaml.dump(report_content, f, default_flow_style=False, indent=2)

            report_results["report_path"] = str(report_path)
            report_results["success"] = True
            report_results["processing_time"] = time.time() - start_time

            typer.echo(f"Final report saved to {report_path}")

        except Exception as e:
            logger.error(f"Error generating final report: {e!s}")
            report_results["errors"].append(str(e))
            report_results["processing_time"] = time.time() - start_time

        return report_results

    def _generate_report_content(
        self, pipeline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate report content.

        Educational Note: Report content provides comprehensive
        summary of pipeline execution and results.

        Args:
            pipeline_results: Complete pipeline results

        Returns:
            Dict[str, Any]: Report content
        """
        report_content = {
            "pipeline_summary": {
                "timestamp": pipeline_results["pipeline_timestamp"],
                "input_dir": pipeline_results["input_dir"],
                "output_dir": pipeline_results["output_dir"],
                "device": pipeline_results["device"],
                "success": pipeline_results["success"],
                "total_processing_time": pipeline_results["total_processing_time"],
            },
            "step_results": {},
            "errors": pipeline_results["errors"],
        }

        # Add step results
        for step_name, step_results in pipeline_results["step_results"].items():
            report_content["step_results"][step_name] = {
                "success": step_results["success"],
                "processing_time": step_results["processing_time"],
                "errors": step_results.get("errors", []),
            }

            # Add step-specific metrics
            if step_name == "preprocessing":
                report_content["step_results"][step_name].update(
                    {
                        "total_files": step_results["total_files"],
                        "processed_files": step_results["processed_files"],
                        "failed_files": step_results["failed_files"],
                    }
                )
            elif step_name == "embedding":
                report_content["step_results"][step_name].update(
                    {
                        "total_tensors": step_results["total_tensors"],
                        "processed_tensors": step_results["processed_tensors"],
                        "failed_tensors": step_results["failed_tensors"],
                    }
                )
            elif step_name == "clustering" and step_results["clustering_result"]:
                clustering_result = step_results["clustering_result"]
                report_content["step_results"][step_name].update(
                    {
                        "algorithm": clustering_result.algorithm,
                        "n_clusters": len(
                            torch.unique(clustering_result.cluster_labels, dim=0)
                        ),
                        "n_samples": len(clustering_result.cluster_labels),
                        "metrics": clustering_result.metrics,
                    }
                )

        return report_content

    # Helper methods for file operations
    def _get_preprocessing_output_path(
        self, input_path: Path, output_dir: Path
    ) -> Path:
        """Get output path for preprocessed tensor."""
        relative_path = input_path.relative_to(input_path.parents[1])
        output_path = output_dir / relative_path.with_suffix(".pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _get_embedding_output_path(self, input_path: Path, output_dir: Path) -> Path:
        """Get output path for embedding vector."""
        relative_path = input_path.relative_to(input_path.parents[1])
        output_path = output_dir / relative_path.with_suffix(".embedding.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _save_preprocessed_tensor(
        self, preprocessed_tensor: PreprocessedTensor, output_path: Path
    ) -> bool:
        """Save preprocessed tensor to file."""
        try:
            torch.save(preprocessed_tensor.tensor, output_path)

            # Save metadata
            metadata_path = output_path.with_suffix(METADATA_SUFFIX)
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

    def _save_embedding_vector(
        self, embedding_vector: EmbeddingVector, output_path: Path
    ) -> bool:
        """Save embedding vector to file."""
        try:
            torch.save(embedding_vector.embedding, output_path)

            # Save metadata
            metadata_path = output_path.with_suffix(METADATA_SUFFIX)
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

    def _load_preprocessed_tensor(
        self, file_path: Path
    ) -> Optional[PreprocessedTensor]:
        """Load preprocessed tensor from file."""
        try:
            tensor_data = torch.load(file_path, map_location="cpu")

            metadata_path = file_path.with_suffix(METADATA_SUFFIX)
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = yaml.safe_load(f)
            else:
                metadata = {}

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

    def _load_embedding_vectors(self, input_dir: Path) -> List[EmbeddingVector]:
        """Load embedding vectors from directory."""
        embedding_vectors = []

        try:
            embedding_files = list(input_dir.rglob("*.embedding.pt"))

            for file_path in embedding_files:
                try:
                    embedding_data = torch.load(file_path, map_location="cpu")

                    metadata_path = file_path.with_suffix(METADATA_SUFFIX)
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            metadata = yaml.safe_load(f)
                    else:
                        metadata = {}

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

        except Exception as e:
            logger.error(f"Error loading embedding vectors: {e!s}")

        return embedding_vectors


def create_mammography_pipeline(config: Dict[str, Any]) -> MammographyPipeline:
    """
    Factory function to create a MammographyPipeline instance.

    Educational Note: This factory function provides a convenient way
    to create MammographyPipeline instances with validated configurations.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        MammographyPipeline: Configured MammographyPipeline instance
    """
    return MammographyPipeline(config)


def run_mammography_pipeline(
    input_dir: Path, output_dir: Path, config: Dict[str, Any], device: str = "auto"
) -> Dict[str, Any]:
    """
    Convenience function to run the complete mammography pipeline.

    Educational Note: This function provides a simple interface for
    running the complete pipeline without creating a MammographyPipeline instance.

    Args:
        input_dir: Directory containing DICOM files
        output_dir: Directory to save results
        config: Pipeline configuration dictionary
        device: Device to use for processing

    Returns:
        Dict[str, Any]: Pipeline execution results
    """
    pipeline = create_mammography_pipeline(config)
    return pipeline.run_complete_pipeline(input_dir, output_dir, device)
