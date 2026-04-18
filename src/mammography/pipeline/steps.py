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


class PipelineStepsMixin:
    def _initialize_components(self) -> None:
        """
        Initialize pipeline components from the instance configuration.

        Initializes and assigns the following attributes on self using values from self.config:
        - dicom_reader: DicomReader (propagates self.max_workers into the reader config)
        - preprocessor: ImagePreprocessor
        - extractor: ResNet50Extractor
        - clusterer: ClusteringAlgorithms
        - evaluator: ClusteringEvaluator (constructed from optional "evaluation" config)
        - visualizer: ClusterVisualizer (constructed from optional "visualization" config)

        Logs a success message on completion; any exception raised during initialization is logged and re-raised.
        """
        try:
            # Initialize DICOM reader
            dicom_reader_config = dict(self.config["dicom_reader"])
            dicom_reader_config.setdefault("max_workers", self.max_workers)
            self.dicom_reader = DicomReader(**dicom_reader_config)

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
        Run preprocessing on all DICOM files in input_dir and save preprocessed tensors under output_dir / "preprocessed".

        Processes discovered DICOM files (possibly in parallel), converts each into a standardized preprocessed tensor and returns an aggregate result for the step.

        Returns:
            dict: A summary of the preprocessing step containing:
                - "step" (str): Step name ("preprocessing").
                - "success" (bool): `true` if at least one file was processed successfully, `false` otherwise.
                - "processing_time" (float): Elapsed time in seconds.
                - "total_files" (int): Number of DICOM files discovered.
                - "processed_files" (int): Number of files successfully processed.
                - "failed_files" (int): Number of files that failed processing.
                - "mammography_images" (List[MammographyImage]): Loaded mammography image objects for successful files.
                - "preprocessed_tensors" (List[path-like or tensor]): Saved preprocessed tensor references for successful files.
                - "output_dir" (Path): Directory where preprocessed outputs were written (output_dir / "preprocessed").
                - "errors" (List[str]): Error messages encountered during the step.
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
                message = f"No DICOM files found in {input_dir}"
                logger.warning(message)
                preprocessing_results["errors"].append(message)
                preprocessing_results["processing_time"] = time.time() - start_time
                return preprocessing_results

            typer.echo(f"Found {len(dicom_files)} DICOM files")

            # Process DICOM files in parallel
            results = self._process_preprocessing_files_parallel(
                dicom_files, preprocessing_results["output_dir"], input_dir
            )

            # Aggregate results
            for result in results:
                if result is not None and result["success"]:
                    preprocessing_results["processed_files"] += 1
                    preprocessing_results["mammography_images"].append(
                        result["mammography_image"]
                    )
                    preprocessing_results["preprocessed_tensors"].append(
                        result["preprocessed_tensor"]
                    )
                else:
                    preprocessing_results["failed_files"] += 1
                    if result is not None and "error" in result:
                        preprocessing_results["errors"].append(result["error"])

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

    def _process_single_preprocessing_file(
        self, file_path: Path, output_dir: Path, step_input_dir: Path | None = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single DICOM file: read it, preprocess it to a tensor, and save the preprocessed tensor to disk.

        Parameters:
            file_path (Path): Path to the source DICOM file.
            output_dir (Path): Directory where the preprocessed tensor file will be written.

        Returns:
            result (Dict[str, Any]): A dictionary describing the outcome.
                - 'success' (bool): `True` if processing and save succeeded, `False` otherwise.
                - On success: includes 'mammography_image' (the loaded image object) and 'preprocessed_tensor' (the produced tensor).
                - On failure: includes 'error' (str) with a human-readable failure message.
        """
        try:
            # Read DICOM file
            mammography_image = self.dicom_reader.read_dicom_file(file_path)
            if mammography_image is None:
                mammography_image = self._read_dicom_with_pipeline_defaults(file_path)
            if mammography_image is None:
                return {
                    "success": False,
                    "error": f"Failed to read {file_path}",
                }

            # Preprocess image
            preprocessed_tensor = self.preprocessor.preprocess_image(mammography_image)
            if preprocessed_tensor is None:
                return {
                    "success": False,
                    "error": f"Failed to preprocess {file_path}",
                }

            # Save preprocessed tensor
            output_path = self._get_preprocessing_output_path(
                file_path, output_dir, step_input_dir
            )
            if not self._save_preprocessed_tensor(preprocessed_tensor, output_path):
                return {
                    "success": False,
                    "error": f"Failed to save {file_path}",
                }

            return {
                "success": True,
                "mammography_image": mammography_image,
                "preprocessed_tensor": preprocessed_tensor,
            }

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e!s}")
            return {
                "success": False,
                "error": f"Error processing {file_path}: {e!s}",
            }

    def _read_dicom_with_pipeline_defaults(
        self, file_path: Path
    ) -> Optional[MammographyImage]:
        """Read a DICOM for pipeline processing after strict metadata validation rejects it."""
        try:
            return create_mammography_image_from_dicom(str(file_path))
        except Exception as exc:
            logger.debug(
                "Pipeline fallback DICOM read failed for %s: %s", file_path, exc
            )
            return None

    def _process_preprocessing_files_parallel(
        self,
        dicom_files: List[Path],
        output_dir: Path,
        step_input_dir: Path | None = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Process a list of DICOM files concurrently and return per-file processing results.

        Parameters:
            dicom_files (List[Path]): DICOM file paths to process.
            output_dir (Path): Directory where per-file preprocessing outputs are written.

        Returns:
            List[Optional[Dict[str, Any]]]: List aligned with `dicom_files` where each entry is either the file's result dictionary or `None` if no result was produced; failure entries contain `{"success": False, "error": "<message>"}`.
        """
        results: List[Optional[Dict[str, Any]]] = [None] * len(dicom_files)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(
                    self._process_single_preprocessing_file,
                    file_path,
                    output_dir,
                    step_input_dir,
                ): i
                for i, file_path in enumerate(dicom_files)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    logger.error(
                        "Error processing DICOM file %s: %s",
                        dicom_files[index],
                        exc,
                    )
                    results[index] = {
                        "success": False,
                        "error": f"Error processing {dicom_files[index]}: {exc!s}",
                    }

        return results

    def _run_embedding_step(
        self, input_dir: Path, output_dir: Path, device: str
    ) -> Dict[str, Any]:
        """
        Execute embedding extraction for preprocessed tensors and collect step results.

        Parameters:
            input_dir (Path): Directory containing preprocessed tensor files (*.pt).
            output_dir (Path): Base directory where an "embeddings" subdirectory will be created for outputs.
            device (str): Compute device to use (e.g., "cpu", "cuda:0", or "auto" to keep extractor default).

        Returns:
            Dict[str, Any]: A result dictionary with the following keys:
                - step: "embedding"
                - success: `true` if at least one tensor was processed successfully, `false` otherwise
                - processing_time: elapsed time in seconds for the step
                - total_tensors: number of tensor files discovered under `input_dir`
                - processed_tensors: number of tensors successfully processed
                - failed_tensors: number of tensors that failed processing
                - embedding_vectors: list of extracted embedding objects/vectors
                - output_dir: Path to the embeddings output directory
                - errors: list of error messages encountered during the step
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
        original_device = getattr(self.extractor, "device", None)
        original_model = getattr(self.extractor, "model", None)
        device_overridden = False

        try:
            # Create embedding output directory
            embedding_results["output_dir"].mkdir(parents=True, exist_ok=True)

            # Set device for extractor
            if device != "auto":
                target_device = torch.device(device)
                self.extractor.device = target_device
                self.extractor.model = self.extractor.model.to(target_device)
                device_overridden = True

            # Find raw tensor files, excluding embedding outputs from previous runs.
            tensor_files = [
                path
                for path in input_dir.rglob("*.pt")
                if not path.name.endswith(".embedding.pt")
            ]
            embedding_results["total_tensors"] = len(tensor_files)

            if not tensor_files:
                message = f"No tensor files found in {input_dir}"
                logger.warning(message)
                embedding_results["errors"].append(message)
                embedding_results["processing_time"] = time.time() - start_time
                return embedding_results

            typer.echo(f"Found {len(tensor_files)} tensor files")

            extractor_device = getattr(self.extractor, "device", device)
            device_type = (
                extractor_device.type
                if isinstance(extractor_device, torch.device)
                else str(extractor_device).split(":", maxsplit=1)[0]
            )
            max_workers = max(1, int(self.max_workers or 1))
            if device_type == "cuda":
                max_workers = min(max_workers, 2)

            # Process tensor files in parallel
            results = self._process_embedding_files_parallel(
                tensor_files,
                embedding_results["output_dir"],
                step_input_dir=input_dir,
                max_workers=max_workers,
            )

            # Aggregate results
            for result in results:
                if result is not None and result["success"]:
                    embedding_results["processed_tensors"] += 1
                    embedding_results["embedding_vectors"].append(
                        result["embedding_vector"]
                    )
                else:
                    embedding_results["failed_tensors"] += 1
                    if result is not None and "error" in result:
                        embedding_results["errors"].append(result["error"])

            embedding_results["success"] = embedding_results["processed_tensors"] > 0
            embedding_results["processing_time"] = time.time() - start_time

            typer.echo(
                f"Embedding extraction completed: {embedding_results['processed_tensors']}/{embedding_results['total_tensors']} tensors processed"
            )

        except Exception as e:
            logger.error(f"Error in embedding step: {e!s}")
            embedding_results["errors"].append(str(e))
            embedding_results["processing_time"] = time.time() - start_time
        finally:
            if (
                device_overridden
                and original_device is not None
                and original_model is not None
            ):
                self.extractor.model = original_model.to(original_device)
                self.extractor.device = original_device

        return embedding_results

    def _process_single_embedding_file(
        self, file_path: Path, output_dir: Path, step_input_dir: Path | None = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract an embedding vector from a preprocessed tensor file and persist the embedding to disk.

        Returns:
            dict: A result dictionary with a `success` boolean. On success the dictionary contains
            `embedding_vector` (the extracted embedding). On failure the dictionary contains
            `error` (a human-readable error message).
        """
        try:
            # Load preprocessed tensor
            preprocessed_tensor = self._load_preprocessed_tensor(file_path)
            if preprocessed_tensor is None:
                return {
                    "success": False,
                    "error": f"Failed to load {file_path}",
                }

            # Extract embedding
            embedding_vector = self.extractor.extract_embedding(preprocessed_tensor)
            if embedding_vector is None:
                return {
                    "success": False,
                    "error": f"Failed to extract embedding from {file_path}",
                }

            # Save embedding
            output_path = self._get_embedding_output_path(
                file_path, output_dir, step_input_dir
            )
            if not self._save_embedding_vector(embedding_vector, output_path):
                return {
                    "success": False,
                    "error": f"Failed to save embedding for {file_path}",
                }

            return {
                "success": True,
                "embedding_vector": embedding_vector,
            }

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e!s}")
            return {
                "success": False,
                "error": f"Error processing {file_path}: {e!s}",
            }

    def _process_embedding_files_parallel(
        self,
        tensor_files: List[Path],
        output_dir: Path,
        max_workers: int | None = None,
        step_input_dir: Path | None = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Process tensor files concurrently to extract embeddings and collect per-file results in the original input order.

        Parameters:
            tensor_files (List[Path]): Paths to preprocessed tensor files to process.
            output_dir (Path): Directory where per-file embedding outputs will be written.
            max_workers (int | None): Optional worker count override for this embedding pass.

        Returns:
            List[Optional[Dict[str, Any]]]: A list with the same length and order as `tensor_files`. Each element is either
            the result dictionary produced for that file (containing at minimum a `"success"` boolean and optionally
            an `"error"` string) or `None` if the corresponding task returned `None`.
        """
        results: List[Optional[Dict[str, Any]]] = [None] * len(tensor_files)
        worker_count = max(1, int(max_workers or self.max_workers or 1))

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(
                    self._process_single_embedding_file,
                    file_path,
                    output_dir,
                    step_input_dir,
                ): i
                for i, file_path in enumerate(tensor_files)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    logger.error(
                        "Error processing tensor file %s: %s",
                        tensor_files[index],
                        exc,
                    )
                    results[index] = {
                        "success": False,
                        "error": f"Error processing {tensor_files[index]}: {exc!s}",
                    }

        return results

    def _run_clustering_step(self, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Run clustering on embedding vectors found in the input directory and save the clustering result into the output directory.

        Parameters:
            input_dir (Path): Directory containing saved embedding vectors to load.
            output_dir (Path): Base directory where a `clustering` subdirectory and the clustering result file will be created.

        Returns:
            dict: Result summary with keys:
                - "step" (str): Step name ("clustering").
                - "success" (bool): `True` if clustering completed and result was saved, `False` otherwise.
                - "processing_time" (float): Elapsed time in seconds for the step.
                - "clustering_result" (ClusteringResult or None): The clustering result object when successful; otherwise `None`.
                - "output_dir" (Path): Path to the created clustering output directory.
                - "errors" (List[str]): List of error messages encountered during the step.
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
                message = f"No embedding vectors found in {input_dir}"
                logger.warning(message)
                clustering_results["errors"].append(message)
                clustering_results["success"] = False
                clustering_results["processing_time"] = time.time() - start_time
                return clustering_results

            typer.echo(f"Loaded {len(embedding_vectors)} embedding vectors")

            # Perform clustering
            clustering_result = self.clusterer.cluster_embeddings(embedding_vectors)
            if clustering_result is None:
                clustering_results["errors"].append("Clustering failed")
                clustering_results["success"] = False
                clustering_results["processing_time"] = time.time() - start_time
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

            valid_labels = clustering_result.cluster_labels[
                clustering_result.cluster_labels != -1
            ]
            n_clusters = int(torch.unique(valid_labels, dim=0).numel())
            typer.echo(f"Clustering completed: {n_clusters} clusters found")

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
        Evaluate clustering results and write an evaluation report to the output directory.

        Parameters:
            clustering_result (ClusteringResult): Clustering output to evaluate.
            embedding_vectors (List[EmbeddingVector]): Embedding vectors used for evaluation.
            output_dir (Path): Base directory where `evaluation` subdirectory and results file will be written.

        Returns:
            Dict[str, Any]: Summary of the evaluation step containing:
                - `step`: step name ("evaluation"),
                - `success`: `True` if evaluation and save succeeded, `False` otherwise,
                - `processing_time`: elapsed time in seconds,
                - `evaluation_result`: evaluator output (when successful),
                - `output_dir`: path to the `evaluation` output directory,
                - `errors`: list of error messages encountered.
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
        Create visualizations for clustering results and save them under the provided output directory.

        Parameters:
            clustering_result (ClusteringResult): Clustering outcome containing cluster labels and metadata.
            embedding_vectors (List[EmbeddingVector]): Embedding vectors that correspond to the clustered items.
            mammography_images (List[MammographyImage]): Original mammography images used to generate visualizations.
            output_dir (Path): Base directory where a "visualizations" subdirectory will be created and outputs written.

        Returns:
            result (Dict[str, Any]): A dictionary containing:
                - "step": the step name ("visualization")
                - "success": `true` if visualizations were created, `false` otherwise
                - "processing_time": elapsed time in seconds
                - "visualization_result": the object returned by the visualizer (or `None` on failure)
                - "output_dir": Path to the created "visualizations" directory
                - "errors": list of error messages encountered during the step
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
