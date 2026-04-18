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


class PipelineStorageMixin:
    def _get_preprocessing_output_path(
        self, input_path: Path, output_dir: Path, step_input_dir: Path | None = None
    ) -> Path:
        """
        Compute the destination file path for a preprocessed tensor derived from an input file.

        The returned path preserves the input file's directory structure relative to the step input directory, places it under output_dir, and uses a `.pt` suffix. The function ensures the parent directory for the returned path exists.

        Parameters:
            input_path (Path): Original input file path used to derive a relative structure.
            output_dir (Path): Root directory where the preprocessed tensor file should be placed.
            step_input_dir (Path | None): Root directory for the current pipeline step.

        Returns:
            Path: Filesystem path where the preprocessed tensor should be saved (with `.pt` suffix).
        """
        root = step_input_dir or input_path.parent
        try:
            relative_path = input_path.relative_to(root)
        except ValueError:
            logger.warning(
                "Input path %s is not under step input dir %s; using filename only.",
                input_path,
                root,
            )
            relative_path = Path(input_path.name)
        output_path = output_dir / relative_path.with_suffix(".pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _get_embedding_output_path(
        self, input_path: Path, output_dir: Path, step_input_dir: Path | None = None
    ) -> Path:
        """
        Compute the output file path for an embedding by mirroring the input path structure under the given output directory.

        Parameters:
            input_path (Path): Original input file path used to derive a relative path fragment.
            output_dir (Path): Base directory where the embedding file will be placed.
            step_input_dir (Path | None): Root directory for the current pipeline step.

        Returns:
            Path: File path ending with `.embedding.pt` located under `output_dir`; the parent directories are created if they do not exist.
        """
        root = step_input_dir or input_path.parent
        try:
            relative_path = input_path.relative_to(root)
        except ValueError:
            logger.warning(
                "Input path %s is not under step input dir %s; using filename only.",
                input_path,
                root,
            )
            relative_path = Path(input_path.name)
        output_path = output_dir / relative_path.with_suffix(".embedding.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _save_preprocessed_tensor(
        self, preprocessed_tensor: PreprocessedTensor, output_path: Path
    ) -> bool:
        """
        Persist a preprocessed tensor to disk and write its associated YAML metadata alongside it.

        Parameters:
            preprocessed_tensor (PreprocessedTensor): The tensor data and associated attributes to persist.
            output_path (Path): Filesystem path where the tensor will be saved; a metadata file is created at the same path with the module's `METADATA_SUFFIX`.

        Returns:
            bool: `True` if the tensor and metadata were written successfully, `False` otherwise.
        """
        try:
            torch.save(preprocessed_tensor.tensor_data, output_path)

            # Save metadata
            metadata_path = output_path.with_suffix(METADATA_SUFFIX)
            metadata = {
                "image_id": preprocessed_tensor.image_id,
                "preprocessing_config": preprocessed_tensor.preprocessing_config,
                "normalization_method": preprocessed_tensor.normalization_method,
                "target_size": list(preprocessed_tensor.target_size),
                "input_adapter": preprocessed_tensor.input_adapter,
                "border_removed": preprocessed_tensor.border_removed,
                "original_shape": (
                    list(preprocessed_tensor.original_shape)
                    if preprocessed_tensor.original_shape is not None
                    else None
                ),
                "processing_time": preprocessed_tensor.processing_time,
                "created_at": preprocessed_tensor.created_at.isoformat(),
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
        """
        Write an embedding tensor, a NumPy sidecar, and a YAML metadata file for an EmbeddingVector to the filesystem.

        Parameters:
            embedding_vector (EmbeddingVector): Object containing the embedding tensor and metadata fields
                (image_id, model_config, input_adapter, extraction_time, device_used, created_at).
            output_path (Path): Destination path for the primary embedding file (saved with torch.save).
                A `.npy` sidecar and a metadata file with suffix `.metadata.yaml` will be written alongside it.

        Returns:
            bool: `true` if all files were written successfully, `false` otherwise.
        """
        try:
            torch.save(embedding_vector.embedding, output_path)
            np.save(
                output_path.with_suffix(".npy"),
                embedding_vector.embedding.detach().cpu().numpy(),
            )

            # Save metadata
            metadata_path = output_path.with_suffix(METADATA_SUFFIX)
            metadata = {
                "image_id": embedding_vector.image_id,
                "embedding_dimension": embedding_vector.embedding.shape[0],
                "model_config": embedding_vector.model_config,
                "input_adapter": embedding_vector.input_adapter,
                "extraction_time": embedding_vector.extraction_time,
                "device_used": embedding_vector.device_used,
                "created_at": embedding_vector.created_at.isoformat(),
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
        """
        Reconstruct a PreprocessedTensor by loading tensor data and optional YAML metadata from disk.

        When present, metadata from file_path.with_suffix(METADATA_SUFFIX) is used to populate fields such as image_id, preprocessing_config, normalization_method, target_size, input_adapter, border_removed, original_shape, processing_time, and created_at; missing metadata fields are filled with sensible defaults.

        Parameters:
            file_path (Path): Path to the saved tensor file to load.

        Returns:
            PreprocessedTensor or None: The reconstructed PreprocessedTensor on success, `None` if loading fails.
        """
        try:
            tensor_data = torch.load(file_path, map_location="cpu", weights_only=True)

            metadata_path = file_path.with_suffix(METADATA_SUFFIX)
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = yaml.safe_load(f)
            else:
                metadata = {}

            preprocessed_tensor = PreprocessedTensor(
                image_id=metadata.get("image_id", file_path.stem),
                tensor_data=tensor_data,
                preprocessing_config=metadata.get("preprocessing_config", {}),
                normalization_method=metadata.get(
                    "normalization_method", "z_score_per_image"
                ),
                target_size=tuple(metadata.get("target_size", tensor_data.shape[-2:])),
                input_adapter=metadata.get("input_adapter", "1to3_replication"),
                border_removed=bool(metadata.get("border_removed", False)),
                original_shape=(
                    tuple(metadata["original_shape"])
                    if metadata.get("original_shape") is not None
                    else None
                ),
                processing_time=float(metadata.get("processing_time", 0.0) or 0.0),
                created_at=datetime.fromisoformat(
                    metadata.get("created_at", datetime.now().isoformat())
                ),
            )

            return preprocessed_tensor

        except Exception as e:
            logger.error(f"Error loading preprocessed tensor from {file_path}: {e!s}")
            return None

    def _load_embedding_vectors(self, input_dir: Path) -> List[EmbeddingVector]:
        """
        Collect embedding vectors from an input directory and reconstruct their metadata-backed representations.

        This searches recursively for files matching "*.embedding.pt", loads each embedding tensor, and uses a same-named YAML sidecar (if present) to populate metadata fields; files that fail to load are skipped and logged.

        Returns:
            embedding_vectors (List[EmbeddingVector]): List of successfully loaded EmbeddingVector objects. Files that could not be read or parsed are omitted.
        """
        embedding_vectors = []

        try:
            embedding_files = list(input_dir.rglob("*.embedding.pt"))

            for file_path in embedding_files:
                try:
                    embedding_data = torch.load(
                        file_path, map_location="cpu", weights_only=True
                    )

                    metadata_path = file_path.with_suffix(METADATA_SUFFIX)
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            metadata = yaml.safe_load(f)
                    else:
                        metadata = {}

                    embedding_vector = EmbeddingVector(
                        image_id=metadata.get("image_id", file_path.stem),
                        embedding=embedding_data,
                        model_config=metadata.get(
                            "model_config", EmbeddingVector.default_model_config()
                        ),
                        input_adapter=metadata.get("input_adapter", "1to3_replication"),
                        extraction_time=float(
                            metadata.get("extraction_time", 0.0) or 0.0
                        ),
                        device_used=metadata.get("device_used", "cpu"),
                        created_at=datetime.fromisoformat(
                            metadata.get("created_at", datetime.now().isoformat())
                        ),
                    )

                    embedding_vectors.append(embedding_vector)

                except Exception as e:
                    logger.error(f"Error loading embedding from {file_path}: {e!s}")
                    continue

        except Exception as e:
            logger.error(f"Error loading embedding vectors: {e!s}")

        return embedding_vectors
