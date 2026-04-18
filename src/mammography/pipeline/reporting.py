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


class PipelineReportingMixin:
    def _generate_final_report(
        self, pipeline_results: Dict[str, Any], output_dir: Path
    ) -> Dict[str, Any]:
        """
        Create a timestamped YAML report from pipeline_results, write it to output_dir, and return metadata about the report generation.

        Parameters:
            pipeline_results (Dict[str, Any]): Aggregated pipeline execution data used to build the report.
            output_dir (Path): Directory where the generated YAML report file will be written.

        Returns:
            Dict[str, Any]: Metadata about the report generation containing:
                - step (str): The step name, `"report"`.
                - success (bool): `true` if the report was written successfully, `false` otherwise.
                - processing_time (float): Time in seconds spent generating and writing the report.
                - report_path (str | None): Path to the written report file when successful, otherwise `None`.
                - errors (List[str]): List of error messages encountered during report generation (empty on success).
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
        Builds a structured report payload summarizing pipeline execution and per-step results.

        Parameters:
            pipeline_results (Dict[str, Any]): Pipeline execution data containing at least
                `pipeline_timestamp`, `input_dir`, `output_dir`, `device`, `success`,
                `total_processing_time`, `errors`, and `step_results`. Each entry in
                `step_results` must include `success` and `processing_time`; it may also
                include step-specific fields:
                  - preprocessing: `total_files`, `processed_files`, `failed_files`
                  - embedding: `total_tensors`, `processed_tensors`, `failed_tensors`
                  - clustering: `clustering_result` with `algorithm`, `cluster_labels`, and `metrics`

        Returns:
            Dict[str, Any]: Report content with keys:
              - `pipeline_summary`: summary metadata (timestamp, input/output dirs, device, overall success, total processing time)
              - `step_results`: mapping of step names to per-step summaries (success, processing_time, errors, plus step-specific metrics). Clustering summaries include `algorithm`, `n_clusters`, `n_samples`, and `metrics`.
              - `errors`: list of pipeline-level errors
        """
        report_content = {
            "pipeline_summary": {
                "timestamp": pipeline_results["pipeline_timestamp"],
                "input_dir": pipeline_results["input_dir"],
                "output_dir": pipeline_results["output_dir"],
                "device": pipeline_results["device"],
                "success": pipeline_results.get("success", False),
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
                cluster_labels = clustering_result.cluster_labels
                if isinstance(cluster_labels, torch.Tensor):
                    valid_labels = cluster_labels[cluster_labels != -1]
                    n_clusters = int(torch.unique(valid_labels).numel())
                else:
                    n_clusters = len({label for label in cluster_labels if label != -1})
                report_content["step_results"][step_name].update(
                    {
                        "algorithm": clustering_result.algorithm,
                        "n_clusters": n_clusters,
                        "n_samples": len(clustering_result.cluster_labels),
                        "metrics": clustering_result.metrics,
                    }
                )

        return report_content
