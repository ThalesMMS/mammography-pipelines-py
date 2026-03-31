#
# report_exporter.py
# mammography-pipelines
#
# Report export utility for packaging results from model training runs.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""Report export component for packaging training results and visualizations."""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mammography.vis.export import (
    export_figure,
    export_training_curves,
    export_confusion_matrix,
)

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - optional dependency
    mlflow = None
    MlflowClient = None


LOGGER = logging.getLogger("mammography")


def _require_streamlit() -> None:
    """Raise ImportError if Streamlit is not available."""
    if st is None:
        raise ImportError(
            "Streamlit is required to run the web UI dashboard."
        ) from _STREAMLIT_IMPORT_ERROR


@dataclass
class ExportManifest:
    """Manifest describing exported report contents.

    Attributes:
        run_id: MLflow run ID or local run directory name
        export_dir: Directory where files were exported
        exported_files: List of exported file paths (relative to export_dir)
        missing_files: List of expected but missing files
        generated_at: ISO timestamp of export generation
        metadata: Additional metadata about the run
    """

    run_id: str
    export_dir: str
    exported_files: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "export_dir": self.export_dir,
            "exported_files": self.exported_files,
            "missing_files": self.missing_files,
            "generated_at": self.generated_at,
            "metadata": self.metadata,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save manifest to JSON file.

        Args:
            path: Output path for manifest JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.info(f"Saved export manifest to {path}")


class ReportExporter:
    """Component for exporting training results as publication-ready report packages.

    This component packages training artifacts into a structured export directory
    including:
    - Training curves (loss, accuracy)
    - Confusion matrices
    - ROC curves
    - Metrics summary (JSON)
    - Model checkpoint (optional)
    - Run metadata and manifest

    Supports exporting from:
    - MLflow runs (via run_id)
    - Local directories (via run_dir)

    Example:
        >>> exporter = ReportExporter()
        >>> manifest = exporter.export_from_mlflow(
        ...     run_id="abc123",
        ...     output_dir="reports/run_001",
        ...     formats=["png", "pdf"]
        ... )
        >>> exporter.create_zip_archive("reports/run_001", "reports/run_001.zip")
    """

    def __init__(self) -> None:
        """Initialize the report exporter."""
        self._client: Optional[MlflowClient] = None

    def _get_mlflow_client(self) -> MlflowClient:
        """Get or create MLflow client."""
        if mlflow is None:
            raise ImportError("MLflow is required for exporting from MLflow runs")

        if self._client is None:
            self._client = MlflowClient()
        return self._client

    def export_from_mlflow(
        self,
        run_id: str,
        output_dir: Union[str, Path],
        formats: Optional[List[str]] = None,
        include_checkpoint: bool = False,
    ) -> ExportManifest:
        """Export report from MLflow run.

        Args:
            run_id: MLflow run ID to export
            output_dir: Directory to save exported files
            formats: Image formats for figures (default: ["png", "pdf"])
            include_checkpoint: Whether to include model checkpoint in export

        Returns:
            ExportManifest describing exported contents

        Raises:
            ImportError: If MLflow is not installed
            ValueError: If run_id is not found
        """
        if formats is None:
            formats = ["png", "pdf"]

        client = self._get_mlflow_client()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get run info
        try:
            run = client.get_run(run_id)
        except Exception as e:
            raise ValueError(f"Failed to get MLflow run {run_id}: {e}") from e

        manifest = ExportManifest(
            run_id=run_id,
            export_dir=str(output_path),
            metadata={
                "experiment_id": run.info.experiment_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "status": run.info.status,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
            },
        )

        # Export artifacts
        exported_files = []
        missing_files = []

        # Download artifacts from MLflow
        artifacts = ["confusion_matrix.npy", "val_metrics.json", "train_history.json"]
        if include_checkpoint:
            artifacts.append("checkpoint.pth")

        for artifact in artifacts:
            try:
                artifact_path = client.download_artifacts(run_id, artifact, dst_path=str(output_path))
                if Path(artifact_path).exists():
                    exported_files.append(artifact)
                    LOGGER.info(f"Downloaded artifact: {artifact}")
            except Exception as e:
                LOGGER.warning(f"Failed to download {artifact}: {e}")
                missing_files.append(artifact)

        # Generate visualizations if metrics are available
        metrics_path = output_path / "val_metrics.json"
        history_path = output_path / "train_history.json"

        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
                train_losses = history.get("train_loss", [])
                val_losses = history.get("val_loss", [])
                train_accs = history.get("train_acc", [])
                val_accs = history.get("val_acc", [])

                # Export training curves
                curve_paths = export_training_curves(
                    train_losses=train_losses,
                    val_losses=val_losses,
                    train_accs=train_accs if train_accs else None,
                    val_accs=val_accs if val_accs else None,
                    base_path=output_path / "training_curves",
                    formats=formats,
                    title="Training and Validation Curves",
                )
                exported_files.extend([p.name for p in curve_paths])
            except Exception as e:
                LOGGER.warning(f"Failed to generate training curves: {e}")
                missing_files.append("training_curves")

        # Generate confusion matrix if available
        cm_path = output_path / "confusion_matrix.npy"
        if cm_path.exists():
            try:
                cm = np.load(cm_path)
                class_names = ["A", "B", "C", "D"]  # BI-RADS density classes

                # Export normalized and unnormalized versions
                cm_paths = export_confusion_matrix(
                    cm=cm,
                    class_names=class_names,
                    base_path=output_path / "confusion_matrix",
                    formats=formats,
                    title="Confusion Matrix",
                    normalize=True,
                )
                exported_files.extend([p.name for p in cm_paths])

                cm_paths_raw = export_confusion_matrix(
                    cm=cm,
                    class_names=class_names,
                    base_path=output_path / "confusion_matrix_raw",
                    formats=formats,
                    title="Confusion Matrix (Raw Counts)",
                    normalize=False,
                )
                exported_files.extend([p.name for p in cm_paths_raw])
            except Exception as e:
                LOGGER.warning(f"Failed to generate confusion matrix: {e}")
                missing_files.append("confusion_matrix")

        # Save manifest
        manifest.exported_files = exported_files
        manifest.missing_files = missing_files
        manifest_path = output_path / "export_manifest.json"
        manifest.save(manifest_path)

        return manifest

    def export_from_directory(
        self,
        run_dir: Union[str, Path],
        output_dir: Union[str, Path],
        formats: Optional[List[str]] = None,
        include_checkpoint: bool = False,
    ) -> ExportManifest:
        """Export report from local run directory.

        Args:
            run_dir: Local directory containing training artifacts
            output_dir: Directory to save exported files
            formats: Image formats for figures (default: ["png", "pdf"])
            include_checkpoint: Whether to include model checkpoint in export

        Returns:
            ExportManifest describing exported contents

        Raises:
            FileNotFoundError: If run_dir doesn't exist
        """
        if formats is None:
            formats = ["png", "pdf"]

        run_path = Path(run_dir)
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        manifest = ExportManifest(
            run_id=run_path.name,
            export_dir=str(output_path),
            metadata={"source_dir": str(run_path)},
        )

        exported_files = []
        missing_files = []

        # Copy artifacts from run directory
        artifacts = {
            "val_metrics.json": run_path / "val_metrics.json",
            "train_history.json": run_path / "train_history.json",
            "summary.json": run_path / "summary.json",
            "confusion_matrix.npy": run_path / "confusion_matrix.npy",
        }

        if include_checkpoint:
            artifacts["checkpoint.pth"] = run_path / "checkpoint.pth"

        for artifact_name, artifact_path in artifacts.items():
            if artifact_path.exists():
                dest_path = output_path / artifact_name
                shutil.copy2(artifact_path, dest_path)
                exported_files.append(artifact_name)
                LOGGER.info(f"Copied artifact: {artifact_name}")
            else:
                missing_files.append(artifact_name)
                LOGGER.warning(f"Missing artifact: {artifact_name}")

        # Generate visualizations
        history_path = output_path / "train_history.json"
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
                train_losses = history.get("train_loss", [])
                val_losses = history.get("val_loss", [])
                train_accs = history.get("train_acc", [])
                val_accs = history.get("val_acc", [])

                curve_paths = export_training_curves(
                    train_losses=train_losses,
                    val_losses=val_losses,
                    train_accs=train_accs if train_accs else None,
                    val_accs=val_accs if val_accs else None,
                    base_path=output_path / "training_curves",
                    formats=formats,
                    title="Training and Validation Curves",
                )
                exported_files.extend([p.name for p in curve_paths])
            except Exception as e:
                LOGGER.warning(f"Failed to generate training curves: {e}")
                missing_files.append("training_curves")

        cm_path = output_path / "confusion_matrix.npy"
        if cm_path.exists():
            try:
                cm = np.load(cm_path)
                class_names = ["A", "B", "C", "D"]

                cm_paths = export_confusion_matrix(
                    cm=cm,
                    class_names=class_names,
                    base_path=output_path / "confusion_matrix",
                    formats=formats,
                    title="Confusion Matrix",
                    normalize=True,
                )
                exported_files.extend([p.name for p in cm_paths])

                cm_paths_raw = export_confusion_matrix(
                    cm=cm,
                    class_names=class_names,
                    base_path=output_path / "confusion_matrix_raw",
                    formats=formats,
                    title="Confusion Matrix (Raw Counts)",
                    normalize=False,
                )
                exported_files.extend([p.name for p in cm_paths_raw])
            except Exception as e:
                LOGGER.warning(f"Failed to generate confusion matrix: {e}")
                missing_files.append("confusion_matrix")

        # Save manifest
        manifest.exported_files = exported_files
        manifest.missing_files = missing_files
        manifest_path = output_path / "export_manifest.json"
        manifest.save(manifest_path)

        return manifest

    @staticmethod
    def create_zip_archive(
        source_dir: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Path:
        """Create ZIP archive of exported report directory.

        Args:
            source_dir: Directory to archive
            output_path: Output path for ZIP file

        Returns:
            Path to created ZIP archive

        Raises:
            FileNotFoundError: If source_dir doesn't exist
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_path.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_path)
                    zipf.write(file_path, arcname)
                    LOGGER.debug(f"Added to archive: {arcname}")

        LOGGER.info(f"Created ZIP archive: {output_file}")
        return output_file

    def render_export_ui(
        self,
        run_id: Optional[str] = None,
        run_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Render export UI in Streamlit.

        DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
        It must NOT be used for clinical or medical diagnostic purposes.
        No medical decision should be based on these results.

        Args:
            run_id: Optional MLflow run ID (for MLflow export)
            run_dir: Optional local run directory (for local export)

        Raises:
            ImportError: If Streamlit is not available
            ValueError: If neither run_id nor run_dir is provided
        """
        _require_streamlit()

        if run_id is None and run_dir is None:
            raise ValueError("Either run_id or run_dir must be provided")

        st.subheader("📦 Export Report Package")
        st.markdown(
            "Generate a publication-ready report package with training curves, "
            "confusion matrices, and metrics in multiple formats."
        )

        # Export configuration
        col1, col2 = st.columns(2)
        with col1:
            formats = st.multiselect(
                "Export Formats",
                options=["png", "pdf", "svg"],
                default=["png", "pdf"],
                help="Select image formats for exported figures",
            )

        with col2:
            include_checkpoint = st.checkbox(
                "Include Model Checkpoint",
                value=False,
                help="Include trained model weights in export (increases package size)",
            )

        # Output directory configuration
        default_output = f"reports/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = st.text_input(
            "Output Directory",
            value=default_output,
            help="Directory where exported files will be saved",
        )

        # Export button
        if st.button("🚀 Generate Report Package", type="primary"):
            with st.spinner("Generating report package..."):
                try:
                    # Export based on source type
                    if run_id is not None:
                        manifest = self.export_from_mlflow(
                            run_id=run_id,
                            output_dir=output_dir,
                            formats=formats,
                            include_checkpoint=include_checkpoint,
                        )
                        st.success(f"✅ Exported MLflow run {run_id}")
                    else:
                        manifest = self.export_from_directory(
                            run_dir=run_dir,  # type: ignore[arg-type]
                            output_dir=output_dir,
                            formats=formats,
                            include_checkpoint=include_checkpoint,
                        )
                        st.success(f"✅ Exported local run from {run_dir}")

                    # Display summary
                    st.markdown("### Export Summary")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Exported Files", len(manifest.exported_files))
                    with col2:
                        st.metric("Missing Files", len(manifest.missing_files))

                    # Show file list
                    if manifest.exported_files:
                        with st.expander("📄 Exported Files", expanded=True):
                            for file in sorted(manifest.exported_files):
                                st.text(f"✓ {file}")

                    if manifest.missing_files:
                        with st.expander("⚠️ Missing Files"):
                            for file in sorted(manifest.missing_files):
                                st.text(f"✗ {file}")

                    # Create ZIP archive
                    st.markdown("### Create Archive")
                    if st.button("📦 Create ZIP Archive"):
                        zip_path = Path(output_dir).with_suffix(".zip")
                        zip_file = self.create_zip_archive(output_dir, zip_path)

                        # Provide download button
                        with open(zip_file, "rb") as f:
                            st.download_button(
                                label="⬇️ Download ZIP Archive",
                                data=f.read(),
                                file_name=zip_file.name,
                                mime="application/zip",
                            )
                        st.success(f"✅ Created archive: {zip_file.name}")

                except Exception as e:
                    st.error(f"❌ Export failed: {e}")
                    LOGGER.exception("Export failed")
