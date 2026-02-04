"""
Interactive dashboard components for mammography analysis pipeline.

This module provides comprehensive dashboard components for interactive
visualization, real-time metrics monitoring, and cluster exploration
in the breast density exploration pipeline.

DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- Interactive dashboards enable real-time exploration of analysis results
- Real-time metrics monitoring provides immediate feedback on processing
- Cluster exploration tools facilitate understanding of clustering results
- Dashboard components support both research and educational objectives

Author: Research Team
Version: 1.0.0
"""

from datetime import datetime
import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional dashboard deps
    px = None
    go = None
    make_subplots = None
    st = None
    _DASHBOARD_IMPORT_ERROR = exc
else:
    _DASHBOARD_IMPORT_ERROR = None

from ..clustering.clustering_result import ClusteringResult

# Import pipeline components

# Configure logging
logger = logging.getLogger(__name__)

# Dashboard configuration
DASHBOARD_CONFIG = {
    "title": "Mammography Analysis Dashboard",
    "subtitle": "Interactive Breast Density Exploration Pipeline",
    "disclaimer": "DISCLAIMER: This is an EDUCATIONAL RESEARCH project. It must NOT be used for clinical or medical diagnostic purposes.",
    "layout": "wide",
    "theme": "light",
}


def _require_dashboard_deps() -> None:
    if px is None or go is None or st is None:
        raise ImportError(
            "Dashboard dependencies missing. Install plotly and streamlit to use this module."
        ) from _DASHBOARD_IMPORT_ERROR


class MetricsMonitor:
    """
    Real-time metrics monitoring for dashboard.

    This class provides real-time monitoring of processing metrics,
    performance indicators, and system status for the dashboard.

    Educational Notes:
    - Real-time monitoring enables immediate feedback on processing status
    - Performance metrics help optimize processing parameters
    - System status monitoring ensures reliable operation
    - Metrics visualization supports understanding of processing efficiency
    """

    def __init__(self):
        """Initialize metrics monitor."""
        self.metrics = {
            "processing_time": 0.0,
            "memory_usage": 0.0,
            "gpu_usage": 0.0,
            "files_processed": 0,
            "clusters_found": 0,
            "silhouette_score": 0.0,
            "davies_bouldin_score": 0.0,
            "calinski_harabasz_score": 0.0,
        }
        self.history = []
        self.start_time = time.time()

    def update_metrics(self, new_metrics: Dict[str, Any]):
        """Update metrics with new values."""
        self.metrics.update(new_metrics)
        self.metrics["timestamp"] = datetime.now()
        self.history.append(self.metrics.copy())

        # Keep only last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()

    def get_metrics_history(self) -> pd.DataFrame:
        """Get metrics history as DataFrame."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)


class ClusterExplorer:
    """
    Interactive cluster exploration tools.

    This class provides interactive tools for exploring clustering results,
    including cluster visualization, prototype selection, and analysis tools.

    Educational Notes:
    - Cluster exploration enables understanding of clustering results
    - Interactive tools facilitate discovery of patterns and relationships
    - Prototype selection helps identify representative samples
    - Analysis tools support interpretation of clustering quality
    """

    def __init__(self, clustering_result: ClusteringResult):
        """Initialize cluster explorer with clustering results."""
        self.clustering_result = clustering_result
        self.selected_clusters = []
        self.selected_samples = []

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary of clustering results."""
        labels = self.clustering_result.cluster_labels.numpy()
        unique_labels = np.unique(labels)

        summary = {
            "total_samples": len(labels),
            "num_clusters": len(unique_labels),
            "cluster_sizes": {
                int(label): int(np.sum(labels == label)) for label in unique_labels
            },
            "algorithm": self.clustering_result.algorithm,
            "metrics": self.clustering_result.metrics,
        }

        return summary

    def get_cluster_prototypes(
        self, cluster_id: int, n_prototypes: int = 4
    ) -> List[int]:
        """Get prototype samples for a cluster."""
        labels = self.clustering_result.cluster_labels.numpy()
        cluster_indices = np.where(labels == cluster_id)[0]

        if len(cluster_indices) <= n_prototypes:
            return cluster_indices.tolist()

        # Select diverse prototypes (simplified selection)
        step = len(cluster_indices) // n_prototypes
        prototypes = cluster_indices[::step][:n_prototypes]

        return prototypes.tolist()

    def filter_clusters(self, cluster_ids: List[int]) -> Dict[str, Any]:
        """Filter data by selected clusters."""
        labels = self.clustering_result.cluster_labels.numpy()
        mask = np.isin(labels, cluster_ids)

        return {
            "filtered_indices": np.where(mask)[0],
            "filtered_labels": labels[mask],
            "num_samples": np.sum(mask),
        }


class InteractiveVisualizer:
    """
    Interactive visualization components for dashboard.

    This class provides interactive visualization components including
    UMAP plots, cluster montages, metrics plots, and real-time monitoring.

    Educational Notes:
    - Interactive visualizations enable exploration of high-dimensional data
    - Real-time updates provide immediate feedback on processing
    - Multiple visualization types support different analysis needs
    - Interactive features facilitate discovery and understanding
    """

    def __init__(self):
        """Initialize interactive visualizer."""
        _require_dashboard_deps()
        self.plot_config = {
            "width": 800,
            "height": 600,
            "template": "plotly_white",
            "color_scale": "viridis",
        }

    def create_umap_plot(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[List[Dict]] = None,
    ) -> go.Figure:
        """Create interactive UMAP plot."""
        # Create DataFrame for easy manipulation
        df = pd.DataFrame(
            {
                "x": embeddings[:, 0],
                "y": embeddings[:, 1],
                "cluster": labels,
                "sample_id": range(len(embeddings)),
            }
        )

        # Add metadata if available
        if metadata:
            for key in metadata[0].keys():
                df[key] = [meta[key] for meta in metadata]

        # Create interactive scatter plot
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster",
            hover_data=["sample_id"],
            title="Interactive UMAP Visualization",
            width=self.plot_config["width"],
            height=self.plot_config["height"],
        )

        # Customize layout
        fig.update_layout(
            template=self.plot_config["template"],
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
        )

        return fig

    def create_metrics_plot(self, metrics: Dict[str, Any]) -> go.Figure:
        """Create metrics visualization plot."""
        # Extract metrics
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        # Create bar plot
        fig = go.Figure(
            data=[go.Bar(x=metric_names, y=metric_values, marker_color="lightblue")]
        )

        fig.update_layout(
            title="Clustering Quality Metrics",
            xaxis_title="Metrics",
            yaxis_title="Values",
            template=self.plot_config["template"],
            width=self.plot_config["width"],
            height=self.plot_config["height"],
        )

        return fig

    def create_cluster_size_plot(self, cluster_sizes: Dict[int, int]) -> go.Figure:
        """Create cluster size distribution plot."""
        clusters = list(cluster_sizes.keys())
        sizes = list(cluster_sizes.values())

        fig = go.Figure(data=[go.Bar(x=clusters, y=sizes, marker_color="lightcoral")])

        fig.update_layout(
            title="Cluster Size Distribution",
            xaxis_title="Cluster ID",
            yaxis_title="Number of Samples",
            template=self.plot_config["template"],
            width=self.plot_config["width"],
            height=self.plot_config["height"],
        )

        return fig

    def create_metrics_timeline(self, metrics_history: pd.DataFrame) -> go.Figure:
        """Create metrics timeline plot."""
        if metrics_history.empty:
            return go.Figure()

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Processing Time",
                "Memory Usage",
                "Silhouette Score",
                "Files Processed",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Add traces
        if "processing_time" in metrics_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_history.index,
                    y=metrics_history["processing_time"],
                    name="Processing Time",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )

        if "memory_usage" in metrics_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_history.index,
                    y=metrics_history["memory_usage"],
                    name="Memory Usage",
                    line=dict(color="green"),
                ),
                row=1,
                col=2,
            )

        if "silhouette_score" in metrics_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_history.index,
                    y=metrics_history["silhouette_score"],
                    name="Silhouette Score",
                    line=dict(color="red"),
                ),
                row=2,
                col=1,
            )

        if "files_processed" in metrics_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_history.index,
                    y=metrics_history["files_processed"],
                    name="Files Processed",
                    line=dict(color="orange"),
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title="Real-time Metrics Timeline",
            template=self.plot_config["template"],
            width=self.plot_config["width"],
            height=self.plot_config["height"] * 1.5,
        )

        return fig


class DashboardApp:
    """
    Main dashboard application for mammography analysis.

    This class provides the main dashboard application with interactive
    components, real-time monitoring, and cluster exploration tools.

    Educational Notes:
    - Dashboard applications provide user-friendly interfaces for complex analysis
    - Real-time monitoring enables immediate feedback on processing
    - Interactive components facilitate exploration and understanding
    - Educational context helps users understand the analysis process
    """

    def __init__(self):
        """Initialize dashboard application."""
        _require_dashboard_deps()
        self.metrics_monitor = MetricsMonitor()
        self.visualizer = InteractiveVisualizer()
        self.cluster_explorer = None
        self.current_data = None

    def setup_page_config(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title=DASHBOARD_CONFIG["title"],
            page_icon="M",
            layout=DASHBOARD_CONFIG["layout"],
            initial_sidebar_state="expanded",
        )

    def render_header(self):
        """Render dashboard header."""
        st.title(DASHBOARD_CONFIG["title"])
        st.subtitle(DASHBOARD_CONFIG["subtitle"])

        # Display disclaimer
        st.warning(DASHBOARD_CONFIG["disclaimer"])

        # Display current time
        st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def render_sidebar(self):
        """Render dashboard sidebar."""
        st.sidebar.header("Dashboard Controls")

        # Data upload section
        st.sidebar.subheader("Data Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Clustering Results",
            type=["json", "pkl"],
            help="Upload clustering results file",
        )

        # Processing controls
        st.sidebar.subheader("Processing Controls")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 60, 5)

        # Visualization controls
        st.sidebar.subheader("Visualization Controls")
        plot_type = st.sidebar.selectbox("Plot Type", ["UMAP", "PCA", "t-SNE"], index=0)

        color_by = st.sidebar.selectbox(
            "Color By", ["Cluster", "Projection", "Laterality"], index=0
        )

        return {
            "uploaded_file": uploaded_file,
            "auto_refresh": auto_refresh,
            "refresh_interval": refresh_interval,
            "plot_type": plot_type,
            "color_by": color_by,
        }

    def render_metrics_overview(self):
        """Render metrics overview section."""
        st.header("Real-time Metrics Overview")

        # Get current metrics
        metrics = self.metrics_monitor.get_current_metrics()

        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Files Processed",
                value=metrics.get("files_processed", 0),
                delta=None,
            )

        with col2:
            st.metric(
                label="Clusters Found",
                value=metrics.get("clusters_found", 0),
                delta=None,
            )

        with col3:
            st.metric(
                label="Silhouette Score",
                value=f"{metrics.get('silhouette_score', 0):.3f}",
                delta=None,
            )

        with col4:
            st.metric(
                label="Processing Time",
                value=f"{metrics.get('processing_time', 0):.2f}s",
                delta=None,
            )

    def render_cluster_analysis(self):
        """Render cluster analysis section."""
        st.header("Cluster Analysis")

        if self.cluster_explorer is None:
            st.info("Please upload clustering results to view cluster analysis.")
            return

        # Get cluster summary
        summary = self.cluster_explorer.get_cluster_summary()

        # Display cluster summary
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cluster Summary")
            st.write(f"**Total Samples:** {summary['total_samples']}")
            st.write(f"**Number of Clusters:** {summary['num_clusters']}")
            st.write(f"**Algorithm:** {summary['algorithm']}")

        with col2:
            st.subheader("Cluster Sizes")
            cluster_sizes = summary["cluster_sizes"]
            for cluster_id, size in cluster_sizes.items():
                st.write(f"Cluster {cluster_id}: {size} samples")

        # Cluster selection
        st.subheader("Cluster Selection")
        selected_clusters = st.multiselect(
            "Select Clusters to Analyze",
            options=list(cluster_sizes.keys()),
            default=list(cluster_sizes.keys())[:2] if len(cluster_sizes) > 1 else [],
        )

        if selected_clusters:
            # Filter data by selected clusters
            filtered_data = self.cluster_explorer.filter_clusters(selected_clusters)
            st.write(f"**Selected Samples:** {filtered_data['num_samples']}")

            # Display cluster prototypes
            st.subheader("Cluster Prototypes")
            for cluster_id in selected_clusters:
                prototypes = self.cluster_explorer.get_cluster_prototypes(cluster_id)
                st.write(f"**Cluster {cluster_id} Prototypes:** {prototypes}")

    def render_visualizations(self, controls: Dict[str, Any]):
        """Render visualization section."""
        st.header("Interactive Visualizations")

        if self.current_data is None:
            st.info("Please upload data to view visualizations.")
            return

        # Create visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["UMAP Plot", "Metrics Plot", "Cluster Sizes", "Timeline"]
        )

        with tab1:
            # UMAP visualization
            if "embeddings" in self.current_data and "labels" in self.current_data:
                embeddings = self.current_data["embeddings"]
                labels = self.current_data["labels"]
                metadata = self.current_data.get("metadata", None)

                fig = self.visualizer.create_umap_plot(embeddings, labels, metadata)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No embedding data available for UMAP visualization.")

        with tab2:
            # Metrics visualization
            if self.cluster_explorer:
                summary = self.cluster_explorer.get_cluster_summary()
                metrics = summary.get("metrics", {})

                if metrics:
                    fig = self.visualizer.create_metrics_plot(metrics)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No metrics data available.")
            else:
                st.info("No clustering results available for metrics visualization.")

        with tab3:
            # Cluster sizes visualization
            if self.cluster_explorer:
                summary = self.cluster_explorer.get_cluster_summary()
                cluster_sizes = summary.get("cluster_sizes", {})

                if cluster_sizes:
                    fig = self.visualizer.create_cluster_size_plot(cluster_sizes)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No cluster size data available.")
            else:
                st.info(
                    "No clustering results available for cluster size visualization."
                )

        with tab4:
            # Metrics timeline
            metrics_history = self.metrics_monitor.get_metrics_history()

            if not metrics_history.empty:
                fig = self.visualizer.create_metrics_timeline(metrics_history)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No metrics history available for timeline visualization.")

    def render_export_section(self):
        """Render export section."""
        st.header("Export Results")

        col1, col2 = st.columns(2)

        with col1:
            # Export clustering results
            if self.cluster_explorer:
                if st.button("Export Clustering Results"):
                    # Create export data
                    export_data = {
                        "clustering_result": self.cluster_explorer.clustering_result.to_dict(),
                        "cluster_summary": self.cluster_explorer.get_cluster_summary(),
                        "export_timestamp": datetime.now().isoformat(),
                    }

                    # Convert to JSON
                    json_data = json.dumps(export_data, indent=2, default=str)

                    # Create download button
                    st.download_button(
                        label="Download Clustering Results",
                        data=json_data,
                        file_name=f"clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )

        with col2:
            # Export metrics
            metrics_history = self.metrics_monitor.get_metrics_history()

            if not metrics_history.empty:
                if st.button("Export Metrics History"):
                    # Convert to CSV
                    csv_data = metrics_history.to_csv(index=False)

                    # Create download button
                    st.download_button(
                        label="Download Metrics History",
                        data=csv_data,
                        file_name=f"metrics_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

    def load_data(self, uploaded_file) -> bool:
        """Load data from uploaded file."""
        if uploaded_file is None:
            return False

        try:
            # Load data based on file type
            if uploaded_file.name.endswith(".json"):
                data = json.load(uploaded_file)
            elif uploaded_file.name.endswith(".pkl"):
                import pickle

                data = pickle.load(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload JSON or PKL files.")
                return False

            # Process loaded data
            self.current_data = data

            # Initialize cluster explorer if clustering results are available
            if "clustering_result" in data:
                # Create ClusteringResult object from data
                clustering_result = ClusteringResult.from_dict(
                    data["clustering_result"]
                )
                self.cluster_explorer = ClusterExplorer(clustering_result)

            # Update metrics
            if "metrics" in data:
                self.metrics_monitor.update_metrics(data["metrics"])

            st.success("Data loaded successfully!")
            return True

        except Exception as e:
            st.error(f"Error loading data: {e!s}")
            return False

    def run_dashboard(self):
        """Run the main dashboard application."""
        # Setup page configuration
        self.setup_page_config()

        # Render header
        self.render_header()

        # Render sidebar and get controls
        controls = self.render_sidebar()

        # Load data if file uploaded
        if controls["uploaded_file"] is not None:
            self.load_data(controls["uploaded_file"])

        # Render main sections
        self.render_metrics_overview()
        self.render_cluster_analysis()
        self.render_visualizations(controls)
        self.render_export_section()

        # Auto refresh if enabled
        if controls["auto_refresh"]:
            time.sleep(controls["refresh_interval"])
            st.rerun()


def create_dashboard_app() -> DashboardApp:
    """Create and configure dashboard application."""
    return DashboardApp()


def run_dashboard():
    """Run the dashboard application."""
    app = create_dashboard_app()
    app.run_dashboard()


if __name__ == "__main__":
    # Run dashboard
    run_dashboard()
