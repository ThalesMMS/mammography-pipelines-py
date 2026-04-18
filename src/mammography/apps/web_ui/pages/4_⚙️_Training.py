#
# 4_⚙️_Training.py
# mammography-pipelines
#
# Streamlit page for configuring and launching training jobs with custom hyperparameters.
#
# Thales Matheus Mendonca Santos - February 2026
#
"""Training configuration page for launching new training jobs."""

from __future__ import annotations

import shlex
import time
from typing import Dict, Any, Optional

from mammography.apps.web_ui.pages.training import config_sections as training_config_sections
from mammography.apps.web_ui.pages.training import data_loading as training_data_loading
from mammography.apps.web_ui.pages.training import live_metrics as training_live_metrics
from mammography.apps.web_ui.pages.training import mlflow_polling as training_mlflow_polling
from mammography.apps.web_ui.pages.training import process as training_process
from mammography.apps.web_ui.pages.training import state as training_state
from mammography.apps.web_ui.pages.training import tracking as training_tracking
from mammography.apps.web_ui.utils import ensure_shared_session_state

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None

try:
    from mlflow.tracking import MlflowClient
except Exception as exc:  # pragma: no cover - optional MLflow dependency
    MlflowClient = None
    _MLFLOW_IMPORT_ERROR = exc
else:
    _MLFLOW_IMPORT_ERROR = None


def _require_streamlit() -> None:
    """Raise ImportError if Streamlit is not available."""
    if st is None:
        raise ImportError(
            "Streamlit is required to run the web UI dashboard."
        ) from _STREAMLIT_IMPORT_ERROR


def _get_mlflow_client(tracking_uri: str) -> Optional[MlflowClient]:
    """Get or create MLflow client with the specified tracking URI."""
    return training_mlflow_polling.get_mlflow_client(tracking_uri, st.session_state)


def _poll_mlflow_metrics() -> bool:
    """Poll MLflow for the latest metrics from the active run.

    Returns:
        True if metrics were successfully fetched, False otherwise.
    """
    return training_mlflow_polling.poll_mlflow_metrics(st.session_state)


def _sync_shared_state() -> None:
    """Sync values written by the background thread back into session state.

    The ``stream_output`` thread writes to a plain ``dict`` (``_training_shared``)
    instead of ``st.session_state`` to avoid "missing ScriptRunContext" warnings.
    This helper copies those values back on each main-thread rerun.
    """
    training_state.sync_shared_state(st.session_state)


def _ensure_session_defaults() -> None:
    """Initialize session state with default values."""
    training_state.ensure_session_defaults(st.session_state)


def _render_data_section() -> Dict[str, Any]:
    """Render data configuration section and return parameters."""
    return training_config_sections.render_data_section()

def _render_model_section() -> Dict[str, Any]:
    """Render model configuration section and return parameters."""
    return training_config_sections.render_model_section()

def _render_training_section() -> Dict[str, Any]:
    """Render training hyperparameters section and return parameters."""
    return training_config_sections.render_training_section()

def _render_optimization_section() -> Dict[str, Any]:
    """Render optimization settings section and return parameters."""
    return training_config_sections.render_optimization_section()

def _render_data_loading_section() -> Dict[str, Any]:
    """Render data loading settings section and return parameters."""
    return training_data_loading.render_data_loading_section()

def _render_class_balancing_section() -> Dict[str, Any]:
    """Render class balancing settings section and return parameters."""
    return training_data_loading.render_class_balancing_section()

def _render_augmentation_section() -> Dict[str, Any]:
    """Render augmentation settings section and return parameters."""
    return training_data_loading.render_augmentation_section()

def _render_analysis_section() -> Dict[str, Any]:
    """Render analysis and outputs section and return parameters."""
    return training_tracking.render_analysis_section()

def _render_tracking_section() -> Dict[str, Any]:
    """Render experiment tracking section and return parameters."""
    return training_tracking.render_tracking_section()

def _render_live_metrics_section() -> None:
    """Render live training metrics monitoring section."""
    training_live_metrics.render_live_metrics_section()

def _launch_training(command: list[str]) -> None:
    """Launch training process in background and capture output."""
    training_process.launch_training(
        command,
        session_state=st.session_state,
        get_mlflow_client=_get_mlflow_client,
    )


def _stop_training() -> None:
    """Stop the running training process."""
    training_process.stop_training(st.session_state)


def _build_command_line(config: Dict[str, Any]) -> list[str]:
    """Build the command-line argv list from configuration."""
    return training_process.build_command_line(config)


def main() -> None:
    """Render the training configuration page."""
    _require_streamlit()

    st.set_page_config(
        page_title="Training - Mammography Pipelines",
        page_icon="⚙️",
        layout="wide",
    )

    # Initialize shared session state for cross-page data persistence
    try:
        ensure_shared_session_state()
        _ensure_session_defaults()
        _sync_shared_state()
    except Exception as exc:
        st.error(f"❌ Failed to initialize session state: {exc}")
        st.stop()

    st.title("⚙️ Training Configuration")

    st.header("Configure Training Job")

    st.markdown("""
    Configure hyperparameters for training breast density classification models.
    All settings are organized into logical sections below. When you're done,
    copy the generated command to run training in your terminal.
    """)

    st.info(
        "💡 **Quick Start:** Configure your training parameters using the sections below, "
        "then click 'Generate Command' to get the CLI command for training, or use the "
        "'Launch Training' button to start training directly in the background."
    )

    # Collect all configuration sections
    config = {}

    with st.form("training_config_form"):
        config.update(_render_data_section())
        st.markdown("---")

        config.update(_render_model_section())
        st.markdown("---")

        config.update(_render_training_section())
        st.markdown("---")

        config.update(_render_optimization_section())
        st.markdown("---")

        config.update(_render_data_loading_section())
        st.markdown("---")

        config.update(_render_class_balancing_section())
        st.markdown("---")

        config.update(_render_augmentation_section())
        st.markdown("---")

        config.update(_render_analysis_section())
        st.markdown("---")

        config.update(_render_tracking_section())

        st.markdown("---")

        # Submit button
        submitted = st.form_submit_button("🚀 Generate Command", type="primary")

        if submitted:
            st.session_state.training_config = config

    # Display generated command
    if st.session_state.training_config:
        st.header("Generated Command")

        st.markdown("""
        Copy the command below to run training in your terminal, or use the Launch Training
        button to start training directly from the web interface.
        """)

        command = _build_command_line(st.session_state.training_config)

        st.code(shlex.join(command), language="bash")

        # Training control buttons
        col1, col2, col3 = st.columns([1, 1, 3])

        with col1:
            if st.session_state.training_status == "running":
                if st.button("⏹️ Stop Training", type="secondary"):
                    _stop_training()
                    st.rerun()
            else:
                if st.button("🚀 Launch Training", type="primary"):
                    _launch_training(command)
                    st.rerun()

        with col2:
            if st.session_state.training_status != "idle":
                status_emoji = {
                    "running": "🔄",
                    "completed": "✅",
                    "failed": "❌",
                }
                st.markdown(
                    f"**Status:** {status_emoji.get(st.session_state.training_status, '❓')} "
                    f"{st.session_state.training_status.title()}"
                )

        # Show training output
        if st.session_state.training_output:
            st.subheader("Training Output")

            # Auto-refresh if training is running
            if st.session_state.training_status == "running":
                st.markdown("🔄 *Auto-refreshing every 2 seconds...*")
                time.sleep(2)
                st.rerun()

            # Display output in a scrollable container
            output_text = "\n".join(st.session_state.training_output[-100:])  # Show last 100 lines
            st.text_area(
                "Output Log",
                value=output_text,
                height=400,
                disabled=True,
                label_visibility="collapsed",
            )

            # Show line count
            st.caption(f"Showing last {min(100, len(st.session_state.training_output))} of {len(st.session_state.training_output)} lines")

        # Live metrics monitoring section
        st.markdown("---")
        _render_live_metrics_section()

        # Configuration summary
        with st.expander("📋 Configuration Summary"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Key Settings")
                st.metric("Architecture", st.session_state.training_config.get("arch", "efficientnet_b0"))
                st.metric("Task", st.session_state.training_config.get("classes", "multiclass"))
                st.metric("Epochs", st.session_state.training_config.get("epochs", 100))
                st.metric("Batch Size", st.session_state.training_config.get("batch_size", 16))

            with col2:
                st.subheader("Optimization")
                st.metric("Learning Rate", f"{st.session_state.training_config.get('lr', 1e-4):.1e}")
                st.metric("Backbone LR", f"{st.session_state.training_config.get('backbone_lr', 1e-5):.1e}")
                st.metric("Weight Decay", f"{st.session_state.training_config.get('weight_decay', 1e-4):.1e}")
                st.metric("Scheduler", st.session_state.training_config.get("scheduler", "auto"))

    # Help section
    with st.expander("ℹ️ Help & Documentation"):
        st.markdown("""
        ### How to Use This Page

        1. **Configure Each Section**: Fill in the configuration options for each section
        2. **Review Settings**: Check your configuration in each section
        3. **Generate Command**: Click "Generate Command" to create the CLI command
        4. **Launch Training**: Either:
           - Click "Launch Training" to run training directly in the web interface
           - Copy the generated command and run it manually in your terminal

        ### Configuration Sections

        - **Data Configuration**: Dataset selection, paths, caching options
        - **Model Configuration**: Architecture, task type, pretrained weights
        - **Training Hyperparameters**: Epochs, batch size, learning rates
        - **Optimization Settings**: Backbone training, performance options, schedulers
        - **Data Loading**: Worker processes, prefetching, caching
        - **Class Balancing**: Class weights and weighted sampling
        - **Data Augmentation**: Image transformations for training
        - **Analysis & Outputs**: GradCAM, predictions export, profiling
        - **Experiment Tracking**: MLflow or Weights & Biases integration

        ### Dataset Presets

        - **archive**: DICOMs in `archive/` + `classificacao.csv`
        - **mamografias**: PNGs in subfolders with `featureS.txt`
        - **patches_completo**: PNGs at root with `featureS.txt`

        ### Model Architectures

        - **efficientnet_b0**: Efficient and lightweight CNN
        - **resnet50**: Deep residual network with 50 layers

        ### Task Types

        - **multiclass**: BI-RADS density classification (A, B, C, D or 1-4)
        - **binary**: Binary classification (A/B vs C/D)

        ### Learning Rate Schedulers

        - **auto**: Automatically selects appropriate scheduler
        - **none**: No learning rate scheduling
        - **plateau**: Reduce LR when validation metric plateaus
        - **cosine**: Cosine annealing schedule
        - **step**: Step decay at fixed intervals

        ### Cache Modes

        - **auto**: Automatically selects best caching strategy
        - **none**: No caching (load from disk each time)
        - **memory**: Cache decoded images in RAM
        - **disk**: Cache preprocessed images on disk
        - **tensor-disk**: Cache preprocessed tensors on disk
        - **tensor-memmap**: Memory-mapped tensor caching

        ### Tips for Better Training

        1. **Start Small**: Use `--subset` to test with a small number of samples first
        2. **Use Pretrained Weights**: Enable pretrained weights for faster convergence
        3. **Monitor with MLflow**: Enable experiment tracking to visualize training progress
        4. **Enable AMP**: Use automatic mixed precision on CUDA/MPS for faster training
        5. **Adjust Batch Size**: Increase batch size if you have enough GPU memory
        6. **Use Early Stopping**: Prevent overfitting with early stopping patience
        7. **Enable Augmentation**: Data augmentation improves generalization
        8. **Balance Classes**: Use class weights or weighted sampling for imbalanced datasets

        ### Troubleshooting

        - **Out of Memory**: Reduce batch size or image size
        - **Slow Training**: Increase num_workers, enable AMP, or use tensor caching
        - **Overfitting**: Enable augmentation, increase weight decay, or use early stopping
        - **Poor Convergence**: Adjust learning rate, use warmup epochs, or change scheduler
        - **Class Imbalance**: Enable class weights or weighted sampling

        ### Command Line Execution

        After generating the command:

        ```bash
        # Copy the command and run it in your terminal
        python -m mammography.cli train-density --dataset mamografias --epochs 10

        # Or save to a script file
        # save_script.sh:
        #!/bin/bash
        python -m mammography.cli train-density \\
            --dataset mamografias \\
            --epochs 100 \\
            --tracker mlflow

        # Make executable and run
        chmod +x save_script.sh
        ./save_script.sh
        ```

        ### References

        - [CLAUDE.md](https://github.com/yourusername/mammography-pipelines/blob/main/CLAUDE.md)
        - [CLI cheatsheet](https://github.com/yourusername/mammography-pipelines/blob/main/docs/guides/cli-cheatsheet.md)
        - [README.md](https://github.com/yourusername/mammography-pipelines/blob/main/README.md)
        """)


if __name__ == "__main__":
    main()
