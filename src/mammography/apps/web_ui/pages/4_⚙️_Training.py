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

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

from mammography.apps.web_ui.utils import ensure_shared_session_state

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None


def _require_streamlit() -> None:
    """Raise ImportError if Streamlit is not available."""
    if st is None:
        raise ImportError(
            "Streamlit is required to run the web UI dashboard."
        ) from _STREAMLIT_IMPORT_ERROR


def _ensure_session_defaults() -> None:
    """Initialize session state with default values."""
    if "training_config" not in st.session_state:
        st.session_state.training_config = {}
    if "training_process" not in st.session_state:
        st.session_state.training_process = None
    if "training_output" not in st.session_state:
        st.session_state.training_output = []
    if "training_status" not in st.session_state:
        st.session_state.training_status = "idle"  # idle, running, completed, failed


def _render_data_section() -> Dict[str, Any]:
    """Render data configuration section and return parameters."""
    st.subheader("📂 Data Configuration")

    config = {}

    col1, col2 = st.columns(2)

    with col1:
        config["dataset"] = st.selectbox(
            "Dataset Preset",
            options=["", "archive", "mamografias", "patches_completo"],
            help="Preset dataset configuration (archive/mamografias/patches_completo)",
        )

        config["csv"] = st.text_input(
            "CSV Path",
            value="",
            help="Path to CSV file, directory with featureS.txt, or manual path",
        )

        config["dicom_root"] = st.text_input(
            "DICOM Root Path",
            value="",
            help="Root directory for DICOM files (used with classificacao.csv)",
        )

        config["outdir"] = st.text_input(
            "Output Directory",
            value="outputs/run",
            help="Directory for saving training outputs",
        )

    with col2:
        config["cache_mode"] = st.selectbox(
            "Cache Mode",
            options=["auto", "none", "memory", "disk", "tensor-disk", "tensor-memmap"],
            help="Data caching strategy for faster loading",
        )

        config["cache_dir"] = st.text_input(
            "Cache Directory",
            value="",
            help="Directory for cached data (optional)",
        )

        config["embeddings_dir"] = st.text_input(
            "Embeddings Directory",
            value="",
            help="Directory with features.npy + metadata.csv (optional)",
        )

        config["log_level"] = st.selectbox(
            "Log Level",
            options=["critical", "error", "warning", "info", "debug"],
            index=3,  # default to "info"
            help="Logging verbosity level",
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        config["subset"] = st.number_input(
            "Subset Size",
            min_value=0,
            value=0,
            step=1,
            help="Limit number of samples (0 = use all)",
        )

    with col2:
        config["include_class_5"] = st.checkbox(
            "Include Class 5",
            value=False,
            help="Keep samples with classification 5 when loading classificacao.csv",
        )

    with col3:
        config["val_frac"] = st.number_input(
            "Validation Fraction",
            min_value=0.01,
            max_value=0.99,
            value=0.20,
            step=0.05,
            help="Fraction of data to use for validation",
        )

    return config


def _render_model_section() -> Dict[str, Any]:
    """Render model configuration section and return parameters."""
    st.subheader("🏗️ Model Configuration")

    config = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        config["arch"] = st.selectbox(
            "Architecture",
            options=["efficientnet_b0", "resnet50"],
            help="Model architecture to use",
        )

    with col2:
        config["classes"] = st.selectbox(
            "Task",
            options=["density", "binary", "multiclass"],
            help="Classification task: density (BI-RADS 1-4), binary (A/B vs C/D), multiclass",
        )

    with col3:
        config["pretrained"] = st.checkbox(
            "Use Pretrained Weights",
            value=True,
            help="Initialize with ImageNet pretrained weights when available",
        )

    # View-specific training
    with st.expander("🔍 View-Specific Training"):
        config["view_specific_training"] = st.checkbox(
            "Enable View-Specific Training",
            value=False,
            help="Train separate models for each view (CC/MLO)",
        )

        if config["view_specific_training"]:
            config["views"] = st.text_input(
                "Views to Train",
                value="CC,MLO",
                help="Comma-separated list of views to train",
            )

            config["ensemble_method"] = st.selectbox(
                "Ensemble Method",
                options=["none", "average", "weighted", "max"],
                help="Method for combining predictions from different views",
            )
        else:
            config["views"] = ""
            config["ensemble_method"] = "none"

    return config


def _render_training_section() -> Dict[str, Any]:
    """Render training hyperparameters section and return parameters."""
    st.subheader("🎯 Training Hyperparameters")

    config = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        config["epochs"] = st.number_input(
            "Epochs",
            min_value=1,
            value=100,
            step=1,
            help="Number of training epochs",
        )

        config["batch_size"] = st.number_input(
            "Batch Size",
            min_value=1,
            value=16,
            step=1,
            help="Training batch size",
        )

        config["lr"] = st.number_input(
            "Learning Rate",
            min_value=1e-7,
            max_value=1.0,
            value=1e-4,
            format="%.1e",
            help="Learning rate for the classification head",
        )

    with col2:
        config["backbone_lr"] = st.number_input(
            "Backbone Learning Rate",
            min_value=1e-7,
            max_value=1.0,
            value=1e-5,
            format="%.1e",
            help="Learning rate for the backbone (if trainable)",
        )

        config["weight_decay"] = st.number_input(
            "Weight Decay",
            min_value=0.0,
            max_value=1.0,
            value=1e-4,
            format="%.1e",
            help="L2 regularization weight decay",
        )

        config["img_size"] = st.number_input(
            "Image Size",
            min_value=32,
            max_value=1024,
            value=512,
            step=32,
            help="Input image size (will be resized to this)",
        )

    with col3:
        config["seed"] = st.number_input(
            "Random Seed",
            min_value=0,
            value=42,
            step=1,
            help="Random seed for reproducibility",
        )

        config["device"] = st.selectbox(
            "Device",
            options=["auto", "cpu", "cuda", "mps"],
            help="Compute device (auto will select the best available)",
        )

        config["amp"] = st.checkbox(
            "Enable AMP",
            value=False,
            help="Enable Automatic Mixed Precision for faster training on CUDA/MPS",
        )

    return config


def _render_optimization_section() -> Dict[str, Any]:
    """Render optimization settings section and return parameters."""
    st.subheader("⚡ Optimization Settings")

    config = {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Backbone Training**")

        config["train_backbone"] = st.checkbox(
            "Train Backbone",
            value=False,
            help="Train the backbone feature extractor",
        )

        config["unfreeze_last_block"] = st.checkbox(
            "Unfreeze Last Block",
            value=True,
            help="Unfreeze the last block of the backbone",
        )

        config["warmup_epochs"] = st.number_input(
            "Warmup Epochs",
            min_value=0,
            value=0,
            step=1,
            help="Number of warmup epochs before full training",
        )

    with col2:
        st.markdown("**Performance Options**")

        config["deterministic"] = st.checkbox(
            "Deterministic Mode",
            value=False,
            help="Enable deterministic algorithms (may be slower)",
        )

        config["allow_tf32"] = st.checkbox(
            "Allow TF32",
            value=True,
            help="Allow TensorFloat-32 for faster computation on Ampere GPUs",
        )

        config["fused_optim"] = st.checkbox(
            "Fused Optimizer",
            value=False,
            help="Use fused AdamW optimizer on CUDA (faster but more memory)",
        )

        config["torch_compile"] = st.checkbox(
            "Torch Compile",
            value=False,
            help="Optimize model with torch.compile (PyTorch 2.0+)",
        )

    # Learning rate scheduling
    with st.expander("📉 Learning Rate Scheduling"):
        config["scheduler"] = st.selectbox(
            "Scheduler Type",
            options=["auto", "none", "plateau", "cosine", "step"],
            help="Learning rate scheduler strategy",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            config["lr_reduce_patience"] = st.number_input(
                "LR Reduce Patience",
                min_value=0,
                value=0,
                step=1,
                help="Epochs to wait before reducing LR on plateau (0 = disabled)",
            )

            config["lr_reduce_factor"] = st.number_input(
                "LR Reduce Factor",
                min_value=0.01,
                max_value=0.99,
                value=0.5,
                step=0.05,
                help="Factor by which to reduce learning rate",
            )

        with col2:
            config["lr_reduce_min_lr"] = st.number_input(
                "Min Learning Rate",
                min_value=1e-10,
                max_value=1e-3,
                value=1e-7,
                format="%.1e",
                help="Minimum learning rate threshold",
            )

            config["lr_reduce_cooldown"] = st.number_input(
                "LR Reduce Cooldown",
                min_value=0,
                value=0,
                step=1,
                help="Epochs to wait before resuming normal operation after LR reduction",
            )

        with col3:
            config["scheduler_step_size"] = st.number_input(
                "Step Size",
                min_value=1,
                value=5,
                step=1,
                help="Step size for step scheduler",
            )

            config["scheduler_gamma"] = st.number_input(
                "Gamma",
                min_value=0.01,
                max_value=0.99,
                value=0.5,
                step=0.05,
                help="Multiplicative factor for step scheduler",
            )

    # Early stopping
    with st.expander("⏹️ Early Stopping"):
        config["early_stop_patience"] = st.number_input(
            "Early Stop Patience",
            min_value=0,
            value=0,
            step=1,
            help="Epochs to wait for improvement before stopping (0 = disabled)",
        )

        config["early_stop_min_delta"] = st.number_input(
            "Min Delta",
            min_value=0.0,
            value=0.0,
            step=0.0001,
            format="%.4f",
            help="Minimum change to qualify as improvement",
        )

    return config


def _render_data_loading_section() -> Dict[str, Any]:
    """Render data loading settings section and return parameters."""
    st.subheader("💾 Data Loading")

    config = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        config["num_workers"] = st.number_input(
            "Number of Workers",
            min_value=0,
            value=4,
            step=1,
            help="Number of worker processes for data loading (0 = main process)",
        )

    with col2:
        config["prefetch_factor"] = st.number_input(
            "Prefetch Factor",
            min_value=0,
            value=4,
            step=1,
            help="Number of batches to prefetch per worker",
        )

    with col3:
        config["persistent_workers"] = st.checkbox(
            "Persistent Workers",
            value=True,
            help="Keep workers alive between epochs",
        )

    config["loader_heuristics"] = st.checkbox(
        "Loader Heuristics",
        value=True,
        help="Apply automatic heuristics for optimal data loading",
    )

    return config


def _render_class_balancing_section() -> Dict[str, Any]:
    """Render class balancing settings section and return parameters."""
    st.subheader("⚖️ Class Balancing")

    config = {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Class Weights**")

        config["class_weights"] = st.selectbox(
            "Class Weights Mode",
            options=["none", "auto"],
            help="How to weight classes in loss function (none/auto)",
        )

        config["class_weights_alpha"] = st.number_input(
            "Class Weights Alpha",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Exponent for auto class weights calculation",
        )

    with col2:
        st.markdown("**Weighted Sampling**")

        config["sampler_weighted"] = st.checkbox(
            "Weighted Sampler",
            value=False,
            help="Use weighted random sampling to balance classes",
        )

        config["sampler_alpha"] = st.number_input(
            "Sampler Alpha",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Exponent for weighted sampler calculation",
        )

    return config


def _render_augmentation_section() -> Dict[str, Any]:
    """Render augmentation settings section and return parameters."""
    st.subheader("🎨 Data Augmentation")

    config = {}

    config["augment"] = st.checkbox(
        "Enable Augmentation",
        value=True,
        help="Enable data augmentation during training",
    )

    if config["augment"]:
        col1, col2, col3 = st.columns(3)

        with col1:
            config["augment_vertical"] = st.checkbox(
                "Vertical Flip",
                value=False,
                help="Enable random vertical flips",
            )

        with col2:
            config["augment_color"] = st.checkbox(
                "Color Jitter",
                value=False,
                help="Enable random brightness/contrast adjustment",
            )

        with col3:
            config["augment_rotation_deg"] = st.number_input(
                "Rotation (degrees)",
                min_value=0.0,
                max_value=45.0,
                value=5.0,
                step=1.0,
                help="Maximum rotation angle for random rotation",
            )
    else:
        config["augment_vertical"] = False
        config["augment_color"] = False
        config["augment_rotation_deg"] = 0.0

    return config


def _render_analysis_section() -> Dict[str, Any]:
    """Render analysis and outputs section and return parameters."""
    st.subheader("📊 Analysis & Outputs")

    config = {}

    col1, col2 = st.columns(2)

    with col1:
        config["gradcam"] = st.checkbox(
            "Generate GradCAM",
            value=False,
            help="Generate GradCAM visualizations during validation",
        )

        if config["gradcam"]:
            config["gradcam_limit"] = st.number_input(
                "GradCAM Limit",
                min_value=1,
                value=4,
                step=1,
                help="Maximum number of GradCAM visualizations to generate",
            )
        else:
            config["gradcam_limit"] = 4

    with col2:
        config["save_val_preds"] = st.checkbox(
            "Save Validation Predictions",
            value=False,
            help="Save validation predictions to CSV",
        )

        config["export_val_embeddings"] = st.checkbox(
            "Export Validation Embeddings",
            value=False,
            help="Export validation embeddings for analysis",
        )

    # Profiling
    with st.expander("🔍 Profiling"):
        config["profile"] = st.checkbox(
            "Enable Profiling",
            value=False,
            help="Enable PyTorch profiler in first epoch",
        )

        if config["profile"]:
            config["profile_dir"] = st.text_input(
                "Profile Output Directory",
                value="outputs/profiler",
                help="Directory for saving profiler traces",
            )
        else:
            config["profile_dir"] = "outputs/profiler"

    return config


def _render_tracking_section() -> Dict[str, Any]:
    """Render experiment tracking section and return parameters."""
    st.subheader("📈 Experiment Tracking")

    config = {}

    config["tracker"] = st.selectbox(
        "Tracking Backend",
        options=["none", "mlflow", "wandb"],
        help="Experiment tracking backend (none/mlflow/wandb)",
    )

    if config["tracker"] != "none":
        col1, col2 = st.columns(2)

        with col1:
            config["tracker_project"] = st.text_input(
                "Project/Experiment Name",
                value="",
                help="Name of the project or experiment for tracking",
            )

        with col2:
            config["tracker_run_name"] = st.text_input(
                "Run Name",
                value="",
                help="Optional name for this specific run",
            )

        if config["tracker"] == "mlflow":
            config["tracker_uri"] = st.text_input(
                "MLflow Tracking URI",
                value="./mlruns",
                help="MLflow tracking server URI or local path",
            )
        else:
            config["tracker_uri"] = ""
    else:
        config["tracker_project"] = ""
        config["tracker_run_name"] = ""
        config["tracker_uri"] = ""

    return config


def _launch_training(command: str) -> None:
    """Launch training process in background and capture output."""
    st.session_state.training_output = []
    st.session_state.training_status = "running"

    # Convert command to list for subprocess
    # Handle multi-line command by removing backslashes and extra whitespace
    clean_command = command.replace("\\\n", " ").replace("  ", " ").strip()

    try:
        # Start the process
        process = subprocess.Popen(
            clean_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        st.session_state.training_process = process

        # Stream output in a separate thread
        def stream_output():
            """Stream process output to session state."""
            try:
                for line in process.stdout:
                    st.session_state.training_output.append(line.rstrip())
                    if len(st.session_state.training_output) > 1000:
                        # Keep only last 1000 lines
                        st.session_state.training_output = st.session_state.training_output[-1000:]

                # Wait for process to complete
                process.wait()

                if process.returncode == 0:
                    st.session_state.training_status = "completed"
                    st.session_state.training_output.append("\n✅ Training completed successfully!")
                else:
                    st.session_state.training_status = "failed"
                    st.session_state.training_output.append(
                        f"\n❌ Training failed with exit code {process.returncode}"
                    )
            except Exception as e:
                st.session_state.training_status = "failed"
                st.session_state.training_output.append(f"\n❌ Error during training: {e}")
                st.session_state.training_output.append(
                    "\n💡 Check the output above for error details. Common issues:\n"
                    "- Dataset path not found or inaccessible\n"
                    "- Insufficient memory or disk space\n"
                    "- Missing dependencies or incompatible versions\n"
                    "- Invalid hyperparameter values"
                )

        # Start output streaming thread
        output_thread = threading.Thread(target=stream_output, daemon=True)
        output_thread.start()

    except Exception as e:
        st.session_state.training_status = "failed"
        st.session_state.training_output.append(f"❌ Failed to start training: {e}")
        st.session_state.training_output.append(
            "\n💡 This may happen if:\n"
            "- The mammography command is not in your PATH\n"
            "- Python environment is not activated\n"
            "- The command syntax is malformed\n"
            "- System resources are exhausted"
        )


def _stop_training() -> None:
    """Stop the running training process."""
    if st.session_state.training_process is not None:
        try:
            st.session_state.training_process.terminate()
            st.session_state.training_output.append("\n⚠️ Training stopped by user")
            st.session_state.training_status = "failed"
        except Exception as e:
            st.session_state.training_output.append(f"❌ Error stopping training: {e}")
        finally:
            st.session_state.training_process = None


def _build_command_line(config: Dict[str, Any]) -> str:
    """Build the command-line string from configuration."""
    cmd_parts = ["mammography train-density"]

    # Data configuration
    if config.get("dataset"):
        cmd_parts.append(f"--dataset {config['dataset']}")
    if config.get("csv"):
        cmd_parts.append(f"--csv \"{config['csv']}\"")
    if config.get("dicom_root"):
        cmd_parts.append(f"--dicom-root \"{config['dicom_root']}\"")
    if config.get("include_class_5"):
        cmd_parts.append("--include-class-5")
    if config.get("outdir") != "outputs/run":
        cmd_parts.append(f"--outdir \"{config['outdir']}\"")

    # Cache configuration
    if config.get("cache_mode") != "auto":
        cmd_parts.append(f"--cache-mode {config['cache_mode']}")
    if config.get("cache_dir"):
        cmd_parts.append(f"--cache-dir \"{config['cache_dir']}\"")
    if config.get("embeddings_dir"):
        cmd_parts.append(f"--embeddings-dir \"{config['embeddings_dir']}\"")

    # Model configuration
    if config.get("arch") != "efficientnet_b0":
        cmd_parts.append(f"--arch {config['arch']}")
    if config.get("classes") != "density":
        cmd_parts.append(f"--classes {config['classes']}")
    if not config.get("pretrained", True):
        cmd_parts.append("--no-pretrained")

    # View-specific training
    if config.get("view_specific_training"):
        cmd_parts.append("--view-specific-training")
        if config.get("views"):
            cmd_parts.append(f"--views {config['views']}")
        if config.get("ensemble_method") != "none":
            cmd_parts.append(f"--ensemble-method {config['ensemble_method']}")

    # Training hyperparameters
    if config.get("epochs") != 100:
        cmd_parts.append(f"--epochs {config['epochs']}")
    if config.get("batch_size") != 16:
        cmd_parts.append(f"--batch-size {config['batch_size']}")
    if config.get("lr") != 1e-4:
        cmd_parts.append(f"--lr {config['lr']}")
    if config.get("backbone_lr") != 1e-5:
        cmd_parts.append(f"--backbone-lr {config['backbone_lr']}")
    if config.get("weight_decay") != 1e-4:
        cmd_parts.append(f"--weight-decay {config['weight_decay']}")
    if config.get("img_size") != 512:
        cmd_parts.append(f"--img-size {config['img_size']}")
    if config.get("seed") != 42:
        cmd_parts.append(f"--seed {config['seed']}")
    if config.get("device") != "auto":
        cmd_parts.append(f"--device {config['device']}")
    if config.get("val_frac") != 0.20:
        cmd_parts.append(f"--val-frac {config['val_frac']}")

    # Optimization settings
    if config.get("train_backbone"):
        cmd_parts.append("--train-backbone")
    if not config.get("unfreeze_last_block", True):
        cmd_parts.append("--no-unfreeze-last-block")
    if config.get("warmup_epochs") != 0:
        cmd_parts.append(f"--warmup-epochs {config['warmup_epochs']}")
    if config.get("deterministic"):
        cmd_parts.append("--deterministic")
    if not config.get("allow_tf32", True):
        cmd_parts.append("--no-allow-tf32")
    if config.get("fused_optim"):
        cmd_parts.append("--fused-optim")
    if config.get("torch_compile"):
        cmd_parts.append("--torch-compile")
    if config.get("amp"):
        cmd_parts.append("--amp")

    # Learning rate scheduling
    if config.get("scheduler") != "auto":
        cmd_parts.append(f"--scheduler {config['scheduler']}")
    if config.get("lr_reduce_patience") != 0:
        cmd_parts.append(f"--lr-reduce-patience {config['lr_reduce_patience']}")
    if config.get("lr_reduce_factor") != 0.5:
        cmd_parts.append(f"--lr-reduce-factor {config['lr_reduce_factor']}")
    if config.get("lr_reduce_min_lr") != 1e-7:
        cmd_parts.append(f"--lr-reduce-min-lr {config['lr_reduce_min_lr']}")
    if config.get("lr_reduce_cooldown") != 0:
        cmd_parts.append(f"--lr-reduce-cooldown {config['lr_reduce_cooldown']}")

    # Early stopping
    if config.get("early_stop_patience") != 0:
        cmd_parts.append(f"--early-stop-patience {config['early_stop_patience']}")
    if config.get("early_stop_min_delta") != 0.0:
        cmd_parts.append(f"--early-stop-min-delta {config['early_stop_min_delta']}")

    # Data loading
    if config.get("num_workers") != 4:
        cmd_parts.append(f"--num-workers {config['num_workers']}")
    if config.get("prefetch_factor") != 4:
        cmd_parts.append(f"--prefetch-factor {config['prefetch_factor']}")
    if not config.get("persistent_workers", True):
        cmd_parts.append("--no-persistent-workers")
    if not config.get("loader_heuristics", True):
        cmd_parts.append("--no-loader-heuristics")

    # Class balancing
    if config.get("class_weights") != "none":
        cmd_parts.append(f"--class-weights {config['class_weights']}")
    if config.get("class_weights_alpha") != 1.0:
        cmd_parts.append(f"--class-weights-alpha {config['class_weights_alpha']}")
    if config.get("sampler_weighted"):
        cmd_parts.append("--sampler-weighted")
    if config.get("sampler_alpha") != 1.0:
        cmd_parts.append(f"--sampler-alpha {config['sampler_alpha']}")

    # Augmentation
    if not config.get("augment", True):
        cmd_parts.append("--no-augment")
    if config.get("augment_vertical"):
        cmd_parts.append("--augment-vertical")
    if config.get("augment_color"):
        cmd_parts.append("--augment-color")
    if config.get("augment_rotation_deg") != 5.0:
        cmd_parts.append(f"--augment-rotation-deg {config['augment_rotation_deg']}")

    # Analysis and outputs
    if config.get("gradcam"):
        cmd_parts.append("--gradcam")
        if config.get("gradcam_limit") != 4:
            cmd_parts.append(f"--gradcam-limit {config['gradcam_limit']}")
    if config.get("save_val_preds"):
        cmd_parts.append("--save-val-preds")
    if config.get("export_val_embeddings"):
        cmd_parts.append("--export-val-embeddings")

    # Profiling
    if config.get("profile"):
        cmd_parts.append("--profile")
        if config.get("profile_dir") != "outputs/profiler":
            cmd_parts.append(f"--profile-dir \"{config['profile_dir']}\"")

    # Experiment tracking
    if config.get("tracker") != "none":
        cmd_parts.append(f"--tracker {config['tracker']}")
        if config.get("tracker_project"):
            cmd_parts.append(f"--tracker-project \"{config['tracker_project']}\"")
        if config.get("tracker_run_name"):
            cmd_parts.append(f"--tracker-run-name \"{config['tracker_run_name']}\"")
        if config.get("tracker_uri"):
            cmd_parts.append(f"--tracker-uri \"{config['tracker_uri']}\"")

    # Other settings
    if config.get("subset") != 0:
        cmd_parts.append(f"--subset {config['subset']}")
    if config.get("log_level") != "info":
        cmd_parts.append(f"--log-level {config['log_level']}")

    return " \\\n    ".join(cmd_parts)


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
    except Exception as exc:
        st.error(f"❌ Failed to initialize session state: {exc}")
        st.stop()

    st.title("⚙️ Training Configuration")

    st.markdown("""
    <div style="background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107; margin-bottom: 1rem;">
    <h3 style="color: #856404; margin-top: 0;">⚠️ EDUCATIONAL RESEARCH USE ONLY</h3>
    <p style="color: #856404; margin-bottom: 0;">
    This tool is for <strong>educational and research purposes only</strong>. It is <strong>NOT</strong>
    intended for clinical diagnosis or treatment. All results should be validated by qualified
    medical professionals before any clinical decision-making.
    </p>
    </div>
    """, unsafe_allow_html=True)

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

        st.code(command, language="bash")

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

        # Configuration summary
        with st.expander("📋 Configuration Summary"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Key Settings")
                st.metric("Architecture", st.session_state.training_config.get("arch", "efficientnet_b0"))
                st.metric("Task", st.session_state.training_config.get("classes", "density"))
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

        - **density**: BI-RADS density classification (A, B, C, D or 1-4)
        - **binary**: Binary classification (A/B vs C/D)
        - **multiclass**: Multi-class classification

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
        - [CLI_CHEATSHEET.md](https://github.com/yourusername/mammography-pipelines/blob/main/CLI_CHEATSHEET.md)
        - [README.md](https://github.com/yourusername/mammography-pipelines/blob/main/README.md)
        """)


if __name__ == "__main__":
    main()
