#
# 1_📊_Inference.py
# mammography-pipelines
#
# Streamlit page for running inference on mammography images using trained models.
#
# Thales Matheus Mendonca Santos - February 2026
#
"""Inference page for breast density classification with trained models."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Sequence
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mammography.data.dataset import MammoDensityDataset, mammo_collate
from mammography.models.nets import build_model
from mammography.utils.common import resolve_device, configure_runtime
from mammography.io.dicom import is_dicom_path
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
    if "inference_results" not in st.session_state:
        st.session_state.inference_results = None
    if "uploaded_files_count" not in st.session_state:
        st.session_state.uploaded_files_count = 0
    if "loaded_model" not in st.session_state:
        st.session_state.loaded_model = None
    if "loaded_checkpoint_path" not in st.session_state:
        st.session_state.loaded_checkpoint_path = None


def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix from state dict keys if present.

    This is needed when loading models that were trained with DataParallel.
    """
    if not state_dict:
        return state_dict
    if all(str(k).startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _load_checkpoint(
    checkpoint_path: str,
    arch: str,
    classes_mode: str,
    device: str,
) -> tuple[torch.nn.Module, str]:
    """Load a model checkpoint and return the initialized model.

    Args:
        checkpoint_path: Path to the checkpoint file
        arch: Model architecture ('resnet50' or 'efficientnet_b0')
        classes_mode: Classification mode ('binary', 'density', or 'multiclass')
        device: Device to load model on ('auto', 'cuda', 'cpu', or 'mps')

    Returns:
        Tuple of (loaded model, device string)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Determine number of classes
    num_classes = 2 if classes_mode == "binary" else 4

    # Resolve device
    device_obj = resolve_device(device)
    configure_runtime(device_obj, deterministic=False, allow_tf32=True)

    # Build model architecture
    model = build_model(
        arch=arch,
        num_classes=num_classes,
        train_backbone=False,
        unfreeze_last_block=False,
        pretrained=False,
    )

    # Load checkpoint
    try:
        state = torch.load(checkpoint_path, map_location=device_obj)

        # Extract state_dict if checkpoint is a dict with metadata
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Strip 'module.' prefix if present (from DataParallel training)
        if isinstance(state, dict):
            state = _strip_module_prefix(state)

        # Load state dict into model
        model.load_state_dict(state, strict=False)

    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {exc}") from exc

    # Move model to device and set to eval mode
    model.to(device_obj)
    model.eval()

    return model, str(device_obj)


def main() -> None:
    """Render the inference page."""
    _require_streamlit()

    st.set_page_config(
        page_title="Inference - Mammography Pipelines",
        page_icon="📊",
        layout="wide",
    )

    # Initialize shared session state for cross-page data persistence
    try:
        ensure_shared_session_state()
        _ensure_session_defaults()
    except Exception as exc:
        st.error(f"❌ Failed to initialize session state: {exc}")
        st.stop()

    st.title("📊 Model Inference")

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

    st.header("Upload Images for Classification")

    st.markdown("""
    Upload mammography images (DICOM or PNG format) to classify breast density using
    a trained deep learning model. The model will predict BI-RADS density categories:

    - **Class 1**: Fatty (BI-RADS A)
    - **Class 2**: Mostly Fatty (BI-RADS B)
    - **Class 3**: Mostly Dense (BI-RADS C)
    - **Class 4**: Dense (BI-RADS D)
    """)

    st.info(
        "💡 **Quick Start:** (1) Enter the path to your trained model checkpoint, "
        "(2) Select the model architecture and classification mode, "
        "(3) Upload one or more mammography images, and (4) Click 'Run Inference'."
    )

    # Model selection
    st.subheader("1. Select Model Checkpoint")

    col1, col2 = st.columns([2, 1])

    with col1:
        checkpoint_path = st.text_input(
            "Model Checkpoint Path",
            placeholder="path/to/checkpoint.pt",
            help="Path to the trained model checkpoint file (.pt or .pth)",
        )

    with col2:
        arch = st.selectbox(
            "Model Architecture",
            options=["resnet50", "efficientnet_b0"],
            help="Architecture used for training the model",
        )

    # Classification mode
    st.subheader("2. Classification Mode")

    classes_mode = st.radio(
        "Select classification type:",
        options=["multiclass", "binary", "density"],
        horizontal=True,
        help="Multiclass: 4 classes (1-4), Binary: 2 classes (non-dense/dense), Density: 4 density levels",
    )

    # File upload
    st.subheader("3. Upload Images")

    uploaded_files = st.file_uploader(
        "Choose mammography images",
        type=["png", "jpg", "jpeg", "dcm", "dicom"],
        accept_multiple_files=True,
        help="Upload one or more mammography images in PNG, JPEG, or DICOM format",
    )

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)

        with col1:
            img_size = st.number_input(
                "Image Size",
                min_value=128,
                max_value=1024,
                value=224,
                step=32,
                help="Input image size for the model",
            )

            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=128,
                value=16,
                help="Number of images to process at once",
            )

        with col2:
            use_amp = st.checkbox(
                "Use Mixed Precision (AMP)",
                value=False,
                help="Use automatic mixed precision for faster inference on CUDA/MPS devices",
            )

            device = st.selectbox(
                "Device",
                options=["auto", "cuda", "cpu", "mps"],
                help="Compute device for inference",
            )

    # Model status indicator
    if st.session_state.loaded_model is not None:
        st.success(
            f"✅ Model loaded: {st.session_state.loaded_checkpoint_path} "
            f"(Architecture: {arch}, Mode: {classes_mode})"
        )
        if st.button("🗑️ Clear Loaded Model"):
            st.session_state.loaded_model = None
            st.session_state.loaded_checkpoint_path = None
            st.rerun()

    # Run inference button
    st.subheader("4. Run Inference")

    if st.button("🚀 Run Inference", type="primary", disabled=not (checkpoint_path and uploaded_files)):
        if not checkpoint_path:
            st.error("❌ Please provide a model checkpoint path.")
            st.info("💡 Enter the file path to your trained model (e.g., 'outputs/run/model_best.pt')")
        elif not os.path.exists(checkpoint_path):
            st.error(f"❌ Checkpoint file not found: {checkpoint_path}")
            st.info("💡 Please verify the file path exists and try again.")
        elif not uploaded_files:
            st.error("❌ Please upload at least one image.")
            st.info("💡 Use the file uploader above to select mammography images (DICOM or PNG).")
        else:
            # Load checkpoint if not already loaded or if path changed
            if (
                st.session_state.loaded_model is None
                or st.session_state.loaded_checkpoint_path != checkpoint_path
            ):
                with st.spinner(f"Loading model checkpoint from {checkpoint_path}..."):
                    try:
                        model, device_str = _load_checkpoint(
                            checkpoint_path=checkpoint_path,
                            arch=arch,
                            classes_mode=classes_mode,
                            device=device,
                        )
                        st.session_state.loaded_model = model
                        st.session_state.loaded_checkpoint_path = checkpoint_path

                        # Store in shared session state for cross-page access
                        st.session_state.shared_model = model
                        st.session_state.shared_checkpoint_path = checkpoint_path
                        st.session_state.shared_model_arch = arch
                        st.session_state.shared_model_num_classes = 2 if classes_mode == "binary" else 4

                        st.success(
                            f"✅ Model loaded successfully! "
                            f"({arch}, {classes_mode} mode, device: {device_str})"
                        )
                    except FileNotFoundError as exc:
                        st.error(f"❌ Checkpoint file not found: {exc}")
                        st.info("💡 Verify the checkpoint path is correct and the file exists.")
                        st.stop()
                    except RuntimeError as exc:
                        st.error(f"❌ Failed to load checkpoint: {exc}")
                        st.info(
                            "💡 This may happen if:\n"
                            "- The checkpoint was trained with a different architecture\n"
                            "- The checkpoint file is corrupted\n"
                            "- The classification mode doesn't match the trained model"
                        )
                        st.stop()
                    except Exception as exc:
                        st.error(f"❌ Unexpected error loading checkpoint: {exc}")
                        st.info("💡 Check that PyTorch is installed and the checkpoint file is valid.")
                        st.stop()
            else:
                st.info(f"Using previously loaded model from {checkpoint_path}")

            # Save uploaded files to temporary directory
            with st.spinner("Saving uploaded files..."):
                temp_dir = tempfile.mkdtemp(prefix="mammography_inference_")
                saved_paths: list[str] = []

                for idx, uploaded_file in enumerate(uploaded_files):
                    # Determine file extension
                    file_name = uploaded_file.name
                    file_path = os.path.join(temp_dir, file_name)

                    # Write file to disk
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    saved_paths.append(file_path)

                    # Store first image in shared state for cross-page access
                    if idx == 0:
                        st.session_state.shared_uploaded_image_path = file_path
                        st.session_state.shared_uploaded_image_name = file_name
                        # Load and store PIL image for explainability
                        try:
                            if is_dicom_path(file_path):
                                from mammography.io.dicom import dicom_to_pil_rgb
                                pil_img = dicom_to_pil_rgb(file_path)
                            else:
                                from PIL import Image
                                pil_img = Image.open(file_path).convert("RGB")
                            st.session_state.shared_uploaded_image = pil_img
                        except Exception:
                            # Best effort - don't fail inference if image loading fails
                            st.session_state.shared_uploaded_image = None

                st.success(f"✅ Saved {len(saved_paths)} file(s) to temporary directory")

            # Create dataset rows for inference
            with st.spinner("Preparing dataset..."):
                rows = []
                for path in saved_paths:
                    rows.append(
                        {
                            "image_path": path,
                            "professional_label": None,
                            "accession": (
                                os.path.basename(os.path.dirname(path))
                                if is_dicom_path(path)
                                else None
                            ),
                        }
                    )

                # Create label mapper for binary mode
                mapper = None
                if classes_mode == "binary":
                    def _mapper(y: int) -> int:
                        if y in [1, 2]:
                            return 0  # Non-dense
                        if y in [3, 4]:
                            return 1  # Dense
                        return y - 1
                    mapper = _mapper

                # Create dataset
                try:
                    dataset = MammoDensityDataset(
                        rows,
                        img_size=img_size,
                        train=False,
                        cache_mode="none",
                        split_name="inference",
                        label_mapper=mapper,
                        mean=None,  # Use defaults
                        std=None,   # Use defaults
                    )

                    loader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=mammo_collate,
                    )

                    st.success(f"✅ Dataset created with {len(dataset)} image(s)")

                except Exception as exc:
                    st.error(f"❌ Failed to create dataset: {exc}")
                    st.info(
                        "💡 This may happen if:\n"
                        "- The uploaded images are corrupted or in an unsupported format\n"
                        "- DICOM files are missing required metadata\n"
                        "- The image size parameters are invalid"
                    )
                    st.stop()

            # Run inference
            with st.spinner("Running inference... This may take a while."):
                model = st.session_state.loaded_model
                device_obj = next(model.parameters()).device
                use_amp_final = use_amp and device_obj.type in {"cuda", "mps"}

                results: list[dict[str, object]] = []

                try:
                    with torch.no_grad():
                        progress_bar = st.progress(0)
                        total_batches = len(loader)

                        for batch_idx, batch in enumerate(loader):
                            if batch is None:
                                continue

                            imgs, _, metas, _ = batch
                            imgs = imgs.to(device_obj)

                            # Run inference with optional AMP
                            if use_amp_final:
                                with torch.autocast(device_obj.type, dtype=torch.float16):
                                    logits = model(imgs)
                            else:
                                logits = model(imgs)

                            # Compute probabilities and predictions
                            probs = torch.softmax(logits, dim=1).cpu().numpy()
                            preds = np.argmax(probs, axis=1)

                            # Store results
                            for meta, pred, prob in zip(metas, preds, probs):
                                row = {
                                    "file": os.path.basename(meta.get("path", "")),
                                    "pred_class": int(pred) + 1,  # Convert to 1-indexed for BI-RADS
                                }
                                for i, p in enumerate(prob.tolist()):
                                    row[f"prob_class_{i + 1}"] = float(p)
                                results.append(row)

                            # Update progress
                            progress_bar.progress((batch_idx + 1) / total_batches)

                        progress_bar.empty()

                    # Convert to DataFrame
                    results_df = pd.DataFrame(results)

                    # Add BI-RADS labels
                    birads_map = {
                        1: "Fatty (A)",
                        2: "Mostly Fatty (B)",
                        3: "Mostly Dense (C)",
                        4: "Dense (D)",
                    }

                    if classes_mode == "binary":
                        birads_map = {
                            1: "Non-Dense",
                            2: "Dense",
                        }

                    results_df["prediction"] = results_df["pred_class"].map(birads_map)

                    # Reorder columns for better readability
                    cols = ["file", "prediction", "pred_class"] + [
                        c for c in results_df.columns if c.startswith("prob_")
                    ]
                    results_df = results_df[cols]

                    # Store results in session state
                    st.session_state.inference_results = results_df
                    st.session_state.uploaded_files_count = len(uploaded_files)

                    # Store in shared session state for cross-page access
                    st.session_state.shared_inference_results = results_df

                    # Store prediction data for first image
                    if len(results) > 0:
                        first_result = results[0]
                        pred_class = first_result["pred_class"] - 1  # Convert back to 0-indexed
                        probs = np.array([
                            first_result.get(f"prob_class_{i+1}", 0.0)
                            for i in range(num_classes)
                        ])
                        st.session_state.shared_prediction_data = {
                            "class": pred_class,
                            "probs": probs,
                        }

                    st.success(
                        f"✅ Inference complete! Processed {len(results_df)} image(s)."
                    )

                except Exception as exc:
                    st.error(f"❌ Inference failed: {exc}")
                    st.info(
                        "💡 This may happen if:\n"
                        "- The model runs out of memory (try reducing batch size or image size)\n"
                        "- The images are incompatible with the model input size\n"
                        "- There's a CUDA/device error (try using CPU device)\n"
                        "- The uploaded files are corrupted"
                    )
                    with st.expander("🔍 View Full Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
                    st.stop()
                finally:
                    # Clean up temporary files
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass  # Best effort cleanup

    # Display results placeholder
    if st.session_state.inference_results is not None:
        st.header("Results")
        st.dataframe(st.session_state.inference_results, use_container_width=True)

        # Download results
        csv = st.session_state.inference_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="inference_results.csv",
            mime="text/csv",
        )
    elif st.session_state.uploaded_files_count > 0:
        st.info(f"Ready to process {st.session_state.uploaded_files_count} file(s).")

    # Help section
    with st.expander("ℹ️ Help & Documentation"):
        st.markdown("""
        ### How to Use This Page

        1. **Select a Model**: Enter the path to your trained model checkpoint file
        2. **Choose Architecture**: Select the architecture that matches your checkpoint
        3. **Set Classification Mode**: Choose between multiclass (4 classes) or binary classification
        4. **Upload Images**: Select one or more mammography images to classify
        5. **Adjust Settings** (optional): Configure image size, batch size, and device
        6. **Run Inference**: Click the button to process your images

        ### Supported File Formats

        - **DICOM**: `.dcm`, `.dicom` - Medical imaging standard format
        - **PNG**: `.png` - Portable Network Graphics
        - **JPEG**: `.jpg`, `.jpeg` - Joint Photographic Experts Group

        ### Model Checkpoints

        Checkpoints are saved during training and typically stored in:
        - `checkpoints/` directory in your project
        - MLflow experiment artifacts
        - Custom output paths specified during training

        ### Troubleshooting

        - **Out of Memory**: Reduce batch size or image size
        - **Slow Performance**: Enable AMP if using CUDA/MPS device
        - **Wrong Predictions**: Verify checkpoint matches the selected architecture
        """)


if __name__ == "__main__":
    main()
