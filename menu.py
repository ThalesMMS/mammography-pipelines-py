#!/usr/bin/env python3
#
# menu.py
# mammography-pipelines-py
#
# Provides an interactive CLI menu to launch training, feature extraction, and visualization pipelines.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""
Interactive Menu Launcher — CLI interface to run mammography pipelines.

Usage:
  python menu.py                 # Interactive CLI menu
  python menu.py --ui web        # Simple HTTP form (localhost:8000)
  python menu.py --quick train   # Quick mode: skip menus
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import parse_qs

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable or "python3"

# Available options
DATASETS = {
    "archive": "Kaggle RSNA Breast Density Archive",
    "mamografias": "Local mammography DICOM dataset",
    "patches_completo": "Pre-extracted patches dataset",
    "custom": "Custom dataset (specify CSV/DICOM paths)",
}

MODELS = {
    "efficientnet_b0": "EfficientNet-B0 (recommended, balanced)",
    "resnet50": "ResNet-50 (larger, higher capacity)",
}

TASKS = {
    "binary": "Binary classification (Low vs High density)",
    "multiclass": "Multiclass classification (BI-RADS A/B/C/D)",
    "density": "Density classification (same as multiclass)",
}

OPERATIONS = {
    "train": "Train a neural network on mammography data",
    "extract": "Extract embeddings/features from images",
    "visualize": "Generate visualizations from embeddings",
    "evaluate": "Evaluate model and export metrics",
}

CACHE_MODES = {
    "auto": "Auto-select based on dataset size",
    "memory": "Cache all in RAM (fast, high memory)",
    "disk": "Cache to disk as images",
    "tensor-disk": "Cache as PyTorch tensors on disk",
    "tensor-memmap": "Memory-mapped tensor cache",
    "none": "No caching (slowest, lowest memory)",
}

VISUALIZATIONS = {
    "tsne": "t-SNE 2D embedding plot",
    "tsne_3d": "t-SNE 3D embedding plot",
    "pca": "PCA scatter plot",
    "umap": "UMAP embedding plot",
    "compare": "Compare PCA, t-SNE, UMAP side by side",
    "heatmap": "Feature correlation heatmap",
    "confusion": "Confusion matrix heatmap",
    "scatter_matrix": "Pairwise scatter plot matrix",
    "distribution": "Feature distribution plots",
    "class_separation": "Class separation analysis",
    "learning_curves": "Training learning curves",
    "report": "Generate comprehensive report (all visualizations)",
}


def clear_screen():
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title: str, width: int = 60):
    """Print a formatted header."""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def print_menu(title: str, options: Dict[str, str], show_back: bool = True) -> None:
    """Print a numbered menu with descriptions."""
    print_header(title)
    keys = list(options.keys())
    for i, key in enumerate(keys, 1):
        print(f"  [{i}] {key:<20} - {options[key]}")
    if show_back:
        print(f"\n  [0] Back / Cancel")
    print()


def get_choice(options: Dict[str, str], prompt: str = "Select option") -> Optional[str]:
    """Get user choice from numbered menu."""
    keys = list(options.keys())
    while True:
        try:
            choice = input(f"{prompt} [1-{len(keys)}, 0=back]: ").strip()
            if choice == "0" or choice.lower() in ("q", "quit", "exit", "back"):
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
            print(f"  Invalid choice. Enter 1-{len(keys)} or 0 to go back.")
        except ValueError:
            # Check if they typed the key directly
            if choice.lower() in [k.lower() for k in keys]:
                for k in keys:
                    if k.lower() == choice.lower():
                        return k
            print(f"  Invalid input. Enter a number 1-{len(keys)}.")


def get_bool(prompt: str, default: bool = False) -> bool:
    """Yes/no prompt with default."""
    default_str = "Y/n" if default else "y/N"
    val = input(f"{prompt}? [{default_str}]: ").strip().lower()
    if not val:
        return default
    return val.startswith("y")


def get_int(prompt: str, default: int, min_val: int = 1, max_val: int = 1000) -> int:
    """Integer prompt with validation."""
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if not val:
            return default
        try:
            num = int(val)
            if min_val <= num <= max_val:
                return num
            print(f"  Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("  Please enter a valid number.")


def get_string(prompt: str, default: str = "") -> str:
    """String prompt with default."""
    default_display = f" [{default}]" if default else ""
    val = input(f"{prompt}{default_display}: ").strip()
    return val if val else default


def get_multi_choice(title: str, options: Dict[str, str]) -> List[str]:
    """Allow selecting multiple options."""
    print_header(title)
    keys = list(options.keys())
    for i, key in enumerate(keys, 1):
        print(f"  [{i}] {key:<20} - {options[key]}")
    print(f"\n  Enter numbers separated by comma (e.g., 1,2,3)")
    print(f"  Or 'all' for all options, 'none' to skip")
    print()
    
    while True:
        val = input("Select options: ").strip().lower()
        if val == "none" or val == "0":
            return []
        if val == "all":
            return keys
        try:
            indices = [int(x.strip()) - 1 for x in val.split(",")]
            selected = [keys[i] for i in indices if 0 <= i < len(keys)]
            if selected:
                return selected
            print("  No valid options selected. Try again.")
        except (ValueError, IndexError):
            print("  Invalid input. Enter numbers like: 1,2,3")


def _build_dataset_args(dataset: str, csv_path: Optional[str], dicom_root: Optional[str]) -> List[str]:
    """Translate dataset selection into CLI flags understood by downstream scripts."""
    if dataset != "custom":
        return ["--dataset", dataset]
    args: List[str] = []
    if csv_path:
        args.extend(["--csv", csv_path])
    if dicom_root:
        args.extend(["--dicom-root", dicom_root])
    return args


def build_train_command(params: Dict[str, Any]) -> List[str]:
    """Compose the train.py command from collected UI/CLI parameters."""
    cmd = [
        PYTHON,
        "-u",
        str(REPO_ROOT / "mammography" / "scripts" / "train.py"),
    ]
    cmd += _build_dataset_args(params["dataset"], params.get("csv"), params.get("dicom_root"))
    cmd += [
        "--arch", params["arch"],
        "--classes", params["task"],
        "--epochs", str(params["epochs"]),
        "--batch-size", str(params["batch_size"]),
        "--outdir", params["outdir"],
        "--cache-mode", params["cache_mode"],
    ]
    if params.get("gradcam"):
        cmd.append("--gradcam")
    if params.get("save_val_preds"):
        cmd.append("--save-val-preds")
    if params.get("export_val_embeddings"):
        cmd.append("--export-val-embeddings")
    if params.get("amp"):
        cmd.append("--amp")
    if params.get("lr"):
        cmd.extend(["--lr", str(params["lr"])])
    if params.get("weight_decay"):
        cmd.extend(["--weight-decay", str(params["weight_decay"])])
    return cmd


def build_extract_command(params: Dict[str, Any]) -> List[str]:
    """Compose the extract_features.py command from collected UI/CLI parameters."""
    cmd = [
        PYTHON,
        "-u",
        str(REPO_ROOT / "mammography" / "scripts" / "extract_features.py"),
    ]
    cmd += _build_dataset_args(params["dataset"], params.get("csv"), params.get("dicom_root"))
    cmd += [
        "--arch", params["arch"],
        "--classes", params["task"],
        "--batch-size", str(params["batch_size"]),
        "--outdir", params["outdir"],
        "--cache-mode", params["cache_mode"],
    ]
    if params.get("amp"):
        cmd.append("--amp")
    if params.get("pca"):
        cmd.append("--pca")
    if params.get("tsne"):
        cmd.append("--tsne")
    if params.get("umap"):
        cmd.append("--umap")
    if params.get("cluster"):
        cmd.append("--cluster")
    if params.get("save_csv"):
        cmd.append("--save-csv")
    return cmd


def build_visualize_command(params: Dict[str, Any]) -> List[str]:
    """Compose the visualize.py command from collected parameters."""
    cmd = [
        PYTHON,
        "-u",
        str(REPO_ROOT / "mammography" / "scripts" / "visualize.py"),
    ]
    
    if params.get("from_run"):
        cmd.extend(["--from-run", "--input", params["input"]])
    else:
        cmd.extend(["--input", params["input"]])
        if params.get("labels"):
            cmd.extend(["--labels", params["labels"]])
    
    cmd.extend(["--output", params.get("outdir", "outputs/visualizations")])
    
    if params.get("binary"):
        cmd.append("--binary")
    
    # Add visualization flags
    viz_flags = params.get("visualizations", [])
    if "report" in viz_flags:
        cmd.append("--report")
    else:
        flag_map = {
            "tsne": "--tsne",
            "tsne_3d": "--tsne-3d",
            "pca": "--pca",
            "umap": "--umap",
            "compare": "--compare-embeddings",
            "heatmap": "--heatmap",
            "confusion": "--confusion-matrix",
            "scatter_matrix": "--scatter-matrix",
            "distribution": "--distribution",
            "class_separation": "--class-separation",
            "learning_curves": "--learning-curves",
        }
        for viz in viz_flags:
            if viz in flag_map:
                cmd.append(flag_map[viz])
    
    if params.get("predictions"):
        cmd.extend(["--predictions", params["predictions"]])
    if params.get("history"):
        cmd.extend(["--history", params["history"]])
    
    return cmd


def run_command(cmd: List[str]) -> int:
    """Print and run a subprocess, returning its exit code."""
    print()
    print("-" * 60)
    print(f"Command: {' '.join(shlex.quote(p) for p in cmd)}")
    print("-" * 60)
    print()
    proc = subprocess.run(cmd)
    return proc.returncode


def select_dataset() -> Tuple[str, Optional[str], Optional[str]]:
    """Interactive dataset selection menu."""
    print_menu("Select Dataset", DATASETS)
    dataset = get_choice(DATASETS, "Choose dataset")
    if dataset is None:
        return "archive", None, None
    
    csv_path = None
    dicom_root = None
    
    if dataset == "custom":
        print("\n  Custom dataset requires path configuration:")
        csv_path = get_string("  CSV file or directory path")
        if not csv_path:
            print("  No CSV path provided, falling back to 'archive'")
            return "archive", None, None
        dicom_root = get_string("  DICOM root directory (optional)")
        dicom_root = dicom_root if dicom_root else None
    
    return dataset, csv_path, dicom_root


def select_model() -> str:
    """Interactive model selection menu."""
    print_menu("Select Neural Network", MODELS)
    model = get_choice(MODELS, "Choose model")
    return model if model else "efficientnet_b0"


def select_task() -> str:
    """Interactive task selection menu."""
    print_menu("Select Classification Task", TASKS)
    task = get_choice(TASKS, "Choose task")
    return task if task else "binary"


def configure_training() -> Dict[str, Any]:
    """Configure training parameters interactively."""
    print_header("Training Configuration")
    
    params: Dict[str, Any] = {}
    
    # Dataset
    dataset, csv_path, dicom_root = select_dataset()
    params["dataset"] = dataset
    params["csv"] = csv_path
    params["dicom_root"] = dicom_root
    
    # Model
    params["arch"] = select_model()
    
    # Task
    params["task"] = select_task()
    
    # Training parameters
    print_header("Training Hyperparameters")
    params["epochs"] = get_int("Number of epochs", 10, 1, 500)
    params["batch_size"] = get_int("Batch size", 16, 1, 256)
    params["lr"] = float(get_string("Learning rate", "1e-3"))
    
    # Output
    params["outdir"] = get_string("Output directory", "outputs/train_run")
    
    # Cache mode
    print_menu("Cache Mode", CACHE_MODES, show_back=False)
    params["cache_mode"] = get_choice(CACHE_MODES, "Choose cache mode") or "auto"
    
    # Options
    print_header("Additional Options")
    params["amp"] = get_bool("Use Automatic Mixed Precision (AMP)", True)
    params["gradcam"] = get_bool("Generate Grad-CAM visualizations", False)
    params["save_val_preds"] = get_bool("Save validation predictions to CSV", True)
    params["export_val_embeddings"] = get_bool("Export validation embeddings", True)
    
    return params


def configure_extraction() -> Dict[str, Any]:
    """Configure feature extraction parameters interactively."""
    print_header("Feature Extraction Configuration")
    
    params: Dict[str, Any] = {}
    
    # Dataset
    dataset, csv_path, dicom_root = select_dataset()
    params["dataset"] = dataset
    params["csv"] = csv_path
    params["dicom_root"] = dicom_root
    
    # Model
    params["arch"] = select_model()
    
    # Task
    params["task"] = select_task()
    
    # Extraction parameters
    print_header("Extraction Parameters")
    params["batch_size"] = get_int("Batch size", 32, 1, 256)
    params["outdir"] = get_string("Output directory", "outputs/features")
    
    # Cache mode
    print_menu("Cache Mode", CACHE_MODES, show_back=False)
    params["cache_mode"] = get_choice(CACHE_MODES, "Choose cache mode") or "auto"
    
    # Analysis options
    print_header("Analysis Options")
    params["amp"] = get_bool("Use Automatic Mixed Precision (AMP)", True)
    params["pca"] = get_bool("Run PCA dimensionality reduction", True)
    params["tsne"] = get_bool("Run t-SNE visualization", True)
    params["umap"] = get_bool("Run UMAP visualization", False)
    params["cluster"] = get_bool("Run k-means clustering analysis", True)
    params["save_csv"] = get_bool("Save joined results CSV", True)
    
    return params


def configure_visualization() -> Dict[str, Any]:
    """Configure visualization parameters interactively."""
    print_header("Visualization Configuration")
    
    params: Dict[str, Any] = {}
    
    # Input source
    print("\n  Input source options:")
    print("  [1] From training/extraction run directory")
    print("  [2] From separate feature and label files")
    print()
    
    source = input("Choose input source [1]: ").strip()
    
    if source == "2":
        params["from_run"] = False
        params["input"] = get_string("Path to features file (.npy/.npz)")
        params["labels"] = get_string("Path to labels CSV (optional)")
        params["predictions"] = get_string("Path to predictions CSV (optional)")
        params["history"] = get_string("Path to training history (optional)")
    else:
        params["from_run"] = True
        params["input"] = get_string("Path to run directory", "outputs/train_run")
    
    # Output
    params["outdir"] = get_string("Output directory", "outputs/visualizations")
    
    # Task type for labels
    params["binary"] = get_bool("Binary classification (Low/High density)", False)
    
    # Visualization selection
    params["visualizations"] = get_multi_choice("Select Visualizations", VISUALIZATIONS)
    
    return params


def menu_train() -> None:
    """Training workflow menu."""
    params = configure_training()
    
    print_header("Configuration Summary")
    print(f"  Dataset:     {params['dataset']}")
    print(f"  Model:       {params['arch']}")
    print(f"  Task:        {params['task']}")
    print(f"  Epochs:      {params['epochs']}")
    print(f"  Batch Size:  {params['batch_size']}")
    print(f"  Output:      {params['outdir']}")
    print(f"  Cache Mode:  {params['cache_mode']}")
    print()
    
    if not get_bool("Execute training", True):
        print("Aborted.")
        return
    
    cmd = build_train_command(params)
    code = run_command(cmd)
    print(f"\nTraining finished with exit code {code}")
    input("\nPress Enter to continue...")


def menu_extract() -> None:
    """Feature extraction workflow menu."""
    params = configure_extraction()
    
    print_header("Configuration Summary")
    print(f"  Dataset:     {params['dataset']}")
    print(f"  Model:       {params['arch']}")
    print(f"  Task:        {params['task']}")
    print(f"  Batch Size:  {params['batch_size']}")
    print(f"  Output:      {params['outdir']}")
    print(f"  PCA:         {'Yes' if params['pca'] else 'No'}")
    print(f"  t-SNE:       {'Yes' if params['tsne'] else 'No'}")
    print(f"  UMAP:        {'Yes' if params['umap'] else 'No'}")
    print(f"  Clustering:  {'Yes' if params['cluster'] else 'No'}")
    print()
    
    if not get_bool("Execute extraction", True):
        print("Aborted.")
        return
    
    cmd = build_extract_command(params)
    code = run_command(cmd)
    print(f"\nExtraction finished with exit code {code}")
    input("\nPress Enter to continue...")


def menu_visualize() -> None:
    """Visualization workflow menu."""
    params = configure_visualization()
    
    if not params.get("visualizations"):
        print("\nNo visualizations selected. Aborted.")
        input("\nPress Enter to continue...")
        return
    
    print_header("Configuration Summary")
    print(f"  Input:          {params['input']}")
    print(f"  From Run:       {'Yes' if params.get('from_run') else 'No'}")
    print(f"  Output:         {params['outdir']}")
    print(f"  Binary Labels:  {'Yes' if params.get('binary') else 'No'}")
    print(f"  Visualizations: {', '.join(params['visualizations'])}")
    print()
    
    if not get_bool("Generate visualizations", True):
        print("Aborted.")
        return
    
    cmd = build_visualize_command(params)
    code = run_command(cmd)
    print(f"\nVisualization finished with exit code {code}")
    input("\nPress Enter to continue...")


def menu_quick_train() -> None:
    """Quick training with minimal prompts."""
    print_header("Quick Train Mode")
    
    dataset, csv_path, dicom_root = select_dataset()
    model = select_model()
    
    params = {
        "dataset": dataset,
        "csv": csv_path,
        "dicom_root": dicom_root,
        "arch": model,
        "task": "binary",
        "epochs": 5,
        "batch_size": 16,
        "outdir": "outputs/quick_train",
        "cache_mode": "auto",
        "amp": True,
        "gradcam": False,
        "save_val_preds": True,
        "export_val_embeddings": True,
    }
    
    print(f"\n  Quick training: {model} on {dataset}")
    print(f"  5 epochs, batch size 16")
    print()
    
    if not get_bool("Start quick training", True):
        print("Aborted.")
        return
    
    cmd = build_train_command(params)
    code = run_command(cmd)
    print(f"\nQuick training finished with exit code {code}")
    input("\nPress Enter to continue...")


def cli_menu() -> None:
    """Main interactive CLI menu loop."""
    while True:
        clear_screen()
        print_header("Mammography Pipeline Menu")
        print("  Welcome to the mammography analysis pipeline!")
        print("  Select an operation to get started.\n")
        
        main_ops = {
            "train": "Train neural network classifier",
            "extract": "Extract features/embeddings",
            "visualize": "Generate visualizations",
            "quick": "Quick train (minimal config)",
        }
        print_menu("Main Operations", main_ops, show_back=False)
        print("  [0] Exit\n")
        
        choice = input("Select operation: ").strip().lower()
        
        if choice in ("0", "q", "quit", "exit"):
            print("\nGoodbye!")
            break
        elif choice in ("1", "train"):
            menu_train()
        elif choice in ("2", "extract"):
            menu_extract()
        elif choice in ("3", "visualize"):
            menu_visualize()
        elif choice in ("4", "quick"):
            menu_quick_train()
        else:
            print("\n  Invalid option. Press Enter to try again...")
            input()


HTML_FORM = """<!doctype html>
<html><body>
<h3>Mammography Menu (Web)</h3>
<form method="post">
Action:
  <select name="action">
    <option value="train">train</option>
    <option value="extract">extract</option>
  </select><br/>
Dataset:
  <select name="dataset">
    <option value="archive">archive</option>
    <option value="mamografias">mamografias</option>
    <option value="patches_completo">patches_completo</option>
    <option value="custom">custom</option>
  </select><br/>
CSV (custom): <input type="text" name="csv"/><br/>
DICOM root (custom): <input type="text" name="dicom_root"/><br/>
Model:
  <select name="arch">
    <option value="efficientnet_b0">efficientnet_b0</option>
    <option value="resnet50">resnet50</option>
  </select><br/>
Task:
  <select name="task">
    <option value="binary">binary</option>
    <option value="multiclass">multiclass</option>
  </select><br/>
Epochs: <input type="number" name="epochs" value="5"/><br/>
Batch: <input type="number" name="batch_size" value="16"/><br/>
Outdir: <input type="text" name="outdir" value="outputs/menu_run"/><br/>
Cache mode: <input type="text" name="cache_mode" value="auto"/><br/>
Flags: 
  <label><input type="checkbox" name="amp" value="1"/>amp</label>
  <label><input type="checkbox" name="gradcam" value="1"/>gradcam</label>
  <label><input type="checkbox" name="save_val_preds" value="1"/>save val preds</label>
  <label><input type="checkbox" name="export_val_embeddings" value="1"/>val embeddings</label>
  <label><input type="checkbox" name="pca" value="1"/>pca</label>
  <label><input type="checkbox" name="tsne" value="1"/>tsne</label>
  <label><input type="checkbox" name="umap" value="1"/>umap</label>
  <label><input type="checkbox" name="cluster" value="1"/>cluster</label><br/>
<input type="submit" value="Run"/>
</form>
<p>Results appear below after submission.</p>
{output}
</body></html>
"""


def _bool_from_form(data: Dict[str, List[str]], key: str) -> bool:
    """Interpret checkbox values from the web form payload."""
    return key in data and bool(data[key])


def _run_and_capture(cmd: List[str]) -> str:
    """Execute a command and capture stdout/stderr for embedding in HTML."""
    err = None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
        text = (out.stdout or "") + "\n" + (out.stderr or "")
        return f"<h4>Command</h4><pre>{' '.join(shlex.quote(p) for p in cmd)}</pre><h4>Output</h4><pre>{text}</pre>"
    except Exception as exc:  # pragma: no cover - best effort Web mode
        err = str(exc)
    return f"<pre>Failed to run command: {err}</pre>"


class MenuHandler(BaseHTTPRequestHandler):
    output_html = ""

    def do_GET(self):  # pragma: no cover - simple server
        body = HTML_FORM.format(output=self.output_html)
        self._send(body)

    def do_POST(self):  # pragma: no cover - simple server
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        data = parse_qs(raw)
        action = data.get("action", ["train"])[0]
        params = {
            "dataset": data.get("dataset", ["archive"])[0],
            "csv": data.get("csv", [""])[0] or None,
            "dicom_root": data.get("dicom_root", [""])[0] or None,
            "arch": data.get("arch", ["efficientnet_b0"])[0],
            "task": data.get("task", ["binary"])[0],
            "epochs": int(data.get("epochs", ["5"])[0]),
            "batch_size": int(data.get("batch_size", ["16"])[0]),
            "outdir": data.get("outdir", ["outputs/menu_run"])[0],
            "cache_mode": data.get("cache_mode", ["auto"])[0],
            "amp": _bool_from_form(data, "amp"),
        }
        if action == "train":
            params.update(
                {
                    "gradcam": _bool_from_form(data, "gradcam"),
                    "save_val_preds": _bool_from_form(data, "save_val_preds"),
                    "export_val_embeddings": _bool_from_form(data, "export_val_embeddings"),
                }
            )
            cmd = build_train_command(params)
        else:
            params.update(
                {
                    "pca": _bool_from_form(data, "pca"),
                    "tsne": _bool_from_form(data, "tsne"),
                    "umap": _bool_from_form(data, "umap"),
                    "cluster": _bool_from_form(data, "cluster"),
                }
            )
            cmd = build_extract_command(params)
        MenuHandler.output_html = _run_and_capture(cmd)
        body = HTML_FORM.format(output=MenuHandler.output_html)
        self._send(body)

    def log_message(self, *_):
        return  # silence default logging

    def _send(self, body: str):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_web(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = HTTPServer((host, port), MenuHandler)
    print(f"Serving menu at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Menu para pipelines de mamografia (CLI ou Web).")
    parser.add_argument("--ui", choices=["cli", "web"], default="cli")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.ui == "cli":
        cli_menu()
    else:
        run_web(args.host, args.port)


if __name__ == "__main__":
    main()
