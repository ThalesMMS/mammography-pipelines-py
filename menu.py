#!/usr/bin/env python3
#
# menu.py
# mammography-pipelines-py
#
# Provides a CLI/web menu to launch training and feature extraction pipelines with preset options.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""
Menu Launcher — CLI/Web wrapper to run train/extract pipelines with presets.

Usage:
  python menu.py                 # CLI menu
  python menu.py --ui web        # Simple HTTP form (localhost:8000)
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
from typing import Dict, List, Optional
from urllib.parse import parse_qs

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable or "python3"

DATASETS = ["archive", "mamografias", "patches_completo", "custom"]
MODELS = ["efficientnet_b0", "resnet50"]
TASKS = ["binary", "multiclass"]


def _prompt_choice(title: str, options: List[str], default: str) -> str:
    """Ask the user to pick from a set of options with a sensible default."""
    print(f"{title} ({'/'.join(options)}), default [{default}]: ", end="", flush=True)
    val = input().strip().lower()
    return val if val in options else default


def _prompt_bool(title: str, default: bool = False) -> bool:
    """Yes/no prompt that respects the provided default when left blank."""
    dv = "Y/n" if default else "y/N"
    print(f"{title}? [{dv}]: ", end="", flush=True)
    val = input().strip().lower()
    if not val:
        return default
    return val.startswith("y")


def _prompt_int(title: str, default: int) -> int:
    """Integer prompt with graceful fallback to the default value."""
    print(f"{title} (default={default}): ", end="", flush=True)
    val = input().strip()
    try:
        return int(val)
    except Exception:
        return default


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


def build_train_command(params: Dict[str, str]) -> List[str]:
    """Compose the train.py command from collected UI/CLI parameters."""
    cmd = [
        PYTHON,
        "-u",
        str(REPO_ROOT / "mammography" / "scripts" / "train.py"),
    ]
    cmd += _build_dataset_args(params["dataset"], params.get("csv"), params.get("dicom_root"))
    cmd += [
        "--arch",
        params["arch"],
        "--classes",
        params["task"],
        "--epochs",
        str(params["epochs"]),
        "--batch-size",
        str(params["batch_size"]),
        "--outdir",
        params["outdir"],
        "--cache-mode",
        params["cache_mode"],
    ]
    if params.get("gradcam"):
        cmd.append("--gradcam")
    if params.get("save_val_preds"):
        cmd.append("--save-val-preds")
    if params.get("export_val_embeddings"):
        cmd.append("--export-val-embeddings")
    if params.get("amp"):
        cmd.append("--amp")
    return cmd


def build_extract_command(params: Dict[str, str]) -> List[str]:
    """Compose the extract_features.py command from collected UI/CLI parameters."""
    cmd = [
        PYTHON,
        "-u",
        str(REPO_ROOT / "mammography" / "scripts" / "extract_features.py"),
    ]
    cmd += _build_dataset_args(params["dataset"], params.get("csv"), params.get("dicom_root"))
    cmd += [
        "--arch",
        params["arch"],
        "--classes",
        params["task"],
        "--batch-size",
        str(params["batch_size"]),
        "--outdir",
        params["outdir"],
        "--cache-mode",
        params["cache_mode"],
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
    return cmd


def run_command(cmd: List[str]) -> int:
    """Print and run a subprocess, returning its exit code."""
    print(f"\n[cmd] {' '.join(shlex.quote(p) for p in cmd)}\n", flush=True)
    proc = subprocess.run(cmd)
    return proc.returncode


def cli_menu() -> None:
    """Interactive CLI menu that mirrors the web form."""
    print("=== Mammography Menu (CLI) ===")
    action = _prompt_choice("Action", ["train", "extract"], "train")
    dataset = _prompt_choice("Dataset", DATASETS, "archive")
    csv_path = dicom_root = None
    if dataset == "custom":
        csv_path = input("Path for --csv: ").strip()
        dicom_root = input("Dicom root (blank if not needed): ").strip() or None
    arch = _prompt_choice("Model", MODELS, "efficientnet_b0")
    task = _prompt_choice("Task", TASKS, "binary" if action == "train" else "multiclass")
    epochs = _prompt_int("Epochs (train only)", 5 if action == "train" else 1)
    batch = _prompt_int("Batch size", 16 if action == "train" else 32)
    outdir = input("Outdir (default outputs/menu_run): ").strip() or "outputs/menu_run"
    cache_mode = _prompt_choice("Cache mode", ["auto", "memory", "disk", "tensor-disk", "tensor-memmap", "none"], "auto")

    common_params = {
        "dataset": dataset,
        "csv": csv_path,
        "dicom_root": dicom_root,
        "arch": arch,
        "task": task,
        "batch_size": batch,
        "outdir": outdir,
        "cache_mode": cache_mode,
        "amp": _prompt_bool("Use AMP"),
    }

    if action == "train":
        params = {
            **common_params,
            "epochs": epochs,
            "gradcam": _prompt_bool("Enable Grad-CAM"),
            "save_val_preds": _prompt_bool("Save val predictions CSV"),
            "export_val_embeddings": _prompt_bool("Save val embeddings"),
        }
        cmd = build_train_command(params)
    else:
        params = {
            **common_params,
            "pca": _prompt_bool("Run PCA"),
            "tsne": _prompt_bool("Run t-SNE"),
            "umap": _prompt_bool("Run UMAP"),
            "cluster": _prompt_bool("Auto k-means"),
        }
        cmd = build_extract_command(params)

    confirm = _prompt_bool("Execute", True)
    if not confirm:
        print("Aborted.")
        return
    code = run_command(cmd)
    print(f"Finished with exit code {code}")


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
