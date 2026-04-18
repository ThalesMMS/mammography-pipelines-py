"""Training process helpers for the Streamlit Training page."""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from mammography.config import HP, TrainConfig


_BASE_COMMAND = "mammography train-density"
_TRAIN_DEFAULTS = {
    name: field.default for name, field in TrainConfig.model_fields.items()
}

_FLAG_OPTIONS = {
    "--include-class-5",
    "--no-pretrained",
    "--view-specific-training",
    "--train-backbone",
    "--no-unfreeze-last-block",
    "--deterministic",
    "--no-allow-tf32",
    "--fused-optim",
    "--torch-compile",
    "--amp",
    "--no-persistent-workers",
    "--no-loader-heuristics",
    "--sampler-weighted",
    "--no-augment",
    "--augment-vertical",
    "--augment-color",
    "--gradcam",
    "--save-val-preds",
    "--export-val-embeddings",
    "--profile",
}

_VALUE_OPTIONS = {
    "--dataset",
    "--csv",
    "--dicom-root",
    "--outdir",
    "--cache-mode",
    "--cache-dir",
    "--embeddings-dir",
    "--arch",
    "--classes",
    "--views",
    "--ensemble-method",
    "--epochs",
    "--batch-size",
    "--lr",
    "--backbone-lr",
    "--weight-decay",
    "--img-size",
    "--seed",
    "--device",
    "--val-frac",
    "--warmup-epochs",
    "--scheduler",
    "--lr-reduce-patience",
    "--lr-reduce-factor",
    "--lr-reduce-min-lr",
    "--lr-reduce-cooldown",
    "--early-stop-patience",
    "--early-stop-min-delta",
    "--num-workers",
    "--prefetch-factor",
    "--class-weights",
    "--class-weights-alpha",
    "--sampler-alpha",
    "--augment-rotation-deg",
    "--gradcam-limit",
    "--profile-dir",
    "--tracker",
    "--tracker-project",
    "--tracker-run-name",
    "--tracker-uri",
    "--subset",
    "--log-level",
}


def _append_value_arg(argv: list[str], option: str, value: Any) -> None:
    argv.extend([option, str(value)])


def _train_default(name: str) -> Any:
    return _TRAIN_DEFAULTS[name]


def build_command_line(config: Mapping[str, Any]) -> list[str]:
    """Build the training argv list from UI configuration."""
    argv = ["mammography", "train-density"]

    if config.get("dataset"):
        _append_value_arg(argv, "--dataset", config["dataset"])
    if config.get("csv") and config["csv"] != "classificacao.csv":
        _append_value_arg(argv, "--csv", config["csv"])
    if config.get("dicom_root") and config["dicom_root"] != "archive":
        _append_value_arg(argv, "--dicom-root", config["dicom_root"])
    if config.get("include_class_5"):
        argv.append("--include-class-5")
    if config.get("outdir") != _train_default("outdir"):
        _append_value_arg(argv, "--outdir", config["outdir"])
    if config.get("cache_mode") != HP.CACHE_MODE:
        _append_value_arg(argv, "--cache-mode", config["cache_mode"])
    if config.get("cache_dir") and config["cache_dir"] != "outputs/cache":
        _append_value_arg(argv, "--cache-dir", config["cache_dir"])
    if config.get("embeddings_dir"):
        _append_value_arg(argv, "--embeddings-dir", config["embeddings_dir"])
    if config.get("arch") != _train_default("arch"):
        _append_value_arg(argv, "--arch", config["arch"])
    if config.get("classes") != _train_default("classes"):
        _append_value_arg(argv, "--classes", config["classes"])
    if not config.get("pretrained", _train_default("pretrained")):
        argv.append("--no-pretrained")
    if config.get("view_specific_training"):
        argv.append("--view-specific-training")
        if config.get("views"):
            _append_value_arg(argv, "--views", config["views"])
        if config.get("ensemble_method") != "none":
            _append_value_arg(argv, "--ensemble-method", config["ensemble_method"])
    if config.get("epochs") != HP.EPOCHS:
        _append_value_arg(argv, "--epochs", config["epochs"])
    if config.get("batch_size") != HP.BATCH_SIZE:
        _append_value_arg(argv, "--batch-size", config["batch_size"])
    if config.get("lr") != HP.LR:
        _append_value_arg(argv, "--lr", config["lr"])
    if config.get("backbone_lr") != HP.BACKBONE_LR:
        _append_value_arg(argv, "--backbone-lr", config["backbone_lr"])
    if config.get("weight_decay") != _train_default("weight_decay"):
        _append_value_arg(argv, "--weight-decay", config["weight_decay"])
    if config.get("img_size") != HP.IMG_SIZE:
        _append_value_arg(argv, "--img-size", config["img_size"])
    if config.get("seed") != HP.SEED:
        _append_value_arg(argv, "--seed", config["seed"])
    if config.get("device") != HP.DEVICE:
        _append_value_arg(argv, "--device", config["device"])
    if config.get("val_frac") != HP.VAL_FRAC:
        _append_value_arg(argv, "--val-frac", config["val_frac"])
    if config.get("train_backbone"):
        argv.append("--train-backbone")
    if not config.get("unfreeze_last_block", HP.UNFREEZE_LAST_BLOCK):
        argv.append("--no-unfreeze-last-block")
    if config.get("warmup_epochs") != HP.WARMUP_EPOCHS:
        _append_value_arg(argv, "--warmup-epochs", config["warmup_epochs"])
    if config.get("deterministic"):
        argv.append("--deterministic")
    if not config.get("allow_tf32", HP.ALLOW_TF32):
        argv.append("--no-allow-tf32")
    if config.get("fused_optim"):
        argv.append("--fused-optim")
    if config.get("torch_compile"):
        argv.append("--torch-compile")
    if config.get("amp"):
        argv.append("--amp")
    if config.get("scheduler") != _train_default("scheduler"):
        _append_value_arg(argv, "--scheduler", config["scheduler"])
    if config.get("lr_reduce_patience") != HP.LR_REDUCE_PATIENCE:
        _append_value_arg(argv, "--lr-reduce-patience", config["lr_reduce_patience"])
    if config.get("lr_reduce_factor") != HP.LR_REDUCE_FACTOR:
        _append_value_arg(argv, "--lr-reduce-factor", config["lr_reduce_factor"])
    if config.get("lr_reduce_min_lr") != HP.LR_REDUCE_MIN_LR:
        _append_value_arg(argv, "--lr-reduce-min-lr", config["lr_reduce_min_lr"])
    if config.get("lr_reduce_cooldown") != HP.LR_REDUCE_COOLDOWN:
        _append_value_arg(argv, "--lr-reduce-cooldown", config["lr_reduce_cooldown"])
    if config.get("early_stop_patience") != HP.EARLY_STOP_PATIENCE:
        _append_value_arg(argv, "--early-stop-patience", config["early_stop_patience"])
    if config.get("early_stop_min_delta") != HP.EARLY_STOP_MIN_DELTA:
        _append_value_arg(argv, "--early-stop-min-delta", config["early_stop_min_delta"])
    if config.get("num_workers") != HP.NUM_WORKERS:
        _append_value_arg(argv, "--num-workers", config["num_workers"])
    if config.get("prefetch_factor") != HP.PREFETCH_FACTOR:
        _append_value_arg(argv, "--prefetch-factor", config["prefetch_factor"])
    if not config.get("persistent_workers", HP.PERSISTENT_WORKERS):
        argv.append("--no-persistent-workers")
    if not config.get("loader_heuristics", HP.LOADER_HEURISTICS):
        argv.append("--no-loader-heuristics")
    if config.get("class_weights") != HP.CLASS_WEIGHTS:
        _append_value_arg(argv, "--class-weights", config["class_weights"])
    if config.get("class_weights_alpha") != _train_default("class_weights_alpha"):
        _append_value_arg(argv, "--class-weights-alpha", config["class_weights_alpha"])
    if config.get("sampler_weighted"):
        argv.append("--sampler-weighted")
    if config.get("sampler_alpha") != _train_default("sampler_alpha"):
        _append_value_arg(argv, "--sampler-alpha", config["sampler_alpha"])
    if not config.get("augment", HP.TRAIN_AUGMENT):
        argv.append("--no-augment")
    if config.get("augment_vertical"):
        argv.append("--augment-vertical")
    if config.get("augment_color"):
        argv.append("--augment-color")
    if config.get("augment_rotation_deg") != _train_default("augment_rotation_deg"):
        _append_value_arg(argv, "--augment-rotation-deg", config["augment_rotation_deg"])
    if config.get("gradcam"):
        argv.append("--gradcam")
        if config.get("gradcam_limit") != _train_default("gradcam_limit"):
            _append_value_arg(argv, "--gradcam-limit", config["gradcam_limit"])
    if config.get("save_val_preds"):
        argv.append("--save-val-preds")
    if config.get("export_val_embeddings"):
        argv.append("--export-val-embeddings")
    if config.get("profile"):
        argv.append("--profile")
        if config.get("profile_dir") != _train_default("profile_dir"):
            _append_value_arg(argv, "--profile-dir", config["profile_dir"])
    if config.get("tracker") != "none":
        _append_value_arg(argv, "--tracker", config["tracker"])
        if config.get("tracker_project"):
            _append_value_arg(argv, "--tracker-project", config["tracker_project"])
        if config.get("tracker_run_name"):
            _append_value_arg(argv, "--tracker-run-name", config["tracker_run_name"])
        if config.get("tracker_uri"):
            _append_value_arg(argv, "--tracker-uri", config["tracker_uri"])
    if config.get("subset") != _train_default("subset"):
        _append_value_arg(argv, "--subset", config["subset"])
    if config.get("log_level") != HP.LOG_LEVEL:
        _append_value_arg(argv, "--log-level", config["log_level"])

    return argv


def normalize_training_command(
    command: str | Sequence[str], *, executable: str = sys.executable
) -> str:
    """Normalize a training command for display/logging only.

    String commands are normalized with a narrow ``_BASE_COMMAND`` replacement
    for legacy display. Use ``build_training_argv`` for safe subprocess argv
    construction.
    """
    if not isinstance(command, str):
        return shlex.join(build_training_argv(command, executable=executable))

    clean_command = command.replace("\\\n", " ").replace("  ", " ").strip()
    return clean_command.replace(
        _BASE_COMMAND,
        f'"{executable}" -m mammography.cli train-density',
        1,
    )


def _validate_training_args(args: list[str]) -> None:
    i = 0
    while i < len(args):
        arg = args[i]
        if "\x00" in arg:
            raise ValueError("Training command arguments cannot contain NUL bytes")

        option, has_inline_value, inline_value = arg.partition("=")
        if has_inline_value and option in _VALUE_OPTIONS:
            if not inline_value:
                raise ValueError(f"Missing value for {option}")
            i += 1
            continue

        if arg in _FLAG_OPTIONS:
            i += 1
            continue

        if arg in _VALUE_OPTIONS:
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                raise ValueError(f"Missing value for {arg}")
            if "\x00" in args[i + 1]:
                raise ValueError(f"Invalid NUL byte in value for {arg}")
            i += 2
            continue

        raise ValueError(f"Unsupported training command argument: {arg}")


def build_training_argv(
    command: str | Sequence[str], *, executable: str = sys.executable
) -> list[str]:
    """Build a validated argv list for launching training without a shell."""
    if not isinstance(command, str):
        argv = [str(part) for part in command]
        if argv[:2] == ["mammography", "train-density"]:
            args = argv[2:]
            _validate_training_args(args)
            return [executable, "-m", "mammography.cli", "train-density", *args]
        if len(argv) >= 4 and argv[1:4] == ["-m", "mammography.cli", "train-density"]:
            args = argv[4:]
            _validate_training_args(args)
            return [argv[0], "-m", "mammography.cli", "train-density", *args]
        raise ValueError("Training command must start with 'mammography train-density'")

    clean_command = command.replace("\\\n", " ").replace("  ", " ").strip()
    if clean_command.startswith(_BASE_COMMAND):
        raw_args = clean_command[len(_BASE_COMMAND):].strip()
        args = shlex.split(raw_args) if raw_args else []
        _validate_training_args(args)
        return [executable, "-m", "mammography.cli", "train-density", *args]

    argv = shlex.split(clean_command)
    if len(argv) < 4 or argv[1:4] != ["-m", "mammography.cli", "train-density"]:
        raise ValueError("Training command must start with 'mammography train-density'")
    args = argv[4:]
    _validate_training_args(args)
    return [argv[0], "-m", "mammography.cli", "train-density", *args]


def launch_training(
    command: str | Sequence[str],
    *,
    session_state: Any,
    get_mlflow_client: Callable[[str], Any],
    popen: Any = subprocess.Popen,
) -> None:
    """Launch training in the background and capture output into session state."""
    session_state.training_output = []
    session_state.training_status = "running"
    session_state.active_run_id = None

    config = session_state.training_config
    if config.get("tracker") == "mlflow" and config.get("tracker_uri"):
        client = get_mlflow_client(config["tracker_uri"])
        if client:
            session_state.mlflow_client = client

    try:
        argv = build_training_argv(command)
        process = popen(
            argv,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        session_state.training_process = process
        output_list: list[str] = []
        shared = {
            "training_status": "running",
            "active_run_id": None,
            "training_output": output_list,
            "finished": False,
        }
        session_state._training_shared = shared

        def stream_output() -> None:
            mlflow_run_pattern = re.compile(
                r"MLFLOW_RUN_ID[:\s]*([a-f0-9\-]+)", re.IGNORECASE
            )
            try:
                for line in process.stdout:
                    output_list.append(line.rstrip())
                    if shared["active_run_id"] is None:
                        match = mlflow_run_pattern.search(line)
                        if match:
                            shared["active_run_id"] = match.group(1)
                    if len(output_list) > 1000:
                        del output_list[:-1000]

                process.wait()
                if process.returncode == 0:
                    shared["training_status"] = "completed"
                    output_list.append("\n✅ Training completed successfully!")
                else:
                    shared["training_status"] = "failed"
                    output_list.append(
                        f"\n❌ Training failed with exit code {process.returncode}"
                    )
            except Exception as exc:
                shared["training_status"] = "failed"
                output_list.append(f"\n❌ Error during training: {exc}")
                output_list.append(
                    "\n💡 Check the output above for error details. Common issues:\n"
                    "- Dataset path not found or inaccessible\n"
                    "- Insufficient memory or disk space\n"
                    "- Missing dependencies or incompatible versions\n"
                    "- Invalid hyperparameter values"
                )
            finally:
                shared["finished"] = True

        threading.Thread(target=stream_output, daemon=True).start()
    except ValueError as exc:
        session_state.training_status = "failed"
        session_state.active_run_id = None
        session_state.training_process = None
        session_state.training_output.append(f"❌ Failed to start training: {exc}")
    except Exception as exc:
        session_state.training_status = "failed"
        session_state.active_run_id = None
        session_state.training_process = None
        session_state.training_output.append(f"❌ Failed to start training: {exc}")
        session_state.training_output.append(
            "\n💡 This may happen if:\n"
            "- The mammography command is not in your PATH\n"
            "- Python environment is not activated\n"
            "- The command syntax is malformed\n"
            "- System resources are exhausted"
        )


def stop_training(session_state: Any) -> None:
    """Stop the running training process."""
    shared = getattr(session_state, "_training_shared", None)
    if shared is not None and shared.get("finished"):
        session_state.training_process = None
        return

    if session_state.training_process is not None:
        try:
            session_state.training_process.terminate()
            session_state.training_output.append("\n⚠️ Training stopped by user")
            session_state.training_status = "failed"
            if shared is not None:
                shared["training_status"] = "failed"
                shared["finished"] = True
        except Exception as exc:
            session_state.training_output.append(f"❌ Error stopping training: {exc}")
        finally:
            session_state.training_process = None
