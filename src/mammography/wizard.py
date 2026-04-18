# ruff: noqa: F401
"""Interactive wizard facade."""

from __future__ import annotations

from mammography import wizard_core as _core
from mammography import wizard_flows as _flows
from mammography.wizard_core import (
    WizardCommand,
    _ask_choice,
    _ask_config_args,
    _ask_extra_args,
    _ask_float,
    _ask_int,
    _ask_optional,
    _ask_string,
    _ask_yes_no,
    _build_cli_command,
    _print_progress,
    _run_command,
    detect_dataset_format,
    suggest_preprocessing,
    validate_format,
)

_ORIGINAL_VALIDATE_DATASET_PATH = _core._validate_dataset_path
_ORIGINAL_DATASET_PROMPT = _core._dataset_prompt


def _validate_dataset_path(path: str, expected_type: str | None = None) -> bool:
    _core.detect_dataset_format = detect_dataset_format
    _core.validate_format = validate_format
    _core.suggest_preprocessing = suggest_preprocessing
    return _ORIGINAL_VALIDATE_DATASET_PATH(path, expected_type)


def _dataset_prompt() -> tuple[list[str], str | None, str | None]:
    _sync_wizard_flows()
    _core._validate_dataset_path = _validate_dataset_path
    return _ORIGINAL_DATASET_PROMPT()


def _sync_wizard_flows() -> None:
    _flows._dataset_prompt = _dataset_prompt
    _flows._validate_dataset_path = _validate_dataset_path


def _wizard_train() -> WizardCommand:
    _sync_wizard_flows()
    return _flows._wizard_train()


def _wizard_quick_train() -> WizardCommand:
    _sync_wizard_flows()
    return _flows._wizard_quick_train()


def _wizard_embed() -> WizardCommand:
    _sync_wizard_flows()
    return _flows._wizard_embed()


def _wizard_visualize() -> WizardCommand:
    _sync_wizard_flows()
    return _flows._wizard_visualize()


def _wizard_embeddings_baselines() -> WizardCommand:
    _sync_wizard_flows()
    return _flows._wizard_embeddings_baselines()


def _wizard_inference() -> WizardCommand:
    _sync_wizard_flows()
    return _flows._wizard_inference()


def _wizard_batch_inference() -> WizardCommand:
    _sync_wizard_flows()
    return _flows._wizard_batch_inference()


def _wizard_augment() -> WizardCommand:
    _sync_wizard_flows()
    return _flows._wizard_augment()


def _wizard_label_density() -> WizardCommand:
    return _flows._wizard_label_density()


def _wizard_label_patches() -> WizardCommand:
    return _flows._wizard_label_patches()


def _wizard_eda() -> WizardCommand:
    return _flows._wizard_eda()


def _wizard_eval_export() -> WizardCommand:
    return _flows._wizard_eval_export()


def _wizard_data_audit() -> WizardCommand:
    return _flows._wizard_data_audit()


def _wizard_report_pack() -> WizardCommand:
    return _flows._wizard_report_pack()


def run_wizard(dry_run: bool = False) -> int:
    """
    Display an interactive wizard menu of tasks, run the selected task's command, and return its exit code.

    Parameters:
        dry_run (bool): If True, commands are not actually executed and are only shown.

    Returns:
        int: Exit code of the executed command, or 0 if the wizard was exited without running a command.
    """
    print("\n=== MAMMOGRAPHY WIZARD ===")
    options = [
        "Treinamento de densidade",
        "Treino rapido de densidade",
        "Extracao de embeddings",
        "Baselines classicos (embeddings)",
        "Visualizacao de embeddings",
        "Inferencia",
        "Inferencia em lote (batch)",
        "Augmentacao de dados",
        "Rotular densidade (GUI)",
        "Rotular patches (GUI)",
        "EDA cancer",
        "Auditoria de dados",
        "Eval-export",
        "Report-pack (Article)",
        "Sair",
    ]
    choice = _ask_choice("Selecione a tarefa:", options, default=0)
    if choice == 0:
        cmd = _wizard_train()
    elif choice == 1:
        cmd = _wizard_quick_train()
    elif choice == 2:
        cmd = _wizard_embed()
    elif choice == 3:
        cmd = _wizard_embeddings_baselines()
    elif choice == 4:
        cmd = _wizard_visualize()
    elif choice == 5:
        cmd = _wizard_inference()
    elif choice == 6:
        cmd = _wizard_batch_inference()
    elif choice == 7:
        cmd = _wizard_augment()
    elif choice == 8:
        cmd = _wizard_label_density()
    elif choice == 9:
        cmd = _wizard_label_patches()
    elif choice == 10:
        cmd = _wizard_eda()
    elif choice == 11:
        cmd = _wizard_data_audit()
    elif choice == 12:
        cmd = _wizard_eval_export()
    elif choice == 13:
        cmd = _wizard_report_pack()
    else:
        print("Saindo do wizard.")
        return 0

    return _run_command(cmd, dry_run)
