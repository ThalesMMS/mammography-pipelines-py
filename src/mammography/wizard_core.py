# ruff: noqa
#
# wizard.py
# mammography-pipelines
#
# Interactive wizard for guided mammography pipeline workflows.
# DISCLAIMER: Educational project only - NOT for clinical or medical diagnostic purposes.
#
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from mammography.config import HP
from mammography.data.csv_loader import DATASET_PRESETS
from mammography.data.format_detection import (
    detect_dataset_format,
    validate_format,
    suggest_preprocessing,
)
from mammography.utils.smart_defaults import SmartDefaults
from mammography.utils.wizard_helpers import (
    ask_choice_with_help,
    ask_float_with_help,
    ask_int_with_help,
    ask_optional_with_help,
    ask_with_help,
    ask_yes_no_with_help,
    print_help,
)

@dataclass
class WizardCommand:
    label: str
    argv: list[str]

def _print_progress(current: int, total: int, section: str = "") -> None:
    """
    Print progress indicator for wizard steps.

    Args:
        current: Current step number (1-indexed)
        total: Total number of steps
        section: Optional section description
    """
    progress_text = f"\n{'='*60}\nPasso {current} de {total}"
    if section:
        progress_text += f": {section}"
    progress_text += f"\n{'='*60}"
    print(progress_text)

def _ask_choice(title: str, options: Sequence[str], default: int = 0) -> int:
    print(f"\n{title}")
    for i, option in enumerate(options):
        print(f"  [{i}] {option}")
    while True:
        raw = input(f"Escolha [{default}]: ").strip()
        if raw == "":
            return default
        if raw.isdigit():
            value = int(raw)
            if 0 <= value < len(options):
                return value
        print(f"Opcao invalida. Informe um numero entre 0 e {len(options) - 1}.")

def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{suffix}]: ").strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes", "s", "sim"}:
            return True
        if raw in {"n", "no", "nao"}:
            return False
        print("Resposta invalida. Use y/n.")

def _ask_string(prompt: str, default: str | None = None) -> str:
    if default:
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw or default
    return input(f"{prompt}: ").strip()

def _ask_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        if raw.isdigit():
            return int(raw)
        print("Informe um numero inteiro valido.")

def _ask_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print("Informe um numero valido.")

def _ask_optional(prompt: str) -> str | None:
    raw = input(f"{prompt} (enter para pular): ").strip()
    return raw or None

def _ask_config_args() -> list[str]:
    path = _ask_optional("Config YAML/JSON")
    if path:
        return ["--config", path]
    return []

def _ask_extra_args() -> list[str]:
    raw = input("Args extras (opcional, ex: --foo 1 --bar): ").strip()
    return shlex.split(raw) if raw else []

def _validate_dataset_path(path: str, expected_type: str | None = None) -> bool:
    """
    Validate dataset path and show format detection results.

    Args:
        path: Path to validate (file or directory)
        expected_type: Expected dataset type (archive, mamografias, patches_completo, custom)

    Returns:
        True if user wants to proceed, False if validation failed and user wants to retry
    """
    if not path:
        print("\nERRO: Caminho vazio fornecido.")
        return False

    path_obj = Path(path)

    # Check if path exists
    if not path_obj.exists():
        print(f"\nERRO: Caminho nao encontrado: {path}")
        print("\nSugestoes:")
        print("  - Verifique se o caminho esta correto")
        print("  - Use caminho absoluto ou relativo valido")
        print("  - Verifique se tem permissoes de leitura")

        # Try to provide helpful suggestions based on path
        parent = path_obj.parent
        if parent.exists():
            print(f"\nDiretorio pai existe: {parent}")
            similar = [
                str(p.name)
                for p in parent.iterdir()
                if path_obj.stem.lower() in p.name.lower()
            ]
            if similar:
                print(f"Arquivos/diretorios similares encontrados: {', '.join(similar)}")

        return False

    # For CSV files, just check if they're files
    if path_obj.suffix.lower() in {".csv", ".tsv", ".txt"}:
        if not path_obj.is_file():
            print(f"\nERRO: Caminho existe mas nao e um arquivo: {path}")
            return False
        print(f"\n✓ Arquivo encontrado: {path}")
        return True

    # For directories, run format detection
    if not path_obj.is_dir():
        print(f"\nAVISO: Caminho existe mas nao e um diretorio: {path}")
        if _ask_yes_no("Prosseguir mesmo assim?", False):
            return True
        return False

    print(f"\n→ Detectando formato do dataset em: {path}")

    try:
        # Run format detection
        fmt = detect_dataset_format(str(path))

        # Display detection results
        print(f"\nFormato detectado:")
        print(f"  Tipo: {fmt.dataset_type}")
        print(f"  Formato de imagem: {fmt.image_format}")
        print(f"  Total de imagens: {fmt.image_count}")

        if fmt.csv_path:
            print(f"  Metadata: {fmt.csv_path}")
        if fmt.dicom_root:
            print(f"  DICOM root: {fmt.dicom_root}")

        if fmt.format_counts:
            print(f"\n  Distribuicao de formatos:")
            for ext, count in fmt.format_counts.items():
                percentage = (count / fmt.image_count * 100) if fmt.image_count > 0 else 0
                print(f"    {ext}: {count} ({percentage:.1f}%)")

        # Run validation and show warnings
        warnings = validate_format(fmt)
        if warnings:
            print(f"\n⚠ Avisos de validacao ({len(warnings)}):")
            for i, warning in enumerate(warnings[:5], 1):  # Show first 5 warnings
                print(f"  {i}. {warning}")
            if len(warnings) > 5:
                print(f"  ... e mais {len(warnings) - 5} avisos")

        # Check if detected type matches expected type
        if expected_type and expected_type != "custom" and fmt.dataset_type != expected_type:
            print(
                f"\n⚠ AVISO: Tipo detectado ({fmt.dataset_type}) difere do esperado ({expected_type})"
            )

        # Show preprocessing suggestions if any issues found
        if warnings or fmt.image_count == 0:
            suggestions = suggest_preprocessing(fmt)
            if suggestions:
                print(f"\nSugestoes de preprocessamento:")
                for i, suggestion in enumerate(suggestions[:3], 1):  # Show first 3
                    print(f"  {i}. {suggestion}")
                if len(suggestions) > 3:
                    print(f"  ... e mais {len(suggestions) - 3} sugestoes")

        # Critical errors - don't allow proceeding
        if fmt.image_count == 0:
            print("\nERRO CRITICO: Nenhuma imagem encontrada no dataset.")
            print("Nao e possivel prosseguir com dataset vazio.")
            return False

        # Ask user if they want to proceed despite warnings
        if warnings:
            if not _ask_yes_no("\nProsseguir mesmo com avisos?", True):
                return False

        print("\n✓ Validacao concluida")
        return True

    except ValueError as exc:
        print(f"\nERRO: {exc}")
        return False
    except Exception as exc:
        print(f"\nERRO inesperado durante validacao: {exc!r}")
        print("\nSugestoes:")
        print("  - Verifique se o diretorio tem permissoes de leitura")
        print("  - Verifique se o diretorio contem imagens validas")
        if _ask_yes_no("Prosseguir mesmo assim?", False):
            return True
        return False

def _run_command(cmd: WizardCommand, dry_run: bool) -> int:
    print("\nResumo do comando:")
    print(f"  {cmd.label}")
    print("  " + " ".join(shlex.quote(part) for part in cmd.argv))
    if not _ask_yes_no("Executar agora?"):
        print("Comando cancelado.")
        return 0
    if dry_run:
        print("Dry-run habilitado; comando nao sera executado.")
        return 0
    completed = subprocess.run(cmd.argv, check=False)
    return completed.returncode

def _dataset_prompt() -> tuple[list[str], str | None, str | None]:
    presets = list(DATASET_PRESETS.keys())
    options = presets + ["custom"]
    idx = _ask_choice("Formato do dataset:", options, default=0)
    selection = options[idx]
    args: list[str] = []
    csv_path = None
    dicom_root = None
    if selection != "custom":
        args.extend(["--dataset", selection])

    # Loop until valid paths are provided
    while True:
        if selection == "archive":
            csv_path = _ask_string("CSV (classificacao.csv)", "classificacao.csv")
            dicom_root = _ask_string("Diretorio DICOM", "archive")

            # Validate CSV path
            if csv_path and not _validate_dataset_path(csv_path, None):
                if not _ask_yes_no("Tentar novamente com outro caminho?", True):
                    break
                continue

            # Validate DICOM root directory
            if dicom_root and not _validate_dataset_path(dicom_root, "archive"):
                if not _ask_yes_no("Tentar novamente com outro caminho?", True):
                    break
                continue

            # Both paths validated successfully
            break

        elif selection in {"mamografias", "patches_completo"}:
            csv_path = _ask_string("Diretorio com featureS.txt", selection)

            # Validate directory path
            if csv_path and not _validate_dataset_path(csv_path, selection):
                if not _ask_yes_no("Tentar novamente com outro caminho?", True):
                    break
                continue

            # Path validated successfully
            break

        else:  # custom
            csv_path = _ask_string("CSV ou diretorio com featureS.txt")
            dicom_root = _ask_string("Diretorio DICOM (opcional)", "")
            if not dicom_root:
                dicom_root = None

            # Validate CSV/directory path
            if csv_path and not _validate_dataset_path(csv_path, "custom"):
                if not _ask_yes_no("Tentar novamente com outro caminho?", True):
                    break
                continue

            # Validate DICOM root if provided
            if dicom_root and not _validate_dataset_path(dicom_root, None):
                if not _ask_yes_no("Tentar novamente com outro caminho?", True):
                    break
                continue

            # Paths validated successfully
            break

    if csv_path:
        args.extend(["--csv", csv_path])
    if dicom_root:
        args.extend(["--dicom-root", dicom_root])
    return args, csv_path, dicom_root

def _build_cli_command(subcommand: str, args: Iterable[str]) -> list[str]:
    return [sys.executable, "-m", "mammography.cli", subcommand, *args]
