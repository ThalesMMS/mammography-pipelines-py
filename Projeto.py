#!/usr/bin/env python3
"""Projeto.py — Windows-friendly orchestrator for the mammography pipelines.

Sample usage
------------
python Projeto.py embed -- --data_dir ./archive --csv_path classificacao.csv
python Projeto.py train-density --dry-run -- --epochs 1 --subset 32
python Projeto.py eval-export --config configs/paths.yaml
python Projeto.py rl-refine --policy-config configs/rl/policy.yaml --episodes 4 --dry-run

Unknown arguments are forwarded verbatim to the wrapped scripts, so existing
CLI flags keep working. Use ``--dry-run`` to preview commands without
executing subprocesses.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

try:  # Optional dependency for YAML configs.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML is not guaranteed to exist.
    yaml = None

LOGGER = logging.getLogger("projeto")
REPO_ROOT = Path(__file__).resolve().parent
# Preserve the original sys.executable (even if it's a venv shim) so subprocesses reuse the same environment.
PYTHON = Path(sys.executable)

DEFAULT_CONFIGS: dict[str, Path | None] = {
    "embed": REPO_ROOT / "configs" / "paths.yaml",
    "train-density": REPO_ROOT / "configs" / "density.yaml",
    "eval-export": None,
    "rl-refine": None,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Projeto.py",
        description=(
            "Orquestra extração de embeddings, treino EfficientNet e refinamento RL "
            "para os pipelines de mamografia."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exibe o comando planejado e evita executar subprocessos/loops pesados.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Define o nível de log (default: INFO).",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="subcommand")

    def _add_config_argument(sp: argparse.ArgumentParser, summary: str) -> None:
        sp.description = summary
        sp.add_argument(
            "--config",
            type=Path,
            help=(
                "Arquivo YAML/JSON com parâmetros extras. Valores são expandidos "
                "antes dos argumentos encaminhados, e a linha de comando prevalece."
            ),
        )

    embed_parser = subparsers.add_parser(
        "embed",
        help="Envolve extract_mammo_resnet50.py para gerar embeddings.",
    )
    _add_config_argument(
        embed_parser,
        "Encaminha argumentos ao extract_mammo_resnet50.py (Stage 1).",
    )

    density_parser = subparsers.add_parser(
        "train-density",
        help="Envolve RSNA_Mammo_EfficientNetB0_Density.py (Stage 2).",
    )
    _add_config_argument(
        density_parser,
        "Executa o treinamento EfficientNetB0; argumentos desconhecidos são encaminhados.",
    )

    eval_parser = subparsers.add_parser(
        "eval-export",
        help="Exibe um checklist para empacotar métricas/figuras da Etapa 2.",
    )
    _add_config_argument(
        eval_parser,
        "Checklist de exportação e apontamento de diretórios para Article/assets.",
    )
    eval_parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        type=Path,
        help="Diretório results_* a validar (pode repetir o argumento para múltiplos).",
    )
    pack_parser = subparsers.add_parser(
        "report-pack",
        help="Empacota runs da Etapa 2 em Article/assets e atualiza o LaTeX.",
    )
    pack_parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        type=Path,
        help="Diretório results_* com summary.json/métricas. Use múltiplas vezes para vários seeds.",
    )
    pack_parser.add_argument(
        "--assets-dir",
        dest="assets_dir",
        type=Path,
        default=Path("Article") / "assets",
        help="Destino das figuras copiadas (default: Article/assets).",
    )
    pack_parser.add_argument(
        "--tex",
        dest="tex_path",
        type=Path,
        default=Path("Article") / "sections" / "stage2_model.tex",
        help="Arquivo LaTeX que receberá a seção Stage 2 (default: Article/sections/stage2_model.tex).",
    )
    pack_parser.add_argument(
        "--gradcam-limit",
        dest="gradcam_limit",
        type=int,
        default=4,
        help="Quantidade máxima de Grad-CAMs combinadas em cada collage (default: 4).",
    )

    rl_parser = subparsers.add_parser(
        "rl-refine",
        help="Aciona o stub de refinamento por RL (Stage 3).",
    )
    _add_config_argument(
        rl_parser,
        "Invoca rl_refinement.train.main; aceita --policy-config via encaminhamento.",
    )

    return parser


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s | %(message)s", force=True)


def _format_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return " ".join(shlex.quote(part) for part in command)


def _read_config(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    if not text.strip():
        return {}
    return json.loads(text)


def _dict_to_cli_args(payload: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in payload.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                args.extend([flag, str(item)])
            continue
        args.extend([flag, str(value)])
    return args


def _coerce_cli_args(payload: Any) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, str):
        return shlex.split(payload)
    if isinstance(payload, dict):
        return _dict_to_cli_args(payload)
    if isinstance(payload, (list, tuple)):
        return [str(item) for item in payload]
    return [str(payload)]


def _default_config(command: str) -> Path | None:
    candidate = DEFAULT_CONFIGS.get(command)
    if candidate and candidate.exists():
        return candidate
    return None


def _load_config_args(config_arg: Path | None, command: str) -> list[str]:
    config_path = config_arg or _default_config(command)
    if not config_path:
        return []
    resolved = config_path.resolve()
    if not resolved.exists():
        LOGGER.warning("Config %s não encontrado; ignorando.", resolved)
        return []
    try:
        data = _read_config(resolved) or {}
    except Exception as exc:
        LOGGER.warning("Falha ao ler %s: %s", resolved, exc)
        return []

    args: list[str] = []
    if isinstance(data, dict):
        global_payload = data.get("global")
        if global_payload is not None:
            args.extend(_coerce_cli_args(global_payload))
        normalized_keys = {command, command.replace("-", "_")}
        for key in normalized_keys:
            if key in data:
                args.extend(_coerce_cli_args(data[key]))
    else:
        args.extend(_coerce_cli_args(data))

    if args:
        LOGGER.debug("Args carregados de %s (%s): %s", resolved, command, args)
    return args


def _run_passthrough(script_fragment: str, args: argparse.Namespace, forwarded: Sequence[str]) -> None:
    script = (REPO_ROOT / script_fragment).resolve()
    if not script.exists():
        raise FileNotFoundError(f"Script não encontrado: {script}")
    config_args = _load_config_args(getattr(args, "config", None), args.command)
    command = [str(PYTHON), str(script), *config_args, *forwarded]
    LOGGER.info("Executando: %s", _format_command(command))
    if args.dry_run:
        LOGGER.info("Dry-run habilitado; subprocesso não será iniciado.")
        return
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def _print_eval_guidance(args: argparse.Namespace, forwarded: Sequence[str]) -> None:
    config_args = _load_config_args(getattr(args, "config", None), args.command)
    if config_args:
        LOGGER.info("Argumentos sugeridos (config): %s", " ".join(config_args))
    if forwarded:
        LOGGER.info("Argumentos adicionais ignorados: %s", " ".join(forwarded))
    checklist = [
        "Reutilize checkpoints aprovados (outputs/mammo_efficientnetb0_density/results_*).",
        "Exporte val_predictions.csv, embeddings_val.*, metrics/val_metrics.{json,png}.",
        "Gere figuras (ROC, confusion, Grad-CAM) e copie para Article/assets via report-pack.",
        "Versione os arquivos (timestamp + git SHA) na pasta Article/assets/.",
    ]
    LOGGER.info("Checklist de exportação:")
    for item in checklist:
        LOGGER.info(" • %s", item)
    run_list = getattr(args, "runs", None) or []
    if run_list:
        required = [
            "summary.json",
            "train_history.csv",
            "train_history.png",
            "metrics/val_metrics.json",
            "metrics/val_metrics.png",
            "val_predictions.csv",
            "embeddings_val.csv",
            "embeddings_val.npz",
            "run.log",
            "best_model.pt",
        ]
        for run in run_list:
            run_path = (REPO_ROOT / run) if not run.is_absolute() else run
            LOGGER.info("Auditando artefatos em: %s", run_path)
            missing = [rel for rel in required if not (run_path / rel).exists()]
            if missing:
                LOGGER.warning("Itens faltantes: %s", ", ".join(missing))
            else:
                LOGGER.info("Todos os arquivos obrigatórios estão presentes.")
            summary_path = run_path / "summary.json"
            if summary_path.exists():
                try:
                    payload = json.loads(summary_path.read_text(encoding="utf-8"))
                    metrics = payload.get("val_metrics", {})
                    LOGGER.info(
                        "Resumo: seed=%s | acc=%.3f | κ=%.3f | macro-F1=%.3f | AUC=%.3f",
                        payload.get("seed", "?"),
                        float(metrics.get("accuracy", 0.0)),
                        float(metrics.get("kappa_quadratic", 0.0)),
                        float(metrics.get("macro_f1", 0.0)),
                        float(metrics.get("auc_ovr", 0.0)),
                    )
                except Exception as exc:
                    LOGGER.warning("Falha ao ler summary.json em %s: %s", summary_path, exc)


def _stub_rl_refine(cli_args: Sequence[str], dry_run: bool) -> None:
    LOGGER.info("rl_refinement não disponível; executando stub.")
    LOGGER.info("Args encaminhados: %s", " ".join(cli_args) or "<nenhum>")
    if dry_run:
        LOGGER.info("Dry-run global; nada a executar.")
    else:
        LOGGER.info("Crie o pacote rl_refinement para substituir este stub.")


def _invoke_rl_refine(args: argparse.Namespace, forwarded: Sequence[str]) -> None:
    config_args = _load_config_args(getattr(args, "config", None), args.command)
    cli_args = [*config_args, *forwarded]
    if args.dry_run and "--dry-run" not in cli_args:
        cli_args = ["--dry-run", *cli_args]
    try:
        from rl_refinement import train as rl_train  # type: ignore
    except Exception as exc:  # pragma: no cover - optional module
        LOGGER.debug("Falha ao importar rl_refinement: %s", exc)
        _stub_rl_refine(cli_args, args.dry_run)
        return

    runner = getattr(rl_train, "main", None)
    if not callable(runner):
        LOGGER.warning("rl_refinement.train.main não encontrado; usando stub.")
        _stub_rl_refine(cli_args, args.dry_run)
        return

    LOGGER.info("Delegando ao rl_refinement.train.main")
    exit_code = runner(cli_args)
    if exit_code:
        raise SystemExit(exit_code)


def _run_report_pack(args: argparse.Namespace, forwarded: Sequence[str]) -> None:
    if forwarded:
        LOGGER.warning("Argumentos adicionais ignorados pelo report-pack: %s", " ".join(forwarded))
    if not args.runs:
        raise SystemExit("Informe pelo menos um --run outputs/.../results_* para empacotar.")
    from tools import report_pack  # Import adiado para evitar dependência desnecessária.

    run_paths = []
    for run in args.runs:
        path = (REPO_ROOT / run) if not run.is_absolute() else run
        run_paths.append(path)
    assets_dir = (REPO_ROOT / args.assets_dir) if not args.assets_dir.is_absolute() else args.assets_dir
    tex_path = None
    if args.tex_path:
        tex_path = (REPO_ROOT / args.tex_path) if not args.tex_path.is_absolute() else args.tex_path

    LOGGER.info("Empacotando runs: %s", ", ".join(str(p) for p in run_paths))
    report_pack.package_stage2_runs(
        run_paths,
        assets_dir,
        tex_path=tex_path,
        gradcam_limit=args.gradcam_limit,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args, forwarded = parser.parse_known_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    _configure_logging(args.log_level)
    LOGGER.debug("Forwarded args: %s", forwarded)

    try:
        if args.command == "embed":
            _run_passthrough("extract_mammo_resnet50.py", args, forwarded)
        elif args.command == "train-density":
            _run_passthrough("RSNA_Mammo_EfficientNetB0_Density.py", args, forwarded)
        elif args.command == "eval-export":
            _print_eval_guidance(args, forwarded)
        elif args.command == "report-pack":
            _run_report_pack(args, forwarded)
        elif args.command == "rl-refine":
            _invoke_rl_refine(args, forwarded)
        else:
            parser.error(f"Subcomando desconhecido: {args.command}")
        return 0
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 1
    except subprocess.CalledProcessError as exc:
        LOGGER.error("Comando falhou (exit=%s)", exc.returncode)
        return exc.returncode


if __name__ == "__main__":
    sys.exit(main())
