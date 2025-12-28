#!/usr/bin/env python3
#
# cli.py
# mammography-pipelines
#
# Orchestrates CLI subcommands for embedding extraction, density training, eval export, and visualization.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""CLI entrypoint for the mammography pipelines.

Sample usage
------------
mammography embed -- --data_dir ./archive --csv_path classificacao.csv
mammography train-density --dry-run -- --epochs 1 --subset 32
mammography eval-export --config configs/paths.yaml
mammography wizard
Compatibilidade:
python -m mammography.cli embed -- --data_dir ./archive --csv_path classificacao.csv

Unknown arguments are forwarded verbatim to the internal command modules, so
existing CLI flags keep working. Use ``--dry-run`` to preview commands without
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
REPO_ROOT = Path(__file__).resolve().parents[2]
# Preserve the original sys.executable (even if it's a venv shim) so subprocesses reuse the same environment.
PYTHON = Path(sys.executable)


DEFAULT_CONFIGS: dict[str, Path | None] = {
    "embed": REPO_ROOT / "configs" / "paths.yaml",
    "train-density": REPO_ROOT / "configs" / "density.yaml",
    "eval-export": None,
    "visualize": None,
    "embeddings-baselines": None,
    "data-audit": None,
}


def _build_parser() -> argparse.ArgumentParser:
    """Define the CLI with subcommands that wrap the internal command modules."""
    parser = argparse.ArgumentParser(
        prog="mammography",
        description=(
            "Orquestra extração de embeddings, treino EfficientNet/ResNet e análises "
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
        help="Extrai embeddings e opcionalmente executa reducao/clustering.",
    )
    _add_config_argument(
        embed_parser,
        "Encaminha argumentos ao comando de extracao de embeddings.",
    )

    density_parser = subparsers.add_parser(
        "train-density",
        help="Treina modelos de densidade (EfficientNetB0/ResNet50).",
    )
    _add_config_argument(
        density_parser,
        "Executa o treinamento; argumentos desconhecidos sao encaminhados.",
    )

    eval_parser = subparsers.add_parser(
        "eval-export",
        help="Exibe um checklist para empacotar métricas/figuras do treino de densidade.",
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
        help="Empacota runs de densidade em Article/assets e atualiza o LaTeX.",
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
        default=Path("Article") / "sections" / "density_model.tex",
        help="Arquivo LaTeX que receberá a seção de densidade (default: Article/sections/density_model.tex).",
    )
    pack_parser.add_argument(
        "--gradcam-limit",
        dest="gradcam_limit",
        type=int,
        default=4,
        help="Quantidade máxima de Grad-CAMs combinadas em cada collage (default: 4).",
    )

    audit_parser = subparsers.add_parser(
        "data-audit",
        help="Audita o acervo DICOM e gera manifest/logs.",
    )
    _add_config_argument(
        audit_parser,
        "Inventario de arquivos DICOM e checagem de integridade/rotulos.",
    )
    audit_parser.add_argument(
        "--archive",
        type=Path,
        default=Path("archive"),
        help="Diretorio raiz com acessos DICOM (default: archive).",
    )
    audit_parser.add_argument(
        "--csv",
        type=Path,
        default=Path("classificacao.csv"),
        help="CSV com rotulos (default: classificacao.csv).",
    )
    audit_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data_manifest.json"),
        help="Arquivo JSON resumido (default: data_manifest.json).",
    )
    audit_parser.add_argument(
        "--audit-csv",
        dest="audit_csv",
        type=Path,
        default=Path("outputs/embeddings_resnet50/data_audit.csv"),
        help="CSV detalhado de auditoria (default: outputs/embeddings_resnet50/data_audit.csv).",
    )
    audit_parser.add_argument(
        "--log",
        type=Path,
        default=Path("Article/assets/data_qc.log"),
        help="Log textual para o artigo (default: Article/assets/data_qc.log).",
    )

    # Visualization subcommand
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Gera visualizações (t-SNE, heatmaps, scatterplots) de embeddings.",
    )
    _add_config_argument(
        viz_parser,
        "Encaminha argumentos ao comando de visualizacao de embeddings.",
    )
    viz_parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Caminho para features (.npy/.npz) ou diretório de run (com --from-run).",
    )
    viz_parser.add_argument(
        "--labels",
        type=Path,
        help="CSV com rótulos (coluna raw_label/label/class).",
    )
    viz_parser.add_argument(
        "--label-col",
        dest="label_col",
        default="raw_label",
        help="Nome da coluna de rótulos (default: raw_label).",
    )
    viz_parser.add_argument(
        "--predictions",
        type=Path,
        help="CSV com predições (true/pred) para matriz de confusão.",
    )
    viz_parser.add_argument(
        "--history",
        type=Path,
        help="CSV/JSON com histórico de treino para curvas de aprendizado.",
    )
    viz_parser.add_argument(
        "--from-run",
        action="store_true",
        help="Trata --input como diretório de run e descobre artefatos automaticamente.",
    )
    viz_parser.add_argument(
        "--output", "-o", "--outdir",
        dest="output",
        type=Path,
        default=Path("outputs/visualizations"),
        help="Diretório de saída para visualizações (default: outputs/visualizations).",
    )
    viz_parser.add_argument(
        "--prefix",
        default="",
        help="Prefixo para nomes dos arquivos de saída.",
    )
    viz_parser.add_argument(
        "--report",
        action="store_true",
        help="Gera relatório completo de visualizações.",
    )
    viz_parser.add_argument(
        "--tsne",
        action="store_true",
        help="Gera plot t-SNE 2D.",
    )
    viz_parser.add_argument(
        "--tsne-3d",
        action="store_true",
        help="Gera plot t-SNE 3D.",
    )
    viz_parser.add_argument(
        "--pca",
        action="store_true",
        help="Gera scatter plot PCA.",
    )
    viz_parser.add_argument(
        "--umap",
        action="store_true",
        help="Gera scatter plot UMAP.",
    )
    viz_parser.add_argument(
        "--compare-embeddings",
        action="store_true",
        help="Compara PCA, t-SNE e UMAP lado a lado.",
    )
    viz_parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Gera heatmap de correlação de features.",
    )
    viz_parser.add_argument(
        "--feature-heatmap",
        action="store_true",
        help="Gera heatmap clusterizado de features.",
    )
    viz_parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Gera heatmap de matriz de confusão (requer predictions).",
    )
    viz_parser.add_argument(
        "--scatter-matrix",
        action="store_true",
        help="Gera matriz de scatter plots pareados.",
    )
    viz_parser.add_argument(
        "--distribution",
        action="store_true",
        help="Gera plots de distribuição de features.",
    )
    viz_parser.add_argument(
        "--class-separation",
        action="store_true",
        help="Gera análise de separação de classes.",
    )
    viz_parser.add_argument(
        "--learning-curves",
        action="store_true",
        help="Gera curvas de aprendizado (requer history).",
    )
    viz_parser.add_argument(
        "--binary",
        action="store_true",
        help="Usa nomes de classes binários (Low/High Density).",
    )
    viz_parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Perplexity para t-SNE (default: 30).",
    )
    viz_parser.add_argument(
        "--tsne-iter",
        dest="tsne_iter",
        type=int,
        default=1000,
        help="Iterações do t-SNE (default: 1000).",
    )
    viz_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reduções/plots (default: 42).",
    )
    viz_parser.add_argument(
        "--pca-svd-solver",
        dest="pca_svd_solver",
        default="auto",
        choices=["auto", "full", "randomized", "arpack"],
        help="Solver do PCA (auto/full/randomized/arpack).",
    )

    wizard_parser = subparsers.add_parser(
        "wizard",
        help="Abre o menu interativo com passos guiados.",
    )
    _add_config_argument(
        wizard_parser,
        "Assistente interativo para escolher tarefas e parametros.",
    )

    infer_parser = subparsers.add_parser(
        "inference",
        help="Executa inferencia em imagens/DICOMs usando checkpoint treinado.",
    )
    _add_config_argument(
        infer_parser,
        "Encaminha argumentos ao comando de inferencia.",
    )

    augment_parser = subparsers.add_parser(
        "augment",
        help="Gera augmentations a partir de um diretorio de imagens.",
    )
    _add_config_argument(
        augment_parser,
        "Encaminha argumentos ao comando de augmentacao.",
    )

    label_density_parser = subparsers.add_parser(
        "label-density",
        help="Abre a interface de rotulagem de densidade.",
    )
    _add_config_argument(
        label_density_parser,
        "Encaminha argumentos ao rotulador de densidade.",
    )

    label_patches_parser = subparsers.add_parser(
        "label-patches",
        help="Abre a interface de rotulagem de patches.",
    )
    _add_config_argument(
        label_patches_parser,
        "Encaminha argumentos ao rotulador de patches.",
    )

    eda_parser = subparsers.add_parser(
        "eda-cancer",
        help="Executa o notebook/script de EDA de cancer.",
    )
    _add_config_argument(
        eda_parser,
        "Encaminha argumentos ao fluxo de EDA para cancer.",
    )

    baselines_parser = subparsers.add_parser(
        "embeddings-baselines",
        help="Compara embeddings com descritores classicos.",
    )
    _add_config_argument(
        baselines_parser,
        "Encaminha argumentos ao comparativo de baselines.",
    )

    return parser


def _configure_logging(level: str) -> None:
    """Configure root logging for both CLI feedback and debug traces."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s | %(message)s", force=True)


def _format_command(command: Sequence[str]) -> str:
    """Format a subprocess command so it is easy to copy/paste/debug."""
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return " ".join(shlex.quote(part) for part in command)


def _read_config(path: Path) -> Any:
    """Read a YAML/JSON config file and return its payload."""
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    if not text.strip():
        return {}
    return json.loads(text)


def _dict_to_cli_args(payload: dict[str, Any]) -> list[str]:
    """Convert a dictionary of arguments into flat CLI flags."""
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
    """Accept bool/iterable/string inputs and normalize to a list of CLI tokens."""
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
    """Return the default config path for the subcommand if it exists."""
    candidate = DEFAULT_CONFIGS.get(command)
    if candidate and candidate.exists():
        return candidate
    return None


def _load_config_args(config_arg: Path | None, command: str) -> list[str]:
    """Load YAML/JSON payloads and convert them to CLI arguments for forwarding."""
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


def _run_command(module: str, args: argparse.Namespace, forwarded: Sequence[str]) -> None:
    """Invoke an internal command module, merging config-derived args with forwarded CLI tokens."""
    config_args = _load_config_args(getattr(args, "config", None), args.command)
    cmd_args = [*config_args, *forwarded]
    _run_module_passthrough(module, args, cmd_args)


def _run_module_passthrough(
    module: str, args: argparse.Namespace, cmd_args: Sequence[str]
) -> None:
    """Invoke a python module with the provided arguments."""
    command = [str(PYTHON), "-m", module, *cmd_args]
    LOGGER.info("Executando: %s", _format_command(command))
    if args.dry_run:
        LOGGER.info("Dry-run habilitado; subprocesso não será iniciado.")
        return
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def _print_eval_guidance(args: argparse.Namespace, forwarded: Sequence[str]) -> None:
    """Emit a short checklist of artifacts needed for evaluation/export."""
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


def _run_visualize(args: argparse.Namespace, forwarded: Sequence[str]) -> None:
    """Execute the visualization script with assembled arguments."""
    cmd_args: list[str] = []
    cmd_args.extend(_load_config_args(getattr(args, "config", None), args.command))
    
    # Build command from parsed arguments
    if hasattr(args, "input") and args.input:
        cmd_args.extend(["--input", str(args.input)])
    if hasattr(args, "labels") and args.labels:
        cmd_args.extend(["--labels", str(args.labels)])
        if hasattr(args, "label_col") and args.label_col:
            cmd_args.extend(["--label-col", args.label_col])
    if hasattr(args, "predictions") and args.predictions:
        cmd_args.extend(["--predictions", str(args.predictions)])
    if hasattr(args, "history") and args.history:
        cmd_args.extend(["--history", str(args.history)])
    if hasattr(args, "from_run") and args.from_run:
        cmd_args.append("--from-run")
    if hasattr(args, "output") and args.output:
        cmd_args.extend(["--output", str(args.output)])
    if hasattr(args, "prefix") and args.prefix:
        cmd_args.extend(["--prefix", args.prefix])
    if hasattr(args, "report") and args.report:
        cmd_args.append("--report")
    if hasattr(args, "tsne") and args.tsne:
        cmd_args.append("--tsne")
    if hasattr(args, "tsne_3d") and args.tsne_3d:
        cmd_args.append("--tsne-3d")
    if hasattr(args, "pca") and args.pca:
        cmd_args.append("--pca")
    if hasattr(args, "umap") and args.umap:
        cmd_args.append("--umap")
    if hasattr(args, "compare_embeddings") and args.compare_embeddings:
        cmd_args.append("--compare-embeddings")
    if hasattr(args, "heatmap") and args.heatmap:
        cmd_args.append("--heatmap")
    if hasattr(args, "feature_heatmap") and args.feature_heatmap:
        cmd_args.append("--feature-heatmap")
    if hasattr(args, "confusion_matrix") and args.confusion_matrix:
        cmd_args.append("--confusion-matrix")
    if hasattr(args, "scatter_matrix") and args.scatter_matrix:
        cmd_args.append("--scatter-matrix")
    if hasattr(args, "distribution") and args.distribution:
        cmd_args.append("--distribution")
    if hasattr(args, "class_separation") and args.class_separation:
        cmd_args.append("--class-separation")
    if hasattr(args, "learning_curves") and args.learning_curves:
        cmd_args.append("--learning-curves")
    if hasattr(args, "binary") and args.binary:
        cmd_args.append("--binary")
    if hasattr(args, "perplexity") and args.perplexity is not None:
        cmd_args.extend(["--perplexity", str(args.perplexity)])
    if hasattr(args, "tsne_iter") and args.tsne_iter is not None:
        cmd_args.extend(["--tsne-iter", str(args.tsne_iter)])
    if hasattr(args, "seed") and args.seed is not None:
        cmd_args.extend(["--seed", str(args.seed)])
    if hasattr(args, "pca_svd_solver") and args.pca_svd_solver:
        cmd_args.extend(["--pca-svd-solver", args.pca_svd_solver])
    
    # Add any forwarded arguments
    cmd_args.extend(forwarded)

    _run_module_passthrough("mammography.commands.visualize", args, cmd_args)


def _run_data_audit(args: argparse.Namespace, forwarded: Sequence[str]) -> None:
    """Execute the data audit tool with assembled arguments."""
    cmd_args: list[str] = []
    cmd_args.extend(_load_config_args(getattr(args, "config", None), args.command))
    if hasattr(args, "archive") and args.archive:
        cmd_args.extend(["--archive", str(args.archive)])
    if hasattr(args, "csv") and args.csv:
        cmd_args.extend(["--csv", str(args.csv)])
    if hasattr(args, "manifest") and args.manifest:
        cmd_args.extend(["--manifest", str(args.manifest)])
    if hasattr(args, "audit_csv") and args.audit_csv:
        cmd_args.extend(["--audit-csv", str(args.audit_csv)])
    if hasattr(args, "log") and args.log:
        cmd_args.extend(["--log", str(args.log)])
    cmd_args.extend(forwarded)
    _run_module_passthrough("mammography.tools.data_audit", args, cmd_args)


def _run_report_pack(args: argparse.Namespace, forwarded: Sequence[str]) -> None:
    """Call the report_pack helper and normalize paths provided via CLI."""
    if forwarded:
        LOGGER.warning("Argumentos adicionais ignorados pelo report-pack: %s", " ".join(forwarded))
    if not args.runs:
        raise SystemExit("Informe pelo menos um --run outputs/.../results_* para empacotar.")
    
    from mammography.tools import report_pack

    run_paths = []
    for run in args.runs:
        path = (REPO_ROOT / run) if not run.is_absolute() else run
        run_paths.append(path)
    assets_dir = (REPO_ROOT / args.assets_dir) if not args.assets_dir.is_absolute() else args.assets_dir
    tex_path = None
    if args.tex_path:
        tex_path = (REPO_ROOT / args.tex_path) if not args.tex_path.is_absolute() else args.tex_path

    LOGGER.info("Empacotando runs: %s", ", ".join(str(p) for p in run_paths))
    report_pack.package_density_runs(
        run_paths,
        assets_dir,
        tex_path=tex_path,
        gradcam_limit=args.gradcam_limit,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point that routes subcommands and applies common logging/error handling."""
    parser = _build_parser()
    args, forwarded = parser.parse_known_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    _configure_logging(args.log_level)
    LOGGER.debug("Forwarded args: %s", forwarded)

    try:
        if args.command == "embed":
            _run_command("mammography.commands.extract_features", args, forwarded)
        elif args.command == "train-density":
            _run_command("mammography.commands.train", args, forwarded)
        elif args.command == "eval-export":
            _print_eval_guidance(args, forwarded)
        elif args.command == "report-pack":
            _run_report_pack(args, forwarded)
        elif args.command == "data-audit":
            _run_data_audit(args, forwarded)
        elif args.command == "visualize":
            _run_visualize(args, forwarded)
        elif args.command == "wizard":
            from mammography import wizard

            return wizard.run_wizard(dry_run=args.dry_run)
        elif args.command == "inference":
            _run_command("mammography.commands.inference", args, forwarded)
        elif args.command == "augment":
            _run_command("mammography.commands.augment", args, forwarded)
        elif args.command == "label-density":
            _run_command("mammography.commands.label_density", args, forwarded)
        elif args.command == "label-patches":
            _run_command("mammography.commands.label_patches", args, forwarded)
        elif args.command == "eda-cancer":
            _run_command("mammography.commands.eda_cancer", args, forwarded)
        elif args.command == "embeddings-baselines":
            _run_command("mammography.commands.embeddings_baselines", args, forwarded)
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
