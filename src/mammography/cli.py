#!/usr/bin/env python3
#
# cli.py
# mammography-pipelines
#
# Orchestrates CLI subcommands for embedding extraction, density training, eval export, and visualization.
# DISCLAIMER: Educational project only - NOT for clinical or medical diagnostic purposes.
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
executing the internal handlers.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import shlex
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

try:  # Optional dependency for YAML configs.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML is not guaranteed to exist.
    yaml = None

LOGGER = logging.getLogger("projeto")
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIGS: dict[str, Path | None] = {
    "embed": REPO_ROOT / "configs" / "paths.yaml",
    "train-density": REPO_ROOT / "configs" / "density.yaml",
    "eval-export": None,
    "visualize": None,
    "explain": None,
    "embeddings-baselines": None,
    "data-audit": None,
    "tune": None,
    "preprocess": None,
    "cross-validate": None,
    "batch-inference": None,
    "compare-models": None,
    "benchmark-report": None,
    "automl": None,
}


def _build_parser() -> argparse.ArgumentParser:
    """
    Builds the top-level CLI ArgumentParser with global options and registered subcommands.

    Each subcommand accepts an optional `--config` and is prepared to forward unknown arguments to the corresponding internal command module (examples: embed, train-density, eval-export, report-pack, data-audit, visualize, explain, wizard, inference, augment, preprocess, label-density, label-patches, web, eda-cancer, embeddings-baselines, tune, cross-validate, batch-inference, compare-models, automl). Adds global flags such as `--dry-run` and `--log-level`.


    Returns:
        argparse.ArgumentParser: The configured top-level CLI parser.
    """
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
        help="Exibe o comando planejado e evita executar rotinas pesadas.",
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
        help="Exporta artefatos de avaliacao e registra no MLflow/registry.",
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
    eval_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/exports"),
        help="Diretorio base para exportacoes (default: outputs/exports).",
    )
    eval_parser.add_argument("--run-name", default="", help="Nome do run no MLflow")
    eval_parser.add_argument("--tracking-uri", default="", help="Tracking URI para MLflow")
    eval_parser.add_argument("--experiment", default="", help="Experimento MLflow")
    eval_parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("results/registry.csv"),
        help="Arquivo CSV do registry local",
    )
    eval_parser.add_argument(
        "--registry-md",
        type=Path,
        default=Path("results/registry.md"),
        help="Arquivo Markdown do registry local",
    )
    eval_parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Nao registrar no MLflow",
    )
    eval_parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Nao atualizar registry local",
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
    pack_parser.add_argument("--run-name", default="", help="Nome do run no MLflow")
    pack_parser.add_argument("--tracking-uri", default="", help="Tracking URI para MLflow")
    pack_parser.add_argument("--experiment", default="", help="Experimento MLflow")
    pack_parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("results/registry.csv"),
        help="Arquivo CSV do registry local",
    )
    pack_parser.add_argument(
        "--registry-md",
        type=Path,
        default=Path("results/registry.md"),
        help="Arquivo Markdown do registry local",
    )
    pack_parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Nao registrar no MLflow",
    )
    pack_parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Nao atualizar registry local",
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
    viz_parser.add_argument(
        "--run-name",
        default="",
        help="Nome do run no MLflow",
    )
    viz_parser.add_argument(
        "--tracking-uri",
        default="",
        help="Tracking URI para MLflow",
    )
    viz_parser.add_argument(
        "--experiment",
        default="",
        help="Experimento MLflow",
    )
    viz_parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("results/registry.csv"),
        help="Arquivo CSV do registry local",
    )
    viz_parser.add_argument(
        "--registry-md",
        type=Path,
        default=Path("results/registry.md"),
        help="Arquivo Markdown do registry local",
    )
    viz_parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Nao registrar no MLflow",
    )
    viz_parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Nao atualizar registry local",
    )

    explain_parser = subparsers.add_parser(
        "explain",
        help="Gera visualizações de explicabilidade (GradCAM, mapas de atenção).",
    )
    _add_config_argument(
        explain_parser,
        "Encaminha argumentos ao comando de explicabilidade.",
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

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocessa datasets de mamografia para formatos padronizados.",
    )
    _add_config_argument(
        preprocess_parser,
        "Encaminha argumentos ao comando de preprocessamento.",
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

    web_parser = subparsers.add_parser(
        "web",
        help="Abre o dashboard web com interface para inferencia e visualizacao.",
    )
    _add_config_argument(
        web_parser,
        "Lanca a interface web Streamlit para o dashboard de mamografia.",
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

    tune_parser = subparsers.add_parser(
        "tune",
        help="Executa hyperparameter tuning automatizado.",
    )
    _add_config_argument(
        tune_parser,
        "Encaminha argumentos ao comando de hyperparameter tuning.",
    )

    cv_parser = subparsers.add_parser(
        "cross-validate",
        help="Executa validacao cruzada k-fold para densidade.",
    )
    _add_config_argument(
        cv_parser,
        "Encaminha argumentos ao comando de validacao cruzada.",
    )

    batch_inference_parser = subparsers.add_parser(
        "batch-inference",
        help="Executa inferencia em lote sobre multiplas imagens ou DICOMs.",
    )
    _add_config_argument(
        batch_inference_parser,
        "Encaminha argumentos ao comando de inferencia em lote.",
    )

    compare_models_parser = subparsers.add_parser(
        "compare-models",
        help="Compara o desempenho de diferentes modelos de densidade.",
    )
    _add_config_argument(
        compare_models_parser,
        "Encaminha argumentos ao comando de comparacao de modelos.",
    )

    benchmark_report_parser = subparsers.add_parser(
        "benchmark-report",
        help="Valida o rerun oficial e gera a tabela mestre e o artefato do artigo.",
    )
    _add_config_argument(
        benchmark_report_parser,
        "Gera a consolidacao oficial do rerun_2026q1 em CSV/MD/JSON/TEX.",
    )
    benchmark_report_parser.add_argument(
        "--namespace",
        type=Path,
        default=Path("outputs/rerun_2026q1"),
        help="Namespace oficial do rerun (default: outputs/rerun_2026q1).",
    )
    benchmark_report_parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/rerun_2026q1_master"),
        help="Prefixo de saida para a tabela mestre (default: results/rerun_2026q1_master).",
    )
    benchmark_report_parser.add_argument(
        "--docs-report",
        type=Path,
        default=Path("docs/reports/rerun_2026q1_technical_report.md"),
        help="Relatorio tecnico consolidado em docs/.",
    )
    benchmark_report_parser.add_argument(
        "--article-table",
        type=Path,
        default=Path("Article/sections/rerun_2026q1_benchmark_table.tex"),
        help="Arquivo LaTeX da tabela consolidada do artigo.",
    )
    benchmark_report_parser.add_argument(
        "--exports-search-root",
        type=Path,
        default=Path("outputs"),
        help="Raiz usada para localizar manifests do eval-export.",
    )

    automl_parser = subparsers.add_parser(
        "automl",
        help="Executa AutoML para busca automatica de arquitetura e hiperparametros.",
    )
    _add_config_argument(
        automl_parser,
        "Encaminha argumentos ao comando de AutoML.",
    )

    return parser


def _configure_logging(level: str) -> None:
    """Configure root logging for both CLI feedback and debug traces."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s | %(message)s", force=True)


def _format_command(command: Sequence[str]) -> str:
    """Format a command preview so it is easy to copy/paste/debug."""
    return " ".join(shlex.quote(str(part)) for part in command)


@contextmanager
def _working_directory(path: Path) -> Iterator[Path]:
    """Temporarily chdir to preserve legacy CLI behavior."""
    current = Path.cwd()
    os.chdir(path)
    try:
        yield current
    finally:
        os.chdir(current)


def _resolve_entrypoint(module: str, entrypoint: str | None = None) -> Callable[..., Any]:
    """Import a module and return its CLI entrypoint callable."""
    module_obj = importlib.import_module(module)
    if entrypoint:
        handler = getattr(module_obj, entrypoint)
    else:
        handler = getattr(module_obj, "main", None) or getattr(module_obj, "run", None)
    if not callable(handler):
        raise AttributeError(f"Entrypoint não encontrado em {module}.")
    return handler


def _entrypoint_accepts_args(handler: Callable[..., Any]) -> bool:
    """Detect if the entrypoint accepts a positional argv-like payload."""
    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            return True
    return bool(sig.parameters)


def _invoke_entrypoint(handler: Callable[..., Any], cmd_args: Sequence[str]) -> int:
    """Call a module entrypoint, forwarding argv when supported."""
    if _entrypoint_accepts_args(handler):
        result = handler(list(cmd_args))
    else:
        result = handler()
    if result is None:
        return 0
    if isinstance(result, int) and not isinstance(result, bool):
        return result
    return 0


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


def _forwarded_has_flag(forwarded: Sequence[str], flag: str) -> bool:
    flag_prefix = f"{flag}="
    return any(token == flag or token.startswith(flag_prefix) for token in forwarded)


def _strip_flags_with_values(args: Sequence[str], flags: set[str]) -> list[str]:
    if not flags:
        return list(args)
    cleaned: list[str] = []
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token in flags:
            skip_next = True
            continue
        if any(token.startswith(f"{flag}=") for flag in flags):
            continue
        cleaned.append(token)
    return cleaned


def _filter_embed_config_args(config_args: Sequence[str], forwarded: Sequence[str]) -> list[str]:
    """Avoid config CSV overrides when the user explicitly selects a dataset preset."""
    if not config_args:
        return list(config_args)
    dataset_flag = _forwarded_has_flag(forwarded, "--dataset")
    data_dir_flag = _forwarded_has_flag(forwarded, "--data_dir")
    csv_flag = _forwarded_has_flag(forwarded, "--csv")
    dicom_root_flag = _forwarded_has_flag(forwarded, "--dicom-root")

    if not (dataset_flag or data_dir_flag):
        return list(config_args)

    flags_to_drop: set[str] = set()
    if dataset_flag:
        flags_to_drop.add("--dataset")
    if csv_flag:
        flags_to_drop.add("--csv")
    if dicom_root_flag:
        flags_to_drop.add("--dicom-root")

    if not csv_flag:
        flags_to_drop.add("--csv")
        flags_to_drop.add("--dicom-root")

    return _strip_flags_with_values(config_args, flags_to_drop)


def _run_command(
    module: str,
    args: argparse.Namespace,
    forwarded: Sequence[str],
    entrypoint: str | None = None,
) -> int:
    """Invoke an internal command module, merging config-derived args with forwarded CLI tokens."""
    config_args = _load_config_args(getattr(args, "config", None), args.command)
    if args.command == "embed":
        config_args = _filter_embed_config_args(config_args, forwarded)
    cmd_args = [*config_args, *forwarded]
    return _run_module_passthrough(module, args, cmd_args, entrypoint=entrypoint)


def _run_module_passthrough(
    module: str,
    args: argparse.Namespace,
    cmd_args: Sequence[str],
    entrypoint: str | None = None,
) -> int:
    """Invoke a module entrypoint in-process with the provided arguments."""
    command = [module, *cmd_args]
    LOGGER.info("Executando (in-process): %s", _format_command(command))
    if args.dry_run:
        LOGGER.info("Dry-run habilitado; comando não será executado.")
        return 0
    handler = _resolve_entrypoint(module, entrypoint=entrypoint)
    with _working_directory(REPO_ROOT):
        return _invoke_entrypoint(handler, cmd_args)


def _print_eval_guidance(args: argparse.Namespace, forwarded: Sequence[str]) -> int:
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
    return 0


def _run_eval_export(args: argparse.Namespace, forwarded: Sequence[str]) -> int:
    """Execute the eval-export routine with assembled arguments."""
    cmd_args: list[str] = []
    cmd_args.extend(_load_config_args(getattr(args, "config", None), args.command))
    if getattr(args, "runs", None):
        for run in args.runs:
            cmd_args.extend(["--run", str(run)])
    if hasattr(args, "output_dir") and args.output_dir:
        cmd_args.extend(["--output-dir", str(args.output_dir)])
    if hasattr(args, "run_name") and args.run_name:
        cmd_args.extend(["--run-name", args.run_name])
    if hasattr(args, "tracking_uri") and args.tracking_uri:
        cmd_args.extend(["--tracking-uri", args.tracking_uri])
    if hasattr(args, "experiment") and args.experiment:
        cmd_args.extend(["--experiment", args.experiment])
    if hasattr(args, "registry_csv") and args.registry_csv:
        cmd_args.extend(["--registry-csv", str(args.registry_csv)])
    if hasattr(args, "registry_md") and args.registry_md:
        cmd_args.extend(["--registry-md", str(args.registry_md)])
    if hasattr(args, "no_mlflow") and args.no_mlflow:
        cmd_args.append("--no-mlflow")
    if hasattr(args, "no_registry") and args.no_registry:
        cmd_args.append("--no-registry")
    cmd_args.extend(forwarded)
    return _run_module_passthrough("mammography.commands.eval_export", args, cmd_args)


def _run_visualize(args: argparse.Namespace, forwarded: Sequence[str]) -> int:
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
    if hasattr(args, "run_name") and args.run_name:
        cmd_args.extend(["--run-name", args.run_name])
    if hasattr(args, "tracking_uri") and args.tracking_uri:
        cmd_args.extend(["--tracking-uri", args.tracking_uri])
    if hasattr(args, "experiment") and args.experiment:
        cmd_args.extend(["--experiment", args.experiment])
    if hasattr(args, "registry_csv") and args.registry_csv:
        cmd_args.extend(["--registry-csv", str(args.registry_csv)])
    if hasattr(args, "registry_md") and args.registry_md:
        cmd_args.extend(["--registry-md", str(args.registry_md)])
    if hasattr(args, "no_mlflow") and args.no_mlflow:
        cmd_args.append("--no-mlflow")
    if hasattr(args, "no_registry") and args.no_registry:
        cmd_args.append("--no-registry")

    # Add any forwarded arguments
    cmd_args.extend(forwarded)

    return _run_module_passthrough("mammography.commands.visualize", args, cmd_args)


def _run_benchmark_report(args: argparse.Namespace, forwarded: Sequence[str]) -> int:
    """Execute the benchmark-report command with assembled arguments."""
    cmd_args: list[str] = []
    cmd_args.extend(_load_config_args(getattr(args, "config", None), args.command))
    if hasattr(args, "namespace") and args.namespace:
        cmd_args.extend(["--namespace", str(args.namespace)])
    if hasattr(args, "output_prefix") and args.output_prefix:
        cmd_args.extend(["--output-prefix", str(args.output_prefix)])
    if hasattr(args, "docs_report") and args.docs_report:
        cmd_args.extend(["--docs-report", str(args.docs_report)])
    if hasattr(args, "article_table") and args.article_table:
        cmd_args.extend(["--article-table", str(args.article_table)])
    if hasattr(args, "exports_search_root") and args.exports_search_root:
        cmd_args.extend(["--exports-search-root", str(args.exports_search_root)])
    cmd_args.extend(forwarded)
    return _run_module_passthrough("mammography.commands.benchmark_report", args, cmd_args)


def _run_data_audit(args: argparse.Namespace, forwarded: Sequence[str]) -> int:
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
    return _run_module_passthrough("mammography.tools.data_audit", args, cmd_args)


def _run_report_pack(args: argparse.Namespace, forwarded: Sequence[str]) -> int:
    """Call the report_pack helper and normalize paths provided via CLI."""
    if forwarded:
        LOGGER.warning("Argumentos adicionais ignorados pelo report-pack: %s", " ".join(forwarded))
    if not args.runs:
        raise SystemExit("Informe pelo menos um --run outputs/.../results_* para empacotar.")
    
    from mammography.tools import report_pack
    from mammography.tools import report_pack_registry

    run_paths = []
    for run in args.runs:
        path = (REPO_ROOT / run) if not run.is_absolute() else run
        run_paths.append(path)
    assets_dir = (REPO_ROOT / args.assets_dir) if not args.assets_dir.is_absolute() else args.assets_dir
    tex_path = None
    if args.tex_path:
        tex_path = (REPO_ROOT / args.tex_path) if not args.tex_path.is_absolute() else args.tex_path

    LOGGER.info("Empacotando runs: %s", ", ".join(str(p) for p in run_paths))
    summarized = report_pack.package_density_runs(
        run_paths,
        assets_dir,
        tex_path=tex_path,
        gradcam_limit=args.gradcam_limit,
    )
    if args.no_registry:
        return 0
    asset_names = sorted(
        {
            asset
            for run in summarized
            for asset in run.assets.values()
            if asset
        }
    )
    output_paths = [
        path for path in (assets_dir / name for name in asset_names) if path.exists()
    ]
    if tex_path and tex_path.exists():
        output_paths.append(tex_path)
    command_parts = ["mammography", "report-pack"]
    for run in args.runs:
        command_parts.extend(["--run", str(run)])
    if args.assets_dir:
        command_parts.extend(["--assets-dir", str(args.assets_dir)])
    if args.tex_path:
        command_parts.extend(["--tex", str(args.tex_path)])
    if args.gradcam_limit is not None:
        command_parts.extend(["--gradcam-limit", str(args.gradcam_limit)])
    if args.run_name:
        command_parts.extend(["--run-name", args.run_name])
    if args.tracking_uri:
        command_parts.extend(["--tracking-uri", args.tracking_uri])
    if args.experiment:
        command_parts.extend(["--experiment", args.experiment])
    if args.registry_csv:
        command_parts.extend(["--registry-csv", str(args.registry_csv)])
    if args.registry_md:
        command_parts.extend(["--registry-md", str(args.registry_md)])
    if args.no_mlflow:
        command_parts.append("--no-mlflow")
    if args.no_registry:
        command_parts.append("--no-registry")
    command = _format_command(command_parts)
    run_name = args.run_name or report_pack_registry.default_run_name(
        report_pack_registry.infer_dataset_name(run_paths)
    )
    report_pack_registry.register_report_pack_run(
        run_paths=run_paths,
        assets_dir=assets_dir,
        tex_path=tex_path,
        output_paths=output_paths,
        run_name=run_name,
        command=command,
        registry_csv=args.registry_csv,
        registry_md=args.registry_md,
        tracking_uri=args.tracking_uri or None,
        experiment=args.experiment or None,
        log_mlflow=not args.no_mlflow,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entry point that parses arguments, dispatches the selected subcommand, and returns a process exit code.
    
    Parameters:
        argv (Sequence[str] | None): Optional list of command-line arguments to parse; if None, the program's argv is used.
    
    Returns:
        int: Exit code produced by the dispatched subcommand (0 on success).
    """
    parser = _build_parser()
    args, forwarded = parser.parse_known_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    _configure_logging(args.log_level)
    LOGGER.debug("Forwarded args: %s", forwarded)

    try:
        if args.command == "embed":
            return _run_command("mammography.commands.extract_features", args, forwarded)
        if args.command == "train-density":
            return _run_command("mammography.commands.train", args, forwarded)
        if args.command == "eval-export":
            return _run_eval_export(args, forwarded)
        if args.command == "report-pack":
            return _run_report_pack(args, forwarded)
        if args.command == "data-audit":
            return _run_data_audit(args, forwarded)
        if args.command == "visualize":
            return _run_visualize(args, forwarded)
        if args.command == "explain":
            return _run_command("mammography.commands.explain", args, forwarded)
        if args.command == "wizard":
            from mammography import wizard

            return wizard.run_wizard(dry_run=args.dry_run)
        if args.command == "inference":
            return _run_command("mammography.commands.inference", args, forwarded)
        if args.command == "augment":
            return _run_command("mammography.commands.augment", args, forwarded)
        if args.command == "preprocess":
            return _run_command("mammography.commands.preprocess", args, forwarded)
        if args.command == "label-density":
            return _run_command("mammography.commands.label_density", args, forwarded)
        if args.command == "label-patches":
            return _run_command("mammography.commands.label_patches", args, forwarded)
        if args.command == "web":
            return _run_command("mammography.commands.web", args, forwarded)
        if args.command == "eda-cancer":
            return _run_command(
                "mammography.commands.eda_cancer",
                args,
                forwarded,
                entrypoint="run_density_classifier_cli",
            )
        if args.command == "embeddings-baselines":
            return _run_command("mammography.commands.embeddings_baselines", args, forwarded)
        if args.command == "tune":
            return _run_command("mammography.commands.tune", args, forwarded)
        if args.command == "cross-validate":
            return _run_command("mammography.commands.cross_validate", args, forwarded)
        if args.command == "batch-inference":
            return _run_command("mammography.commands.batch_inference", args, forwarded)
        if args.command == "compare-models":
            return _run_command("mammography.commands.compare_models", args, forwarded)
        if args.command == "benchmark-report":
            return _run_benchmark_report(args, forwarded)
        if args.command == "automl":
            return _run_command("mammography.commands.automl", args, forwarded)
        parser.error(f"Subcomando desconhecido: {args.command}")
        return 1
    except SystemExit as exc:
        if isinstance(exc.code, str):
            LOGGER.error("%s", exc.code)
            return 1
        return 0 if exc.code is None else int(exc.code)


if __name__ == "__main__":
    sys.exit(main())
