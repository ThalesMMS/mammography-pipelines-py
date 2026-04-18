#!/usr/bin/env python3
"""CLI argument parser for the mammography pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

from mammography.utils.cli_args import add_tracking_args


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
        """
        Attach a standard `--config` Path argument to a subparser and set its description.
        
        Parameters:
            sp (argparse.ArgumentParser): The subparser to configure.
            summary (str): Description text assigned to `sp.description`.
        """
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
    add_tracking_args(eval_parser)
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
    add_tracking_args(pack_parser)

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
    add_tracking_args(viz_parser)

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