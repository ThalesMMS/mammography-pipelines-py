# Funcionamento do Mammography Pipelines (CLI consolidado)

Este documento descreve a arquitetura e o funcionamento do pipeline consolidado em `mammography-pipelines`, com **uma única CLI**, **estrutura modular moderna** e **documentação científica integrada**.

## Visão Geral

A CLI principal é exposta como `mammography` (via `pyproject.toml`). Ela orquestra os fluxos principais do pipeline e os utilitários de visualização/relato.

O subcomando `mammography wizard` oferece um menu interativo com passos guiados para os fluxos principais.

## Estrutura do Projeto

```
├── src/mammography/      # Biblioteca principal (data, models, training, tools)
├── scripts/              # Entrypoints de execução (embed, train, visualize)
├── configs/              # Presets YAML (paths, density)
├── Article/              # Artigo científico + assets + Makefile
├── tools/                # Utilitários de auditoria/report-pack (espelhados em src/)
├── tests/                # Unit, integration, smoke, performance
```

## Fluxos de Trabalho Principais

### 1) Embeddings (`embed`)

- Script: `scripts/extract_features.py`
- Entrada: CSV + DICOMs (`--csv`, `--dicom-root`)
- Saída: `features.npy`, `metadata.csv`, projeções (PCA/t-SNE/UMAP) e clustering opcional
- Cache: `--cache-mode` (`auto`, `disk`, `memory`, `tensor-*`)
- Atalhos: `--run-reduction` (PCA+t-SNE+UMAP) e `--run-clustering` (k-means)
- Preview: `preview/first_image_loaded.png`, `preview/samples_grid.png`, `preview/labels_distribution.png`

### 1b) Baselines classicos (embeddings)

- Script: `scripts/embeddings_baselines.py`
- Entrada: `outputs/embeddings_resnet50/` com `features.npy` + `metadata.csv`
- Saida: `outputs/embeddings_baselines/` com metricas e relatorio comparativo

### 2) Treinamento de Densidade (`train-density`)

- Script: `scripts/train.py`
- Modelos: EfficientNetB0 / ResNet50
- Opções: congelamento de backbone, pesos de classe, AMP, Grad-CAM, embeddings de validação
- Saídas: métricas (`val_metrics.json`), histórico, predições, modelos e gráficos

### 3) Visualização (`visualize`)

- Script: `scripts/visualize.py`
- Entrada: `features.npy`/`metadata.csv` ou diretório de run
- Saídas: gráficos t-SNE, UMAP, heatmaps, learning curves, report completo

### 4) Relatórios e Auditoria

- `report-pack`: empacota figuras/artefatos do treino de densidade e atualiza LaTeX no `Article/`
- `data_audit`: gera manifest/audit de dados para rastreabilidade

### 5) EDA RSNA (cancer)

- Script: `scripts/eda_cancer.py`
- Entrada: CSVs RSNA + PNGs 256x256 + DICOMs originais
- Saida: visualizacoes e datasets balanceados (train/valid) no `--outdir`

## Flags Consistentes

- `--outdir`: diretório base de saída
- `--dicom-root`: raiz DICOM (quando aplicável)
- `--cache-mode`: estratégia de cache

## Artigo Científico

O artigo está em `Article/` com integração direta ao pipeline via `report-pack`. Consulte `Article/README.md` para instruções de build LaTeX e integração de figuras.
