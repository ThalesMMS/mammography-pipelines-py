#
# wizard_helpers.py
# mammography-pipelines
#
# Provides contextual help text and enhanced prompt functions for the interactive wizard.
# This module centralizes parameter documentation to improve user understanding.
#
# Medical Disclaimer: This educational research tool is for academic purposes only.
# Not intended for clinical diagnosis or medical decision-making.
#
# February 2026
#
from __future__ import annotations

from typing import Any

# Comprehensive help text registry for wizard parameters
HELP_TEXTS: dict[str, str] = {
    # Core training parameters
    "batch_size": (
        "Numero de imagens processadas simultaneamente. "
        "Valores maiores (16-32) aceleram o treinamento em GPUs potentes, "
        "mas exigem mais memoria. Valores menores (4-8) sao mais seguros para "
        "hardware limitado ou imagens grandes."
    ),
    "epochs": (
        "Numero de passagens completas pelo dataset durante o treinamento. "
        "Mais epocas permitem melhor aprendizado, mas aumentam risco de overfitting. "
        "Valores tipicos: 50-100 para datasets pequenos, 20-50 para datasets grandes. "
        "Use early stopping para parar automaticamente."
    ),
    "lr": (
        "Taxa de aprendizado (learning rate) para o classificador. "
        "Controla o tamanho dos passos durante otimizacao. "
        "Valores tipicos: 1e-3 a 1e-5. Valores maiores aceleram convergencia, "
        "mas podem causar instabilidade. Use warmup e schedulers para ajuste automatico."
    ),
    "backbone_lr": (
        "Taxa de aprendizado especifica para o backbone (feature extractor). "
        "Geralmente menor que lr (ex: 1e-5) para preservar features pre-treinadas. "
        "Apenas usado quando backbone nao esta congelado (--train-backbone ou --unfreeze-last-block)."
    ),
    "arch": (
        "Arquitetura de rede neural a ser usada. "
        "efficientnet_b0: Leve e eficiente, bom equilibrio precisao/velocidade. "
        "resnet50: Mais robusto, requer mais memoria mas pode ter melhor performance. "
        "Ambas vem com pesos pre-treinados no ImageNet."
    ),
    "classes": (
        "Esquema de classificacao para densidade mamaria: "
        "- density: 4 classes BI-RADS (A, B, C, D) - padrao para pesquisa. "
        "- binary: 2 classes (AB vs CD) - simplifica em baixa/alta densidade. "
        "- multiclass: Alias para density, mantido por compatibilidade."
    ),
    "device": (
        "Dispositivo de computacao para treinamento: "
        "- auto: Detecta automaticamente (cuda > mps > cpu). "
        "- cuda: GPU NVIDIA (requer CUDA instalado). "
        "- mps: GPU Apple Silicon (requer macOS). "
        "- cpu: CPU apenas (mais lento, sem requisitos especiais)."
    ),
    "cache_mode": (
        "Estrategia de cache para imagens preprocessadas: "
        "- auto: Escolhe melhor modo baseado em memoria disponivel. "
        "- memory: Cache em RAM (mais rapido, requer muita memoria). "
        "- disk: Cache em disco (mais lento, economiza memoria). "
        "- tensor-disk: Cache tensors em disco (equilibrio). "
        "- tensor-memmap: Memory-mapped tensors (eficiente). "
        "- none: Sem cache (economiza espaco, mais lento)."
    ),
    "img_size": (
        "Tamanho de entrada das imagens (pixels, quadrado). "
        "Valores comuns: 224 (rapido), 384 (equilibrado), 512 (detalhado). "
        "Imagens maiores capturam mais detalhes mas exigem mais memoria "
        "e tempo de processamento."
    ),
    "num_workers": (
        "Numero de processos paralelos para carregar dados. "
        "Valores tipicos: 2-8 dependendo de CPU cores. "
        "Mais workers aceleram carregamento mas consomem memoria. "
        "Use 0 para debug ou se encontrar erros de multiprocessing."
    ),
    "pretrained": (
        "Usar pesos pre-treinados no ImageNet como ponto de partida. "
        "Recomendado: SIM - Transfer learning acelera convergencia e melhora performance. "
        "Apenas desative se quiser treinar do zero (requer muito mais dados/tempo)."
    ),
    "view_specific_training": (
        "Treinar modelos separados para cada vista mamografica (CC/MLO). "
        "Recomendado: SIM se tiver dados suficientes - Cada vista tem caracteristicas distintas. "
        "Modelos especializados capturam melhor as peculiaridades de cada projecao. "
        "Requer ~2x mais memoria e tempo de treinamento."
    ),
    "views_to_train": (
        "Lista de vistas mamograficas para treinar quando view_specific_training=True. "
        "Opcoes: CC (cranio-caudal), MLO (medio-lateral obliqua), ML, LM, etc. "
        "Padrao: None (treina todas as vistas encontradas no dataset). "
        "Exemplo: ['CC', 'MLO'] para treinar apenas essas duas vistas."
    ),
    "weight_decay": (
        "Regularizacao L2 (penalidade de peso) aplicada ao otimizador. "
        "Padrao: 1e-4. Valores tipicos: 1e-5 a 1e-3. "
        "Weight decay previne overfitting penalizando pesos grandes. "
        "Valores maiores = regularizacao mais forte = modelo mais simples."
    ),
    "outdir": (
        "Diretorio de saida para checkpoints, logs e resultados. "
        "Estrutura criada: outdir/checkpoints/, outdir/logs/, outdir/plots/. "
        "Use nomes descritivos (ex: outputs/efficientnet_density_v2)."
    ),
    # Dataset parameters
    "dataset": (
        "Preset de formato do dataset: "
        "- archive: DICOMs em archive/ + classificacao.csv. "
        "- mamografias: PNGs em subpastas + featureS.txt por pasta. "
        "- patches_completo: PNGs no root + featureS.txt unico. "
        "- custom: Configuracao manual de caminhos."
    ),
    "csv": (
        "Caminho para arquivo CSV com metadados ou diretorio com featureS.txt. "
        "Formatos suportados: classificacao.csv (AccessionNumber + Classification), "
        "featureS.txt (formato IRMA), ou CSV custom com colunas image_path/density_label."
    ),
    "dicom_root": (
        "Diretorio raiz contendo arquivos DICOM organizados por AccessionNumber. "
        "Estrutura esperada: dicom_root/AccessionNumber/*.dcm. "
        "Apenas necessario para datasets DICOM (preset 'archive')."
    ),
    "include_class_5": (
        "Incluir imagens de classe 5 (indefinido/ruido) no dataset. "
        "Recomendado: NAO - Classe 5 geralmente representa imagens inadequadas "
        "para classificacao de densidade."
    ),
    # Advanced training parameters
    "seed": (
        "Semente aleatoria para reproducibilidade. "
        "Fixar seed (ex: 42) garante que splits train/val e augmentacoes sejam identicas "
        "entre execucoes. Use --deterministic para reproducibilidade total (mais lento)."
    ),
    "val_frac": (
        "Fracao do dataset usada para validacao (0.0 a 1.0). "
        "Valores tipicos: 0.15-0.25. Validacao menor = mais dados de treino. "
        "Validacao maior = metricas mais confiaveis. "
        "Sistema garante todas as classes presentes no conjunto de validacao."
    ),
    "split_ensure_all_classes": (
        "Garantir que todas as classes aparecem em train e val splits. "
        "Recomendado: SIM - Evita splits invalidos onde alguma classe esta ausente. "
        "Sistema tenta multiplas seeds aleatorias ate encontrar split valido. "
        "Desative apenas para datasets muito pequenos ou desbalanceados."
    ),
    "split_max_tries": (
        "Numero maximo de tentativas para criar split com todas as classes. "
        "Padrao: 200. Valores tipicos: 100-500. "
        "Se split_ensure_all_classes=True, sistema tenta ate encontrar split valido. "
        "Aumente se receber erros de 'Failed to create valid split'."
    ),
    "augment": (
        "Ativar augmentacao de dados durante treinamento. "
        "Recomendado: SIM - Aumenta variabilidade dos dados e reduz overfitting. "
        "Inclui: rotacoes, flips horizontais, crop/resize, ajustes de contraste. "
        "Use --no-augment apenas para debugging."
    ),
    "augment_vertical": (
        "Adicionar flips verticais alem dos horizontais. "
        "Recomendado: NAO para mamografias - Orientacao vertical tem significado anatomico. "
        "Use apenas se o dataset ja tem imagens em multiplas orientacoes."
    ),
    "augment_color": (
        "Adicionar variacao de cor/brilho (color jitter). "
        "Util para datasets PNG/JPG. Menos relevante para DICOM (usa windowing). "
        "Valores tipicos: brightness=0.2, contrast=0.2."
    ),
    "augment_rotation_deg": (
        "Angulo maximo de rotacao aleatoria (graus). "
        "Padrao: 5 graus. Valores pequenos (5-15) mantém anatomia reconhecivel. "
        "Rotacoes grandes podem distorcer features importantes."
    ),
    "class_weights": (
        "Pesos para balancear contribuicao de cada classe na loss function: "
        "- auto: Calcula pesos inversamente proporcionais a frequencia (recomendado). "
        "- none: Todas as classes tem peso 1.0 (use se dataset ja esta balanceado). "
        "- manual: Especifique pesos customizados (ex: 1.0,0.8,1.2,1.0)."
    ),
    "class_weights_alpha": (
        "Exponente para suavizar class weights automaticos (0.0 a 2.0). "
        "Padrao: 1.0. Valores menores (0.5) reduzem impacto do balanceamento. "
        "Valores maiores (1.5) aumentam enfase em classes raras."
    ),
    "sampler_weighted": (
        "Usar weighted sampler para balancear frequencia de amostragem. "
        "Recomendado: SIM para datasets muito desbalanceados. "
        "Garante que classes raras aparecam mais frequentemente durante treinamento. "
        "Combine com class_weights para melhor resultado."
    ),
    "sampler_alpha": (
        "Exponente para suavizar probabilidades do weighted sampler. "
        "Padrao: 1.0. Valores menores (0.5-0.8) reduzem oversampling de classes raras."
    ),
    "train_backbone": (
        "Treinar todo o backbone (feature extractor) desde o inicio. "
        "Recomendado: NAO - Requer muito mais memoria e tempo. "
        "Use apenas se tiver dataset grande (>10k imagens) e recursos GPU potentes. "
        "Alternativa: --unfreeze-last-block para fine-tuning parcial."
    ),
    "unfreeze_last_block": (
        "Descongelar ultimo bloco do backbone para fine-tuning parcial. "
        "Recomendado: SIM - Permite adaptacao ao dominio sem treinar tudo. "
        "Equilibrio entre velocidade/memoria e capacidade de aprendizado."
    ),
    "warmup_epochs": (
        "Numero de epocas com learning rate crescente no inicio do treinamento. "
        "Padrao: 0 (desativado). Valores tipicos: 3-10. "
        "Warmup estabiliza treinamento inicial, especialmente com batch sizes grandes."
    ),
    "early_stop_patience": (
        "Numero de epocas sem melhora antes de parar treinamento automaticamente. "
        "Padrao: 0 (desativado). Valores tipicos: 10-20. "
        "Early stopping previne overfitting e economiza tempo. "
        "Monitora val_loss por padrao."
    ),
    "early_stop_min_delta": (
        "Melhora minima necessaria para considerar progresso no early stopping. "
        "Padrao: 0.0 (qualquer melhora conta). Valores tipicos: 0.001-0.01. "
        "Evita parar por flutuacoes insignificantes."
    ),
    # Scheduler parameters
    "scheduler": (
        "Scheduler de learning rate para ajuste automatico durante treinamento: "
        "- auto: Escolhe plateau (padrao, seguro). "
        "- none: LR fixo durante todo treinamento. "
        "- plateau: Reduz LR quando val_loss estagnar (recomendado). "
        "- cosine: Decaimento cosseno do LR. "
        "- step: Reduz LR em intervalos fixos."
    ),
    "lr_reduce_patience": (
        "Epocas sem melhora antes de reduzir LR (scheduler plateau). "
        "Padrao: 5-10. Valores menores = ajustes mais frequentes. "
        "Valores maiores = mais estavel."
    ),
    "lr_reduce_factor": (
        "Fator de reducao do LR (scheduler plateau). "
        "Padrao: 0.5 (reduz pela metade). Valores tipicos: 0.1-0.5. "
        "Valores menores = reducoes mais agressivas."
    ),
    "lr_reduce_min_lr": (
        "Learning rate minimo permitido pelos schedulers. "
        "Padrao: 1e-7. Impede que LR fique muito pequeno e pare de aprender."
    ),
    "lr_reduce_cooldown": (
        "Epocas de espera apos reducao de LR antes de permitir nova reducao. "
        "Padrao: 0. Valores tipicos: 2-5. "
        "Cooldown evita reducoes excessivas."
    ),
    "scheduler_min_lr": (
        "LR minimo para scheduler cosine. "
        "Padrao: usa lr_reduce_min_lr. Define LR no final do ciclo cosine."
    ),
    "scheduler_step_size": (
        "Intervalo (epocas) entre reducoes de LR no scheduler step. "
        "Padrao: 5. Valores tipicos: 5-10. "
        "Exemplo: step_size=10 reduz LR nas epocas 10, 20, 30..."
    ),
    "scheduler_gamma": (
        "Fator de multiplicacao do LR no scheduler step. "
        "Padrao: 0.5. Valores tipicos: 0.1-0.5. "
        "novo_lr = lr_atual * gamma"
    ),
    # Performance parameters
    "amp": (
        "Automatic Mixed Precision - Usa float16 para acelerar treinamento. "
        "Recomendado: SIM para GPUs NVIDIA modernas (Volta/Turing/Ampere). "
        "Reduz uso de memoria (~40%) e acelera treinamento (~2x) sem perda de precisao. "
        "NAO use em GPUs antigas ou CPUs."
    ),
    "torch_compile": (
        "Compilar modelo com torch.compile() para otimizacao. "
        "Recomendado: NAO para desenvolvimento. SIM para producao. "
        "Requer PyTorch 2.0+. Primeira epoca e lenta (compilacao), seguintes sao mais rapidas."
    ),
    "fused_optim": (
        "Usar versao fused do otimizador Adam (mais rapida). "
        "Recomendado: SIM para GPUs NVIDIA. "
        "Requer CUDA. Acelera update de parametros (~10-20%)."
    ),
    "prefetch_factor": (
        "Numero de batches carregados antecipadamente por worker. "
        "Padrao: 4. Valores tipicos: 2-8. "
        "Valores maiores = menos stalls, mais memoria. 0 = desativa prefetching."
    ),
    "persistent_workers": (
        "Manter workers de dataloader ativos entre epocas. "
        "Recomendado: SIM - Evita overhead de criar/destruir processos. "
        "Desative apenas se tiver problemas de memoria."
    ),
    "loader_heuristics": (
        "Aplicar heuristicas automaticas para otimizar dataloader. "
        "Recomendado: SIM - Ajusta num_workers e pin_memory baseado em hardware. "
        "Desative se quiser controle manual completo."
    ),
    # Normalization parameters
    "mean": (
        "Media para normalizacao de imagens (formato: R,G,B). "
        "Padrao ImageNet: 0.485,0.456,0.406. "
        "Use --auto-normalize para calcular estatisticas do seu dataset."
    ),
    "std": (
        "Desvio padrao para normalizacao (formato: R,G,B). "
        "Padrao ImageNet: 0.229,0.224,0.225. "
        "Use --auto-normalize para calcular estatisticas do seu dataset."
    ),
    "auto_normalize": (
        "Calcular mean/std automaticamente de amostra do dataset. "
        "Recomendado: SIM para DICOM, NAO para PNG/JPG com pesos pretrained. "
        "Normalizar para distribuicao do dataset melhora convergencia."
    ),
    "auto_normalize_samples": (
        "Numero de imagens para calcular estatisticas de normalizacao. "
        "Padrao: 1000. Valores maiores = estatisticas mais precisas, mais lento."
    ),
    # Output and debugging parameters
    "cache_dir": (
        "Diretorio para cache de imagens preprocessadas. "
        "Padrao: outdir/cache. Cache acelera epocas subsequentes mas ocupa espaco. "
        "Use disco SSD para melhor performance."
    ),
    "embeddings_dir": (
        "Diretorio contendo embeddings pre-extraidos para transfer learning. "
        "Formato esperado: embeddings_dir/train.npz, embeddings_dir/val.npz. "
        "Use este parametro para treinar apenas classificador em cima de features fixas."
    ),
    "save_val_preds": (
        "Salvar predicoes de validacao em CSV a cada epoca. "
        "Util para analise de erros e debugging. Cria outdir/val_predictions_epochN.csv."
    ),
    "gradcam": (
        "Gerar visualizacoes Grad-CAM para interpretar decisoes do modelo. "
        "Grad-CAM destaca regioes importantes da imagem para a predicao. "
        "Util para validar que modelo aprende features corretas."
    ),
    "gradcam_limit": (
        "Numero maximo de Grad-CAMs gerados por epoca de validacao. "
        "Padrao: 4. Valores maiores geram mais exemplos mas ocupam espaco."
    ),
    "export_val_embeddings": (
        "Exportar embeddings do conjunto de validacao ao final do treinamento. "
        "Util para analise posterior, clustering, ou visualization (UMAP/t-SNE)."
    ),
    "subset": (
        "Limitar dataset a N imagens para testes rapidos. "
        "Padrao: 0 (desativado). Valores tipicos: 32-128 para debugging. "
        "Subset mantem balanceamento de classes quando possivel."
    ),
    "profile": (
        "Habilitar profiler do PyTorch para analise de performance. "
        "Gera traces detalhados de CPU/GPU/memoria. "
        "Use para otimizar gargalos. Aumenta overhead ~10%."
    ),
    "profile_dir": (
        "Diretorio de saida para traces do profiler. "
        "Padrao: outputs/profiler. Traces podem ser visualizados com chrome://tracing."
    ),
    "deterministic": (
        "Modo deterministico completo para reproducibilidade total. "
        "Recomendado: NAO - Reduz performance significativamente (~20-30%). "
        "Use apenas para pesquisa que requer reproducibilidade exata."
    ),
    "allow_tf32": (
        "Permitir TensorFloat32 em GPUs Ampere+ para acelerar operacoes. "
        "Recomendado: SIM - Pequena perda de precisao (~1e-4), ganho de velocidade. "
        "Desative se precisar precisao numerica maxima."
    ),
    "log_level": (
        "Nivel de logging (debug, info, warning, error). "
        "Padrao: info. Use debug para troubleshooting detalhado. "
        "Use warning para producao (menos verboso)."
    ),
    # Embedding-specific parameters
    "pooling": (
        "Estrategia de pooling global para extrair embeddings: "
        "- avg: Global Average Pooling (padrao, robusto). "
        "- max: Global Max Pooling (destaca features mais fortes). "
        "- gem: Generalized Mean Pooling (intermediario)."
    ),
    "layer": (
        "Camada do modelo de onde extrair embeddings. "
        "Padrao: penultima camada (antes do classificador). "
        "Opcoes: avgpool, layer4, layer3 (ResNet), features (EfficientNet)."
    ),
}


def format_help_text(param_name: str, width: int = 80) -> str:
    """
    Format help text for display in the wizard.

    Args:
        param_name: Parameter key in HELP_TEXTS
        width: Maximum line width for wrapping

    Returns:
        Formatted help text with proper wrapping and indentation
    """
    if param_name not in HELP_TEXTS:
        return ""

    help_text = HELP_TEXTS[param_name]
    lines = []
    current_line = "ℹ️  "

    for word in help_text.split():
        if len(current_line) + len(word) + 1 > width:
            lines.append(current_line)
            current_line = "   " + word
        else:
            if len(current_line) > 3:
                current_line += " "
            current_line += word

    if current_line.strip():
        lines.append(current_line)

    return "\n".join(lines)


def print_help(param_name: str, width: int = 80) -> None:
    """
    Print help text for a parameter to console.

    Args:
        param_name: Parameter key in HELP_TEXTS
        width: Maximum line width for wrapping
    """
    formatted = format_help_text(param_name, width)
    if formatted:
        print(formatted)


def has_help(param_name: str) -> bool:
    """
    Check if help text exists for a parameter.

    Args:
        param_name: Parameter key to check

    Returns:
        True if help text exists, False otherwise
    """
    return param_name in HELP_TEXTS


def get_help_summary() -> dict[str, int]:
    """
    Get summary statistics about the help text registry.

    Returns:
        Dictionary with counts of help texts by category
    """
    categories = {
        "core": ["batch_size", "epochs", "lr", "arch", "device", "cache_mode", "img_size", "num_workers"],
        "dataset": ["dataset", "csv", "dicom_root", "include_class_5"],
        "advanced": ["seed", "val_frac", "augment", "class_weights", "sampler_weighted"],
        "training": ["train_backbone", "unfreeze_last_block", "warmup_epochs", "early_stop_patience"],
        "scheduler": ["scheduler", "lr_reduce_patience", "lr_reduce_factor"],
        "performance": ["amp", "torch_compile", "fused_optim", "prefetch_factor"],
        "output": ["outdir", "save_val_preds", "gradcam", "export_val_embeddings"],
    }

    summary = {}
    for category, params in categories.items():
        summary[category] = sum(1 for p in params if p in HELP_TEXTS)

    summary["total"] = len(HELP_TEXTS)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Basic prompt functions (mirroring wizard.py patterns)
# ─────────────────────────────────────────────────────────────────────────────


def _ask_choice(title: str, options: list[str], default: int = 0) -> int:
    """
    Ask user to choose from a list of options.

    Args:
        title: Question or prompt text
        options: List of option strings
        default: Default option index

    Returns:
        Selected option index
    """
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
    """
    Ask user a yes/no question.

    Args:
        prompt: Question text
        default: Default answer

    Returns:
        True for yes, False for no
    """
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
    """
    Ask user for a string value.

    Args:
        prompt: Question text
        default: Default value if user presses enter

    Returns:
        User-provided string or default
    """
    if default:
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw or default
    return input(f"{prompt}: ").strip()


def _ask_int(prompt: str, default: int) -> int:
    """
    Ask user for an integer value.

    Args:
        prompt: Question text
        default: Default value

    Returns:
        User-provided integer or default
    """
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            return int(raw)
        print("Informe um numero inteiro valido.")


def _ask_float(prompt: str, default: float) -> float:
    """
    Ask user for a float value.

    Args:
        prompt: Question text
        default: Default value

    Returns:
        User-provided float or default
    """
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print("Informe um numero valido.")


def _ask_optional(prompt: str) -> str | None:
    """
    Ask user for an optional string value.

    Args:
        prompt: Question text

    Returns:
        User-provided string or None if skipped
    """
    raw = input(f"{prompt} (enter para pular): ").strip()
    return raw or None


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced prompt functions with contextual help
# ─────────────────────────────────────────────────────────────────────────────


def ask_with_help(
    prompt: str,
    param_name: str | None = None,
    default: str | None = None,
    show_help: bool = True,
) -> str:
    """
    Ask user for a string value with optional contextual help.

    Args:
        prompt: Question text
        param_name: Parameter name to look up help text
        default: Default value if user presses enter
        show_help: Whether to display help text before prompting

    Returns:
        User-provided string or default
    """
    if show_help and param_name and has_help(param_name):
        print_help(param_name)
        print()
    return _ask_string(prompt, default)


def ask_choice_with_help(
    title: str,
    options: list[str],
    param_name: str | None = None,
    default: int = 0,
    show_help: bool = True,
) -> int:
    """
    Ask user to choose from options with optional contextual help.

    Args:
        title: Question or prompt text
        options: List of option strings
        param_name: Parameter name to look up help text
        default: Default option index
        show_help: Whether to display help text before prompting

    Returns:
        Selected option index
    """
    if show_help and param_name and has_help(param_name):
        print_help(param_name)
        print()
    return _ask_choice(title, options, default)


def ask_int_with_help(
    prompt: str,
    param_name: str | None = None,
    default: int = 0,
    show_help: bool = True,
) -> int:
    """
    Ask user for an integer value with optional contextual help.

    Args:
        prompt: Question text
        param_name: Parameter name to look up help text
        default: Default value
        show_help: Whether to display help text before prompting

    Returns:
        User-provided integer or default
    """
    if show_help and param_name and has_help(param_name):
        print_help(param_name)
        print()
    return _ask_int(prompt, default)


def ask_float_with_help(
    prompt: str,
    param_name: str | None = None,
    default: float = 0.0,
    show_help: bool = True,
) -> float:
    """
    Ask user for a float value with optional contextual help.

    Args:
        prompt: Question text
        param_name: Parameter name to look up help text
        default: Default value
        show_help: Whether to display help text before prompting

    Returns:
        User-provided float or default
    """
    if show_help and param_name and has_help(param_name):
        print_help(param_name)
        print()
    return _ask_float(prompt, default)


def ask_yes_no_with_help(
    prompt: str,
    param_name: str | None = None,
    default: bool = True,
    show_help: bool = True,
) -> bool:
    """
    Ask user a yes/no question with optional contextual help.

    Args:
        prompt: Question text
        param_name: Parameter name to look up help text
        default: Default answer
        show_help: Whether to display help text before prompting

    Returns:
        True for yes, False for no
    """
    if show_help and param_name and has_help(param_name):
        print_help(param_name)
        print()
    return _ask_yes_no(prompt, default)


def ask_optional_with_help(
    prompt: str,
    param_name: str | None = None,
    show_help: bool = True,
) -> str | None:
    """
    Ask user for an optional string value with optional contextual help.

    Args:
        prompt: Question text
        param_name: Parameter name to look up help text
        show_help: Whether to display help text before prompting

    Returns:
        User-provided string or None if skipped
    """
    if show_help and param_name and has_help(param_name):
        print_help(param_name)
        print()
    return _ask_optional(prompt)
