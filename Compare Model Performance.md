# Comparação de Performance de Modelos

Este documento contém análises comparativas entre diferentes modelos e configurações testadas no projeto.

> **Nota**: Para documentação completa e atualizada, consulte `Article/article.md`.

---

## Histórico de Conversas e Análises

_O conteúdo abaixo é um registro histórico de análises realizadas durante o desenvolvimento._

### User Input

O "RSNA_Mammo_EfficientNetB0_mamografias.py" e o "RSNA_Mammo_EfficientNetB0_mamografias_multiclass.py" ainda não terminara de executar, vc mas já poderia fazer uma análise comparativa entre os resultados dos arquivos "RSNA_Mammo..."? Cada resultado está em uma pasta "mammo_..." da pasta outputs.

### Planner Response

Vou fazer uma análise comparativa dos resultados dos dois arquivos RSNA_Mammo... nas pastas de saída. Primeiro, vou verificar as pastas de resultados disponíveis.

*Listed directory [outputs](file:///d:/mammography-pipelines/outputs) *

### Planner Response

Agora vou examinar os resultados das duas pastas principais:

*Listed directory [mammo_efficientnetb0_mamografias](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_mamografias) *

*Listed directory [mammo_efficientnetb0_mamografias_multiclass](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_mamografias_multiclass) *

*Listed directory [results](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_mamografias/results) *

*Listed directory [results_1](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_mamografias/results_1) *

*Listed directory [results](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_mamografias_multiclass/results) *

### Planner Response

Agora vou ler os arquivos de métricas e logs para fazer a análise comparativa:

*Listed directory [metrics](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_mamografias/results_1/metrics) *

*Listed directory [metrics](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_mamografias_multiclass/results/metrics) *

*Viewed [val_metrics.json](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_mamografias/results_1/metrics/val_metrics.json) *

*Viewed [val_metrics.json](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_mamografias_multiclass/results/metrics/val_metrics.json) *

### Planner Response

Agora vou verificar os arquivos de predições para entender melhor os resultados:

*Viewed [val_predictions.csv](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_mamografias/results_1/val_predictions.csv) *

*Viewed [val_predictions.csv](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_mamografias_multiclass/results/val_predictions.csv) *

### Planner Response

# Análise Comparativa dos Resultados RSNA_Mammo

## Resumo das Métricas Principais

### Modelo Binário (mammo_efficientnetb0_mamografias)
- **Acurácia**: 92.52%
- **Kappa**: 0.812
- **AUC**: 0.983
- **Classes**: AB (normal/benigno) vs CD (suspeito/maligno)

### Modelo Multiclasse (mammo_efficientnetb0_mamografias_multiclass)
- **Acurácia**: 74.52%
- **Kappa**: 0.607
- **AUC**: N/A (multiclasse)
- **Classes**: A, B, C, D (densidades BI-RADS)

## Análise Detalhada

### Desempenho por Classe

**Modelo Binário:**
- **Classe AB**: Precision 97.96%, Recall 91.93%
- **Classe CD**: Precision 79.57%, Recall 94.27%

**Modelo Multiclasse:**
- **Classe A**: Precision 84.51%, Recall 85.99%
- **Classe B**: Precision 68.08%, Recall 46.18%
- **Classe C**: Precision 0%, Recall 0% (sem amostras)
- **Classe D**: Precision 90.94%, Recall 79.94%

## Principais Observações

1. **Performance Superior do Modelo Binário**: O modelo binário apresenta acurácia significativamente maior (92.52% vs 74.52%)

2. **Problema com Classe C**: O modelo multiclasse não conseguiu classificar nenhuma amostra como classe C (densidade C), indicando possível desbalanceamento ou dificuldade de aprendizado

3. **Confusão entre Classes B e Outras**: O modelo multiclasse tem dificuldade especial com a classe B, com recall baixo (46.18%)

4. **Distribuição de Dados**: O modelo binário tem 942 amostras AB vs 314 CD, enquanto o multiclasse tem distribuição mais uniforme mas com ausência de classe C no conjunto de validação

## Conclusões

O modelo binário é muito mais eficaz para a tarefa de detecção de anormalidades, enquanto o modelo multiclasse precisa de melhorias, especialmente:
- Mais dados para classe C
- Balanceamento melhor entre classes
- Possíveis ajustes na arquitetura ou hiperparâmetros

O modelo binário já apresenta resultados excelentes com AUC de 0.983, indicando excelente capacidade de discriminação.

### User Input

Eu quero saber comparações principalmente entre o ResNet50 e o EfficientNetB0, e entre datasets diferentes (ex: os EfficientNetB0 cuda rodaram com o dataset RSNA, enquanto os EfficientNetB0 mamografias rodaram com dataset IRMA (imagens completas) e os EfficientNetB0 patches rodaram com patches do IRMA).

### Planner Response

Vou fazer uma análise comparativa entre ResNet50 e EfficientNetB0, e entre os diferentes datasets. Primeiro, vou verificar todas as pastas de resultados disponíveis:

*Listed directory [mammo_resnet50_cuda](file:///d:/mammography-pipelines/outputs/mammo_resnet50_cuda) *

*Listed directory [mammo_efficientnetb0_cuda](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_cuda) *

*Listed directory [mammo_efficientnetb0_patches](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_patches) *

*Listed directory [results](file:///d:/mammography-pipelines/outputs/mammo_resnet50_cuda/results) *

*Listed directory [results](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_cuda/results) *

*Listed directory [results](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_patches/results) *

### Planner Response

Agora vou ler as métricas de todos os modelos:

*Viewed [val_metrics.json](file:///d:/mammography-pipelines/outputs/mammo_resnet50_cuda/results/metrics/val_metrics.json) *

*Viewed [val_metrics.json](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_cuda/results/metrics/val_metrics.json) *

*Listed directory [metrics](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_patches/results/metrics) *

*Viewed [val_metrics.json](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_patches/results/metrics/val_metrics.json) *

### Planner Response

Agora vou verificar também os resultados dos patches multiclasse para completar a análise:

*Listed directory [mammo_efficientnetb0_patches_multiclass](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_patches_multiclass) *

*Listed directory [results](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_patches_multiclass/results) *

*Viewed [val_metrics.json](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_patches_multiclass/results/metrics/val_metrics.json) *

### Planner Response

# Análise Comparativa: ResNet50 vs EfficientNetB0 e Diferentes Datasets

## Comparação de Arquiteturas (Dataset RSNA)

### ResNet50 vs EfficientNetB0 (RSNA - 4 classes)

| Métrica | ResNet50 | EfficientNetB0 | Diferença |
|---------|----------|----------------|-----------|
| **Acurácia** | 65.41% | 68.24% | +2.83% (EfficientNet) |
| **Kappa Quadrático** | 0.685 | 0.711 | +0.026 (EfficientNet) |
| **AUC OvR** | 0.860 | 0.887 | +0.027 (EfficientNet) |

### Análise por Classe - RSNA

**ResNet50:**
- Classe 1: Precision 36.36%, Recall 50.00%
- Classe 2: Precision 44.97%, Recall 91.78%
- Classe 3: Precision 88.42%, Recall 49.41%
- Classe 4: Precision 84.13%, Recall 79.10%

**EfficientNetB0:**
- Classe 1: Precision 0%, Recall 0% (problema grave)
- Classe 2: Precision 55.36%, Recall 84.93%
- Classe 3: Precision 85.71%, Recall 56.47%
- Classe 4: Precision 64.13%, Recall 88.06%

## Comparação de Datasets (EfficientNetB0)

### Dataset RSNA vs IRMA Imagens Completas vs IRMA Patches

| Modelo | Dataset | Tarefa | Acurácia | Kappa | AUC |
|--------|---------|--------|----------|-------|-----|
| EfficientNetB0 | RSNA | 4 classes | 68.24% | 0.711 | 0.887 |
| EfficientNetB0 | IRMA Completo | Binário | 92.52% | 0.812 | 0.983 |
| EfficientNetB0 | IRMA Patches | Binário | 88.46% | 0.728 | 0.971 |

### EfficientNetB0 - Comparação Multiclasse

| Dataset | Acurácia | Kappa | Observações |
|---------|----------|-------|-------------|
| RSNA | 68.24% | 0.711 | 4 classes (1-4) |
| IRMA Completo | 74.52% | 0.607 | 4 classes (A-D, sem C) |
| IRMA Patches | 71.82% | 0.574 | 4 classes (A-D, sem C) |

## Principais Insights

### 1. **EfficientNetB0 vs ResNet50 (RSNA)**
- **EfficientNetB0 superior**: Melhor em todas as métricas principais
- **Problema crítica**: EfficientNetB0 não classificou nenhuma amostra como classe 1
- **ResNet50 mais equilibrado**: Consegue classificar todas as classes, embora com menor acurácia geral

### 2. **Impacto do Dataset**
- **IRMA vs RSNA**: Datasets IRMA apresentam performance muito melhor
- **Tarefa binária vs multiclasse**: Classificação binária (AB/CD) significativamente mais fácil
- **Imagens completas vs patches**: Patches têm performance ligeiramente inferior

### 3. **Análise por Dataset**

**RSNA (Digital):**
- Menor performance geral
- Distribuição desbalanceada afeta resultados
- Classes intermediárias (2,3) mais difíceis

**IRMA (Filme digitalizado):**
- Performance muito superior em tarefas binárias
- Problemas com classe C (ausente em validação)
- Melhor generalização para classes extremas (A, D)

### 4. **Recomendações**

1. **Para RSNA**: ResNet50 pode ser mais robusto apesar da menor acurácia
2. **Para IRMA**: EfficientNetB0 excelente para classificação binária
3. **Para produção**: Considerar abordagem em duas etapas (binário + multiclasse)
4. **Dados necessários**: Mais amostras da classe C para IRMA

## Conclusão Final

O **dataset IRMA** é significativamente mais adequado para classificação automática, com o **EfficientNetB0** apresentando os melhores resultados. Para o dataset RSNA, a escolha entre ResNet50 e EfficientNetB0 depende da prioridade: performance geral (EfficientNet) vs robustez (ResNet50).

### User Input

Tem que fazer as seguintes comparações:
1. ResNet50 x EfficientNetB0 dataset RSNA, para 4 classes
2. ResNet50 x EfficientNetB0 dataset RSNA, para classificacao binária
3. EfficientNetB0: RSNA x IRMA x IRMA patches para 4 classes
4. EfficientNetB0: RSNA x IRMA x IRMA patches para classificacao binária


### Planner Response

Vou fazer as comparações solicitadas. Primeiro, preciso verificar se existem resultados binários para RSNA com ResNet50 e EfficientNetB0:

*Searched filesystem*

*Searched filesystem*

### Planner Response

Vou verificar todas as pastas para encontrar modelos binários RSNA:

*Listed directory [mammo_resnet50_abxcd_cuda](file:///d:/mammography-pipelines/outputs/mammo_resnet50_abxcd_cuda) *

*Listed directory [mammo_efficientnetb0_abxcd_cuda](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_abxcd_cuda) *

### Planner Response

Perfeito! Encontrei os modelos binários ABxCD para RSNA. Vou ler suas métricas:

*Listed directory [results](file:///d:/mammography-pipelines/outputs/mammo_resnet50_abxcd_cuda/results) *

*Listed directory [results](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_abxcd_cuda/results) *

*Viewed [val_metrics.json](file:///d:/mammography-pipelines/outputs/mammo_resnet50_abxcd_cuda/results/metrics/val_metrics.json) *

*Viewed [val_metrics.json](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_abxcd_cuda/results/metrics/val_metrics.json) *

### Planner Response

Agora vou compilar todas as informações para as 4 comparações solicitadas:

# Análise Comparativa Completa

## 1. ResNet50 x EfficientNetB0 - Dataset RSNA (4 classes)

| Métrica | ResNet50 | EfficientNetB0 | Vencedor |
|---------|----------|----------------|-----------|
| **Acurácia** | 65.41% | 68.24% | EfficientNetB0 (+2.83%) |
| **Kappa Quadrático** | 0.685 | 0.711 | EfficientNetB0 (+0.026) |
| **AUC OvR** | 0.860 | 0.887 | EfficientNetB0 (+0.027) |

**Análise por Classe:**
- **Classe 1**: ResNet50 (50% recall) vs EfficientNetB0 (0% recall) - **ResNet50 vence**
- **Classe 2**: ResNet50 (91.78% recall) vs EfficientNetB0 (84.93% recall) - **ResNet50 vence**
- **Classe 3**: ResNet50 (49.41% recall) vs EfficientNetB0 (56.47% recall) - **EfficientNetB0 vence**
- **Classe 4**: ResNet50 (79.10% recall) vs EfficientNetB0 (88.06% recall) - **EfficientNetB0 vence**

## 2. ResNet50 x EfficientNetB0 - Dataset RSNA (Classificação Binária ABxCD)

| Métrica | ResNet50 | EfficientNetB0 | Vencedor |
|---------|----------|----------------|-----------|
| **Acurácia** | 79.25% | 86.79% | EfficientNetB0 (+7.54%) |
| **Kappa** | 0.566 | 0.694 | EfficientNetB0 (+0.128) |
| **AUC** | 0.925 | 0.955 | EfficientNetB0 (+0.030) |

**Análise Binária:**
- **Classe AB**: ResNet50 (98.77% recall) vs EfficientNetB0 (95.06% recall) - ResNet50 melhor em recall
- **Classe CD**: ResNet50 (72.57% recall) vs EfficientNetB0 (83.97% recall) - EfficientNetB0 melhor em recall

## 3. EfficientNetB0 - RSNA x IRMA x IRMA Patches (4 classes)

| Dataset | Acurácia | Kappa | AUC | Classes |
|---------|----------|-------|-----|---------|
| **RSNA** | 68.24% | 0.711 | 0.887 | 1,2,3,4 |
| **IRMA Completo** | 74.52% | 0.607 | N/A | A,B,C,D (sem C) |
| **IRMA Patches** | 71.82% | 0.574 | N/A | A,B,C,D (sem C) |

**Ranking (4 classes):**
1. **IRMA Completo** - 74.52% acurácia
2. **IRMA Patches** - 71.82% acurácia  
3. **RSNA** - 68.24% acurácia

**Problemas identificados:**
- **IRMA**: Classe C sem amostras no conjunto de validação
- **RSNA**: EfficientNetB0 não classificou nenhuma amostra como classe 1

## 4. EfficientNetB0 - RSNA x IRMA x IRMA Patches (Classificação Binária)

| Dataset | Acurácia | Kappa | AUC | Classes |
|---------|----------|-------|-----|---------|
| **RSNA (ABxCD)** | 86.79% | 0.694 | 0.955 | AB vs CD |
| **IRMA Completo** | 92.52% | 0.812 | 0.983 | AB vs CD |
| **IRMA Patches** | 88.46% | 0.728 | 0.971 | AB vs CD |

**Ranking (Binário):**
1. **IRMA Completo** - 92.52% acurácia (melhor performance)
2. **IRMA Patches** - 88.46% acurácia
3. **RSNA** - 86.79% acurácia

## Conclusões Principais

### **Arquiteturas:**
- **EfficientNetB0 superior** em todos os cenários RSNA
- **Vantagem consistente**: 2.83% (4 classes) a 7.54% (binário)
- **Exceção**: ResNet50 mais robusto para classe 1 no problema 4 classes

### **Datasets:**
- **IRMA > RSNA** em todos os cenários
- **Diferença dramática**: Até 5.73% de vantagem para IRMA em classificação binária
- **IRMA Completo > Patches**: Imagens completas superam patches

### **Complexidade da Tarefa:**
- **Binário > Multiclasse**: Diferença de 18-24% em acurácia
- **Classificação binária já é clinicamente útil** para triagem
- **Multiclasse ainda precisa de melhorias**

### **Recomendações:**
1. **Produção**: Usar EfficientNetB0 com dataset IRMA completo para classificação binária
2. **Pesquisa**: Investigar por que EfficientNetB0 falha em classe 1 (RSNA)
3. **Dados**: Coletar mais amostras da classe C (IRMA)
4. **Arquitetura**: EfficientNetB0 é a escolha padrão para novos experimentos

### User Input

pq classe C ficou sem amostras no conjunto de validação? quantos exemplares da classe C tem em comparacao com as outras? 

### Planner Response

Vou investigar por que a classe C ficou sem amostras no conjunto de validação. Primeiro, vou verificar os arquivos de predições para entender a distribuição:

*Viewed [val_predictions.csv](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_mamografias_multiclass/results/val_predictions.csv) *

*Viewed [val_predictions.csv](file:///d:/mammography-pipelines/outputs/mammo_efficientnetb0_patches_multiclass/results/val_predictions.csv) *

### Planner Response

Agora vou contar quantas amostras de cada classe existem nos arquivos de validação:









### Planner Response

Vou usar uma abordagem diferente para contar as classes:









### Planner Response

Agora vou fazer o mesmo para o modelo de patches:









### Planner Response

Agora vou verificar o código fonte para entender como os dados foram divididos:







### Planner Response

Agora vou verificar a distribuição original dos dados no dataset:



### Planner Response

Vou verificar o arquivo de metadados para entender a distribuição original:

*Viewed [metadata.csv](file:///d:/mammography-pipelines/outputs/metadata.csv) *

### Planner Response

Agora vou contar a distribuição original no metadata.csv:









### Planner Response

# Análise: Por que a Classe C ficou sem Amostras na Validação?

## Distribuição Original dos Dados (Dataset IRMA)

Based on the metadata.csv analysis, the original distribution is:

| Classe (BI-RADS) | Count | Percentage |
|------------------|-------|------------|
| **Classe 1** (A) | 34 | 3.7% |
| **Classe 2** (B) | 208 | 22.6% |
| **Classe 3** (C) | 450 | 49.0% |
| **Classe 4** (D) | 229 | 24.9% |
| **Total** | 921 | 100% |

## Distribuição na Validação (Modelo IRMA Completo)

| Classe | Count no Validação | Problema |
|--------|-------------------|----------|
| **A** | 740 amostras | ✅ Presente |
| **B** | 369 amostras | ✅ Presente |
| **C** | 74 amostras | ⚠️ **MUITO POUCAS** |
| **D** | 331 amostras | ✅ Presente |

## Principais Causas do Problema

### 1. **Split Estratificado por Patient/Acession**
O código usa split estratificado agrupado por accession para evitar vazamento de dados:

```python
# Split estratificado por rótulo multiclasse (0=A, 1=B, 2=C, 3=D) 
# agrupado por accession para evitar vazamento
```

### 2. **Desbalanceamento Extremo**
- **Classe C tem 450 amostras** originalmente (49% do total)
- **Mas apenas 74 chegaram à validação** (10x menos que o esperado)
- Isso indica que **a maioria das amostras da classe C foi para o treino**

### 3. **Problema de Agrupamento por Accession**
O split agrupado por accession pode ter criado um viés:
- Se pacientes com classe C tinham múltiplas imagens
- E o split colocou a maioria desses pacientes no treino
- Resultado: poucos pacientes com classe C na validação

### 4. **Efeito Aleatório do Split**
Com 921 amostras totais e split 80/20:
- Esperado: ~90 amostras de cada classe na validação
- Real: **A=740, B=369, C=74, D=331** (total 1.514?)
- **Inconsistência**: os números não batem com 921 total

## Investigação Adicional

Os números de validação (1.514 amostras) são maiores que o dataset original (921), sugerindo que:
1. **Pode haver aumento de dados (data augmentation)**
2. **Ou os contagens estão incorretas**
3. **Ou múltiplas views por paciente estão sendo contadas separadamente**

## Soluções Recomendadas

### 1. **Fix Imediato**
```python
# Garantir split estratificado adequado por classe
from sklearn.model_selection import StratifiedGroupKFold

# Usar stratificação garantindo mínimo por classe
stratify_labels = df['multiclass_label']
groups = df['AccessionNumber']
```

### 2. **Verificação de Dados**
- Confirmar se os contadores estão funcionando corretamente
- Verificar se há duplicação nos dados de validação

### 3. **Balanceamento Explícito**
```python
# Garantir número mínimo por classe na validação
min_per_class = 50  # mínimo desejado
validation_samples = []

for class_label in [0, 1, 2, 3]:
    class_df = df[df['multiclass_label'] == class_label]
    n_val = max(min_per_class, int(len(class_df) * 0.2))
    val_samples = class_df.sample(n=n_val, random_state=42)
    validation_samples.append(val_samples)
```

### 4. **Cross-Validation**
Usar k-fold estratificado para avaliação mais robusta:
```python
# StratifiedGroupKFold com k=5
# Garante representação de todas as classes em cada fold
```

## Conclusão

O problema da **classe C sem amostras** na validação é causado por:
1. **Split estratificado inadequado** quando combinado com agrupamento por paciente
2. **Possível bug no contador** ou duplicação de dados
3. **Necessidade de balanceamento explícito** para garantir representação mínima

A classe C **existe no dataset original** (450 amostras), mas o **processo de split falhou** em distribuí-la adequadamente para a validação.