# Relatório Técnico do Backend de Criação do Modelo de Evasão

Este documento detalha a pipeline de construção do modelo preditivo de evasão escolar. O objetivo é identificar alunos com alto risco de evasão (rótulo positivo) para intervenção preventiva. O target é binário, derivado da situação acadêmica atual do aluno, e o modelo utiliza LightGBM com validação cruzada estratificada.

O projeto agora inclui um **Processador de CSV** dedicado, que garante a padronização e filtragem dos dados *antes* mesmo de entrarem na pipeline de treinamento.

## 1. Objetivo & Definição do Target

O objetivo do modelo é classificar a probabilidade de um aluno evadir (classe positiva).

- **Definição de Evasão (Target=1):** Alunos com situação "desistente" ou "trancado" (Status de Fracasso).
- **Definição de Sucesso (Target=0):** Alunos com situação "formado" (Status de Sucesso).
- **Tratamento de Outros:** Registros com situações diferentes ("regular", "transferido", "afastado") são **automaticamente descartados** tanto no Processador de CSV quanto na validação de entrada do Treinamento.

**Estratégia de Filtragem:**
A filtragem é aplicada de forma redundante e segura em duas camadas:
1. **No Processador de CSV (`src/csv_processor/etl_service.py`):** Gera um dataset "limpo" contendo apenas as classes de interesse.
2. **No Treinamento (`src/modelo_regressao/trainer.py`):** Valida e reforça a filtragem para garantir a integridade do dataset de treino.

## 2. Fluxo Completo (CSV Processor + Training)

O fluxo foi expandido para incluir o tratamento prévio dos arquivos brutos:

1.  **Upload & Análise (Mapping):** O usuário envia um CSV bruto. O sistema usa `rapidfuzz` e regras heurísticas (`src/csv_processor/mapping_service.py`) para sugerir o mapeamento de colunas para o esquema padrão.
2.  **ETL & Padronização:**
    -   Renomeação de colunas.
    -   Cálculo de idade (se necessário).
    -   **Filtragem de Linhas:** Remoção silenciosa de alunos que não são "formado", "desistente" ou "trancado".
    -   Output: `dados_ml_padronizados.csv` (pronto para treino).
3.  **Pipeline de Treinamento (.pkl):**
    -   **Entrada:** CSV padronizado.
    -   **Mapeamento Binário:** `situacao` -> 0 ou 1.
    -   **Split:** Validação Cruzada (StratifiedKFold).
    -   **Pré-processamento:** Limpeza, imputação e encoding.
    -   **Treino/Tuning:** LightGBM com otimização via Optuna.
    -   **Exportação:** Artefato `.pkl` contendo pipeline completo.

**Arquivos Principais:**
-   `src/csv_processor/etl_service.py`: Lógica de transformação e filtragem de segurança.
-   `src/csv_processor/schemas.py`: Definições de colunas e status válidos (`STATUS_VALIDOS`).
-   `src/modelo_regressao/trainer.py`: Motor de treinamento e validação.
-   `src/preprocessing.py`: Pipeline scikit-learn (Imputers, Scalers, Encoders).

## 3. Detalhes do Pré-processamento (Pipeline .pkl)

O pipeline salvo no `.pkl` garante que os dados de produção sofram as mesmas transformações do treino.

**Features Numéricas:**
Inclui notas (Enem/Vestibular), dados acadêmicos (CR, frequência) e a nova feature **`calouro`**.
-   *Tratamento:* `SimpleImputer(strategy="median")` + `StandardScaler`.

**Features Categóricas:**
Inclui `sexo`, `municipio_residencia`, `curso`, `campus`, `turno`, `modalidade_ingresso`, `tipo_cota`.
*Nota: `cor_raca` e `uf_residencia` foram removidas do escopo atual.*
-   *Tratamento:* `SimpleImputer(strategy="constant")` + `AdaptiveCategoricalEncoder` (OneHot para baixa cardinalidade, TargetEncoder para alta).

**Engenharia de Features:**
-   `nota_enem_total`: Soma das notas parciais.
-   `nota_vestibular_total`: Soma das notas parciais.
-   `aprovacao_ratio`: Razão entre disciplinas aprovadas e total cursado.

## 4. Estratégia de Validação

-   **Método:** Stratified K-Fold Cross-Validation (`n_splits=5`).
-   **Consistência:** A estratificação garante que cada fold tenha a mesma proporção de formados/evadidos, crucial para datasets desbalanceados.

## 5. Modelo e Hiperparâmetros

-   **Algoritmo:** LightGBM (`LGBMClassifier`) com `class_weight="balanced"`.
-   **Otimização:** `optuna` busca os melhores hiperparâmetros (num_leaves, learning_rate, feature_fraction, min_child_samples).
-   **Fallback:** LogisticRegression (se configurado).

## 6. Métricas e Logs

O sistema registra logs detalhados de cada etapa.
-   **Métricas:** ROC-AUC e Brier Score.
-   **Metadados:** O artefato final contém um dicionário de metadados com:
    -   Performance (CV score).
    -   Importância das features.
    -   Hiperparâmetros utilizados.
    -   Timestamp e versão.

## 7. Exemplos de Uso

**Via API (Processamento de CSV):**
```http
POST /process-csv
Content-Type: multipart/form-data
file: @meu_arquivo_bruto.csv
mapping: {"Nome Aluno": "nome", "Situação": "situacao", ...}
```
*Retorno:* URL para download do CSV limpo (apenas alunos Sucesso/Fracasso).

**Via API (Treinamento):**
```http
POST /train
Content-Type: multipart/form-data
file: @dados_ml_padronizados.csv
config: {"optimize_trials": 20}
```

## 8. Stack Tecnológico & Versões

As versões críticas foram fixadas para garantir reprodutibilidade (`requirements.txt`):

-   **Core:** Python 3.11 (Docker Slim)
-   **API:** FastAPI 0.128.0
-   **Dados:** Pandas 2.3.3, Numpy 2.4.1
-   **ML:** Scikit-learn 1.8.0, LightGBM 4.6.0, Category Encoders 2.9.0
-   **Otimização:** Optuna 4.7.0
-   **Utils:** Unidecode 1.3.8, RapidFuzz 3.10.1

## 9. Considerações Finais

A arquitetura atual prioriza a **integridade dos dados**. Ao forçar a filtragem de status irrelevantes ("regular", etc.) logo na etapa de processamento de CSV, eliminamos ruídos que poderiam contaminar o treinamento, garantindo que o modelo aprenda exclusivamente a distinguir entre sucesso confirmado e evasão confirmada.
