# Sistema de Predição de Evasão Escolar

## 1. Visão Geral

Este projeto implementa um sistema completo de **predição de evasão escolar** composto por:

- **Backend FastAPI** para:
  - Receber arquivos CSV com histórico de alunos.
  - Treinar modelos de classificação (Logistic Regression e LightGBM).
  - Salvar artefatos `.pkl` contendo pipeline de pré-processamento, modelo e metadados.
- **Frontend Streamlit** para:
  - Interface amigável para upload de CSV.
  - Disparo do treinamento via API.
  - Exibição de métricas de desempenho e identificador do modelo.
- **Módulo de Treino (trainer)** com:
  - Validação cruzada estratificada (`StratifiedKFold`).
  - Probabilidades calibradas.
  - Otimização de hiperparâmetros com Optuna para LightGBM.
- **Módulo de Inferência (inference)** para carregar artefatos `.pkl` e gerar previsões de risco de evasão.

---

## 2. Instalação e Execução

### 2.1. Pré-requisitos

- Python 3.10+
- pip
- (Opcional) Docker, se desejar executar via container.

### 2.2. Instalação das dependências

Na raiz do projeto:

```bash
pip install -r requirements.txt
```

### 2.3. Executar o Backend FastAPI

Na raiz do projeto:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

A API ficará disponível em:

- Swagger: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

### 2.4. Executar o Frontend Streamlit

Na raiz do projeto:

```bash
streamlit run frontend.py
```

O Streamlit abrirá no navegador (geralmente em `http://localhost:8501`).

### 2.5. Fluxo Básico de Uso

1. Inicie o backend FastAPI (`uvicorn`).
2. Inicie o frontend Streamlit.
3. No Streamlit:
   - Faça upload de um CSV com os dados dos alunos.
   - Opcionalmente, preencha a configuração JSON (por ex., `{"model_type": "lightgbm", "optimize": true}`).
   - Clique em **“Treinar modelo”**.
4. O Streamlit chamará o endpoint `/train` e exibirá:
   - Caminho do artefato `.pkl`.
   - Métricas (ex.: AUC ROC, Brier score).
   - Versão e tipo de modelo.

---

## 3. Arquitetura do Pipeline

### 3.1. Pré-processamento (`src/preprocessing.py`)

Pipeline de pré-processamento construído com `sklearn.pipeline.Pipeline` e `ColumnTransformer`, incluindo:

- **Limpeza de dados (DataCleaningTransformer)**
  - Normalização de strings:
    - `lowercase`.
    - Remoção de acentos.
  - Conversão de placeholders (`"NA"`, `"na"`, `"n/a"`, `"-"`, vazio) para `NaN`.

- **Feature Engineering (FeatureEngineeringTransformer)**
  - `idade`: calculada a partir de `data_nascimento` e `ano_ingresso`.
  - `aprovacoes_ratio`: razão entre `qtd_aprovacoes` e `qtd_disciplinas` (com proteção contra divisão por zero).
  - `tempo_cursado_ratio`: razão entre `tempo_cursado` e `tempo_total_curso` (também com proteção).
  - `ano_ingresso_decada`: década de ingresso (ex.: 2010, 2020, ...).

- **Tratamento de colunas ausentes (EnsureColumnsTransformer)**
  - Garante que as colunas esperadas pelo pipeline existam, criando-as como `NaN` quando necessário.

- **Imputação**
  - Numéricos: `SimpleImputer(strategy="median")`.
  - Categóricos: `SimpleImputer(strategy="constant", fill_value="missing")`.

- **Encoding (AdaptiveCategoricalEncoder)**
  - Baixa cardinalidade: `OneHotEncoder(handle_unknown="ignore")`.
  - Alta cardinalidade: `TargetEncoder` (`category_encoders`) para evitar explosão de dimensionalidade, especialmente importante para modelos de árvore (LightGBM).

### 3.2. Modelos e Treinamento (`src/trainer.py`)

- **Modelos suportados**
  - `LogisticRegression` (baseline, com `class_weight="balanced"`).
  - `LightGBM` (`LGBMClassifier`, também com `class_weight="balanced"`).

- **Calibração de probabilidades**
  - Opcional via `CalibratedClassifierCV`:
    - `method` configurável (`isotonic` por padrão).
    - CV interno configurável.

- **Validação cruzada**
  - `StratifiedKFold` com:
    - `n_splits` configurável (`cv_splits`).
    - `shuffle=True` e `random_state` configurável.
  - Métricas calculadas em Out-Of-Fold:
    - **ROC AUC** (`roc_auc_score`).
    - **Brier Score** (`brier_score_loss`).

- **Otimização de Hiperparâmetros (Optuna)**
  - Função `tune_and_train(df, config, n_trials=None)`:
    - Cria um `Study` Optuna para LightGBM.
    - Otimiza hiperparâmetros como:
      - `num_leaves`
      - `learning_rate`
      - `feature_fraction`
      - `min_child_samples`
    - Objetivo: maximizar ROC AUC (minimizando `1 - AUC`).
    - Após encontrar `best_params`, treina o modelo final em todo o dataset usando esses parâmetros.
    - Os `best_params` são armazenados em `metadata["best_params"]` no artefato `.pkl`.

- **Flag de otimização**
  - `train_model(df, config, optimize: bool = False, ...)`:
    - `optimize=False` (padrão): treino rápido com parâmetros informados.
    - `optimize=True`: chama internamente `tune_and_train` antes do treino final.

- **Artefato `.pkl`**
  - Estrutura salva via `joblib`:
    ```python
    {
      "preprocessor": <pipeline de pré-processamento>,
      "model": <pipeline completo (preprocess + modelo)>,
      "metadata": {
        "model_type": ...,
        "metrics": {"roc_auc": ..., "brier_score": ...},
        "hyperparameters": {...},
        "best_params": {...},   # se Optuna for usado
        "cv_splits": ...,
        "class_mapping": {...},
        "n_samples": ...,
        "n_features": ...,
        "timestamp": ...
      },
      "version": "X.Y.Z"
    }
    ```

### 3.3. Inferência (`src/inference.py`)

- **Carregamento**
  - `load_model(artifact_path)`:
    - Carrega o artefato `.pkl` via `joblib.load`.

- **Predição**
  - `predict_proba(model_bundle, df)`:
    - `model_bundle["model"]` é o `Pipeline` completo treinado (pré-processamento + modelo).
    - O `df` de entrada:
      - **Não deve** conter `situacao` em produção; se contiver, a coluna é descartada.
      - Pode conter colunas como `sexo`, `cor_raca`, `data_nascimento`, `curso`, notas ENEM, etc.
    - Produz um `DataFrame` com:
      - `prob_evasao`: probabilidade de evasão (classe positiva).
      - `class_pred`: 0 ou 1 (threshold 0.5).
      - `risk_level`: `Baixo` / `Médio` / `Alto` conforme a probabilidade.

---

## 4. Dicionário de Dados Simplificado (CSV de Treino)

O CSV de treino deve conter, no mínimo, a coluna **obrigatória**:

- `situacao` (target):
  - Valores mapeados para binário:
    - Classe 0: `"Regular"`, `"Formado"`.
    - Classe 1: `"Desistente"`, `"Trancado"`.

Colunas de features recomendadas (todas opcionais, o pipeline é robusto a ausências):

### 4.1. Dados demográficos

- `sexo`
- `cor_raca`
- `municipio_residencia`
- `uf_residencia`
- `data_nascimento` (formatos aceitos: `YYYY-MM-DD`, `DD/MM/YYYY`, `DD-MM-YYYY`)
- `idade` (pode ser fornecida ou derivada)

### 4.2. Dados acadêmicos

- `curso`
- `campus`
- `turno`
- `modalidade_ingresso`
- `tipo_cota`
- `coeficiente_rendimento`
- `disciplinas_aprovadas`
- `disciplinas_reprovadas_nota`
- `disciplinas_reprovadas_frequencia`
- `total_semestres_cursados`
- `periodo`
- `mudou_curso` (ex.: 0/1, true/false)
- `ano_ingresso`
- `semestre_ingresso`

### 4.3. Notas de ingresso

- `enem_humanas`
- `enem_linguagem`
- `enem_matematica`
- `enem_natureza`
- `enem_redacao`
- `nota_final_ingresso`
- `nota_vestibular_total`

> Observação: os nomes exatos de colunas podem ser adaptados, desde que o pré-processador seja atualizado para refletir o schema real. O pipeline foi projetado para ignorar colunas extras e lidar com colunas ausentes, imputando valores.

---

## 5. API `/train`: Especificação e Consumo

### 5.1. Endpoint

- Método: `POST`
- Caminho: `/train`
- Tipo: `multipart/form-data`

### 5.2. Parâmetros

- `file`: arquivo CSV (campo `UploadFile`), **obrigatório**.
  - Extensão obrigatória: `.csv`.
- `config`: string JSON opcional (campo `Form`), por exemplo:
  ```json
  {
    "model_type": "lightgbm",
    "cv_splits": 5,
    "calibrate": true,
    "calibration_method": "isotonic",
    "n_trials": 20,
    "artifact_path": "artifacts/modelo_evasao.pkl",
    "version": "1.0.0"
  }
  ```

### 5.3. Respostas

- **200 OK** (sucesso):
  ```json
  {
    "artifact_path": "artifacts/model_20260119T123456.pkl",
    "version": "1.0.0",
    "model_version": "1.0.0",
    "metrics": {
      "roc_auc": 0.85,
      "brier_score": 0.17
    },
    "model_type": "lightgbm",
    "timestamp": "2026-01-19T12:34:56.789012+00:00"
  }
  ```

- **400 Bad Request**:
  - Erros de:
    - Extensão não `.csv`.
    - CSV inválido.
    - JSON de configuração inválido.
    - Ausência da coluna `situacao`.
    - Problemas de dados (ex.: valores inválidos).

- **500 Internal Server Error**:
  - Erro inesperado durante o treinamento (logado no backend, resposta genérica ao cliente).

---

## 6. Checklist Operacional (Produção)

### 6.1. Métricas de Desempenho

- **ROC AUC**:
  - Mede a capacidade do modelo de separar alunos que evadem dos que permanecem.
  - Deve ser monitorado em:
    - Validação offline.
    - Acompanhamento periódico com dados recentes (backtesting).

- **Brier Score**:
  - Mede a qualidade das probabilidades estimadas (calibração).
  - Valores menores indicam melhores probabilidades calibradas.
  - Útil para avaliar se o modelo está superestimando ou subestimando riscos.

### 6.2. Monitoração de Drift

- **Drift de dados (input)**:
  - Monitorar distribuições de:
    - `idade`, `ano_ingresso`, `coeficiente_rendimento`, etc.
    - Distribuições de categorias (`curso`, `turno`, `modalidade_ingresso`, `tipo_cota`).
  - Indicadores:
    - Mudanças significativas nas proporções de cursos, turnos, perfis de ingresso.
    - Aumento de valores ausentes em colunas importantes.

- **Drift de target (quando disponível)**:
  - Taxa de evasão real ao longo do tempo.
  - Se a taxa de evasão mudar muito, pode ser necessário re-treinar.

### 6.3. Saúde da API / Infraestrutura

- **Latência do endpoint `/train`**:
  - Treinos com otimização (Optuna) podem ser mais demorados (n_trials alto).
  - Configurar limites de tempo e recursos adequados.

- **Erros HTTP**:
  - Monitorar frequência de erros `400` e `500`.
  - Em especial:
    - Aumentos de `400` por problemas de schema de CSV (colunas faltando, tipos inválidos).
    - `500` indicando falhas de infraestrutura ou de código.

### 6.4. Ciclo de Re-treino

- Definir janelas periódicas de re-treino (ex.: por semestre/ano).
- Verificar:
  - Queda de ROC AUC e aumento de Brier Score em dados recentes.
  - Mudanças na população de alunos (novos cursos, políticas de ingresso, etc.).
- Atualizar:
  - Artefatos `.pkl`.
  - Versionamento (`version` / `model_version`).
  - Documentação interna sobre mudanças de modelo.

---

## 7. Resumo Rápido de Comandos

- Instalar dependências:

  ```bash
  pip install -r requirements.txt
  ```

- Rodar API FastAPI:

  ```bash
  uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
  ```

- Rodar Frontend Streamlit:

  ```bash
  streamlit run frontend.py
  ```

- Rodar testes de integração:

  ```bash
  pytest -q
  ```
- Rodar validador de modelo pkl:

  ```bash
  python tests/validate_model.py artifacts/model_nome.pkl
  ```

- Para subir tudo:

  ```bash
  docker-compose up --build
  ```