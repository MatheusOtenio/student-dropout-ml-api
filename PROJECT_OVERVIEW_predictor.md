# Relatório Técnico do Backend de Predição (Inferência)

Este documento detalha o serviço de inferência responsável por servir o modelo de evasão em produção. O serviço é uma API REST construída com FastAPI que recebe arquivos CSV, realiza ETL e validação, carrega o modelo treinado (`.pkl`) e retorna probabilidades de evasão.

## Sumário Executivo
O objetivo do serviço é fornecer predições em lote (batch) ou unitárias via HTTP.
-   **Entrada:** Arquivo CSV (multipart/form-data) + ID do modelo + Mapeamento opcional de colunas (JSON).
-   **Saída:** JSON contendo o ID do modelo utilizado, os dados processados e a lista de probabilidades de evasão.
-   **Garantia:** O pipeline de pré-processamento é carregado junto com o modelo para garantir consistência total com o treinamento.

## 1. Descrição do serviço de predição

-   **Endpoint Principal:** `POST /predict`
-   **Localização:** `src/api.py`
-   **Contrato de Entrada (Multipart Form):**
    -   `file`: Arquivo CSV binário (obrigatório).
    -   `model_id`: String identificadora do modelo (ex: `model_20260212T190237`).
    -   `mapping`: String JSON opcional para mapear nomes de colunas do CSV para o schema esperado (ex: `{"nome_no_csv": "nome_esperado"}`).
-   **Contrato de Saída (JSON):**
    ```json
    {
      "model_id": "model_20260212T190237",
      "rows": [ { ...features... } ],
      "predictions": [ 0.12, 0.85, ... ]
    }
    ```

## 2. Fluxo de inferência (runtime)

O fluxo de execução, passo a passo:

1.  **Recepção:** O endpoint `/predict` recebe o arquivo e parâmetros.
2.  **Leitura e Validação Básica:** O arquivo é lido para um buffer de memória (`io.BytesIO`).
3.  **ETL e Normalização (`src/sercives/processador_csv/etl_service.py`):**
    -   A função `transformar_dados` aplica o mapeamento de colunas (se fornecido).
    -   Filtra colunas permitidas (Whitelist) e descarta extras (`df_dropped`).
    -   Calcula features derivadas básicas (ex: idade a partir da data de nascimento).
    -   Converte formatação numérica (vírgula para ponto).
    -   Preenche valores ausentes padrão (ex: notas zeradas, colunas obrigatórias com NA).
    -   Aplica normalização de strings (UPPERCASE) para categóricas.
    -   Output: DataFrame padronizado contendo `FEATURE_COLUMNS` e opcionalmente `situacao`.
4.  **Preparação Final (`preparar_para_predict`):**
    -   Remove colunas auxiliares (ex: `data_nascimento`).
    -   Garante a ordem exata das features esperadas pelo modelo.
5.  **Predição (`src/sercives/predicao_ml/backend_logic.py`):**
    -   A função `predict(model_id, features)` é chamada.
    -   **Prevenção de Data Leakage:** Se a coluna `situacao` estiver presente no DataFrame, ela é removida explicitamente antes da inferência.
    -   **Carregamento:** O artefato `.pkl` é carregado do disco (`src/models/`) se não estiver em cache (`_MODEL_CACHE`).
    -   **Pipeline Sklearn:** O artefato carregado é um `Pipeline` completo.
    -   **Inferência:** Chama-se `model.predict_proba(df)`. O pipeline executa internamente as transformações de imputação e encoding.
    -   **Extração de Risco:** O sistema identifica dinamicamente o índice da classe "1" (Evasão/Fracasso) para retornar a probabilidade correta.
6.  **Output:** O resultado é retornado como lista de floats (probabilidade de evasão).

## 3. Consistência entre treino e inferência

A consistência é garantida pelo uso do mesmo código de definição de transformadores e pelo salvamento do pipeline completo.

-   **Pipeline Serializado:** O `.pkl` contém os objetos `EnsureColumnsTransformer`, `DataCleaningTransformer`, `AdaptiveCategoricalEncoder` já ajustados (`fitted`) durante o treino.
-   **Código Compartilhado:** O arquivo `src/preprocessing/preprocessing.py` é idêntico ao usado no treinamento.
-   **Novas Categorias (Unseen):**
    -   O `AdaptiveCategoricalEncoder` e o `OneHotEncoder` (configurado com `handle_unknown='ignore'`) tratam categorias não vistas no treino, garantindo que a predição não falhe.
    -   Para colunas de alta cardinalidade usando Target Encoding, valores desconhecidos recebem a média global do target (aprendida no treino).

## 4. Validação de schema & robustez

-   **Validação de Tipos:** Ocorrem principalmente no ETL (`etl_service.py`), onde strings numéricas com vírgula são convertidas para float.
-   **Colunas Obrigatórias:** Definidas em `FEATURE_COLUMNS` no `schemas.py`. Se faltarem no CSV, são criadas preenchidas com `pd.NA` (que depois viram mediana/constante no pipeline).
-   **Inputs Inválidos:**
    -   CSV corrompido ou JSON inválido retornam **HTTP 400**.
    -   Modelo não encontrado retorna **HTTP 404**.
    -   Erro interno de predição retorna **HTTP 500**.
-   **Manejo de NaNs:**
    -   Notas (Enem/Vestibular) vazias são preenchidas com `0.0` no ETL.
    -   Outros campos numéricos são preenchidos com a Mediana no Pipeline (`SimpleImputer(strategy="median")`).
    -   Categóricos são preenchidos com "missing" (`SimpleImputer(strategy="constant")`).

## 5. Formato de saída e contrato de logs

-   **Logs:** Gravados via `logging` (stdout/stderr).
    -   Campos logados: Nome do arquivo, `model_id`, número de linhas, erros de stack trace.
-   **Exemplo de Saída (JSON):**
    ```json
    {
        "model_id": "model_v1",
        "rows": [
            {"id_aluno": 1, "sexo": "M", "nota_enem_redacao": 800.0, ...}
        ],
        "predictions": [0.05, 0.92]
    }
    ```

## 6. Performance & SLA

-   **Latência:**
    -   O carregamento do modelo é feito sob demanda mas cacheado em memória (`_MODEL_CACHE`). A primeira requisição para um `model_id` novo é mais lenta (I/O disco + deserialização), as subsequentes são rápidas (apenas CPU bound da inferência).
    -   Suporta processamento em batch (CSV com múltiplas linhas), o que é muito mais eficiente que requisições unitárias.
-   **Memória:** Depende do tamanho do CSV enviado e do tamanho do modelo.
-   **Benchmark (Sugestão):**
    ```bash
    # Teste de carga simples com Apache Bench ou similar
    ab -p payload.csv -T multipart/form-data -c 10 -n 100 http://localhost:8000/predict
    ```

## 7. Mecanismo de versionamento & reload do modelo

-   **Seleção de Modelo:** O cliente decide qual versão usar enviando o parâmetro `model_id`.
-   **Carregamento:**
    -   O sistema busca o arquivo `src/models/{model_id}.pkl`.
    -   Se o arquivo existir, ele é carregado.
-   **Reload/Rollback:**
    -   Para atualizar um modelo, basta colocar o novo arquivo `.pkl` na pasta `src/models/`.
    -   O cache (`_MODEL_CACHE`) é simples: uma vez carregado, o modelo persiste em memória. Para forçar reload sem reiniciar, seria necessário implementar uma lógica de invalidação de cache (não presente na versão atual).

## 8. Checks de sanidade operacionais

-   **Script de Verificação:** `src/sercives/predicao_ml/check_env.py` verifica se o diretório de modelos existe e se há arquivos `.pkl` disponíveis.
-   **Health Check API:** `GET /health` retorna `{"status": "ok"}` para verificar se o servidor está de pé.

## 9. Riscos & Observações

-   **Sincronia ETL vs Treino:** A lista `FEATURE_COLUMNS` em `schemas.py` é a fonte da verdade. Ela deve estar perfeitamente alinhada com o que o modelo foi treinado para esperar.
-   **Data Leakage:** A lógica de backend (`backend_logic.py`) possui um *guardrail* explícito que remove a coluna `situacao` antes de passar os dados para o modelo, prevenindo vazamento de informação caso o usuário envie o target junto com as features.
