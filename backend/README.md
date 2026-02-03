# Backend Unificado de Processamento e Predição de Evasão

Este diretório contém o **backend unificado** do sistema de predição de evasão universitária. Ele integra, em um único serviço, as responsabilidades que antes estavam em projetos separados:

- Processar e padronizar CSVs brutos (antigo `processador_csv`).
- Aplicar o mesmo pipeline de pré-processamento usado no treino.
- Carregar um modelo pré-treinado `.pkl` e realizar predições de evasão (função equivalente ao antigo `predicao_ml`).
- Expor tudo isso via API HTTP (FastAPI) e disponibilizar um cliente de teste em Streamlit (`front_test.py`).

---

## 1. Estrutura geral do backend

Principais pastas e arquivos:

- `src/api.py`
  - Define a API FastAPI.
  - Endpoints principais (nomes exemplificativos):
    - `POST /mapping-suggestions`: recebe um CSV e sugere o mapeamento de colunas.
    - `POST /preview-mapped-csv`: aplica o ETL com base no mapeamento informado e retorna uma pré-visualização.
    - `POST /predict`: aplica o ETL completo, carrega o modelo `.pkl` selecionado (`model_id`) e retorna as probabilidades/indicadores de evasão.

- `src/preprocessing/`
  - Implementa o pipeline de pré-processamento compartilhado (imputação, engenharia de atributos, codificação categórica, etc.), alinhado ao pipeline de treino.

- `src/sercives/processador_csv/`
  - Contém a lógica de ETL e mapeamento:
    - Leitura e validação de CSV.
    - Sugestão de mapeamento de colunas para o schema interno.
    - Aplicação do mapeamento e produção de um `DataFrame` pronto para o modelo.

- `src/sercives/predicao_ml/`
  - Lógica de carregamento do modelo `.pkl` e execução da predição:
    - Carrega bundles salvos em `src/models/` via `joblib.load`.
    - Garante que o pipeline de pré-processamento seja desserializado corretamente.
    - Executa `predict_proba` e monta o resultado com `prob_evasao`, `class_pred` e `risk_level`.

- `src/models/`
  - Diretório onde são colocados os artefatos `.pkl` gerados pelo pipeline de treino (fora deste backend).
  - Cada arquivo representa uma versão de modelo pronta para produção (por exemplo, `model_5.pkl`, `model_6.pkl`).

- `front_test.py`
  - Cliente Streamlit para testar a API de forma interativa:
    - Permite informar a URL da API (local ou em produção).
    - Enviar um CSV.
    - Acionar, em passos, os endpoints de mapeamento, pré-visualização e predição.

---

## 2. Como rodar o backend localmente

### 2.1. Ambiente local (API + cliente Streamlit)

Dentro de `backend/`:

1. (Opcional) criar e ativar um ambiente virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux/macOS
   # ou
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

2. Instalar dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Rodar a API diretamente (modo desenvolvimento):

   ```bash
   uvicorn src.api:app --reload --port 10000
   ```

4. Em outro terminal, rodar o cliente Streamlit de teste:

   ```bash
   streamlit run front_test.py
   ```

   - No campo "URL da API" do Streamlit, use `http://localhost:10000`.
   - Envie um CSV de alunos, gere o mapeamento, pré-visualize o ETL e execute a predição.

### 2.2. Usando Docker

Ainda dentro de `backend/`, você pode construir e rodar a imagem de produção do backend:

```bash
docker build -t evasao-api .

docker run --rm -p 10000:10000 evasao-api
```

- `docker build -t evasao-api .` constrói a imagem usando o `Dockerfile` do backend.
- `docker run --rm -p 10000:10000 evasao-api` expõe a API em `http://localhost:10000`.
- O comando `CMD` do Dockerfile sobe o Uvicorn com `src.api:app` na porta `10000` (ou na porta definida pela variável `PORT`).

Você pode então apontar o `front_test.py` (rodando localmente via `streamlit run front_test.py`) para essa URL Dockerizada.

---

## 3. Fluxo de uso da API

Em um cenário típico:

1. O usuário inicia o container ou a API local.
2. O cliente (por exemplo, `front_test.py` em Streamlit) envia um CSV para `/mapping-suggestions`.
3. A API analisa os nomes das colunas e sugere um mapeamento para o schema interno.
4. O cliente confirma esse mapeamento e chama `/preview-mapped-csv` para visualizar algumas linhas já transformadas.
5. Depois de ajustar o mapeamento (se necessário), o cliente chama `/predict`, informando:
   - o arquivo CSV,
   - o mapeamento,
   - opcionalmente um `model_id` para escolher o `.pkl` em `src/models/`.
6. A API aplica o pipeline completo e devolve as probabilidades de evasão por aluno, junto com indicadores de risco.

---

## 4. Detalhes de validação cruzada do modelo (conceito)

O backend serve modelos que foram treinados com validação cruzada estratificada. Abaixo está a explicação detalhada de como esse processo funciona (conteúdo equivalente ao documento original de cross-validation).

### 4.1. Como o código decide quantos folds usar

No `_cross_validate`:

```python
min_samples = int(y.value_counts().min())
requested_splits = int(n_splits)
actual_splits = min(requested_splits, min_samples)
```

- `n_splits` vem da configuração (`config["cv_splits"]`, ex.: `5`).
- `y.value_counts().min()` pega o **número mínimo de alunos em qualquer uma das classes** (`0` ou `1`).
- `actual_splits` é o número real de folds usados:
  - Se você pediu `5` folds, mas a classe minoritária só tem `3` alunos, faz **3 folds**.
  - Se você tem bastante dados nas duas classes, usa o valor pedido (ex.: `5`).
- Se `actual_splits < 2`, o código trata como caso especial (treina sem CV), mas assumindo que você tem dados suficientes, consideramos o caso normal (2 ou mais folds).

---

### 4.2. Em números e porcentagens: quanto é treino e quanto é validação

Com `StratifiedKFold`:

```python
skf = StratifiedKFold(
    n_splits=actual_splits,
    shuffle=True,
    random_state=random_state,
)
```

Em cada *fold*:

- Aproximadamente `1 / n_splits` dos alunos vão para **validação**.
- Aproximadamente `(n_splits − 1) / n_splits` vão para **treino**.

Se `n_splits = 5` (caso típico):

- Em cada rodada:
  - ~20% dos alunos → validação  
  - ~80% dos alunos → treino

#### Exemplo 1: 1000 alunos no CSV

Suponha:

- 1000 linhas válidas (após descartar `situacao` desconhecida).
- `cv_splits = 5`.
- Cada classe (`0` e `1`) tem pelo menos 5 exemplos.

Então:

- `actual_splits = min(5, min_samples)` → digamos que vira `5`.
- Em cada *fold*:
  - Treino ≈ `1000 × (4/5) = 800` alunos.
  - Validação ≈ `1000 × (1/5) = 200` alunos.

E isso é estratificado:

- A proporção de **evasão** e **não evasão** fica parecida em treino e validação.

#### Exemplo 2: 300 alunos

- 300 linhas válidas.
- `cv_splits = 5` e `min_samples` das classes é ≥ 5.

Então por *fold*:

- Treino ≈ `300 × 80% = 240` alunos.  
- Validação ≈ `300 × 20% = 60` alunos.

Se a classe minoritária tiver, por exemplo, só `3` alunos:

- `min_samples = 3`.
- `actual_splits = min(5, 3) = 3`.

Então por *fold*:

- ~66% treino, ~33% validação.

---

### 4.3. Como todos os alunos recebem predição de validação (out-of-fold)

No `_cross_validate`, o fluxo é:

```python
oof_probs = np.zeros(len(y), dtype=float)

for train_idx, val_idx in skf.split(X, y):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]

    # Preprocessamento e treino no grupo de treino
    ...

    proba = fold_model.predict_proba(X_val_proc)[:, 1]
    oof_probs[val_idx] = proba
```

- `skf.split(X, y)` gera `actual_splits` partições.
- Em cada partição:
  - `train_idx`: índices usados para treinar o modelo daquele *fold*.
  - `val_idx`: índices usados para **validar** (o modelo nunca viu esses alunos nesse *fold*).

O ponto importante:

- Ao longo de todos os *folds*, cada aluno entra em:
  - **Treino** em `n_splits - 1` *folds*.
  - **Validação** em **exatamente 1** *fold*.

Isso quer dizer:

- Sim, o sistema “volta” e troca quem é treino e quem é validação em cada rodada:
  - No *Fold 1*, o aluno 1 pode estar na validação, e nos outros 4 *folds* ele é usado como treino.
  - No *Fold 2*, o aluno 2 pode estar na validação, e assim por diante.
- O vetor `oof_probs` guarda, para cada aluno, a **probabilidade predita quando ele estava no grupo de validação**, ou seja, uma predição que não foi “contaminada” por treino com aquele mesmo aluno.

---

### 4.4. Resumo em linguagem simples

- Partindo de um CSV completo depois da limpeza (removendo linhas com `situacao` inválida):
  - Gera-se uma base `X` (features) e `y` (evasão 0/1).
- Com `cv_splits = 5`, em cada rodada:
  - Aproximadamente **80%** dos alunos são usados para **treinar**.
  - Aproximadamente **20%** são usados para **validar**.
- O sistema repete esse processo 5 vezes, cada vez com um subconjunto diferente na validação.

No fim:

- Todo aluno é visto como **validação** exatamente **uma vez**.  
- Todo aluno é visto como **treino** em `n_splits - 1` rodadas.

