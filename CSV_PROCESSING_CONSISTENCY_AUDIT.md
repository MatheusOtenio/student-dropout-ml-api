# Auditoria de Consistência de Processamento de CSV

**Data da Auditoria:** 14/02/2026
**Status Global:** ⚠️ **ATENÇÃO** (Inconsistências detectadas na definição de colunas obrigatórias)

## 1. Estado Real do Pipeline Atual

O pipeline de treinamento e inferência foi auditado com base no código fonte em `modelo_pkl/src` e `backend/src`. O fluxo de processamento é o seguinte:

1.  **Leitura e Validação Inicial**:
    *   O endpoint de treino (`modelo_pkl/src/main.py`) valida a presença de colunas definidas em `REQUIRED_COLUMNS`.
    *   **Divergência Crítica**: A lista `REQUIRED_COLUMNS` no `main.py` exige colunas que **não são mais utilizadas** pelo modelo (`disciplinas_reprovadas_frequencia`, `cor_raca`, `uf_residencia`).

2.  **Pré-processamento (`preprocessing.py`)**:
    *   **Drop Técnico**: Remove colunas identificadoras (`codigo_aluno`) e vazamentos de target (`situacao`, `target_evasao`).
    *   **Garantia de Colunas (`EnsureColumnsTransformer`)**: Garante que todas as features numéricas e categóricas existam, preenchendo com `NaN` se ausentes.
    *   **Limpeza (`DataCleaningTransformer`)**: Normaliza strings (minúsculas, sem acentos) e trata placeholders (`na`, `n/a`, `-`).
    *   **Engenharia de Features (`FeatureEngineeringTransformer`)**:
        *   `nota_enem_total`: Soma das notas do ENEM.
        *   `nota_vestibular_total`: Soma das notas do Vestibular.
        *   `aprovacao_ratio`: `disciplinas_aprovadas / (disciplinas_aprovadas + disciplinas_reprovadas_nota)`.
        *   **Mudança Importante**: O cálculo de `aprovacao_ratio` **ignora** explicitamente a coluna `disciplinas_reprovadas_frequencia`, mesmo se ela estiver presente no CSV.

3.  **Transformação e Encoding**:
    *   **Numéricos**: Imputação pela mediana -> StandardScaler.
    *   **Categóricos**: Imputação constante ("missing") -> AdaptiveCategoricalEncoder (OneHot para baixa cardinalidade, TargetEncoder para alta).

## 2. Contrato Oficial do CSV de Entrada

Para garantir compatibilidade com a versão atual (`model.pkl`), o CSV de entrada (tanto para treino quanto para inferência) deve conter as seguintes colunas funcionais.

### 2.1 Colunas Obrigatórias (Features Ativas)

Estas colunas são efetivamente utilizadas pelo modelo para gerar predições.

| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `sexo` | Categórico | Sexo do aluno. |
| `municipio_residencia` | Categórico | Cidade de residência. |
| `curso` | Categórico | Nome do curso. |
| `campus` | Categórico | Campus da instituição. |
| `turno` | Categórico | Turno (Matutino, Noturno, etc). |
| `modalidade_ingresso` | Categórico | Forma de entrada (ENEM, Vestibular, etc). |
| `tipo_cota` | Categórico | Cota utilizada (se houver). |
| `coeficiente_rendimento` | Numérico | CR acumulado. |
| `disciplinas_aprovadas` | Numérico | Qtd. disciplinas aprovadas. |
| `disciplinas_reprovadas_nota` | Numérico | Qtd. reprovações por nota. |
| `periodo` | Numérico | Período atual. |
| `ano_ingresso` | Numérico | Ano de entrada. |
| `semestre_ingresso` | Numérico | Semestre de entrada (1 ou 2). |
| `idade` | Numérico | Idade do aluno. |
| `calouro` | Numérico (0/1) | Indicador se é calouro. |
| `nota_enem_*` | Numérico | 5 colunas: humanas, linguagem, matematica, natureza, redacao. |
| `nota_vestibular_*` | Numérico | 9 colunas de matérias do vestibular. |

### 2.2 Colunas Obsoletas / Ignoradas

As seguintes colunas, embora possam aparecer em documentações antigas ou validadores legados (`main.py`), **NÃO** influenciam o resultado do modelo atual:

*   `disciplinas_reprovadas_frequencia` (Removida da engenharia de features)
*   `cor_raca` (Não listada em `CATEGORICAL_FEATURES`)
*   `uf_residencia` (Não listada em `CATEGORICAL_FEATURES`)

## 3. Pontos de Risco e Inconsistência

### 🔴 Crítico: Validação de API Desatualizada
O arquivo `modelo_pkl/src/main.py` define `REQUIRED_COLUMNS` incluindo campos obsoletos.
*   **Risco**: Requisições de treino válidas podem ser rejeitadas se não enviarem colunas inúteis (ex: `disciplinas_reprovadas_frequencia`).
*   **Ação Recomendada**: Atualizar `modelo_pkl/src/main.py` para remover colunas obsoletas da validação.

### 🟡 Atenção: Duplicação de Código de Pré-processamento
A lógica de pré-processamento existe em duplicidade em:
1.  `modelo_pkl/src/preprocessing.py` (Usado no Treino)
2.  `backend/src/preprocessing/preprocessing.py` (Usado, potencialmente, na API de inferência)
*   **Risco**: Se houver divergência na lógica (ex: como `aprovacao_ratio` é calculado), o modelo em produção (Backend) se comportará de forma diferente do validado no treino. Atualmente, a lógica parece consistente, mas a manutenção duplicada é propensa a erro.

### 🟡 Atenção: Dependência de `TargetEncoder`
O pipeline usa `TargetEncoder` para colunas de alta cardinalidade.
*   **Risco**: O `TargetEncoder` depende estatisticamente do target (`y`) durante o `fit`. O backend deve garantir que carrega o artefato **já treinado** (`.pkl`) e apenas executa `transform`. Jamais deve tentar refazer o `fit` no backend sem o target real.

## 4. Recomendações Imediatas

1.  **Backend**: Garantir que o payload de entrada da API de predição não exija `disciplinas_reprovadas_frequencia`, `cor_raca` e `uf_residencia` como obrigatórios.
2.  **Treino**: Limpar `REQUIRED_COLUMNS` em `modelo_pkl/src/main.py`.
3.  **Documentação**: Adotar este documento como fonte da verdade sobre o esquema de dados.
