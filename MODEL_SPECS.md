# Especificações do Modelo (Model Specs)

**Versão do Documento:** 2.0 (Pós-Auditoria)
**Data:** 14/02/2026
**Artefato Referência:** `model.pkl` (Gerado pelo pipeline em `modelo_pkl/src`)

## 1. Identidade do Modelo

*   **Tipo**: Pipeline Scikit-Learn Composto.
    *   **Pré-processador**: `ColumnTransformer` com imputação e encoding.
    *   **Estimador Principal**: `LGBMClassifier` (LightGBM) ou `LogisticRegression` (dependendo da configuração de treino).
    *   **Calibração**: `CalibratedClassifierCV` (Isotonic Regression, CV=5) envolvendo o estimador base.
*   **Objetivo**: Classificação Binária (Risco de Evasão).
    *   `0`: Permanência/Conclusão (Sucesso).
    *   `1`: Evasão (Desistência/Trancamento).
*   **Target**: `target_evasao` (Prospectivo).
    *   Prevê se o aluno entrará em situação de evasão nos próximos **2 semestres** letivos.

## 2. Estrutura do Pipeline de Dados

### 2.1 Features de Entrada (Raw)

O modelo espera as seguintes 33 features originais no DataFrame de entrada:

**Numéricas (23 features):**
*   `coeficiente_rendimento`
*   `disciplinas_aprovadas`
*   `disciplinas_reprovadas_nota`
*   `periodo`
*   `ano_ingresso`, `semestre_ingresso`
*   `idade`
*   `calouro` (0 ou 1)
*   `nota_enem_humanas`, `nota_enem_linguagem`, `nota_enem_matematica`, `nota_enem_natureza`, `nota_enem_redacao`
*   `nota_vestibular_biologia`, `nota_vestibular_filosofia_sociologia`, `nota_vestibular_fisica`, `nota_vestibular_geografia`, `nota_vestibular_historia`, `nota_vestibular_literatura_brasileira`, `nota_vestibular_lingua_estrangeira`, `nota_vestibular_lingua_portuguesa`, `nota_vestibular_matematica`, `nota_vestibular_quimica`

**Categóricas (7 features):**
*   `sexo`
*   `municipio_residencia`
*   `curso`
*   `campus`
*   `turno`
*   `modalidade_ingresso`
*   `tipo_cota`

### 2.2 Engenharia de Features (Transformações)

Internamente, o pipeline gera 3 features sintéticas antes do treinamento:

1.  **`nota_enem_total`**: Soma das 5 notas do ENEM. (Se todas nulas, retorna 0 ou NaN conforme lógica).
2.  **`nota_vestibular_total`**: Soma das 9 notas do Vestibular.
3.  **`aprovacao_ratio`**: Razão de aprovação.
    *   Fórmula: `disciplinas_aprovadas / (disciplinas_aprovadas + disciplinas_reprovadas_nota)`
    *   **Nota**: Ignora `disciplinas_reprovadas_frequencia`.

### 2.3 Estratégia de Encoding e Imputação

*   **Numéricos**:
    *   Missing: Preenchido com a **Mediana** do treino.
    *   Scaling: `StandardScaler` (Média 0, Desvio Padrão 1).
*   **Categóricos**:
    *   Missing: Preenchido com constante `"missing"`.
    *   Baixa Cardinalidade (<= 20 categorias): `OneHotEncoder`.
    *   Alta Cardinalidade (> 20 categorias): `TargetEncoder` (Suavizado pela média global do target).

## 3. Métricas e Validação

O modelo é validado utilizando uma estratégia de **Split Temporal** (Time-Based Split) para simular o cenário real de produção (treinar no passado, prever no futuro).

*   **Métrica Principal**: `ROC AUC` (Area Under the Receiver Operating Characteristic Curve). Mede a capacidade de ordenação de risco.
*   **Métrica de Calibração**: `Brier Score`. Mede a precisão probabilística (o quão próximo a probabilidade predita está da realidade).
*   **Split Padrão**:
    *   Treino: Anos anteriores (ex: 2014-2018).
    *   Validação/Teste: Anos mais recentes (ex: 2019).

## 4. Integração com Backend

### Observações Técnicas para Consumo (`model.pkl`)

1.  **Carregamento**: Utilizar `joblib.load()`. O objeto retornado é um dicionário contendo:
    *   `"model"`: O pipeline Scikit-Learn treinado.
    *   `"metadata"`: Informações de versão, métricas e importância de features.
2.  **Inferência**:
    *   Chamar `pipeline.predict_proba(df_entrada)`.
    *   **Entrada**: DataFrame Pandas contendo as colunas listadas na seção 2.1.
    *   **Saída**: Array numpy `(N, 2)`. A coluna de índice `1` contém a probabilidade de evasão.
3.  **Prevenção de Data Leakage**:
    *   O Backend **DEVE** remover a coluna `situacao` (se presente) antes de passar os dados para o modelo. O modelo não deve ter acesso ao status atual do aluno, pois seu objetivo é prever o status futuro.
4.  **Tratamento de Erros**:
    *   Se uma categoria nova aparecer em produção (não vista no treino):
        *   OneHot: Será ignorada (zerada).
        *   TargetEncoder: Receberá a média global do target (suavização).
