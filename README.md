 [![Backend](https://img.shields.io/badge/backend-README-blue)](backend/README.md)
 [![Modelo Pkl](https://img.shields.io/badge/modelo__pkl-README-green)](modelo_pkl/README.md)

# Sistema de Predição de Evasão Universitária

Este repositório reúne dois componentes principais que trabalham em conjunto:

- **backend/**: backend unificado que faz o papel do antigo `processador_csv` (ETL/mapeamento de colunas) e do `predicao_ml` (predição com modelo `.pkl`), pronto para rodar localmente ou em Docker.
- **predicao_ml/**: interface Streamlit independente/legado para inferência direta a partir de um artefato `.pkl` (útil para experimentos e validação rápida).

O objetivo é permitir que dados brutos de alunos sejam processados, utilizados em um modelo robusto de predição de evasão e, por fim, consumidos em uma interface simples para uso por docentes e gestores.

---

## backend/

O diretório `backend/` concentra hoje o fluxo completo que o usuário final precisa:

1. **Receber um CSV bruto** enviado pelo usuário.
2. **Sugerir um mapeamento de colunas** do CSV para o esquema interno.
3. **Aplicar o ETL** (limpeza, padronização, engenharia de atributos) para produzir as features esperadas pelo modelo.
4. **Carregar um modelo pré-treinado `.pkl`** a partir de `src/models/`.
5. **Executar a predição de evasão** e retornar as probabilidades/indicadores ao cliente (ex.: Streamlit ou outra UI).

A organização interna segue a mesma separação de responsabilidades dos projetos antigos, mas agora dentro de um único backend:

- `src/preprocessing/`
  - Contém as funções de pré-processamento compartilhadas, alinhadas ao pipeline usado no treinamento.

- `src/sercives/processador_csv/`
  - Lida com leitura de CSV, sugestão de mapeamento de colunas, validação de schema e ETL.
  - Implementa o comportamento do antigo projeto `processador_csv`, agora exposto via API.

- `src/sercives/predicao_ml/`
  - Contém a lógica de carregamento do modelo `.pkl` (via `joblib`) e execução de `predict_proba`.
  - Garante que o pipeline de pré-processamento (definido em `src.preprocessing`) é importável para desserializar corretamente o artefato.

- `src/models/`
  - Armazena os artefatos `.pkl` gerados pelo pipeline de treinamento (antes no projeto `modelo_pkl`).
  - Cada arquivo representa um bundle de modelo pronto para produção.

### Fluxo de ponta a ponta

De forma simplificada, o backend executa o seguinte fluxo lógico:

1. O usuário anexa um CSV com informações de alunos.
2. O backend sugere um mapeamento entre colunas do CSV e campos internos (schemas) e permite pré-visualizar o resultado do ETL.
3. Após confirmar o mapeamento, o backend aplica o ETL completo, reproduzindo o mesmo pré-processamento usado no treino.
4. O backend carrega o modelo `.pkl` selecionado (por `model_id`) a partir de `src/models/`.
5. O pipeline completo (pré-processamento + modelo) calcula:
   - `prob_evasao`: probabilidade estimada de evasão.
   - `class_pred`: rótulo binário (0/1) com threshold típico 0.5.
   - `risk_level`: faixas de risco (Baixo, Médio, Alto) definidas por cortes em 0.33 e 0.66.
6. O resultado é devolvido ao cliente (por exemplo, o cliente Streamlit em `backend/front_test.py`) em formato tabular ou JSON.

O treinamento em si não acontece mais dentro de `backend/`: os modelos são treinados em um pipeline separado (equivalente ao antigo `modelo_pkl`), e apenas os artefatos `.pkl` finais são copiados para `backend/src/models/` para uso em produção.

---

## Pipeline de treinamento (conceito)

Embora o treinamento não esteja mais exposto como um serviço separado neste repositório, o **design do pipeline de modelagem** permanece o mesmo do antigo `modelo_pkl` e é importante para entender o que está por trás dos artefatos `.pkl` usados pelo backend.

Principais responsabilidades do pipeline de treino:

- Mapear a coluna `situacao` (ex.: regular, desistente, trancado, afastado, etc.) para um alvo binário (evasão = 1, não evasão = 0).
- Construir o pipeline de pré-processamento (`preprocessing.py`), incluindo:
  - Imputação de valores faltantes para variáveis numéricas e categóricas.
  - Engenharia de atributos como:
    - `nota_enem_total`
    - `nota_vestibular_total`
    - `aprovacao_ratio`
  - Codificação categórica adaptativa com `category_encoders`:
    - One-Hot Encoding para baixa cardinalidade.
    - Target Encoding para alta cardinalidade (capturando melhor relações entre categoria e alvo quando há muitas categorias raras).
- Treinar o modelo principal (ex.: LightGBM ou Regressão Logística calibrada) usando:
  - `class_weight="balanced"` para lidar com desbalanceamento de classes.
  - Validação cruzada estratificada (`StratifiedKFold`) para avaliar ROC AUC e Brier Score.
- Opcionalmente, otimizar hiperparâmetros com **Optuna**:
  - Optuna oferece uma busca eficiente em espaços de hiperparâmetros contínuos e categóricos, com boa experiência em problemas supervisionados tabulares.

O resultado desse pipeline é um **bundle** `.pkl` que contém, tipicamente:

- `preprocessor`: pipeline de pré-processamento.
- `model`: pipeline completo (preprocessamento + modelo).
- `metadata`: métricas, hiperparâmetros, mapeamento de classes, número de amostras, etc.
- `version`: versão do modelo.

São esses bundles que, uma vez gerados, são copiados para `backend/src/models/` e servidos pela API unificada.

---

## predicao_ml/

A pasta `predicao_ml/` mantém uma **interface Streamlit independente** para inferência direta a partir de um artefato `.pkl`. Ela é útil para:

- Experimentar rapidamente com novos modelos.
- Validar artefatos `.pkl` fora do backend unificado.
- Demonstrar o modelo em um contexto puramente interativo.

Componentes principais:

- `backend_logic.py`:
  - Carrega o artefato `.pkl` via `joblib.load`.
  - Garante que o pipeline de pré-processamento (definido em `src.preprocessing`) seja importável, para que o bundle seja desserializado corretamente.
  - Lê o CSV enviado pelo usuário (caminho ou *file-like*, como o `UploadedFile` do Streamlit).
  - Remove a coluna `situacao` do `DataFrame` de entrada antes da predição para evitar vazamento de dados.
  - Executa `predict_proba` no pipeline completo e calcula `prob_evasao`, `class_pred` e `risk_level`.

- `frontendend.py`:
  - Interface Streamlit para que o usuário:
    - Selecione um modelo `.pkl` disponível (por exemplo, em `../modelo_pkl/artifacts/` ou outro diretório de artefatos).
    - Faça *upload* de um CSV de alunos.
    - Visualize as predições com destaque visual para alunos em risco Médio/Alto.
    - Baixe o CSV enriquecido com as colunas de probabilidade e risco.

- `check_env.py`:
  - Verifica a existência do diretório de artefatos.
  - Lista os arquivos `.pkl` disponíveis.
  - Emite avisos claros se não houver modelos encontrados.

Na prática, `predicao_ml/` é uma ferramenta de apoio; o caminho principal para uso em produção passa pelo `backend/` unificado.

---

## Tecnologias e escolhas de projeto

Algumas decisões adotadas neste repositório:

- **Python + scikit-learn + LightGBM**  
  Usamos scikit-learn como base porque oferece uma API padronizada para pipelines, validação cruzada e métricas. Para modelos mais expressivos, LightGBM foi escolhido por sua boa performance em dados tabulares e suporte nativo a probabilidades bem calibradas quando usado com técnicas de calibração.

- **category_encoders para variáveis categóricas**  
  Em vez de usar apenas One-Hot Encoding, adotamos `category_encoders` para aplicar Target Encoding em colunas de alta cardinalidade. Essa escolha combina boa prática teórica (melhor aproveitamento de informação com muitas categorias raras) com experiência prática em datasets educacionais, onde campos como curso, município ou escola geram muitas categorias.

- **Optuna para otimização de hiperparâmetros**  
  Optuna foi escolhido por ser flexível e eficiente, suportando busca bayesiana e integração limpa com scikit-learn. Na prática, isso permitiu explorar combinações de hiperparâmetros do LightGBM sem escrever lógica de busca manual.

- **FastAPI + Uvicorn no backend**  
  O backend unificado expõe uma API HTTP moderna, com validação de dados automática e documentação interativa.

- **Streamlit para interfaces de predição e cliente de teste**  
  A interface foi pensada para ser usada por professores e analistas sem conhecimento técnico profundo em Python. Streamlit reduz a barreira de entrada, permitindo carregar modelos, anexar CSVs e visualizar resultados com poucas linhas de código e uma curva de aprendizado acessível, tanto em `predicao_ml/` quanto no cliente de teste em `backend/front_test.py`.
