# Sistema de Predição de Evasão Universitária

Este repositório reúne três componentes principais que trabalham em conjunto:

- **processador_csv/**: preparação e padronização dos dados de entrada.
- **modelo_pkl/**: treinamento e validação do modelo de predição.
- **predicao_ml/**: interface de inferência (Streamlit) usando o modelo treinado.

O objetivo é permitir que dados brutos de alunos sejam processados, utilizados para treinar um modelo robusto de predição de evasão e, por fim, consumidos em uma interface simples para uso por docentes e gestores.

---

## processador_csv/

Responsável por:

- Ler arquivos CSV brutos fornecidos pela instituição.
- Fazer limpeza básica e ajustes de formato (tipos, datas, normalização mínima).
- Fazer *drop* de colunas irrelevantes para o problema de evasão, reduzindo ruído.
- Gerar um CSV estruturado com as colunas esperadas pelo pipeline de modelagem em `modelo_pkl/`.

Aqui a preocupação é mais de engenharia de dados do que de modelagem. A ideia é deixar o dataset:

- **Consistente**: tipos de dados previsíveis e valores padronizados.
- **Enxuto**: remoção de campos que não ajudam ou atrapalham na modelagem.

Na prática, este estágio facilita reproduzir o fluxo de treinamento e inferência em ambientes diferentes, garantindo que o modelo receba sempre o mesmo formato de entrada.

---

## modelo_pkl/

Aqui está a lógica de **treinamento**, avaliação e empacotamento do modelo em um artefato `.pkl`.

Principais responsabilidades:

- Mapear a coluna `situacao` (ex.: regular, desistente, trancado, afastado, etc.) para um alvo binário (evasão = 1, não evasão = 0).
- Construir o pipeline de pré-processamento (`src/preprocessing.py`), incluindo:
  - Imputação de valores faltantes para variáveis numéricas e categóricas.
  - Engenharia de atributos como:
    - `nota_enem_total`
    - `nota_vestibular_total`
    - `aprovacao_ratio`
  - Codificação categórica adaptativa com `category_encoders`:
    - One-Hot Encoding para baixa cardinalidade.
    - Target Encoding para alta cardinalidade (usando a teoria de que esse tipo de codificação captura melhor relações entre categoria e alvo quando há muitas categorias raras).
- Treinar o modelo principal (ex.: LightGBM ou Regressão Logística calibrada) usando:
  - `class_weight="balanced"` para lidar com desbalanceamento de classes.
  - Validação cruzada estratificada (`StratifiedKFold`) para avaliar ROC AUC e Brier Score.
- Opcionalmente, otimizar hiperparâmetros com **Optuna**:
  - Escolhemos Optuna porque oferece uma busca eficiente em espaços de hiperparâmetros contínuos e categóricos, com boa experiência em problemas supervisionados tabulares.

O resultado é um **bundle** `.pkl` salvo em `modelo_pkl/artifacts/`, com estrutura semelhante a:

- `preprocessor`: pipeline de pré-processamento.
- `model`: pipeline completo (preprocessamento + modelo).
- `metadata`: métricas, hiperparâmetros, mapeamento de classes, número de amostras, etc.
- `version`: versão do modelo.

Esse artefato é justamente o que será carregado pelo módulo de predição em `predicao_ml/`.

---

## predicao_ml/

Esta pasta contém a lógica de **inferência** e a **interface Streamlit** que o usuário final acessa.

Componentes principais:

- `backend_logic.py`:
  - Carrega o artefato `.pkl` via `joblib.load`.
  - Garante que o pipeline de pré-processamento (definido em `src.preprocessing`) seja importável, para que o bundle seja desserializado corretamente.
  - Lê o CSV enviado pelo usuário (caminho ou *file-like*, como o `UploadedFile` do Streamlit).
  - Remove a coluna `situacao` do `DataFrame` de entrada antes da predição para evitar vazamento de dados.
  - Executa `predict_proba` no pipeline completo e calcula:
    - `prob_evasao` (probabilidade estimada de evasão).
    - `class_pred` (0/1 com *threshold* 0.5).
    - `risk_level` (Baixo, Médio, Alto) com cortes em 0.33 e 0.66.
  - Retorna o `DataFrame` original com essas colunas adicionadas.

- `frontendend.py`:
  - Interface Streamlit para que o professor/usuário:
    - Selecione o modelo `.pkl` disponível em `../modelo_pkl/artifacts/`.
    - Faça *upload* de um CSV de alunos.
    - Visualize as predições com destaque visual para alunos em risco Médio/Alto.
    - Baixe o CSV enriquecido com as colunas de probabilidade e risco.
  - Optamos por Streamlit por ser uma solução leve para criar protótipos de interfaces de dados em Python, permitindo validar o modelo com usuários finais sem esforço de front-end tradicional.

- `check_env.py`:
  - Verifica se o diretório de artefatos (`../modelo_pkl/artifacts/`) existe.
  - Lista os arquivos `.pkl` disponíveis.
  - Emite avisos claros se não houver modelos encontrados.

Em conjunto, `predicao_ml/` é o “ponto de contato” entre o modelo treinado e o usuário final, mantendo a lógica de ML no backend e expondo apenas os controles necessários para uso em sala ou gestão acadêmica.

---

## Tecnologias e escolhas de projeto

Algumas decisões adotadas neste repositório:

- **Python + scikit-learn + LightGBM**  
  Usamos scikit-learn como base porque oferece uma API padronizada para pipelines, validação cruzada e métricas. Para modelos mais expressivos, LightGBM foi escolhido por sua boa performance em dados tabulares e suporte nativo a probabilidades bem calibradas quando usado com técnicas de calibração.

- **category_encoders para variáveis categóricas**  
  Em vez de usar apenas One-Hot Encoding, adotamos `category_encoders` para aplicar Target Encoding em colunas de alta cardinalidade. Essa escolha combina boa prática teórica (melhor aproveitamento de informação com muitas categorias raras) com experiência prática em datasets educacionais, onde campos como curso, município ou escola geram muitas categorias.

- **Optuna para otimização de hiperparâmetros**  
  Optuna foi escolhido por ser flexível e eficiente, suportando busca bayesiana e integração limpa com scikit-learn. Na prática, isso permitiu explorar combinações de hiperparâmetros do LightGBM sem escrever lógica de busca manual.

- **Streamlit para interface de predição**  
  A interface foi pensada para ser usada por professores e analistas sem conhecimento técnico profundo em Python. Streamlit reduz a barreira de entrada, permitindo carregar modelos, anexar CSVs e visualizar resultados com poucas linhas de código e uma curva de aprendizado acessível.

