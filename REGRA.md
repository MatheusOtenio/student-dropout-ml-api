# Regras de Negócio e Plano de Implementação: Modelo de Risco de Evasão Prospectivo

Este documento define a nova arquitetura do pipeline de machine learning, migrando de uma classificação estática ("Formado vs Desistente") para um modelo prospectivo de risco de evasão ("Quem vai desistir no futuro?").

---

## 1. Escopo e Definição do Problema

**Objetivo:** Identificar alunos *regulares* com alto risco de evasão nos próximos $K$ semestres (Horizonte).
**Target ($y$):** Binário. $1$ se o aluno evade em $t + \text{horizon}$, $0$ caso contrário.
**Unidade de Análise:** Aluno-Semestre (Snapshot).

---

## 2. Mudanças de Arquitetura e Código

As mudanças devem ser realizadas na ordem abaixo para garantir integridade.

### Fase 1: ETL Permissivo (Refatoração do `csv_processor`)
O processador atual remove dados cruciais para a construção do histórico.
*   **Mudança:** Alterar `src/csv_processor/etl_service.py` para não filtrar linhas baseadas em `situacao`.
*   **Preservação:** Garantir que colunas identificadoras (`codigo_aluno`, `ano_referencia`, `periodo_referencia`) sejam preservadas no `df_padronizado` e não enviadas para `df_dropped`.
*   **Schema:** Atualizar `FEATURE_COLUMNS` em `src/csv_processor/schemas.py` para incluir identificadores temporais e de indivíduo como "metadados", não features de modelo ainda.

### Fase 2: Engenharia de Target (Novo Módulo)
Criação de lógica para olhar para o futuro.
*   **Novo Arquivo:** `src/target_engineering.py`
*   **Função Principal:** `generate_future_dropout_target(df, horizon=2)`
*   **Lógica:**
    1.  Ordenar `df` por `codigo_aluno`, `ano`, `semestre`.
    2.  Iterar (ou usar `groupby().shift()`) para verificar status futuro.
    3.  Gerar coluna `target_evasao_k_semestres`.
    4.  Gerar flag `valid_row` (excluir último semestre do aluno se não houver futuro observável suficiente ou se já for evento terminal).

### Fase 3: Preparação para Treino (Refatoração do `trainer`)
O treinador deve receber o dataset expandido e preparar para o modelo.
*   **Mudança:** Em `src/modelo_regressao/trainer.py`, remover `_map_situacao_to_binary` (obsoleto).
*   **Feature Selection:** Adicionar etapa `DropTechnicalColumns` no pipeline para remover `codigo_aluno`, `ano`, `semestre` *imediatamente antes* do fit, mas mantê-los para split e validação.
*   **Split:** Refatorar `_get_time_based_split` para considerar `ano` E `semestre`.

---

## 3. Plano de Implementação Passo a Passo

Siga esta ordem estrita. Cada passo deve ser validado antes de prosseguir.

### Passo 1: Preparação do Ambiente e Testes de Regressão
1.  **Branch:** `chore/setup-tests`
2.  **Ação:** Criar um teste de integração que roda o pipeline atual com um CSV dummy pequeno.
3.  **Objetivo:** Garantir que sabemos quando quebramos a lógica atual.
4.  **Comando:** `pytest tests/test_integration_current.py`

### Passo 2: ETL Permissivo
1.  **Branch:** `feat/etl-permissive`
2.  **Alterar `src/csv_processor/schemas.py`:**
    *   Adicionar `codigo_aluno`, `ano`, `semestre` (ou equivalentes) na `COLUNAS_WHITELIST`.
3.  **Alterar `src/csv_processor/etl_service.py`:**
    *   Remover o bloco de filtro: `df = df.loc[mask_validos]`.
    *   Comentar/remover lógica que descarta "Regulares".
4.  **Validação:**
    *   Rodar processador com CSV de teste.
    *   Verificar se output contém alunos "Regular".

### Passo 3: Engenharia de Target
1.  **Branch:** `feat/target-engineering`
2.  **Criar `src/target_engineering.py`:**
    ```python
    def generate_future_dropout_target(df, id_col='codigo_aluno', time_col='semestre_geral', target_horizon=2):
        # Implementar lógica de window function
        # Retorna df com coluna 'target' e 'is_train_ready'
        pass
    ```
3.  **Integrar no `etl_service.py` (Opcional) ou `trainer.py`:**
    *   Recomenda-se integrar no final do ETL para que o CSV salvo já tenha o target.
4.  **Teste Unitário:**
    *   Criar dataframe fake com 1 aluno: [Regular, Regular, Desistente].
    *   Verificar se target em t=1 é 1 (se horizon=2).

### Passo 4: Adaptação do Treinador
1.  **Branch:** `feat/trainer-prospective`
2.  **Alterar `src/modelo_regressao/trainer.py`:**
    *   Remover chamada antiga de mapeamento de target.
    *   Usar coluna `target` gerada no passo anterior.
    *   Implementar `DropTechnicalColumns` transformer no pipeline do scikit-learn.
3.  **Split Temporal:**
    *   Atualizar lógica de split para respeitar a ordem cronológica estrita (não misturar futuro no treino).

---

## 4. Critérios de Validação

Para aceitar o novo modelo, as seguintes métricas devem ser avaliadas no set de teste (out-of-time):

1.  **Curva de Calibração (Reliability Diagram):**
    *   Essencial para risco. O modelo diz 70% de risco? Então 70% desses alunos devem evadir.
2.  **Brier Score:**
    *   Deve ser menor que o Brier Score de um modelo dummy (frequência média de evasão).
3.  **AUC-ROC e PR-AUC:**
    *   Métricas secundárias para discriminação.
4.  **Lift no Decil Superior:**
    *   Quanto mais evasão capturamos nos top 10% de risco comparado à aleatoriedade?

**Instruções de Avaliação:**
Ao final do treino, gerar relatório PDF ou logs contendo:
*   Gráfico de calibração.
*   Distribuição de probabilidade prevista para as classes (0 e 1).

---

## 5. Práticas de Código e Qualidade

### Evitar Código Obsoleto e Duplicado
1.  **Refatoração Modular:** Se uma função de limpeza é usada no ETL e no Treino, mova para `src/common/utils.py`.
2.  **Remoção Segura:** Ao substituir `_map_situacao_to_binary`, remova a função antiga inteira. Não deixe código comentado.
3.  **Busca de Duplicatas:**
    *   Rodar: `grep -r "def limpar_texto" .` (exemplo) para achar funções repetidas.

### Linters e CI
*   Usar `black` e `isort` antes de cada commit.
*   Usar `flake8` para identificar variáveis não usadas.

---

## 6. Políticas de Versão e Deploy

1.  **Branch Naming:**
    *   `feat/nome-da-feature` (novas funcionalidades)
    *   `fix/nome-do-bug` (correções)
    *   `refactor/nome-da-melhoria` (código limpo sem mudança de comportamento)
2.  **PR Checklist:**
    *   [ ] Testes unitários novos passaram?
    *   [ ] Testes de regressão passaram?
    *   [ ] Sem código morto/comentado?
    *   [ ] Linter rodou sem erros?
3.  **Arquivo de Configuração Versionado:**
    *   Manter `model_config.yaml` (ou similar) com hiperparâmetros e definições de colunas. Não "hardcoded" no Python.
4.  **Rollback Mínimo:**
    *   Manter sempre o último artefato `.pkl` funcional em pasta de backup ou registro de modelos (MLflow ou simples versionamento de arquivo).
