# Relatório Técnico de Calibração (Pós-Correção)

## 1. Status Atual da Implementação

### 1.1 Arquitetura Corrigida (Leakage Resolvido)
A arquitetura de treinamento foi refatorada para eliminar o vazamento de dados (data leakage) identificado anteriormente.

*   **Nova Estrutura:**
    ```python
    CalibratedClassifierCV(
        estimator=Pipeline([
            ("preprocess", preprocessor),  # TargetEncoder ajustado por fold
            ("model", base_model)
        ]),
        cv=5,
        method="isotonic"
    )
    ```
*   **Mecanismo de Correção:** Ao mover o `Pipeline` de pré-processamento para dentro do `CalibratedClassifierCV`, garantimos que o `TargetEncoder` (utilizado para `municipio_residencia`) seja ajustado **exclusivamente** com os dados de treino de cada fold de calibração.
*   **Resultado:** O modelo calibrador não tem mais acesso às médias de target das amostras que está tentando calibrar. As probabilidades geradas refletem a capacidade real de generalização do modelo.

### 1.2 Ajustes de Regularização
O encoder de variáveis categóricas (`AdaptiveCategoricalEncoder`) recebeu parâmetros de suavização para evitar overfitting em categorias com poucas amostras:

*   **Smoothing (Suavização):** Aumentado para `10.0` (anteriormente padrão ~1.0). Isso puxa a média de municípios pequenos em direção à média global de evasão.
*   **Min Samples Leaf:** Definido em `20`. Categorias com menos de 20 observações não terão encoding baseado puramente em sua própria média, reduzindo ruído estatístico.

## 2. Impacto Esperado nas Probabilidades

Com as correções aplicadas, espera-se uma mudança significativa no comportamento das predições:

1.  **Fim das Probabilidades Artificiais (0.0 / 1.0):**
    *   Alunos de municípios "perfeitos" (0% evasão histórica) não receberão mais automaticamente probabilidade 0% se tiverem outros fatores de risco (ex: Reprovações, Notas baixas).
    *   A feature `municipio_residencia` deixará de ser um preditor determinante (leakage) para se tornar um preditor contextual.

2.  **Redução da Importância de `municipio_residencia`:**
    *   O coeficiente (Regressão Logística) ou ganho (LGBM) dessa feature deve cair drasticamente, saindo do topo absoluto para um patamar comparável às features acadêmicas.

3.  **Métricas de Validação Mais Realistas:**
    *   É esperado que o **ROC-AUC diminua** ligeiramente na validação interna, pois o modelo não está mais "colando". Isso é positivo, pois a métrica agora é honesta.
    *   O **Brier Score** deve refletir a calibração real.

## 3. Matriz de Riscos Atualizada

| Risco | Nível Anterior | Nível Atual | Status |
| :--- | :---: | :---: | :--- |
| **Leakage no Target Encoding** | **Crítico** | **Resolvido** | Arquitetura corrigida (`CalibratedClassifierCV` encapsulando Pipeline). |
| **Overfitting em Municípios Pequenos** | **Alto** | **Baixo** | Mitigado por `smoothing=10.0` e `min_samples_leaf=20`. |
| **Viés em Feature Importance (LGBM)** | Baixo | Médio | O método 'split' ainda é usado. Sem o leakage, outras features de alta cardinalidade podem ser superestimadas se não alterado para 'gain'. |
| **Multicolinearidade (LogReg)** | Médio | Médio | `aprovacao_ratio` ainda coexiste com features brutas. Monitorar estabilidade dos coeficientes. |

## 4. Próximos Passos Recomendados

### 4.1 Validação de Execução
Executar um treinamento completo (`train_model`) para verificar:
1.  Se o pipeline roda sem erros com a nova estrutura aninhada.
2.  Inspecionar os logs para confirmar as `Top 5 Features`. Espera-se que `municipio_residencia` não seja mais a feature dominante absoluta (ou se for, com importância menor).

### 4.2 Monitoramento de Produção
*   Acompanhar a distribuição de probabilidades dos novos modelos gerados. A curva deve ser menos "aguda" nas pontas (0 e 1) e apresentar mais densidade nas zonas de incerteza (0.3 - 0.7), o que é saudável para problemas sociais complexos como evasão.

### 4.3 Melhoria Contínua (Backlog)
*   **Feature Importance:** Alterar LGBM para `importance_type='gain'`.
*   **Seleção de Features:** Avaliar remoção de features redundantes para Regressão Logística se houver instabilidade.
