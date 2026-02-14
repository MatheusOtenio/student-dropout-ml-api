# Guia de Desenvolvimento e Melhoria de UI/UX - Client de Teste (Streamlit)

Este documento serve como guia técnico para desenvolvedores ou IAs que irão trabalhar na evolução do arquivo `backend/front_test.py`. O objetivo é transformar este cliente de teste em um dashboard analítico robusto, mantendo a integridade da comunicação com a API e a estabilidade da aplicação.

## 1. Contexto Arquitetural
O `front_test.py` é uma aplicação **Streamlit** que atua como cliente frontend para a API de Evasão Escolar (Backend).
*   **Frontend**: Streamlit (Python).
*   **Comunicação**: HTTP REST via biblioteca `requests`.
*   **Backend**: FastAPI (endpoints de predição, mapeamento e observabilidade).

---

## 2. Contrato de API (Integração Obrigatória)
Qualquer alteração visual não pode quebrar a comunicação com os endpoints abaixo. Mantenha os nomes dos parâmetros e formatos de payload.

### A. Observabilidade (Feature Importance)
*   **Endpoint**: `GET /model/importance`
*   **Parâmetros**: `model_id` (str).
*   **Resposta Esperada**: JSON contendo `mapped` (lista de objetos `{'feature': nome, 'importance': valor}`) ou `raw_values`.
*   **Dica de UI**: Ao criar gráficos, verifique sempre se a chave `mapped` existe. Se existir, use os nomes das features para o eixo Y.

### B. Sugestão de Mapeamento (Upload Inicial)
*   **Endpoint**: `POST /mapping-suggestions`
*   **Payload**: Multipart/Form-Data com arquivo `file`.
*   **Resposta**: JSON com chaves `mapping` (dict) e `columns_csv` (list).
*   **Regra Crítica**: O resultado deste passo (`mapping`) deve ser armazenado no `st.session_state` pois é obrigatório para os passos seguintes.

### C. Pré-visualização e Predição
*   **Endpoints**: `/preview-mapped-csv` e `/predict`
*   **Payload**:
    *   `file`: O mesmo arquivo CSV (bytes).
    *   `data`: Dicionário contendo `mapping` como uma **string JSON** (`json.dumps(mapping_dict)`). **Não envie o dicionário Python direto no data, a API espera string.**
*   **Resposta Predição**: JSON com `rows` (dados processados) e `predictions` (probabilidades).

---

## 3. Diretrizes para Melhoria de UI/UX

### Estrutura e Layout
*   **Modularização**: Quebre o código monolítico do `main()` em funções menores (ex: `render_sidebar()`, `render_observability()`, `render_prediction_results()`).
*   **Navegação**: Use `st.tabs` ou `st.sidebar.radio` para separar contextos (Configuração vs. Análise vs. Predição) em vez de botões sequenciais que poluem a tela.
*   **Expansores**: Use `st.expander` para esconder configurações técnicas (URL da API, Model ID) e deixar o foco nos dados.

### Gráficos e Visualização
*   **Bibliotecas**: O Streamlit suporta nativamente **Altair**, **Plotly** e **Matplotlib**. Dê preferência ao **Plotly** para gráficos interativos (hover, zoom).
*   **Ideias de Gráficos**:
    *   **Distribuição de Risco**: Histograma das probabilidades de evasão retornadas em `/predict`.
    *   **Matriz de Confusão**: Se o CSV de entrada tiver a coluna real `situacao`, calcule e plote métricas de performance em tempo real.
    *   **Scatter Plot**: Relacionar `prob_evasao` com variáveis numéricas (ex: `renda`, `notas`).

### Filtros Avançados (Pandas)
Ao exibir a tabela de resultados (`st.dataframe`):
1.  Converta a resposta da API para `pd.DataFrame`.
2.  Use `st.data_editor` ou widgets de filtro (`st.slider`, `st.multiselect`) **antes** de renderizar o dataframe final.
3.  **Exemplo**: Filtro de "Alunos em Risco Crítico" (Probabilidade > 0.8).

---

## 4. Gerenciamento de Estado (Session State)
O Streamlit reexecuta todo o script a cada interação. Para manter a persistência:
*   **Sempre** inicialize variáveis no começo do script:
    ```python
    if "data_frames" not in st.session_state:
        st.session_state["data_frames"] = {}
    ```
*   **Não perca os dados**: Ao receber a resposta da API (`response.json()`), salve imediatamente no `st.session_state` antes de tentar plotar. Isso permite que o usuário mude um filtro de gráfico sem ter que chamar a API de novo.

---

## 5. Checklist de Qualidade para a IA
Antes de entregar o código refatorado:
- [ ] O upload de arquivo persiste após interações na tela? (Use `st.session_state` ou o comportamento padrão do `file_uploader` moderno).
- [ ] O mapeamento de colunas (De/Para) continua funcionando e sendo enviado como JSON string para o backend?
- [ ] O gráfico de Feature Importance lida com casos onde não há nomes de features (apenas índices numéricos)?
- [ ] O código trata erros de conexão (API offline) com mensagens amigáveis (`st.error`) em vez de estourar stack trace na tela?
- [ ] A interface é responsiva? (Uso de `st.columns`).

---

## 6. Exemplo de Refatoração Segura
Se for mover a lógica de predição para uma função:

```python
def run_prediction(file_bytes, mapping_dict, model_id):
    # Lógica de requests.post...
    # Retorna o DataFrame
    return df_result

# No main:
if st.button("Predizer"):
    df = run_prediction(...)
    st.session_state['last_result'] = df

if 'last_result' in st.session_state:
    render_dashboard(st.session_state['last_result'])
```
