import io
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ============================================================================
# CONSTANTES E CONFIGURAÇÕES
# ============================================================================

LOW_RISK_THRESHOLD = 0.33


# ============================================================================
# CONFIGURAÇÃO INICIAL E SESSION STATE
# ============================================================================


def initialize_session_state() -> None:
    """Inicializa todas as variáveis de session_state necessárias."""
    defaults = {
        "mapping": None,
        "mapping_df": None,
        "columns_csv": [],
        "last_prediction": None,
        "last_model_id": "",
        "last_uploaded_file": None,
        "ui_filters": {
            "threshold_risco": 0.4,
            "situacao": [],
            "curso": [],
            "risk_min": 0.0,
            "risk_max": 1.0,
            "ano_min": 2000,
            "ano_max": 2030,
        },
        "obs_data": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# FUNÇÕES DE COMUNICAÇÃO COM API
# ============================================================================


@st.cache_data(show_spinner=False)
def fetch_mapping_suggestions(
    api_url: str, file_bytes: bytes, file_name: str
) -> Optional[Dict[str, Any]]:
    """Busca sugestões de mapeamento da API."""
    try:
        files = {"file": (file_name, io.BytesIO(file_bytes), "text/csv")}
        response = requests.post(
            f"{api_url.rstrip('/')}/mapping-suggestions",
            files=files,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erro na API (status {response.status_code}): {response.text}")
            return None
    except requests.RequestException as exc:
        st.error(f"Erro de conexão com a API: {exc}")
        return None
    except ValueError as exc:
        st.error(f"Resposta da API não é um JSON válido: {exc}")
        return None


@st.cache_data(show_spinner=False)
def fetch_preview_mapped_csv(
    api_url: str, file_bytes: bytes, file_name: str, mapping: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Busca pré-visualização do CSV mapeado."""
    try:
        files = {"file": (file_name, io.BytesIO(file_bytes), "text/csv")}
        data = {"mapping": json.dumps(mapping)}
        response = requests.post(
            f"{api_url.rstrip('/')}/preview-mapped-csv",
            files=files,
            data=data,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erro na API (status {response.status_code}): {response.text}")
            return None
    except requests.RequestException as exc:
        st.error(f"Erro de conexão com a API: {exc}")
        return None
    except ValueError as exc:
        st.error(f"Resposta da API não é um JSON válido: {exc}")
        return None


@st.cache_data(show_spinner=False)
def fetch_prediction(
    api_url: str,
    file_bytes: bytes,
    file_name: str,
    mapping: Dict[str, Any],
    model_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Executa predição via API."""
    try:
        files = {"file": (file_name, io.BytesIO(file_bytes), "text/csv")}
        data = {"mapping": json.dumps(mapping)}
        if model_id and model_id.strip():
            data["model_id"] = model_id.strip()

        response = requests.post(
            f"{api_url.rstrip('/')}/predict",
            files=files,
            data=data,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erro na API (status {response.status_code}): {response.text}")
            return None
    except requests.RequestException as exc:
        st.error(f"Erro de conexão com a API: {exc}")
        return None
    except ValueError as exc:
        st.error(f"Resposta da API não é um JSON válido: {exc}")
        return None


@st.cache_data(show_spinner=False)
def fetch_feature_importance(api_url: str, model_id: str) -> Optional[Dict[str, Any]]:
    """Busca importância das features do modelo."""
    try:
        response = requests.get(
            f"{api_url.rstrip('/')}/model/importance",
            params={"model_id": model_id},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erro na API (status {response.status_code}): {response.text}")
            return None
    except requests.RequestException as exc:
        st.error(f"Erro de conexão com a API: {exc}")
        return None


# ============================================================================
# FUNÇÕES DE PROCESSAMENTO DE DADOS
# ============================================================================


@st.cache_data(show_spinner=False)
def process_prediction_response(payload: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Converte resposta da API em DataFrame com colunas calculadas."""
    rows = payload.get("rows")
    predictions = payload.get("predictions")

    if rows is not None and predictions is not None:
        df = pd.DataFrame(rows)
        df["prob_evasao"] = predictions
    elif predictions is not None:
        df = pd.DataFrame({"prob_evasao": predictions})
    else:
        return None

    # Adicionar colunas calculadas
    if "prob_evasao" in df.columns:
        df["risk_pct"] = (df["prob_evasao"] * 100).round(2)

    return df


def calculate_risk_label(prob: float, threshold: float) -> str:
    """Calcula label de risco baseado na probabilidade."""
    if prob < LOW_RISK_THRESHOLD:
        return "Baixo"
    elif prob < threshold:
        return "Médio"
    else:
        return "Alto"


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Aplica filtros ao DataFrame de forma combinada (AND)."""
    if df is None or df.empty:
        return df

    filtered = df.copy()

    # Filtro de situação
    if filters.get("situacao") and "situacao" in filtered.columns:
        filtered = filtered[filtered["situacao"].isin(filters["situacao"])]

    # Filtro de curso
    if filters.get("curso") and "curso" in filtered.columns:
        filtered = filtered[filtered["curso"].isin(filters["curso"])]

    # Filtro de faixa de risco
    if "prob_evasao" in filtered.columns:
        risk_min = filters.get("risk_min", 0.0)
        risk_max = filters.get("risk_max", 1.0)
        filtered = filtered[
            (filtered["prob_evasao"] >= risk_min)
            & (filtered["prob_evasao"] <= risk_max)
        ]

    # Filtro de ano de ingresso
    if "ano_ingresso" in filtered.columns:
        ano_min = filters.get("ano_min", 2000)
        ano_max = filters.get("ano_max", 2030)
        try:
            filtered["ano_ingresso"] = pd.to_numeric(
                filtered["ano_ingresso"], errors="coerce"
            )
            filtered = filtered[
                (filtered["ano_ingresso"] >= ano_min)
                & (filtered["ano_ingresso"] <= ano_max)
            ]
        except Exception:
            pass

    return filtered


# ============================================================================
# COMPONENTES DE UI - SIDEBAR
# ============================================================================


def render_sidebar(df: Optional[pd.DataFrame]) -> None:
    """Renderiza sidebar com configurações e filtros."""
    st.sidebar.title("⚙️ Configurações")

    # Configurações da API
    with st.sidebar.expander("🔧 API & Modelo", expanded=False):
        default_api_url = os.getenv("API_URL", "http://localhost:10000")
        
        api_url = st.text_input(
            "URL da API",
            value=default_api_url,
            help="Exemplo: http://localhost:10000",
            key="api_url_input",
        )
        st.session_state["api_url"] = api_url

        model_id = st.text_input(
            "Model ID",
            value="model_1",
            help="Identificador do modelo .pkl (Obrigatório)",
            key="model_id_input",
        )
        st.session_state["model_id"] = model_id

    st.sidebar.divider()
    st.sidebar.title("🔍 Filtros")

    # Threshold de risco
    threshold = st.sidebar.number_input(
        "Limite de Risco Alto",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["ui_filters"]["threshold_risco"],
        step=0.05,
        help="Probabilidades acima deste valor são consideradas 'Alto Risco'",
    )

    # Filtros dinâmicos baseados no DataFrame
    situacao_options = []
    curso_options = []
    ano_range = (2000, 2030)

    if df is not None and not df.empty:
        if "situacao" in df.columns:
            situacao_options = sorted(df["situacao"].dropna().unique().tolist())
        if "curso" in df.columns:
            curso_options = sorted(df["curso"].dropna().unique().tolist())
        if "ano_ingresso" in df.columns:
            try:
                anos = pd.to_numeric(df["ano_ingresso"], errors="coerce").dropna()
                if not anos.empty:
                    ano_range = (int(anos.min()), int(anos.max()))
            except Exception:
                pass

    situacao_selected = st.sidebar.multiselect(
        "Situação",
        options=situacao_options,
        default=[
            x
            for x in st.session_state["ui_filters"].get("situacao", [])
            if x in situacao_options
        ],
        help="Filtre por situação acadêmica",
    )

    curso_selected = st.sidebar.multiselect(
        "Curso",
        options=curso_options,
        default=[
            x
            for x in st.session_state["ui_filters"].get("curso", [])
            if x in curso_options
        ],
        help="Filtre por curso",
    )

    risk_range = st.sidebar.slider(
        "Faixa de Probabilidade de Evasão",
        min_value=0.0,
        max_value=1.0,
        value=(
            st.session_state["ui_filters"].get("risk_min", 0.0),
            st.session_state["ui_filters"].get("risk_max", 1.0),
        ),
        step=0.05,
        help="Filtre por faixa de risco",
    )

    ano_range_selected = st.sidebar.slider(
        "Ano de Ingresso",
        min_value=ano_range[0],
        max_value=ano_range[1],
        value=(
            st.session_state["ui_filters"].get("ano_min", ano_range[0]),
            st.session_state["ui_filters"].get("ano_max", ano_range[1]),
        ),
        help="Filtre por período de ingresso",
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("✅ Aplicar Filtros", use_container_width=True):
            st.session_state["ui_filters"] = {
                "threshold_risco": threshold,
                "situacao": situacao_selected,
                "curso": curso_selected,
                "risk_min": risk_range[0],
                "risk_max": risk_range[1],
                "ano_min": ano_range_selected[0],
                "ano_max": ano_range_selected[1],
            }
            st.rerun()

    with col2:
        if st.button("🗑️ Limpar", use_container_width=True):
            st.session_state["ui_filters"] = {
                "threshold_risco": 0.4,
                "situacao": [],
                "curso": [],
                "risk_min": 0.0,
                "risk_max": 1.0,
                "ano_min": ano_range[0],
                "ano_max": ano_range[1],
            }
            st.rerun()


# ============================================================================
# COMPONENTES DE UI - TABS
# ============================================================================


def render_tab_visao_geral(df: pd.DataFrame, threshold: float) -> None:
    """Renderiza aba de Visão Geral com KPIs e gráficos."""
    st.header("📊 Visão Geral")

    if df is None or df.empty:
        st.info("Nenhum dado disponível. Execute a predição primeiro.")
        return

    # Calcular labels de risco
    df_copy = df.copy()
    if "prob_evasao" in df_copy.columns:
        df_copy["risk_label"] = df_copy["prob_evasao"].apply(
            lambda x: calculate_risk_label(x, threshold)
        )

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Alunos", len(df_copy))

    with col2:
        if "risk_label" in df_copy.columns:
            alto_risco = (df_copy["risk_label"] == "Alto").sum()
            pct_alto = (alto_risco / len(df_copy) * 100) if len(df_copy) > 0 else 0
            st.metric("Alto Risco", f"{alto_risco}", f"{pct_alto:.1f}%")

    with col3:
        if "risk_label" in df_copy.columns:
            medio_risco = (df_copy["risk_label"] == "Médio").sum()
            pct_medio = (medio_risco / len(df_copy) * 100) if len(df_copy) > 0 else 0
            st.metric("Médio Risco", f"{medio_risco}", f"{pct_medio:.1f}%")

    with col4:
        if "situacao" in df_copy.columns:
            regulares = (df_copy["situacao"].str.lower() == "regular").sum()
            pct_reg = (regulares / len(df_copy) * 100) if len(df_copy) > 0 else 0
            st.metric("Regulares", f"{regulares}", f"{pct_reg:.1f}%")

    st.divider()

    # Gráfico: Distribuição de Risco
    if "prob_evasao" in df_copy.columns:
        st.subheader("Distribuição de Probabilidade de Evasão")
        fig = px.histogram(
            df_copy,
            x="prob_evasao",
            nbins=30,
            title="Histograma de Probabilidades",
            labels={"prob_evasao": "Probabilidade de Evasão"},
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Resumo textual
    st.subheader("📝 Resumo")
    if "prob_evasao" in df_copy.columns:
        media_prob = df_copy["prob_evasao"].mean()
        mediana_prob = df_copy["prob_evasao"].median()
        st.write(
            f"A probabilidade média de evasão é **{media_prob:.2%}** "
            f"e a mediana é **{mediana_prob:.2%}**."
        )

        if "risk_label" in df_copy.columns:
            baixo = (df_copy["risk_label"] == "Baixo").sum()
            medio = (df_copy["risk_label"] == "Médio").sum()
            alto = (df_copy["risk_label"] == "Alto").sum()
            st.write(
                f"- **{baixo}** alunos com risco baixo (< {LOW_RISK_THRESHOLD*100:.0f}%)\n"
                f"- **{medio}** alunos com risco médio ({LOW_RISK_THRESHOLD*100:.0f}% - {threshold*100:.0f}%)\n"
                f"- **{alto}** alunos com risco alto (≥ {threshold*100:.0f}%)"
            )


def render_tab_resultados(df: pd.DataFrame, threshold: float) -> None:
    """Renderiza aba de Resultados com tabela e ações rápidas."""
    st.header("📋 Resultados e Tabela")

    if df is None or df.empty:
        st.info("Nenhum dado disponível. Execute a predição primeiro.")
        return

    # Calcular colunas adicionais
    df_display = df.copy()
    if "prob_evasao" in df_display.columns:
        df_display["risk_label"] = df_display["prob_evasao"].apply(
            lambda x: calculate_risk_label(x, threshold)
        )
        if "risk_pct" not in df_display.columns:
            df_display["risk_pct"] = (df_display["prob_evasao"] * 100).round(2)

    # Reordenar colunas para priorizar identificação e risco
    all_cols = list(df_display.columns)
    priority_cols = ["nome_aluno", "codigo_aluno", "risk_label", "risk_pct", "prob_evasao", "situacao"]
    
    # Colunas que existem no dataframe
    existing_priority = [c for c in priority_cols if c in all_cols]
    other_cols = [c for c in all_cols if c not in existing_priority]
    
    df_display = df_display[existing_priority + other_cols]

    # Botões de ação rápida
    st.subheader("⚡ Ações Rápidas")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("👥 Somente Regulares"):
            if "situacao" in df_display.columns:
                df_display = df_display[
                    df_display["situacao"].str.lower() == "regular"
                ]
                st.session_state["quick_filter"] = "regulares"

    with col2:
        if st.button(f"⚠️ Regulares com Risco > {threshold*100:.0f}%"):
            if "situacao" in df_display.columns and "prob_evasao" in df_display.columns:
                df_display = df_display[
                    (df_display["situacao"].str.lower() == "regular")
                    & (df_display["prob_evasao"] > threshold)
                ]
                st.session_state["quick_filter"] = "regulares_alto_risco"

    with col3:
        if st.button("🔝 Top 10 Maior Risco"):
            if "prob_evasao" in df_display.columns:
                df_display = df_display.nlargest(10, "prob_evasao")
                st.session_state["quick_filter"] = "top10"

    with col4:
        csv = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Exportar CSV",
            data=csv,
            file_name="predicoes_evasao_filtradas.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()

    # Exibição da tabela completa (sem paginação)
    st.dataframe(
        df_display,
        use_container_width=True,
        height=800,
        column_config={
            "nome_aluno": "Nome do Aluno",
            "codigo_aluno": "Matrícula/Código",
            "risk_label": "Nível de Risco",
            "risk_pct": "Risco (%)",
            "prob_evasao": st.column_config.ProgressColumn(
                "Probabilidade de Evasão",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
        },
    )


def render_tab_observabilidade(api_url: str, model_id: str) -> None:
    """Renderiza aba de Observabilidade com Feature Importance."""
    st.header("🔍 Observabilidade do Modelo")

    st.write(
        "Visualize a importância das variáveis (features) utilizadas pelo modelo "
        "para realizar as predições."
    )

    top_n = st.slider(
        "Quantidade de features para exibir",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
    )

    if st.button("📊 Carregar Feature Importance", type="primary"):
        with st.spinner("Carregando importância das features..."):
            mid = model_id.strip() if model_id.strip() else ""
            if not mid:
                st.error("Informe o Model ID.")
                return

            data = fetch_feature_importance(api_url, mid)

            if data:
                st.session_state["obs_data"] = data
                st.success("Dados de importância carregados com sucesso!")
            else:
                st.error(
                    "Não foi possível carregar os dados. Verifique se o modelo possui feature importance."
                )

    if st.session_state.get("obs_data"):
        data = st.session_state["obs_data"]
        mid = model_id.strip() if model_id.strip() else ""

        st.divider()
        st.subheader(f"Importância Global das Variáveis ({mid})")

        if "mapped" in data and data["mapped"]:
            df_imp = pd.DataFrame(data["mapped"])
            df_imp = df_imp.head(top_n).sort_values("importance", ascending=True)

            fig = px.bar(
                df_imp,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Top {top_n} Features Mais Importantes",
                labels={"importance": "Importância", "feature": "Feature"},
                color="importance",
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=max(400, top_n * 20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Tabela detalhada
            with st.expander("📊 Ver tabela detalhada"):
                st.dataframe(
                    df_imp[["feature", "importance"]].reset_index(drop=True),
                    use_container_width=True,
                )

        elif "raw_values" in data:
            st.warning(
                "O modelo retornou valores brutos de importância (sem mapeamento de nomes)."
            )
            raw_values = data["raw_values"][:top_n]
            df_raw = pd.DataFrame(
                {
                    "feature_index": range(len(raw_values)),
                    "importance": raw_values,
                }
            )
            st.dataframe(df_raw, use_container_width=True)
        else:
            st.info("Estrutura de dados não reconhecida.")
            st.json(data)


# ============================================================================
# COMPONENTE PRINCIPAL - CONFIGURAÇÃO E PREDIÇÃO
# ============================================================================


def render_configuration_section(
    api_url: str, uploaded_file: Any
) -> tuple[bool, Optional[pd.DataFrame]]:
    """Renderiza seção de configuração (upload, mapeamento, predição)."""
    st.header("⚙️ Configuração e Predição")

    if uploaded_file is None:
        st.info("📤 Envie um arquivo CSV para começar.")
        return False, None

    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name or "dados.csv"

    # Etapa 1: Sugestão de Mapeamento
    with st.expander("1️⃣ Sugerir Mapeamento de Colunas", expanded=True):
        if st.button("🔄 Gerar Sugestão de Mapeamento"):
            with st.spinner("Processando sugestões..."):
                payload = fetch_mapping_suggestions(api_url, file_bytes, file_name)

                if payload:
                    mapping = payload.get("mapping", {})
                    columns_csv = payload.get("columns_csv", [])

                    st.session_state["mapping"] = mapping
                    st.session_state["columns_csv"] = columns_csv

                    data = {
                        "Coluna ML (Alvo)": list(mapping.keys()),
                        "Coluna do seu CSV (Origem)": [
                            mapping[k] if mapping[k] is not None else ""
                            for k in mapping.keys()
                        ],
                    }
                    st.session_state["mapping_df"] = pd.DataFrame(data)
                    st.success("✅ Sugestão de mapeamento gerada.")

        if st.session_state.get("mapping_df") is not None:
            st.subheader("Validação do Mapeamento")
            st.caption("Ajuste o mapeamento entre as colunas do seu CSV e o modelo:")

            edited_df = st.data_editor(
                st.session_state["mapping_df"],
                key="mapping_editor",
                column_config={
                    "Coluna do seu CSV (Origem)": st.column_config.SelectboxColumn(
                        "Coluna do seu CSV (Origem)",
                        options=st.session_state.get("columns_csv", []),
                        required=False,
                    )
                },
                hide_index=True,
                use_container_width=True,
            )

            # Atualizar mapping no session_state
            mapping_dict = {}
            for _, row in edited_df.iterrows():
                target = row["Coluna ML (Alvo)"]
                origem = row["Coluna do seu CSV (Origem)"]
                origem_value = None
                if isinstance(origem, str) and origem.strip():
                    origem_value = origem
                mapping_dict[str(target)] = origem_value

            st.session_state["mapping"] = mapping_dict

    # Etapa 2: Pré-visualização
    with st.expander("2️⃣ Pré-visualizar Dados Mapeados"):
        if st.button("👁️ Visualizar ETL"):
            if not st.session_state.get("mapping"):
                st.warning("⚠️ Gere primeiro a sugestão de mapeamento.")
            else:
                with st.spinner("Processando pré-visualização..."):
                    payload = fetch_preview_mapped_csv(
                        api_url, file_bytes, file_name, st.session_state["mapping"]
                    )

                    if payload and payload.get("rows"):
                        st.success("✅ Pré-visualização gerada.")
                        df_preview = pd.DataFrame(payload["rows"])
                        st.dataframe(df_preview, use_container_width=True)
                    else:
                        st.warning("A API não retornou dados para pré-visualização.")

    # Etapa 3: Predição
    with st.expander("3️⃣ Executar Predição", expanded=True):
        if st.button("🚀 Rodar Predição", type="primary"):
            if not st.session_state.get("mapping"):
                st.warning("⚠️ Gere primeiro a sugestão de mapeamento.")
                return

            model_id = st.session_state.get("model_id", "")
            if not model_id or not model_id.strip():
                st.error("Informe o Model ID na barra lateral para continuar.")
                return

            with st.spinner("Executando predição..."):
                payload = fetch_prediction(
                    api_url,
                    file_bytes,
                    file_name,
                    st.session_state["mapping"],
                    model_id,
                )

                if payload:
                    df_pred = process_prediction_response(payload)
                    if df_pred is not None:
                        st.session_state["last_prediction"] = df_pred
                        st.session_state["last_model_id"] = (
                            model_id.strip() if model_id.strip() else ""
                        )
                        st.success(
                            f"✅ Predição realizada com sucesso! {len(df_pred)} registros processados."
                        )
                        st.rerun()
                        return True, df_pred
                    else:
                        st.warning("A API não retornou dados suficientes.")
                else:
                    st.error("Falha ao executar predição.")

    return False, None


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================


def main() -> None:
    st.set_page_config(
        page_title="Dashboard de Evasão Escolar",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()

    # Título principal
    st.title("🎓 Dashboard de Predição de Evasão Escolar")
    st.caption("Sistema de análise e predição de risco de evasão estudantil")

    # Upload de arquivo (sempre visível no topo)
    uploaded_file = st.file_uploader(
        "📂 Envie o arquivo CSV para predição",
        type=["csv"],
        help="Arquivo CSV contendo os dados dos alunos",
    )

    # Obter dados da última predição
    df_raw = st.session_state.get("last_prediction")

    # Renderizar sidebar com filtros
    render_sidebar(df_raw)

    # Separador visual
    st.divider()

    # Seção de configuração (mapeamento e predição)
    success, new_df = render_configuration_section(
        st.session_state.get("api_url", "http://localhost:10000"), uploaded_file
    )

    if success and new_df is not None:
        df_raw = new_df

    # Se há dados, renderizar tabs com análises
    if df_raw is not None and not df_raw.empty:
        st.divider()

        # Aplicar filtros
        filters = st.session_state.get("ui_filters", {})
        df_filtered = apply_filters(df_raw, filters)

        # Mostrar contagem de registros filtrados
        if len(df_filtered) < len(df_raw):
            st.info(
                f"📊 Exibindo **{len(df_filtered)}** de **{len(df_raw)}** registros após aplicar filtros."
            )

        # Tabs principais
        tab1, tab2, tab3 = st.tabs(
            ["📊 Visão Geral", "📋 Resultados / Tabela", "🔍 Observabilidade"]
        )

        with tab1:
            render_tab_visao_geral(
                df_filtered, filters.get("threshold_risco", 0.4)
            )

        with tab2:
            render_tab_resultados(df_filtered, filters.get("threshold_risco", 0.4))

        with tab3:
            render_tab_observabilidade(
                st.session_state.get("api_url", "http://localhost:10000"),
                st.session_state.get("model_id", "model_1"),
            )


if __name__ == "__main__":
    main()