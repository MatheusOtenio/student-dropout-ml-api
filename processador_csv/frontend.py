import io
import json
from typing import Dict, Any

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = "http://localhost:8000"


if "suggestions" not in st.session_state:
    st.session_state["suggestions"] = None
if "mapping_df" not in st.session_state:
    st.session_state["mapping_df"] = None
if "csv_columns" not in st.session_state:
    st.session_state["csv_columns"] = []
if "uploaded_file_bytes" not in st.session_state:
    st.session_state["uploaded_file_bytes"] = None
if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None


st.title("Preparação de Dados para Modelo de Evasão Escolar")

uploaded_file = st.file_uploader("Envie o arquivo CSV de alunos", type=["csv"])

if uploaded_file is not None:
    st.session_state["uploaded_file_bytes"] = uploaded_file.getvalue()
    st.session_state["uploaded_file_name"] = uploaded_file.name


analisar = st.button("Analisar Arquivo", disabled=st.session_state["uploaded_file_bytes"] is None)

if analisar and st.session_state["uploaded_file_bytes"] is not None:
    try:
        file_bytes = st.session_state["uploaded_file_bytes"]
        file_name = st.session_state["uploaded_file_name"] or "dados.csv"

        files = {
            "file": (file_name, io.BytesIO(file_bytes), "text/csv"),
        }
        response = requests.post(f"{BACKEND_URL}/analyze-csv", files=files)

        if response.status_code != 200:
            if response.status_code >= 500:
                st.error(f"Erro no servidor ao analisar CSV: {response.text}")
            else:
                st.error(f"Falha ao analisar CSV: {response.text}")
        else:
            suggestions: Dict[str, Any] = response.json()
            st.session_state["suggestions"] = suggestions

            header_df = pd.read_csv(io.BytesIO(file_bytes), nrows=0)
            csv_columns = list(header_df.columns)
            st.session_state["csv_columns"] = csv_columns

            data = {
                "Coluna ML (Alvo)": list(suggestions.keys()),
                "Coluna do seu CSV (Origem)": [
                    suggestions[k] if suggestions[k] is not None else "" for k in suggestions.keys()
                ],
            }
            st.session_state["mapping_df"] = pd.DataFrame(data)
    except requests.RequestException as exc:
        st.error(f"Erro de conexão com a API: {exc}")


if st.session_state["mapping_df"] is not None and st.session_state["csv_columns"]:
    st.subheader("Valide o mapeamento entre seu CSV e o modelo")

    edited_df = st.data_editor(
        st.session_state["mapping_df"],
        key="mapping_editor",
        column_config={
            "Coluna do seu CSV (Origem)": st.column_config.SelectboxColumn(
                "Coluna do seu CSV (Origem)",
                options=st.session_state["csv_columns"],
                required=False,
            )
        },
        hide_index=True,
    )

    st.session_state["mapping_df"] = edited_df

    processar = st.button(
        "Processar e Baixar",
        disabled=st.session_state["uploaded_file_bytes"] is None,
    )

    if processar and st.session_state["uploaded_file_bytes"] is not None:
        mapping_df = st.session_state["mapping_df"]
        mapping_dict: Dict[str, Any] = {}
        for _, row in mapping_df.iterrows():
            target = row["Coluna ML (Alvo)"]
            origem = row["Coluna do seu CSV (Origem)"]
            origem_value = None
            if isinstance(origem, str) and origem.strip() != "":
                origem_value = origem
            mapping_dict[str(target)] = origem_value

        try:
            file_bytes = st.session_state["uploaded_file_bytes"]
            file_name = st.session_state["uploaded_file_name"] or "dados.csv"

            files = {
                "file": (file_name, io.BytesIO(file_bytes), "text/csv"),
            }
            data = {
                "mapping": json.dumps(mapping_dict),
            }

            response = requests.post(f"{BACKEND_URL}/process-csv", files=files, data=data)

            if response.status_code != 200:
                if response.status_code >= 500:
                    st.error(f"Erro no servidor ao processar CSV: {response.text}")
                else:
                    st.error(f"Falha ao processar CSV: {response.text}")
            else:
                data_json = response.json()
                csv_clean = data_json.get("clean_data")
                csv_dropped = data_json.get("dropped_data")

                if csv_clean is None or csv_dropped is None:
                    st.error("Resposta da API não contém os campos esperados.")
                else:
                    st.success("CSV processado com sucesso. Faça o download abaixo.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Baixar Dataset ML",
                            data=csv_clean,
                            file_name=data_json.get("filename_clean", "dados_ml_padronizados.csv"),
                            mime="text/csv",
                        )
                    with col2:
                        st.download_button(
                            "🗑️ Baixar Relatório de Exclusão",
                            data=csv_dropped,
                            file_name=data_json.get("filename_dropped", "colunas_descartadas.csv"),
                            mime="text/csv",
                        )
        except requests.RequestException as exc:
            st.error(f"Erro de conexão com a API: {exc}")

