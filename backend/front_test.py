import io
import json
from typing import Any

import pandas as pd
import requests
import streamlit as st


def main() -> None:
    st.title("Cliente de Teste para API de Evasão Escolar")

    api_url = st.text_input(
        "URL da API",
        value="http://localhost:10000",
        help="Exemplo: http://localhost:10000 ou URL pública do Docker/Render",
    )

    model_id = st.text_input(
        "Model ID",
        value="model_1",
        help="Identificador do modelo .pkl em src/models (opcional).",
    )

    uploaded_file = st.file_uploader(
        "Envie o arquivo CSV para predição",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info("Envie um arquivo CSV para começar.")
        return

    if "mapping" not in st.session_state:
        st.session_state["mapping"] = None

    if st.button("1. Sugerir mapeamento"):
        try:
            file_bytes = uploaded_file.getvalue()
            if not file_bytes:
                st.error("Arquivo enviado está vazio.")
                return

            files = {
                "file": (
                    uploaded_file.name or "dados.csv",
                    io.BytesIO(file_bytes),
                    "text/csv",
                )
            }

            try:
                response = requests.post(
                    f"{api_url.rstrip('/')}/mapping-suggestions",
                    files=files,
                    timeout=60,
                )
            except requests.RequestException as exc:
                st.error(f"Erro de conexão com a API: {exc}")
                return

            if response.status_code != 200:
                st.error(
                    f"Erro na API (status {response.status_code}): "
                    f"{response.text}"
                )
                return

            try:
                payload = response.json()
            except ValueError as exc:
                st.error(f"Resposta da API não é um JSON válido: {exc}")
                return

            mapping = payload.get("mapping")
            columns_csv = payload.get("columns_csv")

            if not isinstance(mapping, dict) or not isinstance(columns_csv, list):
                st.warning("A API não retornou um mapeamento válido.")
                return

            st.session_state["mapping"] = mapping

            csv_to_schema: list[dict[str, Any]] = []
            for original in columns_csv:
                destino = None
                for schema_col, csv_col in mapping.items():
                    if csv_col == original:
                        destino = schema_col
                        break
                csv_to_schema.append(
                    {
                        "coluna_csv": original,
                        "campo_interno": destino,
                    }
                )

            df_map = pd.DataFrame(csv_to_schema)
            st.success("Sugestão de mapeamento gerada.")
            st.subheader("Sugestão de tradução de colunas")
            st.dataframe(df_map)

        except Exception as exc:
            st.error(f"Erro inesperado ao sugerir mapeamento: {exc}")

    if st.button("2. Pré-visualizar ETL"):
        if not st.session_state.get("mapping"):
            st.warning("Gere primeiro a sugestão de mapeamento.")
            return

        try:
            file_bytes = uploaded_file.getvalue()
            if not file_bytes:
                st.error("Arquivo enviado está vazio.")
                return

            files = {
                "file": (
                    uploaded_file.name or "dados.csv",
                    io.BytesIO(file_bytes),
                    "text/csv",
                )
            }

            data = {
                "mapping": json.dumps(st.session_state["mapping"]),
            }

            try:
                response = requests.post(
                    f"{api_url.rstrip('/')}/preview-mapped-csv",
                    files=files,
                    data=data,
                    timeout=60,
                )
            except requests.RequestException as exc:
                st.error(f"Erro de conexão com a API: {exc}")
                return

            if response.status_code != 200:
                st.error(
                    f"Erro na API (status {response.status_code}): "
                    f"{response.text}"
                )
                return

            try:
                payload = response.json()
            except ValueError as exc:
                st.error(f"Resposta da API não é um JSON válido: {exc}")
                return

            rows = payload.get("rows")
            if rows is None:
                st.warning("A API não retornou linhas processadas para pré-visualização.")
                return

            st.success("Pré-visualização do ETL concluída.")
            st.subheader("Tabela processada após mapeamento")
            df_preview = pd.DataFrame(rows)
            st.dataframe(df_preview)

        except Exception as exc:
            st.error(f"Erro inesperado no cliente (pré-visualização com mapeamento): {exc}")

    if st.button("3. Rodar predição"):
        if not st.session_state.get("mapping"):
            st.warning("Gere primeiro a sugestão de mapeamento.")
            return

        try:
            file_bytes = uploaded_file.getvalue()
            if not file_bytes:
                st.error("Arquivo enviado está vazio.")
                return

            files = {
                "file": (
                    uploaded_file.name or "dados.csv",
                    io.BytesIO(file_bytes),
                    "text/csv",
                )
            }

            data: dict[str, Any] = {
                "mapping": json.dumps(st.session_state["mapping"]),
            }
            if model_id.strip():
                data["model_id"] = model_id.strip()

            try:
                response = requests.post(
                    f"{api_url.rstrip('/')}/predict",
                    files=files,
                    data=data,
                    timeout=60,
                )
            except requests.RequestException as exc:
                st.error(f"Erro de conexão com a API: {exc}")
                return

            if response.status_code != 200:
                st.error(
                    f"Erro na API (status {response.status_code}): "
                    f"{response.text}"
                )
                return

            try:
                payload = response.json()
            except ValueError as exc:
                st.error(f"Resposta da API não é um JSON válido: {exc}")
                return

            rows = payload.get("rows")
            predictions = payload.get("predictions")

            if rows is not None and predictions is not None:
                st.success("Predição realizada com sucesso.")
                st.subheader("Tabela com probabilidade de evasão")
                df_pred = pd.DataFrame(rows)
                try:
                    df_pred["prob_evasao"] = predictions
                except Exception:
                    df_pred = df_pred.assign(prob_evasao=predictions)
                st.dataframe(df_pred)
            elif predictions is not None:
                st.success("Predição realizada com sucesso.")
                st.subheader("Predições (lista)")
                try:
                    df_pred = pd.DataFrame({"prob_evasao": predictions})
                except Exception:
                    df_pred = pd.DataFrame(predictions)
                st.dataframe(df_pred)
            else:
                st.warning("A API não retornou dados suficientes para montar a tabela.")

        except Exception as exc:
            st.error(f"Erro inesperado no cliente (predição): {exc}")


if __name__ == "__main__":
    main()
