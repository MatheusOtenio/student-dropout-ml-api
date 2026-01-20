import json
import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")
BACKEND_URL = API_URL.rstrip("/") + "/train"
MAX_FILE_SIZE = 200 * 1024 * 1024


def main():
    st.title("Treinamento de Modelo de Evasão Escolar")
    st.write("Envie um arquivo CSV para treinar um novo modelo.")

    uploaded_file = st.file_uploader("Selecione o arquivo CSV", type=["csv"])
    config_text = st.text_area(
        "Configuração opcional (JSON)",
        value="",
        height=150,
    )

    if uploaded_file is not None:
        size = len(uploaded_file.getvalue())
        st.write(f"Tamanho do arquivo: {size / (1024 * 1024):.2f} MB")
        if size > MAX_FILE_SIZE:
            st.error("O arquivo excede o limite de 200 MB.")

    if st.button("Treinar modelo"):
        if uploaded_file is None:
            st.error("Nenhum arquivo CSV selecionado.")
            return

        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) > MAX_FILE_SIZE:
            st.error("O arquivo excede o limite de 200 MB.")
            return

        data = {}
        if config_text.strip():
            try:
                json.loads(config_text)
                data["config"] = config_text
            except json.JSONDecodeError as exc:
                st.error(f"Configuração JSON inválida: {exc}")
                return

        files = {
            "file": (uploaded_file.name, file_bytes, "text/csv"),
        }

        with st.spinner("Treinando modelo..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    files=files,
                    data=data,
                    timeout=600,
                )
            except requests.RequestException as exc:
                st.error(f"Erro ao conectar ao backend: {exc}")
                return

        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                st.error("Resposta inválida do backend.")
                return

            st.success("Treinamento concluído com sucesso.")
            artifact_path = payload.get("artifact_path")
            version = payload.get("version")
            metrics = payload.get("metrics", {}) or {}

            st.write(f"ID do modelo: {artifact_path}")
            if version is not None:
                st.write(f"Versão: {version}")

            roc_auc = metrics.get("roc_auc")
            accuracy = metrics.get("accuracy")
            brier = metrics.get("brier_score")

            if roc_auc is not None:
                st.write(f"AUC ROC: {roc_auc:.4f}")
            if accuracy is not None:
                st.write(f"Acurácia: {accuracy:.4f}")
            if brier is not None:
                st.write(f"Brier score: {brier:.4f}")

        else:
            try:
                error_payload = response.json()
                detail = error_payload.get("detail", response.text)
            except ValueError:
                detail = response.text
            st.error(f"Erro no treinamento ({response.status_code}): {detail}")


if __name__ == "__main__":
    main()
