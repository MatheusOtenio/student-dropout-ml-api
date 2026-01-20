import os

import pandas as pd
import streamlit as st

from backend_logic import run_prediction


def get_artifacts_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base_dir, "..", "modelo_pkl", "artifacts"))


def list_model_files():
    artifacts_dir = get_artifacts_dir()
    if not os.path.isdir(artifacts_dir):
        return artifacts_dir, []
    files = []
    for name in os.listdir(artifacts_dir):
        full_path = os.path.join(artifacts_dir, name)
        if os.path.isfile(full_path) and name.lower().endswith(".pkl"):
            files.append(name)
    files.sort()
    return artifacts_dir, files


def main():
    st.title("Sistema de Predição ML")

    artifacts_dir, model_files = list_model_files()

    if not model_files:
        st.error(
            "Nenhum modelo .pkl encontrado em '../modelo_pkl/artifacts/'. "
            "Copie os arquivos de modelo treinados para esse diretório."
        )
        return

    selected_model = st.sidebar.selectbox(
        "Selecione o modelo (.pkl)", model_files
    )

    uploaded_file = st.file_uploader(
        "Envie um arquivo CSV com as features", type=["csv"]
    )

    if st.button("Realizar Predição"):
        if uploaded_file is None:
            st.warning("Envie um arquivo CSV antes de realizar a predição.")
            return

        model_path = os.path.join(artifacts_dir, selected_model)

        try:
            result_df = run_prediction(model_path, uploaded_file)
        except Exception as e:
            st.error(f"Ocorreu um erro ao executar a predição: {e}")
            return

        if not isinstance(result_df, pd.DataFrame):
            st.error("A função de predição não retornou um DataFrame pandas.")
            return

        if result_df.empty:
            st.error("A predição retornou um conjunto de dados vazio.")
            return

        if "risk_level" not in result_df.columns:
            st.error(
                "O resultado não contém a coluna 'risk_level'. "
                "Verifique a lógica de backend."
            )
            return

        total_alunos = len(result_df)
        alunos_risco_alto = int((result_df["risk_level"] == "Alto").sum())
        alunos_risco_medio = int((result_df["risk_level"] == "Médio").sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Alunos", total_alunos)
        col2.metric("Alunos em Risco Alto", alunos_risco_alto)
        col3.metric("Alunos em Risco Médio", alunos_risco_medio)

        st.subheader("Resultado da Predição")
        def _highlight_risk(row):
            risk = row.get("risk_level", "")
            if risk == "Alto":
                return ["background-color: #ffcccc"] * len(row)
            if risk == "Médio":
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)

        styled = result_df.style.apply(_highlight_risk, axis=1)
        st.dataframe(styled, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar CSV com predições",
            data=csv_bytes,
            file_name="predicoes.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
