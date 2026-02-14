import io
import json
import os

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")
BACKEND_URL = API_URL.rstrip("/") + "/train"

CSV_PROCESSOR_URL = os.getenv("CSV_PROCESSOR_URL", "http://localhost:8001")
CSV_ANALYZE_URL = CSV_PROCESSOR_URL.rstrip("/") + "/analyze-csv"
CSV_PROCESS_URL = CSV_PROCESSOR_URL.rstrip("/") + "/process-csv"
IMPORTANCE_URL = API_URL.rstrip("/") + "/model/importance"

MAX_FILE_SIZE = 200 * 1024 * 1024


def main():
    st.title("Evasão Escolar – Pipeline de Dados e Treinamento")

    if "suggestions" not in st.session_state:
        st.session_state["suggestions"] = None
    if "mapping_df" not in st.session_state:
        st.session_state["mapping_df"] = None
    if "csv_columns" not in st.session_state:
        st.session_state["csv_columns"] = []
    if "uploaded_file_bytes_raw" not in st.session_state:
        st.session_state["uploaded_file_bytes_raw"] = None
    if "uploaded_file_name_raw" not in st.session_state:
        st.session_state["uploaded_file_name_raw"] = None
    if "model_id" not in st.session_state:
        st.session_state["model_id"] = ""

    tab_process, tab_train, tab_observe = st.tabs(
        [
            "1. Preparar CSV (csv_processor)",
            "2. Treinar modelo (CSV já processado)",
            "3. Observabilidade",
        ]
    )

    with tab_process:
        st.subheader("1. Processar CSV bruto com csv_processor")
        uploaded_file_raw = st.file_uploader(
            "Selecione o CSV bruto para preparação", type=["csv"], key="raw_uploader"
        )

        if uploaded_file_raw is not None:
            st.session_state["uploaded_file_bytes_raw"] = uploaded_file_raw.getvalue()
            st.session_state["uploaded_file_name_raw"] = uploaded_file_raw.name

        analisar = st.button(
            "Analisar cabeçalho",
            disabled=st.session_state["uploaded_file_bytes_raw"] is None,
            key="analyze_button",
        )

        if analisar and st.session_state["uploaded_file_bytes_raw"] is not None:
            try:
                file_bytes = st.session_state["uploaded_file_bytes_raw"]
                file_name = st.session_state["uploaded_file_name_raw"] or "dados.csv"

                files = {
                    "file": (file_name, io.BytesIO(file_bytes), "text/csv"),
                }
                response = requests.post(CSV_ANALYZE_URL, files=files, timeout=120)

                if response.status_code != 200:
                    if response.status_code >= 500:
                        st.error(f"Erro no servidor ao analisar CSV: {response.text}")
                    else:
                        st.error(f"Falha ao analisar CSV: {response.text}")
                else:
                    suggestions = response.json()
                    st.session_state["suggestions"] = suggestions

                    header_df = pd.read_csv(io.BytesIO(file_bytes), nrows=0)
                    csv_columns = list(header_df.columns)
                    st.session_state["csv_columns"] = csv_columns

                    data = {
                        "Coluna ML (Alvo)": list(suggestions.keys()),
                        "Coluna do seu CSV (Origem)": [
                            suggestions[k] if suggestions[k] is not None else ""
                            for k in suggestions.keys()
                        ],
                    }
                    st.session_state["mapping_df"] = pd.DataFrame(data)
            except requests.RequestException as exc:
                st.error(f"Erro de conexão com o serviço de processamento: {exc}")

        if st.session_state["mapping_df"] is not None and st.session_state["csv_columns"]:
            st.subheader("Valide o mapeamento entre seu CSV e o modelo")

            edited_df = st.data_editor(
                st.session_state["mapping_df"],
                key="mapping_editor_root",
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
                "Processar e baixar CSVs",
                disabled=st.session_state["uploaded_file_bytes_raw"] is None,
                key="process_button",
            )

            if processar and st.session_state["uploaded_file_bytes_raw"] is not None:
                mapping_df = st.session_state["mapping_df"]
                mapping_dict = {}
                for _, row in mapping_df.iterrows():
                    target = row["Coluna ML (Alvo)"]
                    origem = row["Coluna do seu CSV (Origem)"]
                    origem_value = None
                    if isinstance(origem, str) and origem.strip() != "":
                        origem_value = origem
                    mapping_dict[str(target)] = origem_value

                try:
                    file_bytes = st.session_state["uploaded_file_bytes_raw"]
                    file_name = st.session_state["uploaded_file_name_raw"] or "dados.csv"

                    files = {
                        "file": (file_name, io.BytesIO(file_bytes), "text/csv"),
                    }
                    data = {
                        "mapping": json.dumps(mapping_dict),
                    }

                    response = requests.post(
                        CSV_PROCESS_URL, files=files, data=data, timeout=300
                    )

                    if response.status_code != 200:
                        if response.status_code >= 500:
                            st.error(
                                f"Erro no servidor ao processar CSV: {response.text}"
                            )
                        else:
                            st.error(f"Falha ao processar CSV: {response.text}")
                    else:
                        data_json = response.json()
                        csv_clean = data_json.get("clean_data")
                        csv_dropped = data_json.get("dropped_data")

                        if csv_clean is None or csv_dropped is None:
                            st.error(
                                "Resposta da API de processamento não contém os campos esperados."
                            )
                        else:
                            st.success(
                                "CSV processado com sucesso. Faça o download abaixo e use o arquivo padronizado na etapa 2."
                            )
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    "📥 Baixar Dataset ML",
                                    data=csv_clean,
                                    file_name=data_json.get(
                                        "filename_clean", "dados_ml_padronizados.csv"
                                    ),
                                    mime="text/csv",
                                )
                            with col2:
                                st.download_button(
                                    "🗑️ Baixar Relatório de Exclusão",
                                    data=csv_dropped,
                                    file_name=data_json.get(
                                        "filename_dropped",
                                        "colunas_descartadas.csv",
                                    ),
                                    mime="text/csv",
                                )
                except requests.RequestException as exc:
                    st.error(f"Erro de conexão com o serviço de processamento: {exc}")

    with tab_train:
        st.subheader("2. Treinar Modelo de Risco (Prospectivo)")
        st.write("Envie o arquivo CSV processado (contendo 'target_evasao' e 'is_target_valid').")

        uploaded_file = st.file_uploader(
            "Selecione o CSV processado (saída do passo 1)", type=["csv"], key="train_uploader"
        )

        # --- Lógica de UI para Split Temporal ---
        split_config = None
        years_available = []

        if uploaded_file is not None:
            # Tentar ler cabeçalho e anos para validar e configurar UI
            try:
                uploaded_file.seek(0)
                # Ler colunas para validação rápida (com detecção automática de separador)
                # O problema relatado pode ser separador ';' ou erro de BOM
                df_header = pd.read_csv(uploaded_file, nrows=0, sep=None, engine="python")
                cols = set(df_header.columns)
                
                required = {"target_evasao", "is_target_valid", "ano_ingresso"}
                missing = required - cols
                
                if missing:
                    # Tentativa de fallback: se faltar, pode ser separador ';' fixo
                    if len(cols) == 1: # Indício de que leu tudo como uma coluna só
                         uploaded_file.seek(0)
                         df_header = pd.read_csv(uploaded_file, nrows=0, sep=";")
                         cols = set(df_header.columns)
                         missing = required - cols
                
                if missing:
                    st.error(f"❌ O CSV fornecido não é compatível com o novo modelo prospectivo.")
                    st.write(f"Colunas ausentes: {missing}")
                    st.warning("Por favor, volte à aba 1 e reprocesse seu CSV bruto para gerar os targets necessários.")
                    uploaded_file = None # Bloqueia o resto
                else:
                    # Se ok, lê anos para o split (usa o mesmo separador detectado ou padrão)
                    uploaded_file.seek(0)
                    # Detectar separador novamente se necessário, ou assumir ',' se funcionou antes
                    # Para garantir, usamos engine python e sep=None novamente
                    df_meta = pd.read_csv(uploaded_file, usecols=["ano_ingresso", "is_target_valid", "target_evasao"], sep=None, engine="python")
                    
                    # Estatísticas Rápidas
                    n_total = len(df_meta)
                    n_valid = df_meta["is_target_valid"].sum()
                    n_target_1 = df_meta.loc[df_meta["is_target_valid"], "target_evasao"].sum()
                    
                    st.info(
                        f"**Análise do Dataset:**\n"
                        f"- Total de Linhas: {n_total}\n"
                        f"- Linhas Válidas para Treino (Futuro Conhecido): {n_valid} ({n_valid/n_total:.1%})\n"
                        f"- Taxa de Evasão Futura (no set válido): {n_target_1/n_valid:.1%}"
                    )
                    
                    years_available = sorted(df_meta["ano_ingresso"].dropna().astype(int).unique().tolist())
                    
            except Exception as e:
                st.warning(f"Erro ao analisar arquivo: {e}")
                uploaded_file.seek(0)

        with st.expander("⚙️ Configuração de Split Temporal (Avançado)", expanded=False):
            use_custom_split = st.checkbox("Usar Split Temporal Customizado")
            
            if use_custom_split:
                if not years_available:
                    st.warning("Coluna 'ano_ingresso' não detectada. Insira os anos manualmente.")
                    min_y, max_y = 2010, datetime.now().year
                    years_options = list(range(min_y, max_y + 1))
                else:
                    min_y, max_y = min(years_available), max(years_available)
                    years_options = years_available
                
                # 1. Train Range
                # Default: Começo até (Fim - 2)
                default_end = max_y - 2 if max_y - 2 >= min_y else min_y
                
                train_range = st.slider(
                    "Intervalo de Treino",
                    min_value=min_y,
                    max_value=max_y,
                    value=(min_y, default_end),
                    step=1,
                    help="Selecione o intervalo de anos para treinar o modelo."
                )
                
                train_end = train_range[1]
                
                # 2. Validation (Select Multiple Years)
                val_options = [y for y in years_options if y > train_end]
                if not val_options:
                    st.error("❌ Sem anos disponíveis para validação (deve ser > fim do treino).")
                    val_years = []
                else:
                    val_years = st.multiselect(
                        "Anos de Validação", 
                        options=val_options,
                        default=[val_options[0]],
                        help="Anos usados para Early Stopping e métricas de validação. Selecione anos com diversidade de casos."
                    )
                
                # 3. Test (Select Multiple Years)
                if val_years:
                    max_val_year = max(val_years)
                    test_options = [y for y in years_options if y > max_val_year]
                else:
                    test_options = []

                if not test_options:
                     st.warning("⚠️ Sem anos disponíveis para teste (deve ser > validação).")
                     test_years = []
                else:
                    test_years = st.multiselect(
                        "Anos de Teste", 
                        options=test_options,
                        default=[test_options[0]] if test_options else [],
                        help="Anos reservados para teste final (opcional mas recomendado)."
                    )
                
                if train_range and val_years and test_years:
                    split_config = {
                        "train_range": list(train_range),
                        "val_years": list(val_years),
                        "test_years": list(test_years)
                    }
                    st.info(
                        f"**Resumo do Split:**\n"
                        f"- Treino: {train_range[0]} a {train_range[1]}\n"
                        f"- Validação: {val_years}\n"
                        f"- Teste: {test_years}\n"
                        f"- Ignorados (Censura): {[y for y in years_options if y > max(test_years)] if test_years else []}"
                    )
                else:
                    st.warning("⚠️ Configuração incompleta: Certifique-se de selecionar anos para Treino, Validação e Teste.")

        config_text = st.text_area(
            "Outras Configurações (JSON)",
            value="",
            height=100,
            help="Configurações avançadas do modelo (ex: {'model_params': {'n_estimators': 500}})"
        )

        if uploaded_file is not None:
            size = len(uploaded_file.getvalue())
            st.write(f"Tamanho do arquivo: {size / (1024 * 1024):.2f} MB")
            if size > MAX_FILE_SIZE:
                st.error("O arquivo excede o limite de 200 MB.")

        if st.button("Treinar modelo", key="train_button"):
            if uploaded_file is None:
                st.error("Nenhum arquivo CSV selecionado.")
                return

            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) > MAX_FILE_SIZE:
                st.error("O arquivo excede o limite de 200 MB.")
                return

            # Montagem da Configuração
            final_config_dict = {}
            
            # 1. Configuração Manual (JSON Text Area)
            if config_text.strip():
                try:
                    manual_cfg = json.loads(config_text)
                    if isinstance(manual_cfg, dict):
                        final_config_dict.update(manual_cfg)
                    else:
                         st.error("O JSON deve ser um objeto (dicionário).")
                         return
                except json.JSONDecodeError as exc:
                    st.error(f"Configuração JSON inválida: {exc}")
                    return
            
            # 2. Configuração de Split (Visual)
            if split_config:
                final_config_dict["split_config"] = split_config

            # Prepara payload
            data = {}
            if final_config_dict:
                data["config"] = json.dumps(final_config_dict)

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
                    st.error(f"Erro ao conectar ao backend de treinamento: {exc}")
                    return

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError:
                    st.error("Resposta inválida do backend de treinamento.")
                    return

                st.success("Treinamento concluído com sucesso.")
                artifact_path = payload.get("artifact_path")
                if artifact_path:
                    st.session_state["model_id"] = artifact_path
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
                st.error(
                    f"Erro no treinamento ({response.status_code}): {detail}"
                )

    with tab_observe:
        st.subheader("3. Observabilidade do Modelo")
        st.write(
            "Analise a importância das features e entenda o porquê de cada predição."
        )

        model_id_input = st.text_input(
            "ID do Modelo (Artefato)", value=st.session_state.get("model_id", "")
        )

        st.markdown("---")
        st.write("### 📊 Importância Global das Features")

        top_n = st.slider("Quantidade de features para exibir", min_value=5, max_value=100, value=20, step=5)

        if st.button("Carregar Importância Global", key="btn_importance"):
            if not model_id_input:
                st.error("Por favor, insira um ID de modelo.")
            else:
                try:
                    resp = requests.get(
                        IMPORTANCE_URL, params={"model_id": model_id_input}, timeout=10
                    )
                    if resp.status_code == 200:
                        importance_data = resp.json()
                        if importance_data:
                            # Prioriza a exibição mapeada (nomes das features)
                            if "mapped" in importance_data and importance_data["mapped"]:
                                df_imp = pd.DataFrame(importance_data["mapped"])
                                # Garante colunas esperadas
                                if "feature" in df_imp.columns and "importance" in df_imp.columns:
                                    df_imp = df_imp.rename(columns={"feature": "Feature", "importance": "Importância"})
                                    df_imp = df_imp.sort_values(by="Importância", ascending=False).head(top_n)
                                    st.bar_chart(df_imp.set_index("Feature"), use_container_width=True)
                                else:
                                    st.warning("Formato de dados mapeados inesperado.")
                                    st.write(importance_data["mapped"])
                            
                            # Fallback para raw_values se não houver mapeamento
                            elif "raw_values" in importance_data:
                                st.warning(
                                    "O modelo retornou valores brutos (sem nomes de features)."
                                )
                                st.bar_chart(importance_data["raw_values"][:top_n])
                            
                            # Legado: dict direto {feature: valor}
                            else:
                                df_imp = pd.DataFrame(
                                    list(importance_data.items()),
                                    columns=["Feature", "Importância"],
                                )
                                df_imp = df_imp.sort_values(
                                    by="Importância", ascending=False
                                ).head(top_n)
                                st.bar_chart(
                                    df_imp.set_index("Feature"), use_container_width=True
                                )
                        else:
                            st.info("Nenhuma informação de importância encontrada.")
                    else:
                        st.error(f"Erro ao buscar importância: {resp.text}")
                except requests.RequestException as e:
                    st.error(f"Erro de conexão: {e}")


if __name__ == "__main__":
    main()
