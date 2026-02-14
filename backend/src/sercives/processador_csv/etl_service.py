from typing import Dict, Any, Tuple, IO
from datetime import datetime
import traceback

import pandas as pd

from src.sercives.processador_csv.schemas import COLUNAS_WHITELIST, FEATURE_COLUMNS  # TODO: Fix typo.


# ---------------------------------------------------------------------------
# Ordem EXATA das colunas que o modelo espera como entrada.
# Definida centralizadamente em schemas.py
# ---------------------------------------------------------------------------
# FEATURE_COLUMNS importada de schemas.py


def process_csv(file_like_obj: IO[bytes] | IO[str]) -> dict | list | pd.DataFrame:
    try:
        if hasattr(file_like_obj, "seek"):
            file_like_obj.seek(0)
        df_bruto = pd.read_csv(file_like_obj, low_memory=False)

        df_ml, _ = transformar_dados(df_bruto, mapa={})
    except Exception as exc:
        traceback.print_exc()
        raise ValueError("Erro ao processar CSV") from exc

    return df_ml


def transformar_dados(
    df_bruto: pd.DataFrame, mapa: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforma o DataFrame bruto em um DataFrame padronizado para o modelo
    e retorna também um DataFrame com as colunas descartadas para auditoria.

    Retorno
    -------
    Tuple[pd.DataFrame, pd.DataFrame]:
        (df_features, df_dropped)

        df_features  — colunas na ordem de FEATURE_COLUMNS, com "situacao"
                       no fim (para uso como label / auditoria).
                       Passe por preparar_para_predict() antes do predict().
        df_dropped   — colunas do CSV original que não foram mapeadas.
    """
    df = df_bruto.copy()

    # --- 1) Monta rename_dict a partir do mapa (filtra valores None) ---
    rename_dict: Dict[str, str] = {}
    for chave, valor in mapa.items():
        if valor is None:
            continue
        if valor in df.columns and chave in COLUNAS_WHITELIST:
            rename_dict[valor] = chave
        elif chave in df.columns and valor in COLUNAS_WHITELIST:
            rename_dict[chave] = valor

    if rename_dict:
        df = df.rename(columns=rename_dict)

    # --- 2) Calcula df_dropped ---
    colunas_mantidas_originais: set = set()
    for col_original in df_bruto.columns:
        col_renomeada = rename_dict.get(col_original, col_original)
        if col_renomeada in COLUNAS_WHITELIST:
            colunas_mantidas_originais.add(col_original)

    colunas_dropped = sorted(set(df_bruto.columns) - colunas_mantidas_originais)
    df_dropped = df_bruto[colunas_dropped].copy() if colunas_dropped else df_bruto.iloc[:, 0:0].copy()

    # --- 3) Filtra apenas colunas da whitelist presentes no df ---
    colunas_presentes = [c for c in COLUNAS_WHITELIST if c in df.columns]
    df = df[colunas_presentes].copy()

    # --- 4) Garante que todas as colunas necessárias existam ---
    todas_necessarias = list(FEATURE_COLUMNS) + ["situacao", "data_nascimento"]
    for col in todas_necessarias:
        if col not in df.columns:
            df[col] = pd.NA

    # --- 5) Calcula idade a partir de data_nascimento quando ausente ---
    if "data_nascimento" in df.columns:
        datas = pd.to_datetime(df["data_nascimento"], errors="coerce", dayfirst=True)
        hoje = pd.Timestamp(datetime.now().date())
        idades_calculadas = ((hoje - datas).dt.days // 365).astype("float")
        if "idade" in df.columns:
            df["idade"] = pd.to_numeric(df["idade"], errors="coerce").astype("float")
            mascara = df["idade"].isna()
            df.loc[mascara, "idade"] = idades_calculadas[mascara]
        else:
            df["idade"] = idades_calculadas

    # --- 6) Converte colunas numéricas (trata vírgula decimal) ---
    colunas_nota = [c for c in df.columns if c.startswith("nota_")]

    colunas_numericas_prioritarias = {
        "coeficiente_rendimento",
        "disciplinas_aprovadas",
        "disciplinas_reprovadas_nota",
        "total_semestres_cursados",
        "ano_ingresso",
        "semestre_ingresso",
        "idade",
        # NOVAS COLUNAS DO x_columns_report.json
        "calouro",
    }
    colunas_numericas = set(colunas_numericas_prioritarias) | set(colunas_nota)
    colunas_numericas_presentes = [c for c in colunas_numericas if c in df.columns]

    for coluna in colunas_numericas_presentes:
        df[coluna] = df[coluna].astype(str).str.replace(",", ".", regex=False)
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    # --- 7) Preenche notas ausentes com 0.0 ---
    if colunas_nota:
        colunas_nota_presentes = [c for c in colunas_nota if c in df.columns]
        if colunas_nota_presentes:
            df[colunas_nota_presentes] = df[colunas_nota_presentes].fillna(0.0)

    # --- 8) Zera notas inaplicáveis por modalidade de ingresso ---
    colunas_enem = [
        "nota_enem_humanas", "nota_enem_linguagem", "nota_enem_matematica",
        "nota_enem_natureza", "nota_enem_redacao",
    ]
    colunas_vestibular = [
        "nota_vestibular_biologia", "nota_vestibular_filosofia_sociologia",
        "nota_vestibular_fisica", "nota_vestibular_geografia",
        "nota_vestibular_historia", "nota_vestibular_literatura_brasileira",
        "nota_vestibular_lingua_estrangeira", "nota_vestibular_lingua_portuguesa",
        "nota_vestibular_matematica", "nota_vestibular_quimica",
    ]

    if "modalidade_ingresso" in df.columns:
        mask_enem = df["modalidade_ingresso"].astype(str).str.contains(
            "ENEM|SISU", case=False, na=False
        )
        mask_vest = df["modalidade_ingresso"].astype(str).str.contains(
            "VESTIBULAR", case=False, na=False
        )

        colunas_enem_presentes = [c for c in colunas_enem if c in df.columns]
        colunas_vestibular_presentes = [c for c in colunas_vestibular if c in df.columns]

        if colunas_enem_presentes:
            df.loc[mask_vest, colunas_enem_presentes] = pd.NA
        if colunas_vestibular_presentes:
            df.loc[mask_enem, colunas_vestibular_presentes] = pd.NA

    # --- 9) Normaliza colunas categóricas para UPPER ---
    # "situacao" é deliberadamente excluída: é o target (label), não feature.
    colunas_categoricas = [
        "sexo", "turno", "curso", "campus", "modalidade_ingresso",
        "tipo_cota", "municipio_residencia",
        "nome_aluno", "codigo_aluno",
    ]
    for coluna in colunas_categoricas:
        if coluna in df.columns:
            mascara = df[coluna].notna()
            df.loc[mascara, coluna] = (
                df.loc[mascara, coluna].astype(str).str.strip().str.upper()
            )

    # --- 10) Reordena: FEATURE_COLUMNS + situacao + identificadores no fim ---
    # data_nascimento já foi usado para calcular idade; não vai para a saída.
    
    # Colunas de identificação que queremos preservar se existirem
    cols_identificacao = ["nome_aluno", "codigo_aluno"]
    
    colunas_finais = [c for c in FEATURE_COLUMNS if c in df.columns] + ["situacao"]
    
    for col in cols_identificacao:
        if col in df.columns:
            colunas_finais.insert(0, col) # Coloca no início para melhor visualização
            
    # Remove duplicatas preservando ordem
    colunas_finais = list(dict.fromkeys(colunas_finais))
            
    df = df[colunas_finais].copy()

    return df, df_dropped


def preparar_para_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe o df saída de transformar_dados() e retorna um DataFrame
    pronto para passar ao predict():
      • A coluna 'situacao' é MANTIDA aqui se existir, pois o backend_logic.py
        precisa removê-la explicitamente apenas no momento da inferência para garantir
        que não houve vazamento, mas pode ser usada para logs ou validações antes.
      • Remove 'data_nascimento' se ainda estiver presente.
      • Garante a ordem exata de FEATURE_COLUMNS.
      • Preenche colunas faltantes com NA.

    Uso típico:
        df_ml, _ = transformar_dados(df_bruto, mapa)
        df_predict = preparar_para_predict(df_ml)
        predictions = predict(model_id, df_predict)
    """
    df = df.copy()

    # Remove campos que não são features do modelo
    for col in ["data_nascimento"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Garante que todas as features existam
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Retorna features + situacao (se existir)
    # Importante: A ordem das features deve ser preservada para o modelo.
    # 'situacao' vai no final se existir, mas NÃO é feature.
    cols_to_return = list(FEATURE_COLUMNS)
    if "situacao" in df.columns:
        cols_to_return.append("situacao")
        
    # Seleciona apenas as colunas relevantes, ignorando extras
    # fillna(np.nan) ou similar não é necessário aqui, o preprocessor cuidará dos NAs
    return df[cols_to_return].copy()