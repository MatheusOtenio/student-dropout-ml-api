from typing import Dict, Any, Tuple
from datetime import datetime

import pandas as pd

from schemas import COLUNAS_WHITELIST


def transformar_dados(
    df_bruto: pd.DataFrame, mapa: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforma o DataFrame bruto em um DataFrame padronizado para o modelo
    e retorna também um DataFrame com as colunas descartadas para auditoria.

    Retorno
    -------
    Tuple[pd.DataFrame, pd.DataFrame]:
        (df_padronizado, df_dropped)
    """
    df = df_bruto.copy()

    rename_dict: Dict[str, str] = {}
    for chave, valor in mapa.items():
        if chave in df.columns and valor in COLUNAS_WHITELIST:
            rename_dict[chave] = valor
        elif valor in df.columns and chave in COLUNAS_WHITELIST:
            rename_dict[valor] = chave

    if rename_dict:
        df = df.rename(columns=rename_dict)

    colunas_originais = set(df_bruto.columns)
    colunas_mapeadas = set(rename_dict.keys())
    colunas_dropped = sorted(colunas_originais - colunas_mapeadas)
    if colunas_dropped:
        df_dropped = df_bruto[colunas_dropped].copy()
    else:
        df_dropped = df_bruto.iloc[:, 0:0].copy()

    colunas_presentes = [c for c in COLUNAS_WHITELIST if c in df.columns]
    df = df[colunas_presentes].copy()

    colunas_obrigatorias = [
        "sexo",
        "cor_raca",
        "municipio_residencia",
        "uf_residencia",
        "data_nascimento",
        "curso",
        "campus",
        "turno",
        "modalidade_ingresso",
        "tipo_cota",
        "coeficiente_rendimento",
        "disciplinas_aprovadas",
        "disciplinas_reprovadas_nota",
        "disciplinas_reprovadas_frequencia",
        "periodo",
        "ano_ingresso",
        "semestre_ingresso",
        "nota_enem_humanas",
        "nota_enem_linguagem",
        "nota_enem_matematica",
        "nota_enem_natureza",
        "nota_enem_redacao",
        "nota_vestibular_biologia",
        "nota_vestibular_filosofia_sociologia",
        "nota_vestibular_fisica",
        "nota_vestibular_geografia",
        "nota_vestibular_historia",
        "nota_vestibular_literatura_brasileira",
        "nota_vestibular_lingua_estrangeira",
        "nota_vestibular_lingua_portuguesa",
        "nota_vestibular_matematica",
        "nota_vestibular_quimica",
        "idade",
        "situacao",
    ]

    for col in colunas_obrigatorias:
        if col not in df.columns:
            df[col] = pd.NA

    if "data_nascimento" in df.columns:
        datas = pd.to_datetime(df["data_nascimento"], errors="coerce")
        hoje = pd.Timestamp(datetime.now().date())
        idades_calculadas = ((hoje - datas).dt.days // 365).astype("float")
        if "idade" in df.columns:
            mascara = df["idade"].isna()
            df.loc[mascara, "idade"] = idades_calculadas[mascara]
        else:
            df["idade"] = idades_calculadas

    colunas_nota = [c for c in df.columns if c.startswith("nota_")]

    colunas_numericas_prioritarias = {
        "coeficiente_rendimento",
        "disciplinas_aprovadas",
        "disciplinas_reprovadas_nota",
        "disciplinas_reprovadas_frequencia",
        "total_semestres_cursados",
        "ano_ingresso",
        "semestre_ingresso",
        "idade",
    }
    colunas_numericas = set(colunas_numericas_prioritarias) | set(colunas_nota)
    colunas_numericas_presentes = [c for c in colunas_numericas if c in df.columns]

    for coluna in colunas_numericas_presentes:
        df[coluna] = df[coluna].astype(str).str.replace(",", ".", regex=False)
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    if colunas_nota:
        colunas_nota_presentes = [c for c in colunas_nota if c in df.columns]
        if colunas_nota_presentes:
            df[colunas_nota_presentes] = df[colunas_nota_presentes].fillna(0.0)

    colunas_enem = [
        "nota_enem_humanas",
        "nota_enem_linguagem",
        "nota_enem_matematica",
        "nota_enem_natureza",
        "nota_enem_redacao",
    ]
    colunas_vestibular = [
        "nota_vestibular_biologia",
        "nota_vestibular_filosofia_sociologia",
        "nota_vestibular_fisica",
        "nota_vestibular_geografia",
        "nota_vestibular_historia",
        "nota_vestibular_literatura_brasileira",
        "nota_vestibular_lingua_estrangeira",
        "nota_vestibular_lingua_portuguesa",
        "nota_vestibular_matematica",
        "nota_vestibular_quimica",
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

    colunas_categoricas = [
        "sexo",
        "turno",
        "curso",
        "campus",
        "modalidade_ingresso",
        "tipo_cota",
        "cor_raca",
        "municipio_residencia",
        "uf_residencia",
    ]
    for coluna in colunas_categoricas:
        if coluna in df.columns:
            mascara = df[coluna].notna()
            df.loc[mascara, coluna] = (
                df.loc[mascara, coluna].astype(str).str.strip().str.upper()
            )

    if "situacao" in df.columns:
        cols_sem_situacao = [c for c in df.columns if c != "situacao"]
        df = df[cols_sem_situacao + ["situacao"]]

    return df, df_dropped
