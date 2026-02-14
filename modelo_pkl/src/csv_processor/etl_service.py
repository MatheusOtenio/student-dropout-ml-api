from typing import Dict, Any, Tuple
from datetime import datetime
import logging

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from .schemas import COLUNAS_WHITELIST, FEATURE_COLUMNS, STATUS_VALIDOS
    from src.target_engineering import generate_future_dropout_target
except ImportError:
    from schemas import COLUNAS_WHITELIST, FEATURE_COLUMNS, STATUS_VALIDOS  # type: ignore
    try:
        from src.target_engineering import generate_future_dropout_target
    except ImportError:
        # Fallback se rodando direto do diretório
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
        from src.target_engineering import generate_future_dropout_target


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

    # Garante existência das colunas de feature + metadados essenciais
    # FEATURE_COLUMNS vem de schemas.py
    colunas_obrigatorias = list(FEATURE_COLUMNS) + ["situacao", "data_nascimento"]

    for col in colunas_obrigatorias:
        if col not in df.columns:
            df[col] = pd.NA

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
        "municipio_residencia",
    ]
    for coluna in colunas_categoricas:
        if coluna in df.columns:
            mascara = df[coluna].notna()
            df.loc[mascara, coluna] = (
                df.loc[mascara, coluna].astype(str).str.strip().str.upper()
            )

    if "situacao" in df.columns:
        cols_target = ["codigo_aluno", "ano_referencia", "periodo_referencia"]
        has_target_cols = all(c in df.columns for c in cols_target)
        
        if has_target_cols:
            try:
                df = generate_future_dropout_target(
                    df, 
                    horizon=2, 
                    id_col="codigo_aluno", 
                    year_col="ano_referencia", 
                    period_col="periodo_referencia", 
                    status_col="situacao"
                )
            except Exception:
                logger.error("Erro ao gerar target prospectivo", exc_info=True)
                raise

        cols_features = [c for c in FEATURE_COLUMNS if c in df.columns]

        other_cols = [c for c in df.columns if c not in cols_features and c != "situacao"]

        final_order = cols_features + other_cols + ["situacao"]
        df = df[final_order]
    else:
        cols_features = [c for c in FEATURE_COLUMNS if c in df.columns]
        other_cols = [c for c in df.columns if c not in cols_features]
        df = df[cols_features + other_cols]

    return df, df_dropped
