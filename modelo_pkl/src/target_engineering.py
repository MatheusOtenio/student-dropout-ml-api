import pandas as pd
import numpy as np
from typing import Optional, List, Set

STATUS_EVASAO = {"desistente", "trancado"}
STATUS_CONCLUSAO = {"formado"}
STATUS_TERMINAIS = STATUS_EVASAO | STATUS_CONCLUSAO

def generate_future_dropout_target(
    df: pd.DataFrame,
    horizon: int = 2,
    id_col: str = "codigo_aluno",
    year_col: str = "ano_referencia",
    period_col: str = "periodo_referencia",
    status_col: str = "situacao"
) -> pd.DataFrame:
    """
    Gera target de evasão prospectiva olhando K semestres à frente, respeitando a distância temporal real.

    Para cada registro de aluno, verifica se existe algum evento de evasão (Desistente/Trancado)
    dentro do horizonte temporal especificado (em semestres letivos), independentemente de gaps
    na matrícula.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo o histórico acadêmico. Deve conter colunas de ID, Ano, Período e Situação.
    horizon : int, optional
        Número de semestres futuros para considerar no target prospectivo (default=2).
    id_col : str, optional
        Nome da coluna de identificação do aluno (default="codigo_aluno").
    year_col : str, optional
        Nome da coluna de ano (default="ano_referencia").
    period_col : str, optional
        Nome da coluna de período/semestre (default="periodo_referencia").
    status_col : str, optional
        Nome da coluna de situação/status (default="situacao").

    Returns
    -------
    pd.DataFrame
        DataFrame original acrescido das colunas:
        - 'target_evasao': 1.0 se houver evasão dentro do horizonte, 0.0 caso contrário.
        - 'is_target_valid': True se o registro atual não é terminal, False caso contrário.

    Raises
    ------
    ValueError
        Se colunas obrigatórias estiverem ausentes.
    """

    required_cols = [id_col, year_col, period_col, status_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes para engenharia de target: {missing}")

    df = df.copy()
    
    # Normalização de status
    s_norm = df[status_col].astype(str).str.strip().str.lower()
    
    # Ordenação crítica para o funcionamento do shift/janela
    df = df.sort_values(by=[id_col, year_col, period_col])
    
    # Cálculo de índice temporal absoluto (semestre corrido)
    # Assumindo semestres 1 e 2.
    # Ex: 2020.1 -> 4040, 2020.2 -> 4041. Diferença de 1 unidade = 1 semestre.
    temp_abs_col = "_abs_semester"
    df[temp_abs_col] = df[year_col] * 2 + (df[period_col] - 1)
    
    grouped = df.groupby(id_col)
    
    future_evasao_mask = pd.Series(False, index=df.index)
    
    # Varredura prospectiva
    # Iteramos k posições à frente no DataFrame agrupado.
    # Para cada k, verificamos:
    # 1. Se o status futuro é de evasão.
    # 2. Se a distância temporal (delta semestres) está dentro do horizonte permitido.
    # Isso corrige o problema de gaps (ex: 2020.1 -> 2022.1 é shift(1) mas delta=4).
    for k in range(1, horizon + 1):

        # Pega status e tempo da k-ésima linha à frente (dentro do grupo do aluno)
        future_status = grouped[status_col].shift(-k)
        future_abs_sem = grouped[temp_abs_col].shift(-k)
        
        # Calcula distância temporal real
        delta_semestres = future_abs_sem - df[temp_abs_col]
        
        # Verifica evasão
        future_status_norm = future_status.astype(str).str.strip().str.lower()
        k_step_evasao = future_status_norm.isin(STATUS_EVASAO)
        
        # Verifica janela temporal
        # delta_semestres <= horizon garante que não estamos pegando evasão muito distante
        valid_window = delta_semestres <= horizon
        
        # Combina condições: É evasão E está dentro do prazo
        # O fillna(False) garante que NaNs (fim da série) sejam tratados como não-evasão
        k_step_target = (k_step_evasao & valid_window).fillna(False)
        
        future_evasao_mask = future_evasao_mask | k_step_target

    df["target_evasao"] = future_evasao_mask.astype(float)
    
    # Define validade do target (não podemos prever evasão para quem já saiu/formou agora)
    current_status_terminal = s_norm.isin(STATUS_TERMINAIS)
    valid_mask = ~current_status_terminal
    
    df["is_target_valid"] = valid_mask
    
    # Limpeza de coluna temporária
    df.drop(columns=[temp_abs_col], inplace=True)
    
    return df
