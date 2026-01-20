import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


_THIS_DIR = Path(__file__).resolve().parent
_MODELO_PKL_ROOT = _THIS_DIR.parent / "modelo_pkl"
if _MODELO_PKL_ROOT.exists():
    root_str = str(_MODELO_PKL_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

try:
    import src.preprocessing  # type: ignore[import]  # noqa: F401
except Exception:
    pass


def _risk_from_prob(prob: float) -> str:
    if prob >= 0.66:
        return "Alto"
    if prob >= 0.33:
        return "Médio"
    return "Baixo"


def run_prediction(model_path, csv_file):
    try:
        bundle = joblib.load(model_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Modelo não encontrado em '{model_path}'."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Falha ao carregar o artefato de modelo de '{model_path}': {e}"
        ) from e

    if not isinstance(bundle, dict):
        raise ValueError(
            "Artefato de modelo inválido: esperado dicionário com chave 'model'."
        )

    if "model" not in bundle:
        raise KeyError("Bundle de modelo não contém a chave 'model'.")

    model = bundle["model"]

    try:
        if hasattr(csv_file, "read"):
            if hasattr(csv_file, "seek"):
                csv_file.seek(0)
            df = pd.read_csv(csv_file)
        else:
            df = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Erro ao ler o CSV para predição: {e}") from e

    X = df.copy()
    if "situacao" in X.columns:
        X = X.drop(columns=["situacao"])

    try:
        proba = model.predict_proba(X)
    except Exception as e:
        raise RuntimeError(
            "Erro ao executar predição com o modelo fornecido. "
            "Verifique se o CSV contém as colunas esperadas pelo pipeline de treinamento. "
            f"Detalhes: {e}"
        ) from e

    proba = np.asarray(proba)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(
            "Saída de predict_proba inesperada. "
            "Esperado array 2D com probabilidade para a classe 1 na coluna 1."
        )

    prob_evasao = proba[:, 1]
    prob_series = pd.Series(prob_evasao, index=X.index, name="prob_evasao")
    class_pred = (prob_series >= 0.5).astype(int)
    risk_level = prob_series.map(_risk_from_prob)

    result = df.copy()
    result["prob_evasao"] = prob_series
    result["class_pred"] = class_pred
    result["risk_level"] = risk_level
    return result
