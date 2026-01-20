import os
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


def load_model(artifact_path: str) -> Dict[str, Any]:
    return joblib.load(artifact_path)


def _risk_from_prob(prob: float) -> str:
    if prob >= 0.66:
        return "Alto"
    if prob >= 0.33:
        return "Médio"
    return "Baixo"


def predict_proba(model_bundle: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in model_bundle:
        raise ValueError("Model bundle must contain key 'model'.")
    model = model_bundle["model"]
    X = df.copy()
    if "situacao" in X.columns:
        X = X.drop(columns=["situacao"])
    probs = model.predict_proba(X)[:, 1]
    prob_series = pd.Series(probs, index=X.index, name="prob_evasao")
    class_pred = (prob_series >= 0.5).astype(int)
    risk_level = prob_series.map(_risk_from_prob)
    result = pd.DataFrame(
        {
            "prob_evasao": prob_series,
            "class_pred": class_pred,
            "risk_level": risk_level,
        }
    )
    return result


def _find_latest_artifact(artifacts_dir: str = "artifacts") -> str | None:
    root = Path(artifacts_dir)
    if not root.exists():
        return None
    candidates = list(root.rglob("*.pkl"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


if __name__ == "__main__":
    path = _find_latest_artifact()
    if not path:
        print("Nenhum artefato encontrado em 'artifacts'.")
    else:
        bundle = load_model(path)
        df_dummy = pd.DataFrame(
            [
                {
                    "sexo": "f",
                    "ano_ingresso": 2020,
                    "curso": "curso_teste",
                }
            ]
        )
        preds = predict_proba(bundle, df_dummy)
        print(f"Usando artefato: {os.path.basename(path)}")
        print(preds)

