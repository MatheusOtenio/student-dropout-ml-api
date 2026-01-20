import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.inference import _find_latest_artifact, load_model


def _build_dummy_row() -> Dict[str, Any]:
    return {
        "situacao": "regular",
        "sexo": "f",
        "cor_raca": "branca",
        "municipio_residencia": "cidade_a",
        "uf_residencia": "sp",
        "data_nascimento": "18/09/1992",
        "idade": 26,
        "curso": "engenharia",
        "campus": "campus_a",
        "turno": "integral",
        "modalidade_ingresso": "ampla_concorrencia",
        "tipo_cota": "nenhuma",
        "coeficiente_rendimento": 7.5,
        "disciplinas_aprovadas": 10,
        "disciplinas_reprovadas_nota": 2,
        "disciplinas_reprovadas_frequencia": 1,
        "periodo": 8,
        "ano_ingresso": 2018,
        "semestre_ingresso": 1,
        "nota_enem_humanas": 600,
        "nota_enem_linguagem": 650,
        "nota_enem_matematica": 700,
        "nota_enem_natureza": 620,
        "nota_enem_redacao": 720,
    }


def validate_latest_artifact() -> int:
    path = _find_latest_artifact()
    if not path:
        print("Nenhum artefato encontrado em 'artifacts'.")
        return 1

    bundle = load_model(path)
    if not isinstance(bundle, dict):
        print("Artefato inválido: objeto raiz não é um dict.")
        return 1

    for key in ("preprocessor", "model", "metadata"):
        if key not in bundle:
            print(f"Artefato inválido: chave ausente: {key}")
            return 1

    metadata = bundle.get("metadata", {})
    required_keys = ["model_type", "metrics", "timestamp", "n_samples", "n_features"]
    missing = [k for k in required_keys if k not in metadata]
    if missing:
        print(f"⚠️  Metadata incompleto. Faltando: {missing}")

    metrics = metadata.get("metrics", {})
    if "roc_auc" in metrics:
        if not (0 <= metrics["roc_auc"] <= 1):
            print(f"❌ ROC-AUC inválido: {metrics['roc_auc']}")
            return 1

    model = bundle["model"]
    df = pd.DataFrame([_build_dummy_row()])

    try:
        proba = model.predict_proba(df)
    except Exception as exc:
        print(f"Falha ao executar predict_proba: {exc}")
        return 1

    if proba is None:
        print("predict_proba retornou None.")
        return 1

    if len(proba.shape) != 2 or proba.shape[0] != 1 or proba.shape[1] < 2:
        print(f"Saída de predict_proba com shape inesperado: {proba.shape}")
        return 1

    prob_row = proba[0]
    prob_sum = float(prob_row.sum())
    if not (0.99 <= prob_sum <= 1.01):
        print(f"❌ Probabilidades não somam 1.0: {prob_sum}")
        return 1

    if not np.all((prob_row >= 0) & (prob_row <= 1)):
        print(f"❌ Probabilidades fora do range [0,1]: {prob_row}")
        return 1

    proba2 = model.predict_proba(df)
    if not np.allclose(proba, proba2, atol=1e-6):
        print("⚠️  AVISO: Modelo não-determinístico!")

    df_multi = pd.DataFrame([_build_dummy_row() for _ in range(100)])
    proba_multi = model.predict_proba(df_multi)

    if len(proba_multi.shape) != 2 or proba_multi.shape[0] != 100:
        print(f"❌ Batch prediction falhou: esperado (100, n_classes), obtido {proba_multi.shape}")
        return 1

    for i, prob_row_multi in enumerate(proba_multi):
        row_sum = float(prob_row_multi.sum())
        if not (0.99 <= row_sum <= 1.01):
            print(f"❌ Amostra {i}: probabilidades não somam 1.0")
            return 1

    roc_auc_value = metrics.get("roc_auc")
    brier_value = metrics.get("brier_score")
    if isinstance(roc_auc_value, (int, float)):
        roc_auc_str = f"{roc_auc_value:.4f}"
    else:
        roc_auc_str = "N/A"
    if isinstance(brier_value, (int, float)):
        brier_str = f"{brier_value:.4f}"
    else:
        brier_str = "N/A"

    print("✅ ARTEFATO VÁLIDO")
    print(f"├─ Caminho: {os.path.basename(path)}")
    print(f"├─ Probabilidade classe 0: {proba[0][0]:.4f}")
    print(f"├─ Probabilidade classe 1: {proba[0][1]:.4f}")
    print(f"├─ ROC-AUC: {roc_auc_str}")
    print(f"├─ Brier Score: {brier_str}")
    print(f"├─ Amostras treinadas: {metadata.get('n_samples', 'N/A')}")
    print(f"└─ Features: {metadata.get('n_features', 'N/A')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate_latest_artifact())
