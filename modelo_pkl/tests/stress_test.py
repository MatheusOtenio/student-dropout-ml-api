import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.main import app


client = TestClient(app)


def build_base_dataframe(
    n_samples: int = 200,
    imbalance_ratio: float = 0.7,
) -> pd.DataFrame:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not 0 < imbalance_ratio < 1:
        raise ValueError("imbalance_ratio must be between 0 and 1")

    cursos = ["engenharia", "direito", "medicina", "administracao", "pedagogia"]
    ufs = ["sp", "rj", "mg", "ba", "rs", "pr", "pe"]
    turnos = ["integral", "noturno", "matutino"]
    modalidades = ["ampla_concorrencia", "cotas", "transferencia"]
    tipos_cota = ["nenhuma", "renda", "etnia", "pcd"]

    n_negative = int(n_samples * imbalance_ratio)

    data = []
    for i in range(n_samples):
        is_negative = i < n_negative
        data.append(
            {
                "sexo": "f" if i % 2 == 0 else "m",
                "cor_raca": np.random.choice(
                    ["branca", "parda", "preta", "amarela", "indigena"]
                ),
                "municipio_residencia": f"cidade_{np.random.randint(0, 50)}",
                "uf_residencia": np.random.choice(ufs),
                "data_nascimento": f"{np.random.randint(1, 29):02d}/"
                f"{np.random.randint(1, 13):02d}/"
                f"{np.random.randint(1985, 2006)}",
                "curso": np.random.choice(cursos),
                "campus": f"campus_{np.random.randint(1, 5)}",
                "turno": np.random.choice(turnos),
                "modalidade_ingresso": np.random.choice(modalidades),
                "tipo_cota": np.random.choice(tipos_cota),
                "coeficiente_rendimento": float(
                    np.random.uniform(5.0, 9.5)
                ),
                "disciplinas_aprovadas": int(
                    np.random.randint(5, 25)
                ),
                "disciplinas_reprovadas_nota": int(
                    np.random.randint(0, 6)
                ),
                "disciplinas_reprovadas_frequencia": int(
                    np.random.randint(0, 3)
                ),
                "periodo": int(np.random.randint(1, 11)),
                "ano_ingresso": int(np.random.randint(2016, 2025)),
                "semestre_ingresso": int(np.random.randint(1, 3)),
                "nota_enem_humanas": int(
                    np.random.randint(400, 801)
                ),
                "nota_enem_linguagem": int(
                    np.random.randint(400, 801)
                ),
                "nota_enem_matematica": int(
                    np.random.randint(400, 801)
                ),
                "nota_enem_natureza": int(
                    np.random.randint(400, 801)
                ),
                "nota_enem_redacao": int(
                    np.random.randint(400, 801)
                ),
                "idade": int(np.random.randint(18, 36)),
                "situacao": "regular" if is_negative else "desistente",
            }
        )
    return pd.DataFrame(data)


def send_dataframe(df: pd.DataFrame):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    file_data = csv_buffer.getvalue()
    files = {
        "file": ("dummy.csv", file_data, "text/csv"),
    }
    config = {"cv_splits": 2, "calibrate": False}
    data = {"config": json.dumps(config)}
    response = client.post("/train", files=files, data=data)
    return response


def scenario_full_schema():
    df = build_base_dataframe()
    response = send_dataframe(df)
    print("Cenário 1 - schema completo:", response.status_code)
    payload = response.json()
    print("Métricas:", payload.get("metrics"))
    assert response.status_code == 200
    assert isinstance(payload.get("metrics"), dict)


def scenario_missing_columns():
    df = build_base_dataframe()
    df = df.drop(columns=["curso", "idade", "nota_enem_redacao"])
    response = send_dataframe(df)
    print("Cenário 2 - faltando colunas:", response.status_code)
    payload = response.json()
    print("Métricas:", payload.get("metrics"))
    assert response.status_code == 200
    assert isinstance(payload.get("metrics"), dict)


def scenario_missing_target():
    df = build_base_dataframe()
    df = df.drop(columns=["situacao"])
    response = send_dataframe(df)
    print("Cenário 3 - sem target:", response.status_code)
    try:
        payload = response.json()
    except Exception:
        payload = {}
    print("Resposta:", payload)
    assert response.status_code == 400


def scenario_realistic_volume():
    df = build_base_dataframe(n_samples=1000, imbalance_ratio=0.7)
    response = send_dataframe(df)
    print("Cenário 4 - volume realista:", response.status_code)
    payload = response.json()
    print("Métricas:", payload.get("metrics"))
    assert response.status_code == 200
    assert isinstance(payload.get("metrics"), dict)


def scenario_high_imbalance():
    df = build_base_dataframe(n_samples=1000, imbalance_ratio=0.9)
    response = send_dataframe(df)
    print("Cenário 5 - alto desbalanceamento:", response.status_code)
    payload = response.json()
    print("Métricas:", payload.get("metrics"))
    assert response.status_code == 200
    assert isinstance(payload.get("metrics"), dict)


def scenario_production_scale():
    df = build_base_dataframe(n_samples=2000, imbalance_ratio=0.8)
    response = send_dataframe(df)
    print("Cenário 6 - escala de produção:", response.status_code)
    payload = response.json()
    print("Métricas:", payload.get("metrics"))
    assert response.status_code == 200
    assert isinstance(payload.get("metrics"), dict)


def scenario_all_nulls_in_feature():
    df = build_base_dataframe(n_samples=50)
    df["nota_enem_humanas"] = np.nan
    response = send_dataframe(df)
    print("Cenário 7 - feature toda nula:", response.status_code)
    payload = response.json()
    print("Métricas:", payload.get("metrics"))
    assert response.status_code == 200
    assert isinstance(payload.get("metrics"), dict)


def scenario_invalid_situacao_values():
    df = build_base_dataframe(n_samples=50)
    df.loc[0:5, "situacao"] = "VALOR_INVALIDO"
    df.loc[6:10, "situacao"] = None
    df.loc[11:15, "situacao"] = ""
    response = send_dataframe(df)
    print("Cenário 8 - target com valores inválidos:", response.status_code)
    payload = response.json()
    print("Métricas:", payload.get("metrics"))
    assert response.status_code == 200
    artifact_path = payload.get("artifact_path")
    if artifact_path:
        artifact = joblib.load(artifact_path)
        metadata = artifact.get("metadata", {})
        n_samples = metadata.get("n_samples", 50)
        print("n_samples usados no treino:", n_samples)
        assert n_samples < 50


def scenario_duplicate_rows():
    df = build_base_dataframe(n_samples=50)
    df_duplicated = pd.concat([df] * 10, ignore_index=True)
    response = send_dataframe(df_duplicated)
    print("Cenário 9 - muitas linhas duplicadas:", response.status_code)
    payload = response.json()
    metrics = payload.get("metrics") or {}
    print("Métricas:", metrics)
    assert response.status_code == 200
    roc_auc = metrics.get("roc_auc")
    brier = metrics.get("brier_score")
    assert roc_auc is not None and 0.0 <= roc_auc <= 1.0
    assert brier is not None and 0.0 <= brier <= 1.0


def scenario_high_cardinality_categorical():
    df = build_base_dataframe(n_samples=100)
    df["municipio_residencia"] = [f"cidade_unica_{i}" for i in range(100)]
    response = send_dataframe(df)
    print("Cenário 10 - alta cardinalidade categórica:", response.status_code)
    payload = response.json()
    print("Métricas:", payload.get("metrics"))
    assert response.status_code == 200
    assert isinstance(payload.get("metrics"), dict)


def scenario_extreme_class_imbalance():
    df = build_base_dataframe(n_samples=1000)
    df.loc[:989, "situacao"] = "regular"
    df.loc[990:, "situacao"] = "desistente"
    response = send_dataframe(df)
    print("Cenário 11 - desbalanceamento extremo:", response.status_code)
    payload = response.json()
    metrics = payload.get("metrics") or {}
    print("Métricas:", metrics)
    assert response.status_code == 200
    roc_auc = metrics.get("roc_auc")
    assert roc_auc is not None and roc_auc > 0.5


def scenario_missing_multiple_features():
    df = build_base_dataframe(n_samples=50)
    df = df.drop(
        columns=[
            "curso",
            "idade",
            "nota_enem_humanas",
            "nota_enem_redacao",
            "campus",
        ]
    )
    response = send_dataframe(df)
    print("Cenário 12 - múltiplas features ausentes:", response.status_code)
    payload = response.json()
    print("Métricas:", payload.get("metrics"))
    assert response.status_code == 200
    assert isinstance(payload.get("metrics"), dict)


def scenario_enem_only():
    df = build_base_dataframe(n_samples=200)
    for col in list(df.columns):
        if col.startswith("nota_vestibular_"):
            df[col] = 0.0
    response = send_dataframe(df)
    print("Cenário 13 - apenas ENEM preenchido:", response.status_code)
    payload = response.json()
    metrics = payload.get("metrics") or {}
    print("Métricas:", metrics)
    assert response.status_code == 200
    assert isinstance(metrics, dict)
    roc_auc = metrics.get("roc_auc")
    brier = metrics.get("brier_score")
    assert roc_auc is not None and 0.0 <= roc_auc <= 1.0
    assert brier is not None and 0.0 <= brier <= 1.0


def scenario_vest_only():
    df = build_base_dataframe(n_samples=200)
    enem_cols = [c for c in df.columns if c.startswith("nota_enem_")]
    df = df.drop(columns=enem_cols)
    df["nota_vestibular_matematica"] = np.random.uniform(5.0, 10.0, size=len(df))
    df["nota_vestibular_lingua_portuguesa"] = np.random.uniform(5.0, 10.0, size=len(df))
    df["nota_vestibular_biologia"] = np.random.uniform(5.0, 10.0, size=len(df))
    response = send_dataframe(df)
    print("Cenário 14 - apenas VESTIBULAR preenchido:", response.status_code)
    payload = response.json()
    metrics = payload.get("metrics") or {}
    print("Métricas:", metrics)
    assert response.status_code == 200
    assert isinstance(metrics, dict)
    roc_auc = metrics.get("roc_auc")
    brier = metrics.get("brier_score")
    assert roc_auc is not None and 0.0 <= roc_auc <= 1.0
    assert brier is not None and 0.0 <= brier <= 1.0


def scenario_mixed_enem_vest():
    df = build_base_dataframe(n_samples=200)
    df["nota_vestibular_matematica"] = np.random.uniform(5.0, 10.0, size=len(df))
    df["nota_vestibular_lingua_portuguesa"] = np.random.uniform(5.0, 10.0, size=len(df))
    response = send_dataframe(df)
    print("Cenário 15 - ENEM e VESTIBULAR preenchidos:", response.status_code)
    payload = response.json()
    metrics = payload.get("metrics") or {}
    print("Métricas:", metrics)
    assert response.status_code == 200
    assert isinstance(metrics, dict)
    roc_auc = metrics.get("roc_auc")
    brier = metrics.get("brier_score")
    assert roc_auc is not None and 0.0 <= roc_auc <= 1.0
    assert brier is not None and 0.0 <= brier <= 1.0


def main():
    scenario_full_schema()
    scenario_missing_columns()
    scenario_missing_target()
    scenario_realistic_volume()
    scenario_high_imbalance()
    scenario_production_scale()
    scenario_all_nulls_in_feature()
    scenario_invalid_situacao_values()
    scenario_duplicate_rows()
    scenario_high_cardinality_categorical()
    scenario_extreme_class_imbalance()
    scenario_missing_multiple_features()
    scenario_enem_only()
    scenario_vest_only()
    scenario_mixed_enem_vest()
    print("Stress test concluído com sucesso.")


if __name__ == "__main__":
    main()
