from pathlib import Path
from typing import Any
import logging

import joblib
import pandas as pd


logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = BASE_DIR / "src" / "models"
_MODEL_CACHE: dict[str, Any] = {}


try:
    import src.preprocessing  # type: ignore[import]  # noqa: F401
except Exception:
    logger.debug("Não foi possível importar src.preprocessing", exc_info=True)


def get_model(model_id: str) -> Any:
    if model_id in _MODEL_CACHE:
        logger.debug("Reutilizando modelo '%s' do cache", model_id)
        return _MODEL_CACHE[model_id]

    model_path = MODELS_DIR / f"{model_id}.pkl"
    if not model_path.is_file():
        logger.error("Modelo '%s' não encontrado em %s", model_id, model_path)
        raise FileNotFoundError(f"Modelo '{model_id}' não encontrado em '{model_path}'.")

    logger.info("Carregando modelo '%s' de %s", model_id, model_path)
    try:
        bundle = joblib.load(model_path)
    except Exception as exc:
        logger.exception("Falha ao carregar o artefato de modelo '%s'", model_id)
        raise RuntimeError(f"Falha ao carregar o artefato de modelo '{model_id}'.") from exc

    _MODEL_CACHE[model_id] = bundle
    return bundle


def predict(model_id: str, features: list | dict) -> list:
    bundle = get_model(model_id)

    if isinstance(bundle, dict):
        if "model" not in bundle:
            logger.error("Bundle de modelo '%s' não contém a chave 'model'", model_id)
            raise KeyError("Bundle de modelo não contém a chave 'model'.")
        model = bundle["model"]
    else:
        model = bundle

    if isinstance(features, dict):
        rows = [features]
    else:
        rows = features

    df = pd.DataFrame(rows)
    logger.debug(
        "Executando predict: model_id=%s, linhas=%d, colunas=%d",
        model_id,
        df.shape[0],
        df.shape[1],
    )

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)
            classes = getattr(model, "classes_", None)
            if classes is not None and 1 in classes:
                pos_index = list(classes).index(1)
            else:
                pos_index = proba.shape[1] - 1
            predictions = proba[:, pos_index]
        else:
            predictions = model.predict(df)
    except Exception as exc:
        logger.exception("Erro em predição para model_id=%s", model_id)
        raise RuntimeError(
            "Erro ao executar predição com o modelo fornecido."
        ) from exc

    return predictions.tolist()
