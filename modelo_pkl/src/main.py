import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.trainer import train_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


REQUIRED_COLUMNS: Set[str] = {
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
}

CRITICAL_COLUMNS: Set[str] = {"situacao"}


def _build_config(config_str: Optional[str]) -> Dict[str, Any]:
    base_config: Dict[str, Any] = {}
    if config_str:
        try:
            user_cfg = json.loads(config_str)
            if isinstance(user_cfg, dict):
                base_config.update(user_cfg)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON config: {exc}")
    if "artifact_path" not in base_config:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        filename = f"model_{ts}.pkl"
        base_config["artifact_path"] = os.path.join("artifacts", filename)
    return base_config


@app.post("/train")
async def train_endpoint(
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")
    logger.info("Received training request for file %s", file.filename)
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        logger.exception("Failed reading CSV")
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {exc}")

    cols = set(df.columns)
    if "situacao" not in cols:
        raise HTTPException(
            status_code=400,
            detail="Coluna target 'situacao' obrigatória.",
        )
    missing_features = REQUIRED_COLUMNS - cols
    missing_features.discard("situacao")
    if missing_features:
        logger.warning(
            "Faltando colunas não críticas: %s. O modelo usará valores padrão/NaN.",
            sorted(missing_features),
        )

    try:
        cfg = _build_config(config)
        artifact_path = cfg["artifact_path"]
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        artifact = train_model(df, cfg)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception("Training error due to invalid data")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected training error")
        raise HTTPException(status_code=500, detail="Internal training error.") from exc

    metadata = artifact.get("metadata", {})
    logger.info(
        "Training completed: path=%s, metrics=%s",
        artifact_path,
        metadata.get("metrics"),
    )

    return {
        "artifact_path": artifact_path,
        "version": artifact.get("version"),
        "model_version": artifact.get("version"),
        "metrics": metadata.get("metrics"),
        "model_type": metadata.get("model_type"),
        "timestamp": metadata.get("timestamp"),
    }
