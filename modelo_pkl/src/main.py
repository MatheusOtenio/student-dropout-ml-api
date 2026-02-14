import io
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.modelo_regressao import train_model


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


def _log_dataset_stats(df: pd.DataFrame, request_id: Optional[str] = None) -> None:
    n_rows, n_cols = df.shape
    prefix = f"[request_id={request_id}] " if request_id is not None else ""
    logger.info(
        "%sDataset shape: rows=%d, cols=%d",
        prefix,
        n_rows,
        n_cols,
    )
    missing_fraction = df.isna().mean(numeric_only=False)
    high_missing = missing_fraction[missing_fraction > 0.3].sort_values(ascending=False)
    if not high_missing.empty:
        logger.warning(
            "%sHigh missing rate columns: %s",
            prefix,
            high_missing.to_dict(),
        )
    if "situacao" in df.columns:
        counts = df["situacao"].value_counts(dropna=False).to_dict()
        logger.info(
            "%sTarget distribution 'situacao': %s",
            prefix,
            counts,
        )


def get_model_artifact(model_path_or_id: str) -> Any:
    """Carrega o artefato do modelo do disco."""
    # Se vier com artifacts/ prefixo, usa direto. Se não, assume que está em artifacts/
    if model_path_or_id.startswith("artifacts") or "/" in model_path_or_id:
        path = model_path_or_id
    else:
        path = os.path.join("artifacts", model_path_or_id)
    
    if not path.endswith(".pkl"):
        path += ".pkl"
        
    if not os.path.exists(path):
        # Tenta sem o path relativo se falhar (caso path absoluto ou relativo diferente)
        if os.path.exists(os.path.basename(path)):
             path = os.path.basename(path)
        # Tenta verificar se existe dentro de artifacts/ mesmo que o input tenha paths estranhos
        elif os.path.exists(os.path.join("artifacts", os.path.basename(path))):
            path = os.path.join("artifacts", os.path.basename(path))
        else:
            raise HTTPException(status_code=404, detail=f"Modelo não encontrado: {path}")
            
    try:
        return joblib.load(path)
    except Exception as e:
        logger.error(f"Erro ao carregar modelo {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao carregar modelo: {e}")


def _aggregate_shap_values(shap_values, feature_names, original_columns, feature_map=None):
    """Agrega SHAP values de colunas One-Hot de volta para a feature original."""
    df_shap = pd.DataFrame([shap_values], columns=feature_names)
    aggregated = {}
    
    if feature_map:
        # Uso do feature_map salvo no metadata (Mais preciso)
        for orig_col, indices in feature_map.items():
            valid_indices = [i for i in indices if 0 <= i < len(feature_names)]
            if valid_indices:
                cols_to_sum = [feature_names[i] for i in valid_indices]
                aggregated[orig_col] = df_shap[cols_to_sum].sum(axis=1).values[0]
            else:
                aggregated[orig_col] = 0.0
                
        # Garante que todas as colunas originais tenham entrada (mesmo que 0)
        for col in original_columns:
            if col not in aggregated:
                aggregated[col] = 0.0
    else:
        # Fallback: Heurística de nomes
        for col in original_columns:
            related_cols = [f for f in feature_names if f.startswith(f"{col}_") or f == col]
            if related_cols:
                aggregated[col] = df_shap[related_cols].sum(axis=1).values[0]
            else:
                aggregated[col] = 0.0
                
    return aggregated


@app.get("/model/importance")
async def get_model_importance(model_id: str) -> Dict[str, Any]:
    """Retorna a feature importance global salva nos metadados do modelo."""
    bundle = get_model_artifact(model_id)
    
    # 1. Tenta buscar pronta no metadata (Ideal)
    if isinstance(bundle, dict) and "metadata" in bundle:
        imp = bundle["metadata"].get("feature_importance")
        # Se tiver mapeado, retorna direto
        if imp and "mapped" in imp and imp["mapped"]:
            return imp
            
    # 2. Fallback: Tenta reconstruir do modelo carregado
    model_obj = bundle["model"] if isinstance(bundle, dict) else bundle
    
    # Tenta extrair o passo do modelo final
    if hasattr(model_obj, "named_steps"):
        step = model_obj.named_steps.get("model") or model_obj.named_steps.get("classifier")
        # Tenta extrair nomes das features do preprocessor
        feature_names = None
        try:
            preprocessor = model_obj.named_steps.get("preprocess")
            if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
                # Pipeline aninhada?
                if hasattr(preprocessor, "named_steps") and "preprocess" in preprocessor.named_steps:
                     feature_names = preprocessor.named_steps["preprocess"].get_feature_names_out()
                else:
                     feature_names = preprocessor.get_feature_names_out()
        except Exception:
            pass
    else:
        step = model_obj
        feature_names = None

    # Extrai valores brutos
    raw_values = []
    if hasattr(step, "feature_importances_"):
        raw_values = step.feature_importances_.tolist()
    elif hasattr(step, "coef_"):
        raw_values = step.coef_[0].tolist()
    else:
        raise HTTPException(status_code=404, detail="Feature importance não encontrada.")

    # Se conseguiu nomes e tamanhos batem, mapeia
    if feature_names is not None and len(feature_names) == len(raw_values):
        mapped = [
            {"feature": str(name), "importance": float(val)}
            for name, val in zip(feature_names, raw_values)
        ]
        # Ordena por importância absoluta
        mapped.sort(key=lambda x: abs(x["importance"]), reverse=True)
        return {"mapped": mapped, "raw_values": raw_values}
        
    # Se não, retorna bruto com aviso
    return {"raw_values": raw_values}


@app.post("/train")
async def train_endpoint(
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
):
    request_id = uuid.uuid4().hex
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")
    logger.info(
        "[request_id=%s] Received training request for file %s",
        request_id,
        file.filename,
    )
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        logger.exception("[request_id=%s] Failed reading CSV", request_id)
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {exc}")

    _log_dataset_stats(df, request_id=request_id)

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
            "[request_id=%s] Faltando colunas não críticas: %s. O modelo usará valores padrão/NaN.",
            request_id,
            sorted(missing_features),
        )

    start_time = datetime.now(timezone.utc)
    try:
        cfg = _build_config(config)
        artifact_path = cfg["artifact_path"]
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        artifact = train_model(df, cfg)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception("[request_id=%s] Training error due to invalid data", request_id)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("[request_id=%s] Unexpected training error", request_id)
        raise HTTPException(status_code=500, detail="Internal training error.") from exc

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()

    metadata = artifact.get("metadata", {})
    logger.info(
        "[request_id=%s] Training completed: path=%s, metrics=%s, duration_s=%.3f",
        request_id,
        artifact_path,
        metadata.get("metrics"),
        duration,
    )

    return {
        "artifact_path": artifact_path,
        "version": artifact.get("version"),
        "model_version": artifact.get("version"),
        "metrics": metadata.get("metrics"),
        "model_type": metadata.get("model_type"),
        "timestamp": metadata.get("timestamp"),
    }
