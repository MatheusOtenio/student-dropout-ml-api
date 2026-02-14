import io
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.sercives.processador_csv import mapping_service
from src.sercives.processador_csv.etl_service import process_csv, transformar_dados, preparar_para_predict
from src.sercives.predicao_ml.backend_logic import predict, get_model


logger = logging.getLogger(__name__)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/mapping-suggestions")
async def mapping_suggestions(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    logger.info(
        "Requisição /mapping-suggestions recebida: file=%s",
        getattr(file, "filename", None),
    )
    try:
        file_bytes = await file.read()
        if not file_bytes:
            logger.warning("Arquivo vazio recebido em /mapping-suggestions")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Arquivo vazio.",
            )

        buffer = io.BytesIO(file_bytes)
        try:
            df_header = pd.read_csv(buffer, nrows=0)
        except Exception as exc:
            logger.warning("Falha ao ler cabeçalho do CSV em /mapping-suggestions: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV inválido.",
            ) from exc

        colunas_csv = list(df_header.columns)
        sugestao = mapping_service.gerar_sugestao_mapeamento(colunas_csv)
        logger.info(
            "Sugestão de mapeamento gerada: colunas=%d",
            len(colunas_csv),
        )
        return {
            "columns_csv": colunas_csv,
            "mapping": sugestao,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erro interno ao processar requisição em /mapping-suggestions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar requisição.",
        ) from exc


@app.post("/preview-mapped-csv")
async def preview_mapped_csv(
    file: UploadFile = File(...),
    mapping: str = Form(...),
) -> dict[str, Any]:
    logger.info(
        "Requisição /preview-mapped-csv recebida: file=%s",
        getattr(file, "filename", None),
    )
    try:
        file_bytes = await file.read()
        if not file_bytes:
            logger.warning("Arquivo vazio recebido em /preview-mapped-csv")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Arquivo vazio.",
            )

        try:
            mapa = json.loads(mapping)
        except json.JSONDecodeError as exc:
            logger.warning("JSON de mapeamento inválido em /preview-mapped-csv: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON de mapeamento inválido.",
            ) from exc

        buffer = io.BytesIO(file_bytes)
        try:
            df_bruto = pd.read_csv(buffer, low_memory=False)
        except Exception as exc:
            logger.warning("Falha ao ler CSV em /preview-mapped-csv: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV inválido.",
            ) from exc

        try:
            df_ml, _ = transformar_dados(df_bruto, mapa)
        except Exception as exc:
            logger.warning("Falha no ETL em /preview-mapped-csv: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Erro ao transformar dados.",
            ) from exc

        rows = df_ml.to_dict(orient="records")
        logger.info(
            "Pré-visualização com mapeamento concluída: linhas=%d",
            len(rows),
        )
        return {"rows": rows}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erro interno ao processar requisição em /preview-mapped-csv")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar requisição.",
        ) from exc


@app.post("/preview-csv")
async def preview_csv(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    logger.info(
        "Requisição /preview-csv recebida: file=%s",
        getattr(file, "filename", None),
    )
    try:
        file_bytes = await file.read()
        if not file_bytes:
            logger.warning("Arquivo vazio recebido em /preview-csv")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Arquivo vazio.",
            )

        buffer = io.BytesIO(file_bytes)

        try:
            df = process_csv(buffer)
        except ValueError as exc:
            logger.warning("Falha no ETL do CSV em /preview-csv: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        rows = df.to_dict(orient="records")
        logger.info(
            "Pré-visualização concluída com sucesso: linhas=%d",
            len(rows),
        )
        return {"rows": rows}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erro interno ao processar requisição em /preview-csv")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar requisição.",
        ) from exc


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    mapping: str | None = Form(None),
) -> dict[str, Any]:
    logger.info(
        "Requisição /predict recebida: file=%s, model_id=%s",
        getattr(file, "filename", None),
        model_id,
    )
    try:
        file_bytes = await file.read()
        if not file_bytes:
            logger.warning("Arquivo vazio recebido em /predict")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Arquivo vazio.",
            )

        buffer = io.BytesIO(file_bytes)

        if mapping is not None and mapping.strip():
            try:
                mapa = json.loads(mapping)
            except json.JSONDecodeError as exc:
                logger.warning("JSON de mapeamento inválido em /predict: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="JSON de mapeamento inválido.",
                ) from exc

            try:
                df_bruto = pd.read_csv(buffer, low_memory=False)
            except Exception as exc:
                logger.warning("Falha ao ler CSV em /predict: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="CSV inválido.",
                ) from exc

            try:
                df_ml, _ = transformar_dados(df_bruto, mapa)
            except Exception as exc:
                logger.warning("Falha no ETL em /predict: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Erro ao transformar dados.",
                ) from exc

            df_features = preparar_para_predict(df_ml)
            features = df_features.to_dict(orient="records")
            rows_for_response = df_ml.to_dict(orient="records")
        else:
            try:
                df = process_csv(buffer)
            except ValueError as exc:
                logger.warning("Falha no ETL do CSV: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc

            features = df.to_dict(orient="records")
            rows_for_response = features

        try:
            predictions = predict(model_id, features)
        except FileNotFoundError as exc:
            logger.error("Modelo não encontrado: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Erro ao executar predição para model_id=%s", model_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Erro ao executar predição.",
            ) from exc

        logger.info(
            "Predição concluída com sucesso: model_id=%s, linhas=%d",
            model_id,
            len(features),
        )
        return {
            "model_id": model_id,
            "rows": rows_for_response,
            "predictions": predictions,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erro interno ao processar requisição em /predict")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar requisição.",
        ) from exc


@app.get("/model/importance")
async def model_importance(model_id: str) -> dict[str, Any]:
    logger.info("Requisição /model/importance recebida para model_id=%s", model_id)
    try:
        bundle = get_model(model_id)
        
        # 1. Tenta buscar pronta no metadata (Ideal, igual ao modelo_pkl)
        if isinstance(bundle, dict) and "metadata" in bundle:
            imp = bundle["metadata"].get("feature_importance")
            if imp and "mapped" in imp and imp["mapped"]:
                return imp

        # 2. Fallback simplificado (apenas retorna raw se não encontrar metadata)
        return {"error": "Feature importance não disponível nos metadados deste modelo."}

    except Exception as exc:
        logger.exception("Erro em /model/importance")
        raise HTTPException(status_code=500, detail=str(exc))
