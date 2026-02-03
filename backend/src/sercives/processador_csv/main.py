import io
import json
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.sercives.processador_csv import mapping_service, etl_service  # TODO: Fix typo.
from src.sercives.processador_csv.etl_service import preparar_para_predict
from src.backend_logic import predict                                   # ajuste o import se necessário


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# 1) Analisa o CSV e retorna sugestões de mapeamento para o frontend.
# --------------------------------------------------------------------------
@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    file_bytes = await file.read()
    buffer = io.BytesIO(file_bytes)
    df_header = pd.read_csv(buffer, nrows=0)
    colunas_csv = list(df_header.columns)
    sugestao = mapping_service.gerar_sugestao_mapeamento(colunas_csv)
    return sugestao


# --------------------------------------------------------------------------
# 2) Processa o CSV com o mapeamento confirmado pelo usuário e retorna
#    os dados limpos + os descartados (para download / auditoria).
# --------------------------------------------------------------------------
@app.post("/process-csv")
async def process_csv(
    file: UploadFile = File(...),
    mapping: str = Form(...),
) -> Dict[str, Any]:
    try:
        mapa_de_para: Dict[str, Any] = json.loads(mapping)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="JSON de mapeamento inválido")

    try:
        file_bytes = await file.read()
        buffer = io.BytesIO(file_bytes)
        df_bruto = pd.read_csv(buffer)

        df_ml, df_audit = etl_service.transformar_dados(df_bruto, mapa_de_para)

        csv_string_ml    = df_ml.to_csv(index=False)
        csv_string_audit = df_audit.to_csv(index=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "success",
        "clean_data": csv_string_ml,
        "dropped_data": csv_string_audit,
        "filename_clean": "dados_ml_padronizados.csv",
        "filename_dropped": "colunas_descartadas.csv",
    }


# --------------------------------------------------------------------------
# 3) Processa o CSV E executa a predição numa única chamada.
#    Fluxo: CSV bruto → mapeamento → ETL → preparar_para_predict → predict()
# --------------------------------------------------------------------------
@app.post("/predict-csv")
async def predict_csv(
    file: UploadFile = File(...),
    mapping: str = Form(...),
    model_id: str = Form(...),
) -> Dict[str, Any]:
    try:
        mapa_de_para: Dict[str, Any] = json.loads(mapping)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="JSON de mapeamento inválido")

    try:
        file_bytes = await file.read()
        buffer    = io.BytesIO(file_bytes)
        df_bruto  = pd.read_csv(buffer)

        # ETL
        df_ml, _ = etl_service.transformar_dados(df_bruto, mapa_de_para)

        # Separa situacao (label) antes de passar ao modelo
        situacoes = df_ml["situacao"].tolist() if "situacao" in df_ml.columns else []

        # Prepara features na ordem exata que o modelo espera
        df_predict = preparar_para_predict(df_ml)

        # Predição
        predictions = predict(model_id, df_predict)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "success",
        "predictions": predictions,
        "situacoes_originais": situacoes,
        "n_registros": len(predictions),
    }