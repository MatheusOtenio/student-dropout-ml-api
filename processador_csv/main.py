import io
import json
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import mapping_service
import etl_service


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    file_bytes = await file.read()
    buffer = io.BytesIO(file_bytes)
    df_header = pd.read_csv(buffer, nrows=0)
    colunas_csv = list(df_header.columns)
    sugestao = mapping_service.gerar_sugestao_mapeamento(colunas_csv)
    return sugestao


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

        csv_string_ml = df_ml.to_csv(index=False)
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
