"""
api.py -- API de inferencia con FastAPI.

Ejecutar:
    uvicorn api:app --reload

Documentación automática:
    http://localhost:8000/docs

Requisitos previos:
    1. Ejecutar primero el pipeline:
       python main.py --input data/raw/Data_CU_venta.csv

    2. Verificar que existan:
       models/model.pkl
       models/feature_names.json

Endpoints:
    GET  /health
    POST /predict
    POST /predict-batch
"""

from pathlib import Path
from typing import Any, Dict, List

import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/model.pkl")
FEATURES_PATH = Path("models/feature_names.json")
METADATA_PATH = Path("data/processed/preprocessing_metadata.json")


app = FastAPI(
    title="API de Inferencia - Pipeline ML End-to-End",
    description=(
        "API para consumir el modelo entrenado del pipeline ML E2E. "
        "Permite obtener probabilidades de predicción para uno o varios registros."
    ),
    version="1.0.0",
)


class PredictRequest(BaseModel):
    """Entrada para inferencia de un único registro."""
    data: Dict[str, Any] = Field(
        ...,
        example={
            "p_codmes": 202412,
            "key_value": 100001,
            "monto": 1500.0,
            "prob_value_contact": 0.35,
            "grp_campecs06m": "G1",
        },
    )


class BatchPredictRequest(BaseModel):
    """Entrada para inferencia por lote."""
    records: List[Dict[str, Any]] = Field(
        ...,
        example=[
            {
                "p_codmes": 202412,
                "key_value": 100001,
                "monto": 1500.0,
                "prob_value_contact": 0.35,
                "grp_campecs06m": "G1",
            },
            {
                "p_codmes": 202412,
                "key_value": 100002,
                "monto": 900.0,
                "prob_value_contact": 0.18,
                "grp_campecs06m": "G3",
            },
        ],
    )


def load_json(path: Path, default):
    """Carga un JSON si existe; si no, devuelve un valor por defecto."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return default


def load_artifacts():
    """Carga modelo, variables y metadata de preprocesamiento."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró models/model.pkl. Ejecuta primero: python main.py"
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "No se encontró models/feature_names.json. Ejecuta primero: python main.py"
        )

    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    metadata = load_json(METADATA_PATH, default={})

    return model, feature_names, metadata


def preprocess_records(records: List[Dict[str, Any]], feature_names: List[str], metadata: Dict) -> pd.DataFrame:
    """
    Alinea los registros recibidos con las variables usadas por el modelo.

    La API soporta dos escenarios:
    1. El usuario envía variables ya numéricas/procesadas.
    2. El usuario envía algunas variables categóricas usadas en el entrenamiento.
       En ese caso, se intenta aplicar el mapeo generado en preprocessing_metadata.json.
    """
    df = pd.DataFrame(records)

    numeric_imputers = metadata.get("numeric_imputers", {})
    categorical_encoders = metadata.get("categorical_encoders", {})

    # Codificar categóricas conocidas con el orden aprendido por LabelEncoder.
    for col, classes in categorical_encoders.items():
        if col in df.columns:
            mapping = {value: idx for idx, value in enumerate(classes)}
            df[col] = df[col].astype(str).map(mapping).fillna(-1)

    # Imputar numéricas conocidas con mediana aprendida.
    for col, median_value in numeric_imputers.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median_value)

    # Crear columnas faltantes.
    for col in feature_names:
        if col not in df.columns:
            if col in numeric_imputers:
                df[col] = numeric_imputers[col]
            else:
                df[col] = 0

    # Eliminar columnas no requeridas y ordenar.
    df = df[feature_names]

    # Cualquier valor que siga no numérico se transforma a 0.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


@app.get("/health")
def health():
    """Verifica el estado de la API y la disponibilidad de artefactos."""
    model_exists = MODEL_PATH.exists()
    features_exists = FEATURES_PATH.exists()
    metadata_exists = METADATA_PATH.exists()

    return {
        "status": "ok" if model_exists and features_exists else "missing_artifacts",
        "model_exists": model_exists,
        "features_exists": features_exists,
        "metadata_exists": metadata_exists,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    """Realiza inferencia para un único registro."""
    try:
        model, feature_names, metadata = load_artifacts()
        X = preprocess_records([request.data], feature_names, metadata)
        probability = float(model.predict_proba(X)[:, 1][0])
        prediction = int(probability >= 0.5)

        return {
            "prediction": prediction,
            "probability": probability,
            "threshold": 0.5,
            "model": "cu_venta_xgb",
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict-batch")
def predict_batch(request: BatchPredictRequest):
    """Realiza inferencia para varios registros."""
    try:
        if not request.records:
            raise ValueError("La lista records no puede estar vacía.")

        model, feature_names, metadata = load_artifacts()
        X = preprocess_records(request.records, feature_names, metadata)
        probabilities = model.predict_proba(X)[:, 1]

        results = []
        for idx, probability in enumerate(probabilities):
            results.append(
                {
                    "row": idx,
                    "prediction": int(probability >= 0.5),
                    "probability": float(probability),
                    "threshold": 0.5,
                }
            )

        return {
            "model": "cu_venta_xgb",
            "n_records": len(results),
            "results": results,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
