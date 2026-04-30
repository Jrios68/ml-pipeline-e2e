"""
api.py -- API de inferencia con FastAPI.

Ejecutar:
    uvicorn api:app --reload

Documentación automática:
    http://localhost:8000/docs

Requisitos previos:
    1. Ejecutar primero el pipeline con la data real:
       python main.py --input "data/raw" --n-trials 5

    2. Verificar que existan:
       models/model.pkl
       models/feature_names.json
       data/processed/preprocessing_metadata.json

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
        "La versión final trabaja con archivos p1_extrac.csv a p10_extrac.csv "
        "ubicados en data/raw/ y permite obtener probabilidades de predicción "
        "para uno o varios registros."
    ),
    version="1.0.0",
)


class PredictRequest(BaseModel):
    """Entrada para inferencia de un único registro."""
    data: Dict[str, Any] = Field(
        ...,
        example={
            "partition": "202403",
            "tip_doc": "1",
            "key_value": 100001,
            "monto": 1500.0,
            "prob_value_contact": 0.35,
            "grp_campecs06m": "G1",
            "tea": 15.5,
            "ingreso_neto": 3500.0,
        },
    )


class BatchPredictRequest(BaseModel):
    """Entrada para inferencia por lote."""
    records: List[Dict[str, Any]] = Field(
        ...,
        example=[
            {
                "partition": "202403",
                "tip_doc": "1",
                "key_value": 100001,
                "monto": 1500.0,
                "prob_value_contact": 0.35,
                "grp_campecs06m": "G1",
                "tea": 15.5,
                "ingreso_neto": 3500.0,
            },
            {
                "partition": "202403",
                "tip_doc": "1",
                "key_value": 100002,
                "monto": 900.0,
                "prob_value_contact": 0.18,
                "grp_campecs06m": "G3",
                "tea": 20.0,
                "ingreso_neto": 2800.0,
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
            'No se encontró models/model.pkl. Ejecuta primero: python main.py --input "data/raw" --n-trials 5'
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            'No se encontró models/feature_names.json. Ejecuta primero: python main.py --input "data/raw" --n-trials 5'
        )

    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    metadata = load_json(METADATA_PATH, default={})

    return model, feature_names, metadata


def _extract_codmes(series: pd.Series) -> pd.Series:
    """
    Extrae codmes YYYYMM desde formatos como:
    - 202403
    - 2024-03-01
    - 2024/03/01
    - partition=202403
    """
    text = series.astype(str).str.strip()

    fechas = pd.to_datetime(text, errors="coerce", dayfirst=False)
    codmes_fecha = fechas.dt.year * 100 + fechas.dt.month

    digits = text.str.replace(r"\D", "", regex=True)
    codmes_texto = digits.str.extract(r"((?:19|20)\d{4})")[0]
    codmes_texto = pd.to_numeric(codmes_texto, errors="coerce")

    codmes = codmes_fecha.fillna(codmes_texto)

    return pd.to_numeric(codmes, errors="coerce")


def standardize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza columnas recibidas por la API para alinearlas con el preprocesamiento.

    - tip_doc se convierte en tipdoc.
    - p_codmes se crea desde partition o p_fecinformacion cuando aplique.
    """
    df = df.copy()

    if "tipdoc" not in df.columns and "tip_doc" in df.columns:
        df = df.rename(columns={"tip_doc": "tipdoc"})

    if "p_codmes" not in df.columns:
        if "partition" in df.columns:
            df["p_codmes"] = _extract_codmes(df["partition"])
        elif "p_fecinformacion" in df.columns:
            df["p_codmes"] = _extract_codmes(df["p_fecinformacion"])

    return df


def preprocess_records(
    records: List[Dict[str, Any]],
    feature_names: List[str],
    metadata: Dict,
) -> pd.DataFrame:
    """
    Alinea los registros recibidos con las variables usadas por el modelo.

    La API soporta dos escenarios:
    1. El usuario envía variables ya numéricas/procesadas.
    2. El usuario envía variables crudas similares a la data real.
       En ese caso, se aplican estandarizaciones básicas y los encoders
       guardados en preprocessing_metadata.json.
    """
    df = pd.DataFrame(records)
    df = standardize_input_columns(df)

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

    # Crear columnas faltantes requeridas por el modelo.
    for col in feature_names:
        if col not in df.columns:
            if col in numeric_imputers:
                df[col] = numeric_imputers[col]
            else:
                df[col] = 0

    # Eliminar columnas no requeridas y ordenar según el entrenamiento.
    df = df[feature_names]

    # Asegurar que todo sea numérico para XGBoost.
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
