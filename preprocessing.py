"""
preprocessing.py -- Limpieza y transformación del dataset.

Produce:
    data/processed/df_train.csv
    data/processed/df_test.csv
    data/processed/df_val.csv

Principales pasos:
    - Lectura del CSV de entrada
    - Eliminación de columnas con exceso de valores nulos
    - Imputación de variables numéricas y categóricas
    - Codificación simple de variables categóricas
    - Split temporal para validación
    - Split aleatorio para entrenamiento y prueba
"""

from pathlib import Path
from typing import Dict, Tuple

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


NAN_THRESHOLD = 80
TEST_SIZE = 0.30
RANDOM_STATE = 123
TARGET_COL = "target"
DATE_COL = "p_codmes"
ID_COL = "key_value"


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Valida que existan las columnas mínimas requeridas."""
    required = [TARGET_COL, DATE_COL]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"El dataset no contiene las columnas obligatorias: {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )


def _select_validation_month(df: pd.DataFrame, validation_codmes=None) -> float:
    """
    Selecciona el mes de validación.

    Si validation_codmes es None, se toma el último p_codmes disponible.
    """
    codmes_values = sorted(df[DATE_COL].dropna().unique())

    if not codmes_values:
        raise ValueError("La columna p_codmes no contiene valores válidos.")

    if validation_codmes is not None:
        if validation_codmes not in codmes_values:
            raise ValueError(
                f"El validation_codmes={validation_codmes} no existe en el dataset. "
                f"Valores disponibles: {codmes_values[:10]} ... {codmes_values[-10:]}"
            )
        return validation_codmes

    return codmes_values[-1]


def _impute_and_encode(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Imputa valores faltantes y codifica variables categóricas.

    Las variables numéricas se imputan con mediana.
    Las variables categóricas se imputan con 'missing' y se codifican con LabelEncoder.
    """
    df = df.copy()
    metadata = {
        "numeric_imputers": {},
        "categorical_encoders": {},
        "categorical_columns": [],
        "numeric_columns": [],
    }

    for col in df.columns:
        if col == TARGET_COL:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            median_value = df[col].median()
            if pd.isna(median_value):
                median_value = 0
            df[col] = df[col].fillna(median_value)
            metadata["numeric_imputers"][col] = float(median_value)
            metadata["numeric_columns"].append(col)
        else:
            df[col] = df[col].astype(str).fillna("missing")
            df[col] = df[col].replace({"nan": "missing", "None": "missing"})

            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])

            metadata["categorical_encoders"][col] = list(encoder.classes_)
            metadata["categorical_columns"].append(col)

    df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)

    return df, metadata


def run_preprocessing(
    data_path: str,
    output_dir: str = "data/processed",
    nan_threshold: int = NAN_THRESHOLD,
    validation_codmes=None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Ejecuta el pipeline completo de preprocesamiento.

    Args:
        data_path: Ruta del archivo CSV de entrada.
        output_dir: Carpeta de salida para los archivos procesados.
        nan_threshold: Porcentaje máximo permitido de nulos por columna.
        validation_codmes: Mes que se reservará para validación.
        test_size: Proporción del set de prueba.
        random_state: Semilla de reproducibilidad.

    Returns:
        df_train, df_test, df_val, metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada: {data_path}. "
            "Coloca el dataset en data/raw/Data_CU_venta.csv o usa --input."
        )

    df = pd.read_csv(data_path)
    _validate_required_columns(df)

    # Normalizar p_codmes como numérico si es posible.
    df[DATE_COL] = pd.to_numeric(df[DATE_COL], errors="coerce")

    # Eliminar columnas con exceso de nulos.
    cols_drop = [
        col for col in df.columns
        if df[col].isna().mean() * 100 > nan_threshold
    ]
    df = df.drop(columns=cols_drop)

    selected_val_codmes = _select_validation_month(df, validation_codmes)

    # Separación temporal.
    df_val_raw = df[df[DATE_COL] == selected_val_codmes].copy()
    df_main_raw = df[df[DATE_COL] != selected_val_codmes].copy()

    if df_val_raw.empty:
        raise ValueError("El set de validación quedó vacío.")
    if df_main_raw.empty:
        raise ValueError("El set de entrenamiento/prueba quedó vacío.")

    # Imputar y codificar todo el dataset de forma simple para mantener consistencia.
    df_processed, encoding_metadata = _impute_and_encode(df)

    df_val = df_processed[df_processed[DATE_COL] == selected_val_codmes].copy()
    df_main = df_processed[df_processed[DATE_COL] != selected_val_codmes].copy()

    stratify = df_main[TARGET_COL] if df_main[TARGET_COL].nunique() == 2 else None

    df_train, df_test = train_test_split(
        df_main,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    df_train.to_csv(output_path / "df_train.csv", index=False)
    df_test.to_csv(output_path / "df_test.csv", index=False)
    df_val.to_csv(output_path / "df_val.csv", index=False)

    metadata = {
        "dropped_columns": cols_drop,
        "validation_codmes": float(selected_val_codmes),
        "n_rows_raw": int(len(df)),
        "n_rows_train": int(len(df_train)),
        "n_rows_test": int(len(df_test)),
        "n_rows_val": int(len(df_val)),
        "target_rate_train": float(df_train[TARGET_COL].mean()),
        "target_rate_test": float(df_test[TARGET_COL].mean()),
        "target_rate_val": float(df_val[TARGET_COL].mean()),
        **encoding_metadata,
    }

    with open(output_path / "preprocessing_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    return df_train, df_test, df_val, metadata
