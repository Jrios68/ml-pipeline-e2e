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

def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Carga uno o varios archivos CSV.

    Casos soportados:
    1. Si data_path es un archivo CSV, lo lee directamente.
    2. Si data_path es una carpeta, busca archivos p*_extrac.csv,
       los ordena correctamente de p1 a p10, los lee y los concatena
       en un único DataFrame.

    Args:
        data_path: Ruta de un archivo CSV o carpeta con varios CSV.

    Returns:
        DataFrame consolidado.
    """
    path = Path(data_path)

    if path.is_file():
        print(f"Leyendo archivo único: {path}")
        return pd.read_csv(path)

    if path.is_dir():
        files = sorted(
            path.glob("p*_extrac.csv"),
            key=lambda file: int(file.stem.replace("p", "").replace("_extrac", ""))
        )

        if not files:
            raise FileNotFoundError(
                f"No se encontraron archivos p*_extrac.csv en la carpeta: {path}"
            )

        dfs = []

        for file in files:
            print(f"Leyendo archivo: {file.name}")
            df_part = pd.read_csv(file)
            df_part["source_file"] = file.name
            dfs.append(df_part)

        df = pd.concat(dfs, ignore_index=True)

        print(f"Archivos leídos: {len(files)}")
        print(f"Filas totales consolidadas: {len(df):,}")

        return df

    raise FileNotFoundError(f"No existe la ruta indicada: {data_path}")

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza nombres de columnas para que el pipeline use una nomenclatura común.

    El pipeline trabaja internamente con:
    - p_codmes
    - tipdoc

    En la data real:
    - El periodo puede venir en partition o p_fecinformacion.
    - El tipo de documento puede venir como tip_doc.
    """
    df = df.copy()

    if "tipdoc" not in df.columns and "tip_doc" in df.columns:
        df = df.rename(columns={"tip_doc": "tipdoc"})

    # Crear p_codmes usando primero partition.
    if "partition" in df.columns:
        codmes_partition = _extract_codmes(df["partition"])
    else:
        codmes_partition = pd.Series([pd.NA] * len(df), index=df.index)

    # Crear p_codmes usando p_fecinformacion como respaldo.
    if "p_fecinformacion" in df.columns:
        codmes_fecha = _extract_codmes(df["p_fecinformacion"])
    else:
        codmes_fecha = pd.Series([pd.NA] * len(df), index=df.index)

    # Usar partition cuando sirva; si no, usar p_fecinformacion.
    df["p_codmes"] = codmes_partition.fillna(codmes_fecha)

    return df

def _extract_codmes(series: pd.Series) -> pd.Series:
    """
    Extrae un codmes numérico YYYYMM desde distintos formatos posibles:
    - 202412
    - 20241231
    - 2024-12-01
    - 2024/12/01
    - 2024-12
    - partition=202412
    """
    text = series.astype(str).str.strip()

    # Primero intenta convertir como fecha.
    fechas = pd.to_datetime(text, errors="coerce", dayfirst=False)
    codmes_fecha = fechas.dt.year * 100 + fechas.dt.month

    # Luego intenta extraer dígitos.
    digits = text.str.replace(r"\D", "", regex=True)

    # Busca un patrón YYYYMM.
    codmes_texto = digits.str.extract(r"((?:19|20)\d{4})")[0]
    codmes_texto = pd.to_numeric(codmes_texto, errors="coerce")

    # Usar fecha si funciona; si no, usar texto.
    codmes = codmes_fecha.fillna(codmes_texto)

    return pd.to_numeric(codmes, errors="coerce")

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
        data_path: Ruta del archivo CSV de entrada o carpeta con varios CSV.
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
            f"No se encontró la ruta de entrada: {data_path}. "
            "Coloca los archivos p1_extrac.csv, p2_extrac.csv, ..., p10_extrac.csv "
            "en data/raw/ o usa --input con la ruta correcta."
        )

    # Cargar dataset: archivo único o múltiples archivos CSV en carpeta.
    df = load_dataset(data_path)

    # Estandarizar nombres de columnas según la data real.
    df = _standardize_columns(df)

    #Validar columnas mínimas.
    _validate_required_columns(df)

    # Normalizar p_codmes como YYYYMM numérico.
    df[DATE_COL] = pd.to_numeric(df[DATE_COL], errors="coerce")

    print("Valores válidos de p_codmes:", df[DATE_COL].notna().sum())
    print("Ejemplos de p_codmes:", sorted(df[DATE_COL].dropna().unique())[:10])

    # Eliminar registros sin p_codmes válido.
    df = df.dropna(subset=[DATE_COL])

    # Eliminar columnas con exceso de nulos.
    cols_drop = [
        col for col in df.columns
        if df[col].isna().mean() * 100 > nan_threshold
    ]

    df = df.drop(columns=cols_drop)

    # Seleccionar mes de validación.
    selected_val_codmes = _select_validation_month(df, validation_codmes)

    # Separación temporal.
    df_val_raw = df[df[DATE_COL] == selected_val_codmes].copy()
    df_main_raw = df[df[DATE_COL] != selected_val_codmes].copy()

    if df_val_raw.empty:
        raise ValueError(
            f"El set de validación quedó vacío para p_codmes={selected_val_codmes}."
        )

    if df_main_raw.empty:
        raise ValueError(
            "El set de entrenamiento/prueba quedó vacío. "
            "Verifica que existan meses diferentes al mes de validación."
        )

    # Imputar y codificar todo el dataset para mantener consistencia.
    df_processed, encoding_metadata = _impute_and_encode(df)

    # Volver a separar después del encoding.
    df_val = df_processed[df_processed[DATE_COL] == selected_val_codmes].copy()
    df_main = df_processed[df_processed[DATE_COL] != selected_val_codmes].copy()

    # Estratificar solo si target tiene dos clases.
    stratify = df_main[TARGET_COL] if df_main[TARGET_COL].nunique() == 2 else None

    df_train, df_test = train_test_split(
        df_main,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Guardar datasets procesados.
    df_train.to_csv(output_path / "df_train.csv", index=False)
    df_test.to_csv(output_path / "df_test.csv", index=False)
    df_val.to_csv(output_path / "df_val.csv", index=False)

    metadata = {
        "input_path": str(data_path),
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
