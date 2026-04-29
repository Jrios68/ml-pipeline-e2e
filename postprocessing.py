"""
postprocessing.py -- Scoring TLV, segmentación y réplica.

Formula TLV:
    puntuacion_tlv = prob * prob_value_contact * log(monto + 1) * prob_frescura

El resultado se segmenta en 10 grupos de ejecución.
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


DIST_GE = [0, 0.035, 0.087, 0.237, 0.393, 0.529, 0.664, 0.787, 0.862, 0.95, 1.0]


def get_groups(scores, df_post: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula puntuación TLV y asigna grupo_ejec_tlv.

    Args:
        scores: Probabilidades del modelo.
        df_post: DataFrame con columnas esperadas:
                 grp_campecs06m, prob_value_contact, monto.

    Returns:
        DataFrame con columnas:
            prob, prob_frescura, puntuacion_tlv, grupo_ejec_tlv.
    """
    df_post = df_post.copy()
    df_post["prob"] = scores

    # Columnas mínimas de negocio.
    if "grp_campecs06m" not in df_post.columns:
        df_post["grp_campecs06m"] = "G5"

    if "prob_value_contact" not in df_post.columns:
        df_post["prob_value_contact"] = 0.000001

    if "monto" not in df_post.columns:
        df_post["monto"] = 0

    df_post["prob_frescura"] = np.where(
        df_post["grp_campecs06m"] == "G1", 0.066,
        np.where(df_post["grp_campecs06m"] == "G2", 0.028,
        np.where(df_post["grp_campecs06m"] == "G3", 0.022,
        np.where(df_post["grp_campecs06m"] == "G4", 0.008, 0.004)))
    )

    df_post["prob_value_contact"] = df_post["prob_value_contact"].fillna(0.000001)
    df_post["monto"] = df_post["monto"].fillna(0).clip(lower=0)

    df_post["puntuacion_tlv"] = (
        df_post["prob"]
        * df_post["prob_value_contact"]
        * np.log(df_post["monto"] + 1)
        * df_post["prob_frescura"]
    )

    # Segmentación por cuantiles definidos por negocio.
    try:
        df_post["grupo_ejec_tlv"] = pd.qcut(
            df_post["puntuacion_tlv"],
            q=DIST_GE,
            labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            duplicates="drop",
        )
    except ValueError:
        # Fallback cuando hay demasiados valores repetidos.
        df_post["grupo_ejec_tlv"] = pd.cut(
            df_post["puntuacion_tlv"].rank(method="first"),
            bins=10,
            labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        )

    df_post["grupo_ejec_tlv"] = df_post["grupo_ejec_tlv"].astype(int)

    return df_post


def run_postprocessing(scores, df_post: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """Ejecuta get_groups y guarda el resultado si se informa output_path."""
    result = get_groups(scores, df_post)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


def _first_existing_column(df: pd.DataFrame, candidates, default_value=""):
    """Devuelve la primera columna existente entre varias candidatas."""
    for col in candidates:
        if col in df.columns:
            return df[col]
    return default_value


def save_replica(
    df_post: pd.DataFrame,
    table: str,
    partition: str,
    dir_s3: str = "data/replica/s3",
    dir_athena: str = "data/replica/athena",
    dir_onpremise: str = "data/replica/onpremise",
):
    """
    Genera archivo de réplica pipe-delimitado para tres destinos.

    Columnas:
        codmes | tipdoc | coddoc | puntuacion | modelo | fec_replica |
        grupo_ejec | score | orden | variable1 | variable2 | variable3
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    replica = pd.DataFrame()
    replica["codmes"] = _first_existing_column(df_post, ["p_codmes", "codmes"], partition)
    replica["tipdoc"] = _first_existing_column(df_post, ["tipdoc", "tipo_documento"], "")
    replica["coddoc"] = _first_existing_column(df_post, ["coddoc", "key_value", "cod_cliente"], "")
    replica["puntuacion"] = df_post["puntuacion_tlv"]
    replica["modelo"] = "cu_venta_xgb"
    replica["fec_replica"] = now
    replica["grupo_ejec"] = df_post["grupo_ejec_tlv"]
    replica["score"] = df_post["prob"]
    replica["orden"] = replica["puntuacion"].rank(method="first", ascending=False).astype(int)
    replica["variable1"] = _first_existing_column(df_post, ["grp_campecs06m"], "")
    replica["variable2"] = _first_existing_column(df_post, ["prob_value_contact"], "")
    replica["variable3"] = _first_existing_column(df_post, ["monto"], "")

    filename = f"{table}_{partition}.txt"

    output_paths = []
    for directory in [dir_s3, dir_athena, dir_onpremise]:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        full_path = path / filename
        replica.to_csv(full_path, sep="|", index=False)
        output_paths.append(str(full_path))

    return output_paths
