"""
monitoring.py -- Monitoreo de deriva y desempeño.

Incluye:
    - PSI sobre scores del modelo
    - AUC de validación
    - Recall por decil
    - Guardado de métricas en JSON y CSV
    - Registro opcional en MLflow
"""

from pathlib import Path
import json

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score


def psi_flag(psi: float) -> str:
    """Retorna la etiqueta de alerta según el valor de PSI."""
    if psi < 0.10:
        return "OK"
    elif psi < 0.25:
        return "WARN"
    return "ALERT"


def calculate_psi(expected, actual, buckets: int = 10, eps: float = 1e-6) -> float:
    """
    Calcula Population Stability Index entre dos distribuciones.

    Args:
        expected: Distribución base, normalmente scores de entrenamiento.
        actual: Distribución actual, normalmente scores de validación.
        buckets: Número de grupos para comparar.
        eps: Valor mínimo para evitar divisiones por cero.

    Returns:
        Valor PSI.
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.quantile(expected, quantiles)
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) <= 2:
        breakpoints = np.linspace(expected.min(), expected.max(), buckets + 1)

    # Asegurar límites abiertos.
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / max(len(expected), 1)
    actual_pct = actual_counts / max(len(actual), 1)

    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)

    return float(np.sum(psi_values))


def compute_recall_by_decile(
    y_true,
    scores,
    n_deciles: int = 10,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Calcula el recall acumulado por decil de score.

    Decil 1 representa el grupo con mayor score.
    """
    df = pd.DataFrame({"score": scores, "target": y_true})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    df["decil"] = pd.qcut(
        df.index + 1,
        q=n_deciles,
        labels=range(1, n_deciles + 1),
    )

    total_positives = df["target"].sum()
    if total_positives == 0:
        total_positives = 1

    result = (
        df.groupby("decil", observed=True)
        .agg(
            registros=("target", "count"),
            positivos=("target", "sum"),
            score_min=("score", "min"),
            score_max=("score", "max"),
        )
        .reset_index()
    )

    result["positivos_acumulados"] = result["positivos"].cumsum()
    result["recall_acumulado"] = result["positivos_acumulados"] / total_positives

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


def run_monitoring(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    train_scores,
    val_scores,
    id_cols=None,
    target_col: str = "target",
    output_dir: str = "data/monitoring",
    mlflow_run_id: str = None,
):
    """
    Calcula PSI sobre scores, AUC y recall en validación.

    Args:
        df_train: Dataset de entrenamiento procesado.
        df_val: Dataset de validación procesado.
        train_scores: Scores del modelo para entrenamiento.
        val_scores: Scores del modelo para validación.
        id_cols: Columnas identificadoras.
        target_col: Nombre de la variable objetivo.
        output_dir: Carpeta donde se guardan las métricas.
        mlflow_run_id: Run ID de MLflow para registrar métricas adicionales.

    Returns:
        Diccionario con métricas de monitoreo.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    psi_score = calculate_psi(train_scores, val_scores)

    auc_val = roc_auc_score(df_val[target_col], val_scores)

    y_pred_val = (np.asarray(val_scores) >= 0.5).astype(int)
    recall_val = recall_score(df_val[target_col], y_pred_val, zero_division=0)

    metrics = {
        "psi_score": float(psi_score),
        "psi_flag": psi_flag(psi_score),
        "val_auc": float(auc_val),
        "val_recall_threshold_0_5": float(recall_val),
        "train_rows": int(len(df_train)),
        "val_rows": int(len(df_val)),
        "train_target_rate": float(df_train[target_col].mean()),
        "val_target_rate": float(df_val[target_col].mean()),
    }

    with open(output_path / "monitoring_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    score_distribution = pd.DataFrame({
        "dataset": ["train"] * len(train_scores) + ["validation"] * len(val_scores),
        "score": list(train_scores) + list(val_scores),
    })
    score_distribution.to_csv(output_path / "score_distribution.csv", index=False)

    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_metric("monitoring.psi_score", metrics["psi_score"])
            mlflow.log_metric("monitoring.val_auc", metrics["val_auc"])
            mlflow.log_metric("monitoring.val_recall_threshold_0_5", metrics["val_recall_threshold_0_5"])

    return metrics
