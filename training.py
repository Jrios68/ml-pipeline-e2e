"""
training.py -- Entrenamiento XGBoost + búsqueda de hiperparámetros con Optuna.

Registra:
    - Parámetros del modelo
    - Métricas de evaluación
    - Modelo entrenado en MLflow
    - Modelo local en models/model.pkl
    - Lista de variables usadas en models/feature_names.json
"""

from pathlib import Path
from typing import List, Tuple

import json
import joblib
import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


# Columnas que no deben entrar al entrenamiento del modelo.
# p_codmes: columna temporal usada para split.
# key_value: identificador del cliente.
# source_file: archivo de origen, usado solo para trazabilidad.
ID_COLS = ["p_codmes", "key_value", "source_file"]

TARGET_COL = "target"
MODEL_DIR = Path("models")


def prepare_features(df: pd.DataFrame, feature_names: List[str] = None) -> pd.DataFrame:
    """
    Prepara las variables predictoras.

    Si feature_names se informa, alinea el DataFrame a esas columnas.
    """
    drop_cols = [col for col in ID_COLS + [TARGET_COL] if col in df.columns]

    X = df.drop(columns=drop_cols)

    if feature_names is not None:
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0

        X = X[feature_names]

    return X


def _xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa variables predictoras y variable objetivo.
    """
    X = prepare_features(df)
    y = df[TARGET_COL]

    if y.nunique() < 2:
        raise ValueError(
            "La variable target debe tener al menos dos clases para entrenar."
        )

    return X, y


def _classification_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    """
    Calcula métricas principales de clasificación.
    """
    y_pred = (scores >= threshold).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def train_and_log(
    train_path: str,
    test_path: str,
    val_path: str,
    n_trials: int = 30,
    experiment_name: str = "cu_venta_e2e",
    registered_model_name: str = "cu_venta_xgb",
):
    """
    Busca hiperparámetros con Optuna, entrena el mejor modelo y registra todo en MLflow.

    Args:
        train_path: Ruta de df_train.csv.
        test_path: Ruta de df_test.csv.
        val_path: Ruta de df_val.csv.
        n_trials: Número de pruebas de hiperparámetros.
        experiment_name: Nombre del experimento en MLflow.
        registered_model_name: Nombre del modelo registrado.

    Returns:
        run_id, model, feature_names
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_val = pd.read_csv(val_path)

    X_train, y_train = _xy(df_train)
    X_test, y_test = _xy(df_test)

    feature_names = list(X_train.columns)

    X_test = X_test[feature_names]
    X_val = prepare_features(df_val, feature_names)
    y_val = df_val[TARGET_COL]

    scale_pos_weight = 1.0
    positives = y_train.sum()
    negatives = len(y_train) - positives

    if positives > 0:
        scale_pos_weight = negatives / positives

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
            "random_state": 123,
            "n_jobs": -1,
        }

        model = xgb.XGBClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        scores = model.predict_proba(X_test)[:, 1]

        return roc_auc_score(y_test, scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {
        **study.best_params,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": 123,
        "n_jobs": -1,
    }

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    test_scores = model.predict_proba(X_test)[:, 1]
    val_scores = model.predict_proba(X_val)[:, 1]

    test_metrics = _classification_metrics(y_test, test_scores)
    val_metrics = _classification_metrics(y_val, val_scores)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(best_params)

        mlflow.log_metric("test.auc", test_metrics["auc"])
        mlflow.log_metric("test.accuracy", test_metrics["accuracy"])
        mlflow.log_metric("test.precision", test_metrics["precision"])
        mlflow.log_metric("test.recall", test_metrics["recall"])

        mlflow.log_metric("val.auc", val_metrics["auc"])
        mlflow.log_metric("val.accuracy", val_metrics["accuracy"])
        mlflow.log_metric("val.precision", val_metrics["precision"])
        mlflow.log_metric("val.recall", val_metrics["recall"])

        mlflow.log_metric("optuna.best_value", float(study.best_value))

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        run_id = run.info.run_id

    # Guardado local para API o uso posterior.
    joblib.dump(model, MODEL_DIR / "model.pkl")

    with open(MODEL_DIR / "feature_names.json", "w", encoding="utf-8") as file:
        json.dump(feature_names, file, indent=4, ensure_ascii=False)

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "run_id": run_id,
                "test_metrics": test_metrics,
                "val_metrics": val_metrics,
                "best_params": best_params,
            },
            file,
            indent=4,
            ensure_ascii=False,
        )

    return run_id, model, feature_names
