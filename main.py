"""
main.py -- Orquestador principal del pipeline ML End-to-End.

Ejecutar:
    python main.py --input data/raw/Data_CU_venta.csv

Etapas:
    1. Preprocesamiento
    2. Entrenamiento y registro en MLflow
    3. Evaluación y monitoreo
    4. Postprocesamiento TLV
    5. Generación de archivos de réplica
"""

import argparse
from pathlib import Path

from preprocessing import run_preprocessing
from training import train_and_log, prepare_features
from monitoring import run_monitoring, compute_recall_by_decile
from postprocessing import run_postprocessing, save_replica


DEFAULT_INPUT_PATH = "data/raw"
DEFAULT_OUTPUT_DIR = "data/processed"
DEFAULT_POST_PATH = "data/postprocessed/output_tlv.csv"
DEFAULT_MONITORING_DIR = "data/monitoring"


def parse_args():
    """Define los argumentos de ejecución del pipeline."""
    parser = argparse.ArgumentParser(description="Pipeline ML End-to-End con MLOps")

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Ruta del dataset CSV de entrada.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde se guardarán los datasets procesados.",
    )
    parser.add_argument(
        "--post-path",
        type=str,
        default=DEFAULT_POST_PATH,
        help="Ruta donde se guardará el resultado postprocesado.",
    )
    parser.add_argument(
        "--monitoring-dir",
        type=str,
        default=DEFAULT_MONITORING_DIR,
        help="Carpeta donde se guardarán los archivos de monitoreo.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Número de pruebas de hiperparámetros con Optuna.",
    )
    parser.add_argument(
        "--validation-codmes",
        type=float,
        default=None,
        help="Mes reservado para validación. Si no se informa, se usará el último p_codmes disponible.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="cu_venta_e2e",
        help="Nombre del experimento en MLflow.",
    )

    return parser.parse_args()


def main():
    """Ejecuta el pipeline completo."""
    args = parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.monitoring_dir).mkdir(parents=True, exist_ok=True)
    Path(args.post_path).parent.mkdir(parents=True, exist_ok=True)

    # 1. Preprocesamiento
    print("[1/5] Ejecutando preprocesamiento...")
    df_train, df_test, df_val, meta = run_preprocessing(
        data_path=args.input,
        output_dir=args.output_dir,
        validation_codmes=args.validation_codmes,
    )

    train_path = str(Path(args.output_dir) / "df_train.csv")
    test_path = str(Path(args.output_dir) / "df_test.csv")
    val_path = str(Path(args.output_dir) / "df_val.csv")

    # 2. Entrenamiento con búsqueda de hiperparámetros y registro en MLflow
    print("[2/5] Entrenando modelo y registrando en MLflow...")
    run_id, model, feature_names = train_and_log(
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        n_trials=args.n_trials,
        experiment_name=args.experiment_name,
    )

    # 3. Monitoreo
    print("[3/5] Ejecutando monitoreo del modelo...")
    X_train = prepare_features(df_train, feature_names)
    X_val = prepare_features(df_val, feature_names)

    train_scores = model.predict_proba(X_train)[:, 1]
    val_scores = model.predict_proba(X_val)[:, 1]

    monitoring_result = run_monitoring(
        df_train=df_train,
        df_val=df_val,
        train_scores=train_scores,
        val_scores=val_scores,
        output_dir=args.monitoring_dir,
        mlflow_run_id=run_id,
    )

    recall_decile = compute_recall_by_decile(
        y_true=df_val["target"],
        scores=val_scores,
        output_path=str(Path(args.monitoring_dir) / "recall_by_decile.csv"),
    )

    # 4. Postprocesamiento TLV
    print("[4/5] Ejecutando postprocesamiento TLV...")
    df_resultado = run_postprocessing(
        scores=val_scores,
        df_post=df_val.copy(),
        output_path=args.post_path,
    )

    # 5. Réplica
    print("[5/5] Generando archivos de réplica...")
    save_replica(
        df_post=df_resultado,
        table="EC_OMNICANAL",
        partition=str(int(meta["validation_codmes"])),
    )

    print("\nPipeline ejecutado correctamente.")
    print(f"Run ID MLflow: {run_id}")
    print(f"Modelo local: models/model.pkl")
    print(f"Resultado TLV: {args.post_path}")
    print(f"Métricas monitoreo: {args.monitoring_dir}/monitoring_metrics.json")
    print(f"Recall por decil: {args.monitoring_dir}/recall_by_decile.csv")


if __name__ == "__main__":
    main()
