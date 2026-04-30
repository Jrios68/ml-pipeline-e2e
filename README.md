# Pipeline ML End-to-End con MLOps

Repositorio GitHub: https://github.com/Jrios68/ml-pipeline-e2e

## 1. Descripción del proyecto

Este proyecto implementa un pipeline de Machine Learning End-to-End para demostrar el valor de la analítica en una organización, aplicando principios de MLOps como reproducibilidad, automatización, trazabilidad, versionado de modelos, monitoreo y preparación para producción.

El pipeline permite procesar múltiples archivos CSV de entrada, consolidarlos en un único dataset, entrenar un modelo de clasificación, registrar el experimento en MLflow, monitorear el desempeño, generar una puntuación TLV, exponer el modelo mediante una API de inferencia y visualizar los resultados en un dashboard.

El pipeline cubre las siguientes etapas:

1. Ingesta de datos desde múltiples archivos CSV
2. Preprocesamiento
3. Entrenamiento del modelo
4. Evaluación
5. Registro del modelo en MLflow
6. Monitoreo básico
7. Postprocesamiento TLV
8. Generación de archivos de réplica
9. API de inferencia
10. Dashboard de resultados

---

## 2. Estructura del proyecto

```text
ml-pipeline-e2e/
│
├── main.py
├── preprocessing.py
├── training.py
├── monitoring.py
├── postprocessing.py
├── api.py
├── dashboard.py
│
├── requirements.txt
├── README.md
├── INSTRUCCIONES_EJECUCION_WINDOWS.md
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   ├── p1_extrac.csv
│   │   ├── p2_extrac.csv
│   │   ├── ...
│   │   └── p10_extrac.csv
│   ├── processed/
│   ├── postprocessed/
│   ├── monitoring/
│   └── replica/
│       ├── s3/
│       ├── athena/
│       └── onpremise/
│
├── models/
│   └── .gitkeep
│
└── reports/
    └── informe_tecnico.pdf
```

> Nota: Los archivos CSV de `data/raw/` no se suben al repositorio si son pesados o sensibles. Deben colocarse localmente antes de ejecutar el pipeline.

---

## 3. Descripción de los archivos principales

### `main.py`

Archivo orquestador del pipeline. Ejecuta de forma secuencial las etapas de preprocesamiento, entrenamiento, evaluación, monitoreo, postprocesamiento y generación de archivos de réplica.

### `preprocessing.py`

Realiza la ingesta y limpieza del dataset. Lee múltiples archivos CSV con patrón `p*_extrac.csv`, los consolida, estandariza columnas, crea la columna `p_codmes` a partir de `partition` o `p_fecinformacion`, trata valores nulos, codifica variables categóricas y separa los datos en entrenamiento, prueba y validación.

### `training.py`

Entrena un modelo de clasificación utilizando XGBoost. Incluye búsqueda de hiperparámetros mediante Optuna y registro del experimento en MLflow. Excluye del entrenamiento columnas identificadoras o de trazabilidad como `p_codmes`, `key_value` y `source_file`.

### `monitoring.py`

Calcula métricas de monitoreo como AUC, PSI y recall por decil. Estas métricas permiten evaluar el desempeño del modelo y detectar posibles cambios en la distribución de los datos.

### `postprocessing.py`

Calcula la puntuación TLV, segmenta los clientes en grupos de ejecución y genera los archivos finales para consumo del negocio.

### `api.py`

Implementa una API de inferencia utilizando FastAPI. Permite enviar datos nuevos y obtener la predicción del modelo.

### `dashboard.py`

Dashboard interactivo desarrollado con Streamlit para visualizar distribución de grupos, top clientes, métricas y resultados del modelo.

---

## 4. Requisitos

Para ejecutar el proyecto se requiere Python 3.8 o superior.

Crear y activar un entorno virtual:

```bash
python -m venv venv
```

En Windows CMD:

```bash
venv\Scripts\activate
```

Instalar dependencias:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Contenido del archivo `requirements.txt`:

```text
pandas
numpy
scikit-learn
xgboost
optuna
mlflow
fastapi
uvicorn
streamlit
matplotlib
joblib
```

---

## 5. Preparación de datos

Colocar los archivos de entrada en la carpeta:

```text
data/raw/
```

La estructura esperada es:

```text
data/raw/
├── p1_extrac.csv
├── p2_extrac.csv
├── p3_extrac.csv
├── p4_extrac.csv
├── p5_extrac.csv
├── p6_extrac.csv
├── p7_extrac.csv
├── p8_extrac.csv
├── p9_extrac.csv
└── p10_extrac.csv
```

El pipeline leerá automáticamente todos los archivos con patrón:

```text
p*_extrac.csv
```

y los consolidará en un solo DataFrame.

---

## 6. Ejecución del pipeline completo

Para ejecutar todo el pipeline desde cero:

```bash
python main.py --input "data/raw" --n-trials 5
```

Para una búsqueda de hiperparámetros más amplia:

```bash
python main.py --input "data/raw" --n-trials 30
```

Este comando ejecutará:

1. Lectura y consolidación de archivos CSV
2. Preprocesamiento de datos
3. Generación de archivos `df_train.csv`, `df_test.csv` y `df_val.csv`
4. Entrenamiento del modelo
5. Registro del modelo en MLflow
6. Evaluación del modelo
7. Monitoreo de drift
8. Postprocesamiento TLV
9. Generación de archivos de réplica

---

## 7. Archivos generados por el pipeline

Después de ejecutar `main.py`, se generan los siguientes archivos:

```text
data/processed/df_train.csv
data/processed/df_test.csv
data/processed/df_val.csv
data/processed/preprocessing_metadata.json

models/model.pkl
models/feature_names.json
models/metrics.json

data/monitoring/monitoring_metrics.json
data/monitoring/recall_by_decile.csv
data/monitoring/score_distribution.csv

data/postprocessed/output_tlv.csv

data/replica/s3/EC_OMNICANAL_YYYYMM.txt
data/replica/athena/EC_OMNICANAL_YYYYMM.txt
data/replica/onpremise/EC_OMNICANAL_YYYYMM.txt
```

Estos archivos son generados automáticamente y no necesariamente deben subirse al repositorio.

---

## 8. Ejecución de MLflow

Para visualizar los experimentos registrados:

```bash
mlflow ui
```

Luego abrir en el navegador:

```text
http://localhost:5000
```

En MLflow se podrán revisar:

- Parámetros del modelo
- Métricas de evaluación
- AUC del modelo
- Modelo registrado
- Versiones del modelo

La ejecución final validada generó el modelo:

```text
Modelo: cu_venta_xgb
Versión: 3
Run ID: cd6aa30e475548b5a11ee8ce19190566
```

---

## 9. Ejecución de la API de inferencia

Antes de iniciar la API, se debe haber ejecutado correctamente el pipeline para generar:

```text
models/model.pkl
models/feature_names.json
data/processed/preprocessing_metadata.json
```

Para iniciar la API:

```bash
uvicorn api:app --reload
```

Luego abrir la documentación automática de FastAPI:

```text
http://localhost:8000/docs
```

Endpoints disponibles:

```text
GET /health
POST /predict
POST /predict-batch
```

La API permite enviar variables de entrada del cliente y obtener como resultado la probabilidad estimada por el modelo.

---

## 10. Ejecución del dashboard

Para iniciar el dashboard:

```bash
streamlit run dashboard.py
```

Luego abrir en el navegador:

```text
http://localhost:8501
```

El dashboard permite visualizar:

- Registros procesados
- Score promedio
- TLV promedio
- PSI
- Distribución de grupos de ejecución
- Top clientes por puntuación TLV
- Métricas de monitoreo
- Recall acumulado por decil
- Distribución de scores

---

## 11. Monitoreo del modelo

El monitoreo básico considera:

### PSI — Population Stability Index

Permite detectar cambios en la distribución de los datos entre el conjunto de entrenamiento y el conjunto de validación.

Interpretación:

```text
PSI < 0.10       Sin deriva relevante
0.10 - 0.25     Deriva moderada
PSI > 0.25      Deriva severa
```

### AUC

Evalúa la capacidad del modelo para discriminar entre clientes con y sin evento objetivo.

### Recall por decil

Permite analizar qué porcentaje de clientes positivos se concentra en los deciles de mayor score.

---

## 12. Valor de negocio

Este pipeline permite convertir datos históricos de clientes en información accionable para la toma de decisiones comerciales.

El modelo prioriza clientes según su probabilidad estimada y una puntuación TLV compuesta, permitiendo enfocar campañas, recursos operativos o acciones comerciales en los segmentos con mayor valor esperado.

Desde una perspectiva empresarial, el pipeline aporta valor porque:

- Automatiza el proceso analítico de inicio a fin.
- Reduce tiempos de preparación y evaluación de modelos.
- Mejora la trazabilidad de experimentos y versiones.
- Permite priorizar clientes de mayor impacto.
- Facilita el monitoreo del desempeño del modelo.
- Prepara el modelo para consumo mediante API.
- Mejora la reproducibilidad del análisis.
- Permite reutilizar el flujo con nuevas particiones de datos.

---

## 13. Reproducibilidad

Para asegurar que cualquier persona pueda ejecutar el proyecto desde cero, se incluyen:

- Código modular
- Archivo `requirements.txt`
- Estructura estándar de carpetas
- Pipeline orquestado desde `main.py`
- Registro de experimentos en MLflow
- Documentación de ejecución
- API documentada automáticamente con FastAPI
- Archivo `INSTRUCCIONES_EJECUCION_WINDOWS.md`

El flujo se puede reconstruir ejecutando:

```bash
python main.py --input "data/raw" --n-trials 5
```

---

## 14. Archivos que no se suben al repositorio

Por buenas prácticas de MLOps y para mantener el repositorio liviano, no se deben subir:

```text
venv/
mlruns/
data/raw/*.csv
models/model.pkl
__pycache__/
data/processed/*.csv
data/postprocessed/*.csv
data/monitoring/*.json
data/monitoring/*.csv
```

Estos archivos se regeneran al ejecutar el pipeline.

---

## 15. Entregables

El proyecto incluye:

- Repositorio GitHub con el código fuente
- Pipeline ML End-to-End funcional
- Modelo registrado en MLflow
- API de inferencia
- Monitoreo básico del modelo
- Dashboard de resultados
- Informe técnico en PDF
- README de ejecución
- Instrucciones de ejecución en Windows

Repositorio GitHub:

```text
https://github.com/Jrios68/ml-pipeline-e2e
```

---

## 16. Autor

Proyecto desarrollado como parte de la Tarea 2: Implementación de Pipeline ML End-to-End.

Autor: Julio Rios
