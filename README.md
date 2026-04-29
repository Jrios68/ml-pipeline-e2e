# Pipeline ML End-to-End con MLOps
https://github.com/Jrios68/ml-pipeline-e2e
## 1. Descripción del proyecto

Este proyecto implementa un pipeline de Machine Learning End-to-End para demostrar el valor de la analítica en una organización, aplicando principios de MLOps como reproducibilidad, automatización, trazabilidad, versionado de modelos, monitoreo y preparación para producción.

El pipeline cubre las siguientes etapas:

1. Ingesta de datos
2. Preprocesamiento
3. Entrenamiento del modelo
4. Evaluación
5. Registro del modelo en MLflow
6. Monitoreo básico
7. Postprocesamiento
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
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── postprocessed/
│   ├── monitoring/
│   └── replica/
│       ├── s3/
│       ├── athena/
│       └── onpremise/
│
├── models/
│
├── reports/
│   └── informe_tecnico.pdf
│
└── mlruns/
```

---

## 3. Descripción de los archivos principales

### `main.py`

Archivo orquestador del pipeline. Ejecuta de forma secuencial las etapas de preprocesamiento, entrenamiento, evaluación, monitoreo, postprocesamiento y generación de archivos de réplica.

### `preprocessing.py`

Realiza la limpieza del dataset, tratamiento de valores nulos, codificación de variables categóricas y separación de los datos en entrenamiento, prueba y validación.

### `training.py`

Entrena un modelo de clasificación utilizando XGBoost. Incluye búsqueda de hiperparámetros mediante Optuna y registro del experimento en MLflow.

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

Instalar las dependencias con:

```bash
pip install -r requirements.txt
```

Contenido sugerido del archivo `requirements.txt`:

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

Colocar el archivo de datos original en la siguiente ruta:

```text
data/raw/Data_CU_venta.csv
```

La ruta puede modificarse en el archivo `main.py`:

```python
INPUT_PATH = "data/raw/Data_CU_venta.csv"
```

---

## 6. Ejecución del pipeline completo

Para ejecutar todo el pipeline desde cero:

```bash
python main.py
```

Este comando ejecutará:

1. Preprocesamiento de datos
2. Generación de archivos `df_train.csv`, `df_test.csv` y `df_val.csv`
3. Entrenamiento del modelo
4. Registro del modelo en MLflow
5. Evaluación del modelo
6. Monitoreo de drift
7. Postprocesamiento TLV
8. Generación de archivos de réplica

---

## 7. Ejecución de MLflow

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

---

## 8. Ejecución de la API de inferencia

Para iniciar la API:

```bash
uvicorn api:app --reload
```

Luego abrir la documentación automática de FastAPI:

```text
http://localhost:8000/docs
```

La API permite enviar variables de entrada del cliente y obtener como resultado la probabilidad estimada por el modelo.

---

## 9. Ejecución del dashboard

Para iniciar el dashboard:

```bash
streamlit run dashboard.py
```

El dashboard permite visualizar:

- Distribución de grupos de ejecución
- Top clientes por puntuación TLV
- Evolución de métricas de monitoreo
- Resultados del modelo

---

## 10. Monitoreo del modelo

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

## 11. Valor de negocio

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

---

## 12. Reproducibilidad

Para asegurar que cualquier persona pueda ejecutar el proyecto desde cero, se incluyen:

- Código modular
- Archivo `requirements.txt`
- Estructura estándar de carpetas
- Pipeline orquestado desde `main.py`
- Registro de experimentos en MLflow
- Documentación de ejecución
- API documentada automáticamente con FastAPI

---

## 13. Entregables

El proyecto incluye:

- Repositorio GitHub con el código fuente
- Pipeline ML End-to-End funcional
- Modelo registrado en MLflow
- API de inferencia
- Monitoreo básico del modelo
- Dashboard de resultados
- Informe técnico en PDF
- README de ejecución

---

## 14. Autor

Proyecto desarrollado como parte de la Tarea 2: Implementación de Pipeline ML End-to-End.

Autor: Julio Rios
