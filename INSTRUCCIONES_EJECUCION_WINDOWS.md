# Instrucciones de ejecución en Windows / VS Code / Terminal

Este documento explica cómo ejecutar el proyecto `ml-pipeline-e2e` desde cero en Windows, usando VS Code o la terminal.

---

## 1. Abrir el proyecto en VS Code

1. Descomprimir el archivo del proyecto.
2. Abrir Visual Studio Code.
3. Ir a `File > Open Folder`.
4. Seleccionar la carpeta `ml-pipeline-e2e`.

La estructura esperada es:

```text
ml-pipeline-e2e/
├── main.py
├── preprocessing.py
├── training.py
├── monitoring.py
├── postprocessing.py
├── api.py
├── dashboard.py
├── README.md
├── requirements.txt
├── data/
├── models/
└── reports/
```

---

## 2. Abrir la terminal integrada

En VS Code abrir:

```text
Terminal > New Terminal
```

Verificar que la terminal esté dentro de la carpeta del proyecto.

Ejemplo:

```powershell
PS C:\Users\jrios\ml-pipeline-e2e>
```

Si no estás en la carpeta correcta:

```powershell
cd C:\Users\jrios\ml-pipeline-e2e
```

---

## 3. Crear entorno virtual

Ejecutar:

```powershell
python -m venv venv
```

Esto creará una carpeta llamada `venv`.

---

## 4. Activar entorno virtual

En PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Cuando esté activo verás algo similar a:

```powershell
(venv) PS C:\Users\jrios\ml-pipeline-e2e>
```

Si aparece el error `running scripts is disabled on this system`, ejecutar:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Luego volver a activar:

```powershell
.\venv\Scripts\Activate.ps1
```

---

## 5. Instalar dependencias

Ejecutar:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Esto instalará las librerías necesarias:

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

## 6. Ubicar el dataset

Colocar el archivo:

```text
Data_CU_venta.csv
```

en la carpeta:

```text
data/raw/
```

La ruta final debe ser:

```text
ml-pipeline-e2e/data/raw/Data_CU_venta.csv
```

---

## 7. Revisar rutas en main.py

Se recomienda usar rutas relativas:

```python
DEFAULT_INPUT_PATH = "data/raw/Data_CU_venta.csv"
DEFAULT_OUTPUT_DIR = "data/processed"
DEFAULT_POST_PATH = "data/postprocessed/output_tlv.csv"
DEFAULT_MONITORING_DIR = "data/monitoring"
```

No se recomienda dejar rutas absolutas como `C:\Users\jrios\...` porque solo funcionarán en tu computadora.

---

## 8. Ejecutar pipeline completo

Desde la terminal:

```powershell
python main.py
```

También puedes indicar explícitamente la ruta:

```powershell
python main.py --input "data/raw/Data_CU_venta.csv"
```

Para reducir el tiempo de búsqueda de hiperparámetros:

```powershell
python main.py --input "data/raw/Data_CU_venta.csv" --n-trials 5
```

Para una ejecución más completa:

```powershell
python main.py --input "data/raw/Data_CU_venta.csv" --n-trials 30
```

---

## 9. Archivos generados por el pipeline

Después de ejecutar `main.py`, deberían generarse:

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

---

## 10. Abrir MLflow

MLflow permite revisar experimentos, parámetros, métricas y modelos registrados.

Ejecutar:

```powershell
mlflow ui
```

Luego abrir en el navegador:

```text
http://localhost:5000
```

En MLflow podrás revisar:

- Experimento
- Run ID
- Parámetros del modelo
- Métricas de evaluación
- Métricas de validación
- Modelo registrado

Para detener MLflow:

```text
Ctrl + C
```

---

## 11. Ejecutar API con FastAPI

Primero ejecutar el pipeline:

```powershell
python main.py
```

La API necesita estos archivos:

```text
models/model.pkl
models/feature_names.json
data/processed/preprocessing_metadata.json
```

Luego ejecutar:

```powershell
uvicorn api:app --reload
```

Abrir en el navegador:

```text
http://localhost:8000/docs
```

Endpoints disponibles:

```text
GET /health
POST /predict
POST /predict-batch
```

---

## 12. Probar endpoint /health

En `http://localhost:8000/docs`:

1. Ir a `GET /health`.
2. Presionar `Try it out`.
3. Presionar `Execute`.

Respuesta esperada:

```json
{
  "status": "ok",
  "model_exists": true,
  "features_exists": true,
  "metadata_exists": true
}
```

---

## 13. Probar endpoint /predict

Ejemplo de JSON:

```json
{
  "data": {
    "p_codmes": 202412,
    "key_value": 100001,
    "monto": 1500.0,
    "prob_value_contact": 0.35,
    "grp_campecs06m": "G1"
  }
}
```

Respuesta esperada:

```json
{
  "prediction": 1,
  "probability": 0.7321,
  "threshold": 0.5,
  "model": "cu_venta_xgb"
}
```

El valor de `probability` dependerá del modelo entrenado.

---

## 14. Probar endpoint /predict-batch

Ejemplo de JSON:

```json
{
  "records": [
    {
      "p_codmes": 202412,
      "key_value": 100001,
      "monto": 1500.0,
      "prob_value_contact": 0.35,
      "grp_campecs06m": "G1"
    },
    {
      "p_codmes": 202412,
      "key_value": 100002,
      "monto": 900.0,
      "prob_value_contact": 0.18,
      "grp_campecs06m": "G3"
    }
  ]
}
```

Respuesta esperada:

```json
{
  "model": "cu_venta_xgb",
  "n_records": 2,
  "results": [
    {
      "row": 0,
      "prediction": 1,
      "probability": 0.7321,
      "threshold": 0.5
    },
    {
      "row": 1,
      "prediction": 0,
      "probability": 0.2845,
      "threshold": 0.5
    }
  ]
}
```

---

## 15. Ejecutar dashboard con Streamlit

Primero verificar que exista:

```text
data/postprocessed/output_tlv.csv
```

Luego ejecutar:

```powershell
streamlit run dashboard.py
```

Se abrirá una página en el navegador, normalmente en:

```text
http://localhost:8501
```

El dashboard muestra:

- Registros procesados
- Score promedio
- TLV promedio
- PSI
- Distribución de grupos de ejecución
- Top clientes por puntuación TLV
- Métricas de monitoreo
- Recall acumulado por decil
- Distribución de scores

Para detener Streamlit:

```text
Ctrl + C
```

---

## 16. Secuencia recomendada para la entrega

Ejecutar en este orden:

```powershell
cd C:\Users\jrios\ml-pipeline-e2e
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --input "data/raw/Data_CU_venta.csv" --n-trials 10
mlflow ui
uvicorn api:app --reload
streamlit run dashboard.py
```

Importante: `mlflow ui`, `uvicorn` y `streamlit` quedan ejecutándose en la terminal. Para tenerlos activos al mismo tiempo, abre tres terminales separadas en VS Code.

---

## 17. Errores comunes y solución

### Error: No se encontró el archivo de entrada

Solución:

```powershell
python main.py --input "data/raw/Data_CU_venta.csv"
```

Verificar que el archivo exista en:

```text
data/raw/Data_CU_venta.csv
```

---

### Error: No module named pandas

Solución:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### Error de PowerShell al activar venv

Solución:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

---

### Error: target no existe

El dataset debe tener una columna llamada:

```text
target
```

Si tiene otro nombre, cambiar en los archivos Python:

```python
TARGET_COL = "target"
```

---

### Error: p_codmes no existe

El dataset debe tener una columna llamada:

```text
p_codmes
```

---

### Error: API no encuentra el modelo

Solución:

```powershell
python main.py
uvicorn api:app --reload
```

---

## 18. Recomendaciones para GitHub

Antes de subir el proyecto, verificar que existan:

```text
README.md
requirements.txt
main.py
preprocessing.py
training.py
monitoring.py
postprocessing.py
api.py
dashboard.py
reports/informe_tecnico.pdf
```

No subir archivos pesados o sensibles:

```text
data/raw/Data_CU_venta.csv
mlruns/
venv/
```

Estos deben estar incluidos en `.gitignore`.

---

## 19. Comando final de validación

Para validar rápidamente:

```powershell
python main.py --input "data/raw/Data_CU_venta.csv" --n-trials 5
```

Si termina sin errores y genera los archivos de salida, el pipeline está funcionando.
