"""
dashboard.py -- Dashboard interactivo con Streamlit.

Ejecutar:
    streamlit run dashboard.py

Muestra:
    - Distribución de grupos de ejecución TLV
    - Top-N clientes por puntuación TLV
    - Métricas de monitoreo
    - Recall acumulado por decil
    - Distribución de scores train vs validación
"""

from pathlib import Path
import json

import pandas as pd
import streamlit as st


POST_PATH = Path("data/postprocessed/output_tlv.csv")
MONITORING_METRICS_PATH = Path("data/monitoring/monitoring_metrics.json")
RECALL_DECILE_PATH = Path("data/monitoring/recall_by_decile.csv")
SCORE_DISTRIBUTION_PATH = Path("data/monitoring/score_distribution.csv")


st.set_page_config(
    page_title="Dashboard Pipeline ML E2E",
    page_icon="📊",
    layout="wide",
)


st.title("Dashboard Pipeline ML End-to-End")
st.caption("Visualización de resultados, segmentación TLV y monitoreo básico del modelo.")


def load_csv(path: Path):
    """Carga un CSV si existe; si no, devuelve None."""
    if path.exists():
        return pd.read_csv(path)
    return None


def load_json(path: Path):
    """Carga un JSON si existe; si no, devuelve None."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return None


df_post = load_csv(POST_PATH)
metrics = load_json(MONITORING_METRICS_PATH)
df_recall = load_csv(RECALL_DECILE_PATH)
df_scores = load_csv(SCORE_DISTRIBUTION_PATH)


if df_post is None:
    st.warning(
        "No se encontró data/postprocessed/output_tlv.csv. "
        "Ejecuta primero el pipeline con: python main.py"
    )
    st.stop()


# KPIs principales
st.subheader("Indicadores principales")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Registros procesados", f"{len(df_post):,}")

with col2:
    if "prob" in df_post.columns:
        st.metric("Score promedio", f"{df_post['prob'].mean():.4f}")
    else:
        st.metric("Score promedio", "N/D")

with col3:
    if "puntuacion_tlv" in df_post.columns:
        st.metric("TLV promedio", f"{df_post['puntuacion_tlv'].mean():.10f}")
    else:
        st.metric("TLV promedio", "N/D")

with col4:
    if metrics:
        st.metric("PSI", f"{metrics.get('psi_score', 0):.4f}", metrics.get("psi_flag", "N/D"))
    else:
        st.metric("PSI", "N/D")


# Distribución de grupos
st.subheader("Distribución de grupos de ejecución TLV")

if "grupo_ejec_tlv" in df_post.columns:
    group_counts = (
        df_post["grupo_ejec_tlv"]
        .value_counts()
        .sort_index()
        .rename_axis("grupo_ejec_tlv")
        .reset_index(name="cantidad")
    )

    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        st.bar_chart(group_counts.set_index("grupo_ejec_tlv"))

    with col_table:
        st.dataframe(group_counts, use_container_width=True)
else:
    st.info("No se encontró la columna grupo_ejec_tlv.")


# Top clientes
st.subheader("Top clientes por puntuación TLV")

top_n = st.slider("Cantidad de registros a mostrar", min_value=5, max_value=100, value=20, step=5)

columns_candidates = [
    "key_value",
    "prob",
    "puntuacion_tlv",
    "grupo_ejec_tlv",
    "monto",
    "prob_value_contact",
    "grp_campecs06m",
]
available_cols = [col for col in columns_candidates if col in df_post.columns]

if "puntuacion_tlv" in df_post.columns:
    top_df = (
        df_post[available_cols]
        .sort_values("puntuacion_tlv", ascending=False)
        .head(top_n)
    )
    st.dataframe(top_df, use_container_width=True)
else:
    st.info("No se encontró la columna puntuacion_tlv.")


# Monitoreo
st.subheader("Monitoreo del modelo")

if metrics:
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("AUC validación", f"{metrics.get('val_auc', 0):.4f}")
    col_b.metric("Recall @ 0.5", f"{metrics.get('val_recall_threshold_0_5', 0):.4f}")
    col_c.metric("Tasa target train", f"{metrics.get('train_target_rate', 0):.4f}")
    col_d.metric("Tasa target validación", f"{metrics.get('val_target_rate', 0):.4f}")

    st.json(metrics)
else:
    st.info("No se encontró data/monitoring/monitoring_metrics.json.")


# Recall por decil
st.subheader("Recall acumulado por decil")

if df_recall is not None:
    if "decil" in df_recall.columns and "recall_acumulado" in df_recall.columns:
        st.line_chart(df_recall.set_index("decil")["recall_acumulado"])
    st.dataframe(df_recall, use_container_width=True)
else:
    st.info("No se encontró data/monitoring/recall_by_decile.csv.")


# Distribución de scores
st.subheader("Distribución de scores")

if df_scores is not None:
    st.dataframe(df_scores.groupby("dataset")["score"].describe(), use_container_width=True)
else:
    st.info("No se encontró data/monitoring/score_distribution.csv.")
