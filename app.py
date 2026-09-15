# -*- coding: utf-8 -*-
"""
Dashboard interactivo de detección de fraude.

Un usuario sube un CSV con transacciones (columnas Time, Amount, V1-V28) y
recibe predicciones al instante. Si el archivo incluye la columna `Class`
(etiqueta real), además se muestran las métricas de evaluación.

Ejecutar con:  streamlit run app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import predict

sns.set_style("whitegrid")
COLOR_NORMAL = "#2e86ab"
COLOR_FRAUDE = "#d7263d"

st.set_page_config(page_title="Fraude en Tarjetas de Crédito", layout="wide")


def obtener_umbral():
    return predict.cargar_artefactos()["umbral"]


def es_csv(upload):
    return upload.name.lower().endswith(".csv")


def grafico_probabilidad(resultado) -> plt.Figure:
    """Histograma de probabilidades por clase predicha."""
    fig, ax = plt.subplots(figsize=(8, 4))
    resultado_normal = resultado[resultado["Fraude_Predicho"] == 0]
    resultado_fraude = resultado[resultado["Fraude_Predicho"] == 1]
    if not resultado_normal.empty:
        ax.hist(resultado_normal["Probabilidad_Fraude"], bins=30, alpha=0.7,
                label="Normal", color=COLOR_NORMAL)
    if not resultado_fraude.empty:
        ax.hist(resultado_fraude["Probabilidad_Fraude"], bins=30, alpha=0.7,
                label="Fraude", color=COLOR_FRAUDE)
    ax.axvline(obtener_umbral(), color="black", linestyle="--", lw=1.5,
               label=f"Umbral {obtener_umbral():.1f}")
    ax.set_xlabel("Probabilidad de fraude")
    ax.set_ylabel("Cantidad de transacciones")
    ax.set_title("Distribución de probabilidades por clase predicha")
    ax.legend()
    return fig


def grafico_matriz_confusion(m) -> plt.Figure:
    """Matriz de confusión cuando la data trae la etiqueta real."""
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = pd.DataFrame(
        [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
        index=["Normal (real)", "Fraude (real)"],
        columns=["Normal (pred)", "Fraude (pred)"],
    )
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Matriz de confusión sobre los datos subidos")
    return fig


st.title("Detección de Fraude en Tarjetas de Crédito")
st.markdown(
    "Subí un archivo **CSV** con transacciones y obtendrás la probabilidad de "
    "fraude y la clasificación al instante. "
    "Si además incluye la columna `Class` (0 = normal, 1 = fraude), verás las "
    "métricas de desempeño calculadas sobre tus datos."
)

try:
    obtener_umbral()
    umbral_ok = True
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Instrucciones: primero ejecutá `python src/train.py` para generar "
            "el modelo y el escalador.")
    umbral_ok = False

with st.sidebar:
    st.header("Información")
    st.markdown(
        "**Esquema esperado:** `Time`, `Amount` y `V1`–`V28` (30 columnas). "
        "La columna `Class` es opcional (etiqueta real)."
    )
    if umbral_ok:
        st.metric("Umbral de decisión", f"{obtener_umbral():.1f}")
    st.markdown(
        "**Modelo:** Random Forest con SMOTE, entrenado sobre el dataset "
        "Credit Card Fraud Detection (Kaggle)."
    )

if not umbral_ok:
    st.stop()

archivo = st.file_uploader("Subí el CSV con las transacciones", type=["csv"])

if archivo is None:
    st.info("Subí un CSV para comenzar.")
    st.stop()

if not es_csv(archivo):
    st.error("El archivo debe ser un CSV (.csv).")
    st.stop()

carga = pd.read_csv(archivo)
st.subheader("Vista previa de los datos")
st.write(f"Filas: {carga.shape[0]:,} | Columnas: {carga.shape[1]}")

faltantes = predict.validar_esquema(carga)
if faltantes:
    st.error(
        "El archivo no tiene el esquema requerido. Columnas faltantes: "
        f"{', '.join(faltantes)}."
    )
    st.stop()

st.dataframe(carga.head(1000), use_container_width=True)

try:
    resultado = predict.predecir(carga)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.subheader("Resultados de la predicción")
st.metric("Transacciones clasificadas como fraude",
          int(resultado["Fraude_Predicho"].sum()))

st.dataframe(
    resultado[["Probabilidad_Fraude", "Fraude_Predicho"]].tail(1000),
    use_container_width=True,
)

col_hist, col_estado = st.columns([2, 1])
with col_hist:
    st.pyplot(grafico_probabilidad(resultado))
with col_estado:
    conteo = resultado["Fraude_Predicho"].value_counts()
    est = pd.DataFrame({
        "Clase": ["Normal (0)", "Fraude (1)"],
        "Transacciones": [int(conteo.get(0, 0)), int(conteo.get(1, 0))],
    })
    st.dataframe(est, use_container_width=True)
    st.markdown("### Interpretación")
    st.write(
        "Cada transacción con probabilidad ≥ {:.1f} se marca como fraude. "
        "Las marcadas como fraude requieren revisión manual.".format(obtener_umbral())
    )

if "Class" in resultado.columns:
    m = predict.metricas(carga, resultado)
    st.subheader("Métricas sobre los datos subidos (con etiqueta real)")
    cols = st.columns(5)
    cols[0].metric("Precision", f"{m['precision']:.3f}")
    cols[1].metric("Recall", f"{m['recall']:.3f}")
    cols[2].metric("F1", f"{m['f1']:.3f}")
    cols[3].metric("Falsos positivos", m["fp"])
    cols[4].metric("Fraudes no detectados (FN)", m["fn"])
    st.pyplot(grafico_matriz_confusion(m))

descarga = resultado.to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar predicciones (CSV)",
    data=descarga,
    file_name="predicciones_fraude.csv",
    mime="text/csv",
)