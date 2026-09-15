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
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import predict

sns.set_style("whitegrid")
COLOR_NORMAL = "#2e86ab"
COLOR_FRAUDE = "#d7263d"


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


def grafico_amount(resultado) -> plt.Figure:
    """Boxplot de Amount por clase predicha (escala simétrica log)."""
    data_normal = resultado.loc[resultado["Fraude_Predicho"] == 0, "Amount"].values
    data_fraude = resultado.loc[resultado["Fraude_Predicho"] == 1, "Amount"].values

    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot([data_normal, data_fraude], tick_labels=["Normal (0)", "Fraude (1)"],
                    patch_artist=True)
    for patch, color in zip(bp["boxes"], [COLOR_NORMAL, COLOR_FRAUDE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_yscale("symlog")
    ax.set_title("Distribución de Amount por clase predicha")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Monto (escala symlog)")
    return fig


def grafico_tiempo(resultado) -> plt.Figure:
    """Histograma de Time por clase predicha."""
    fig, ax = plt.subplots(figsize=(8, 4))
    normal = resultado[resultado["Fraude_Predicho"] == 0]["Time"]
    fraude = resultado[resultado["Fraude_Predicho"] == 1]["Time"]
    ax.hist(normal, bins=30, alpha=0.6, label="Normal", color=COLOR_NORMAL)
    ax.hist(fraude, bins=30, alpha=0.6, label="Fraude", color=COLOR_FRAUDE)
    ax.set_yscale("log")
    ax.set_xlabel("Time (segundos desde la primera transacción)")
    ax.set_ylabel("Frecuencia (log)")
    ax.set_title("Distribución de Time por clase predicha")
    ax.legend()
    return fig


def grafico_top_n(resultado, n=10) -> plt.Figure:
    """Ranking de las N transacciones con mayor probabilidad de fraude."""
    top = resultado.nlargest(n, "Probabilidad_Fraude").sort_values("Probabilidad_Fraude")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * n)))
    colores = [COLOR_FRAUDE if c == 1 else COLOR_NORMAL
               for c in top["Fraude_Predicho"]]
    ax.barh(range(len(top)), top["Probabilidad_Fraude"], color=colores)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"#{i:>2}  {monto:,.0f} $" for i, monto
                        in enumerate(top["Amount"], start=1)])
    ax.axvline(obtener_umbral(), color="black", linestyle="--", lw=1.2,
               label=f"Umbral {obtener_umbral():.1f}")
    ax.set_xlabel("Probabilidad de fraude")
    ax.set_title(f"Top {n} transacciones más riesgosas (monto en la etiqueta)")
    ax.legend()
    fig.tight_layout()
    return fig


def grafico_captura(resultado) -> plt.Figure:
    """Curva de captura acumulada: fraude capturado vs % de transacciones revisadas."""
    orden = resultado.sort_values("Probabilidad_Fraude", ascending=False)
    if "Class" in orden.columns:
        fraude_real = orden["Class"].astype(int).cumsum()
        total = int(orden["Class"].sum()) or 1
        etiqueta = "fraude real"
    else:
        fraude_real = orden["Fraude_Predicho"].cumsum()
        total = int(orden["Fraude_Predicho"].sum()) or 1
        etiqueta = "fraude predicho"

    n_pts = len(orden)
    stride = max(1, n_pts // 2000)
    x = np.arange(1, n_pts + 1, stride) / n_pts
    y = fraude_real.iloc[::stride].values / total

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x * 100, y * 100, color=COLOR_FRAUDE, lw=2)
    ax.plot([0, 100], [0, 100], "k--", lw=1, label="Referencia (aleatorio)")
    ax.set_xlabel("% de transacciones revisadas")
    ax.set_ylabel(f"% de {etiqueta} capturado")
    ax.set_title("Curva de captura acumulada")
    ax.legend()
    return fig


def grafico_roc_pr(resultado) -> plt.Figure:
    """Curvas ROC y Precision-Recall usando la etiqueta real (requiere Class)."""
    y_real = resultado["Class"].astype(int)
    y_prob = resultado["Probabilidad_Fraude"]

    fpr, tpr, _ = roc_curve(y_real, y_prob)
    auc = roc_auc_score(y_real, y_prob)
    prec, rec, _ = precision_recall_curve(y_real, y_prob)
    ap = average_precision_score(y_real, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(fpr, tpr, color=COLOR_FRAUDE, lw=2, label=f"ROC (AUC = {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Aleatorio")
    axes[0].set_xlabel("Falsos positivos (rate)")
    axes[0].set_ylabel("Verdaderos positivos (rate)")
    axes[0].set_title("Curva ROC sobre tus datos")
    axes[0].legend()

    axes[1].plot(rec, prec, color=COLOR_FRAUDE, lw=2, label=f"PR (AP = {ap:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Curva Precision-Recall")
    axes[1].legend()

    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Fraude en Tarjetas de Crédito", layout="wide")

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

    st.subheader("Análisis visual")

    col_amount, col_tiempo = st.columns(2)
    with col_amount:
        st.pyplot(grafico_amount(resultado))
    with col_tiempo:
        st.pyplot(grafico_tiempo(resultado))

    st.markdown("**Top transacciones más riesgosas**")
    n_top = st.slider("Cantidad de transacciones a mostrar", min_value=5,
                      max_value=50, value=10, step=5)
    st.pyplot(grafico_top_n(resultado, n_top))

    st.pyplot(grafico_captura(resultado))

    if "Class" in resultado.columns:
        st.subheader("Evaluación sobre los datos subidos (con etiqueta real)")
        m = predict.metricas(carga, resultado)
        cols = st.columns(5)
        cols[0].metric("Precision", f"{m['precision']:.3f}")
        cols[1].metric("Recall", f"{m['recall']:.3f}")
        cols[2].metric("F1", f"{m['f1']:.3f}")
        cols[3].metric("Falsos positivos", m["fp"])
        cols[4].metric("Fraudes no detectados (FN)", m["fn"])

        if resultado["Class"].nunique() < 2:
            st.info("La data tiene una sola clase: las curvas ROC y Precision-Recall "
                    "no se pueden calcular. Se muestra la matriz de confusión.")
        else:
            col_roc, col_cm = st.columns(2)
            with col_roc:
                st.pyplot(grafico_roc_pr(resultado))
            with col_cm:
                st.pyplot(grafico_matriz_confusion(m))

    descarga = resultado.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar predicciones (CSV)",
        data=descarga,
        file_name="predicciones_fraude.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()