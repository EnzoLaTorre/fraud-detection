# -*- coding: utf-8 -*-
"""
Dashboard interactivo de detección de fraude.

Un usuario sube un CSV con transacciones (columnas Time, Amount, V1-V28) y
recibe predicciones al instante. Si el archivo incluye la columna `Class`
(etiqueta real), además se muestran las métricas de evaluación.

Ejecutar con:  streamlit run app.py
"""

import io
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
RAIZ = Path(__file__).resolve().parent
DATASET_EJEMPLO = RAIZ / "data" / "creditcard.csv"
DATASET_EJEMPLO_MUESTRA = RAIZ / "data" / "creditcard_ejemplo.csv"


def obtener_umbral():
    return predict.cargar_artefactos()["umbral"]


def es_csv(upload):
    return upload.name.lower().endswith(".csv")


def grafico_probabilidad(resultado, umbral) -> plt.Figure:
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
    ax.axvline(umbral, color="black", linestyle="--", lw=1.5,
               label=f"Umbral {umbral:.2f}")
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


def grafico_top_n(resultado, n=10, umbral=None) -> plt.Figure:
    """Ranking de las N transacciones con mayor probabilidad de fraude."""
    top = resultado.nlargest(n, "Probabilidad_Fraude").sort_values("Probabilidad_Fraude")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * n)))
    colores = [COLOR_FRAUDE if c == 1 else COLOR_NORMAL
               for c in top["Fraude_Predicho"]]
    ax.barh(range(len(top)), top["Probabilidad_Fraude"], color=colores)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"#{i:>2}  {monto:,.0f} $" for i, monto
                        in enumerate(top["Amount"], start=1)])
    if umbral is not None:
        ax.axvline(umbral, color="black", linestyle="--", lw=1.2,
                   label=f"Umbral {umbral:.2f}")
        ax.legend()
    ax.set_xlabel("Probabilidad de fraude")
    ax.set_title(f"Top {n} transacciones más riesgosas (monto en la etiqueta)")
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


def resumen_calidad(carga) -> pd.DataFrame:
    """Tabla de tipos de dato y nulos por columna."""
    return pd.DataFrame({
        "Columna": carga.columns,
        "Tipo": carga.dtypes.astype(str).values,
        "Nulos": carga.isnull().sum().values,
    })


def grafico_importancia(n=15) -> plt.Figure:
    """Importancia de las variables del Random Forest (del modelo guardado)."""
    variables, importancias = predict.importancia_modelo(n)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * n)))
    ax.barh(range(len(variables)), importancias[::-1], color=COLOR_NORMAL)
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels(variables[::-1])
    ax.invert_yaxis()
    ax.set_xlabel("Importancia")
    ax.set_title(f"Top {n} variables más importantes del modelo")
    fig.tight_layout()
    return fig


def barrido_umbrales(resultado, umbrales) -> pd.DataFrame:
    """Tabla umbral -> precision/recall/F1/FP/FN usando la etiqueta real."""
    filas = []
    for t in umbrales:
        r = predict.clasificar(resultado, float(t))
        m = predict.metricas(resultado, r)
        if not m:
            continue
        filas.append({
            "Umbral": round(float(t), 2),
            "Precision": round(m["precision"], 3),
            "Recall": round(m["recall"], 3),
            "F1": round(m["f1"], 3),
            "FP": m["fp"],
            "FN": m["fn"],
            "Fraudes detectados": m["tp"],
        })
    return pd.DataFrame(filas)


def extraer_fp_fn(resultado):
    """DataFrames con las transacciones de cada tipo de error (requiere Class)."""
    y_real = resultado["Class"].astype(int)
    fp = resultado[(y_real == 0) & (resultado["Fraude_Predicho"] == 1)]
    fn = resultado[(y_real == 1) & (resultado["Fraude_Predicho"] == 0)]
    cols = ["Time", "Amount", "Probabilidad_Fraude", "Class"]
    return fp[cols].copy(), fn[cols].copy()


def analisis_economico(resultado, costo_revision, perdida_promedio) -> tuple:
    """Costo total por umbral y umbral recomendado (requiere Class).

    Costo = FP * costo_revision  +  FN * perdida_promedio
    """
    umbrales = np.round(np.arange(0.05, 1.0, 0.05), 2)
    filas = []
    for t in umbrales:
        r = predict.clasificar(resultado, float(t))
        m = predict.metricas(resultado, r)
        if not m:
            continue
        costo = m["fp"] * costo_revision + m["fn"] * perdida_promedio
        filas.append({
            "Umbral": round(float(t), 2),
            "FP": m["fp"],
            "FN": m["fn"],
            "Costo total ($)": round(float(costo), 2),
        })
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla, None
    mejor = tabla.loc[tabla["Costo total ($)"].idxmin()]
    return tabla, mejor


def _cargar_ejemplo():
    """Muestra balanceada (normales + todos los fraudes) del dataset.

    Si existe el archivo de ejemplo ya generado (commiteado para el cloud),
    lo usa directo; si no, lo construye desde el dataset completo local.
    """
    if DATASET_EJEMPLO_MUESTRA.exists():
        return pd.read_csv(DATASET_EJEMPLO_MUESTRA).drop_duplicates()
    df = pd.read_csv(DATASET_EJEMPLO).drop_duplicates()
    normales = df[df["Class"] == 0].sample(15000, random_state=42)
    fraudes = df[df["Class"] == 1]
    return pd.concat([normales, fraudes]).sample(frac=1, random_state=42)


@st.cache_data(show_spinner=False)
def _leer_predecir(fuente: bytes) -> pd.DataFrame:
    """Lee el CSV (o ejemplo) y predice probabilidades. Se cachea por contenido."""
    if fuente == b"__ejemplo__":
        carga = _cargar_ejemplo()
    else:
        carga = pd.read_csv(io.BytesIO(fuente))
    return predict.predecir(carga)


def main():
    st.set_page_config(page_title="Fraude en Tarjetas de Crédito", layout="wide")

    st.title("Detección de Fraude en Tarjetas de Crédito")
    st.markdown(
        "Carga un archivo CSV con las transacciones y el sistema analizará "
        "automáticamente cada registro para determinar su **probabilidad de fraude** "
        "y asignarle una **clasificación**.\n\n"
        "También podrás utilizar un **conjunto de datos de ejemplo** para realizar "
        "pruebas.\n\n"
        "Si el archivo contiene la columna **Class** (0 = transacción normal, "
        "1 = fraude), el sistema mostrará adicionalmente las **métricas de evaluación "
        "del modelo**, el análisis de diferentes **umbrales de clasificación** y una "
        "evaluación del **impacto económico** de las decisiones tomadas."
    )

    try:
        umbral_inicial = obtener_umbral()
        umbral_ok = True
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Instrucciones: primero ejecutá `python src/train.py` para generar "
                "el modelo y el escalador.")
        umbral_inicial = 0.7
        umbral_ok = False

    with st.sidebar:
        st.header("Información")
        st.markdown(
            "**Esquema esperado:** `Time`, `Amount` y `V1`–`V28` (30 columnas). "
            "La columna `Class` es opcional (etiqueta real)."
        )
        st.markdown(
            "**Modelo:** Random Forest con SMOTE, entrenado sobre el dataset "
            "Credit Card Fraud Detection (Kaggle)."
        )
        st.header("Configuración")
        umbral = st.slider("Umbral de decisión", 0.10, 0.90, umbral_inicial, 0.05)
        costo_revision = st.number_input("Costo por revisión falsa (FP) $",
                                         min_value=0.0, max_value=100.0,
                                         value=5.0, step=1.0)
        perdida_promedio = st.number_input("Pérdida promedio por fraude no detectado (FN) $",
                                           min_value=0.0, max_value=10000.0,
                                           value=100.0, step=10.0)

    if not umbral_ok:
        st.stop()

    fuente = st.radio("Fuente de datos", ["Subir CSV", "Dataset de ejemplo"],
                      horizontal=True)

    if fuente == "Dataset de ejemplo":
        if not (DATASET_EJEMPLO_MUESTRA.exists() or DATASET_EJEMPLO.exists()):
            st.error("No se encontró `data/creditcard_ejemplo.csv` para el dataset de ejemplo.")
            st.stop()
        bytes_fuente = b"__ejemplo__"
    else:
        archivo = st.file_uploader("Subí el CSV con las transacciones", type=["csv"])
        if archivo is None:
            st.info("Subí un CSV para comenzar.")
            st.stop()
        if not es_csv(archivo):
            st.error("El archivo debe ser un CSV (.csv).")
            st.stop()
        bytes_fuente = archivo.getvalue()

    try:
        resultado = _leer_predecir(bytes_fuente)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    resultado = predict.clasificar(resultado, umbral)
    columnas_orig = [c for c in resultado.columns
                     if c not in {"Probabilidad_Fraude", "Fraude_Predicho"}]
    tiene_class = "Class" in resultado.columns

    tab_datos, tab_pred, tab_visual, tab_eval, tab_eco = st.tabs(
        ["1. Datos", "2. Predicciones", "3. Análisis visual",
         "4. Evaluación", "5. Impacto económico"]
    )

    with tab_datos:
        st.subheader("Calidad de los datos")
        nulos = int(resultado[columnas_orig].isna().sum().sum())
        duplicados = int(resultado[columnas_orig].duplicated().sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas", f"{resultado.shape[0]:,}")
        c2.metric("Columnas", resultado.shape[1])
        c3.metric("Valores nulos", nulos)
        c4.metric("Duplicados", duplicados)
        if tiene_class:
            vc = resultado["Class"].value_counts()
            st.write(f"**Fraudes en la data:** {int(vc.get(1, 0)):,} | "
                     f"**Normales:** {int(vc.get(0, 0)):,} "
                     f"({(vc.get(1, 0) / max(1, vc.sum()) * 100):.3f}%)")
        st.dataframe(resumen_calidad(resultado[columnas_orig]),
                     use_container_width=True)

        st.subheader("Vista previa de los datos")
        st.dataframe(resultado[columnas_orig].head(1000), use_container_width=True)

    with tab_pred:
        st.subheader("Resultados de la predicción")
        pred_fraudes = int(resultado["Fraude_Predicho"].sum())
        p1, p2, p3 = st.columns(3)
        p1.metric("Transacciones analizadas", f"{resultado.shape[0]:,}")
        p2.metric(f"Como fraude (umbral {umbral:.2f})", f"{pred_fraudes:,}")
        p3.metric("% de fraude predicho",
                  f"{pred_fraudes / max(1, resultado.shape[0]) * 100:.2f}%")

        col_hist, col_est = st.columns([2, 1])
        with col_hist:
            st.pyplot(grafico_probabilidad(resultado, umbral))
        with col_est:
            conteo = resultado["Fraude_Predicho"].value_counts()
            est = pd.DataFrame({
                "Clase": ["Normal (0)", "Fraude (1)"],
                "Transacciones": [int(conteo.get(0, 0)), int(conteo.get(1, 0))],
            })
            st.dataframe(est, use_container_width=True)
            st.markdown("### Interpretación")
            st.write(
                "Cada transacción con probabilidad ≥ {:.2f} se marca como fraude. "
                "Las marcadas como fraude requieren revisión manual.".format(umbral)
            )

        st.markdown("**Top transacciones más riesgosas**")
        n_top = st.slider("Cantidad de transacciones a mostrar", min_value=5,
                          max_value=50, value=10, step=5)
        st.pyplot(grafico_top_n(resultado, n_top, umbral))

        st.subheader("Predicción por transacción")
        st.dataframe(
            resultado[["Probabilidad_Fraude", "Fraude_Predicho"]].tail(1000),
            use_container_width=True,
        )

        descarga = resultado.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar predicciones (CSV)",
            data=descarga,
            file_name="predicciones_fraude.csv",
            mime="text/csv",
        )

    with tab_visual:
        st.subheader("Análisis visual")
        col_amount, col_tiempo = st.columns(2)
        with col_amount:
            st.pyplot(grafico_amount(resultado))
        with col_tiempo:
            st.pyplot(grafico_tiempo(resultado))
        st.pyplot(grafico_captura(resultado))

        st.subheader("Interpretabilidad del modelo")
        st.pyplot(grafico_importancia())
        st.caption("Importancia de las variables del Random Forest guardado "
                   "(independiente de los datos subidos).")

    with tab_eval:
        if not tiene_class:
            st.info("Subí un archivo con la columna `Class` para ver la "
                    "evaluación del modelo.")
        else:
            st.subheader("Evaluación sobre los datos subidos (etiqueta real)")
            m = predict.metricas(resultado, resultado)
            cols = st.columns(5)
            cols[0].metric("Precision", f"{m['precision']:.3f}")
            cols[1].metric("Recall", f"{m['recall']:.3f}")
            cols[2].metric("F1", f"{m['f1']:.3f}")
            cols[3].metric("Falsos positivos", m["fp"])
            cols[4].metric("Fraudes no detectados (FN)", m["fn"])

            fp_df, fn_df = extraer_fp_fn(resultado)
            tab_fp, tab_fn, tab_curvas = st.tabs(
                ["Falsos positivos", "Falsos negativos", "Curvas y matriz de confusión"]
            )
            with tab_fp:
                st.caption("Normales marcadas como fraude (falsa alarma).")
                st.dataframe(fp_df, use_container_width=True)
            with tab_fn:
                st.caption("Fraudes reales no detectados (pérdida de dinero).")
                st.dataframe(fn_df, use_container_width=True)
            with tab_curvas:
                if resultado["Class"].nunique() < 2:
                    st.info("La data tiene una sola clase: las curvas ROC y "
                            "Precision-Recall no se pueden calcular. Se muestra la "
                            "matriz de confusión.")
                    st.pyplot(grafico_matriz_confusion(m))
                else:
                    col_roc, col_cm = st.columns(2)
                    with col_roc:
                        st.pyplot(grafico_roc_pr(resultado))
                    with col_cm:
                        st.pyplot(grafico_matriz_confusion(m))

            st.subheader("Barrido de umbrales")
            tabla_umbrales = barrido_umbrales(resultado, np.arange(0.10, 1.0, 0.05))
            st.dataframe(tabla_umbrales, use_container_width=True)
            mejor_f1 = tabla_umbrales.loc[tabla_umbrales["F1"].idxmax()]
            st.markdown(
                f"**Mejor F1:** umbral **{mejor_f1['Umbral']:.2f}** "
                f"(F1 = {mejor_f1['F1']:.3f}, Precision = {mejor_f1['Precision']:.3f}, "
                f"Recall = {mejor_f1['Recall']:.3f})"
            )

    with tab_eco:
        if not tiene_class:
            st.info("Subí un archivo con la columna `Class` para ver el "
                    "análisis económico del umbral.")
        else:
            st.subheader("Análisis económico del umbral")
            tabla_economico, mejor_economico = analisis_economico(
                resultado, costo_revision, perdida_promedio
            )
            col_est, col_rec = st.columns([3, 1])
            with col_est:
                st.dataframe(tabla_economico, use_container_width=True)
            with col_rec:
                st.metric("Mejor umbral (costo)", f"{mejor_economico['Umbral']:.2f}")
                st.metric("Costo mínimo total",
                          f"${mejor_economico['Costo total ($)']:,.2f}")
            st.caption(
                f"Costo = FP × ${costo_revision:,.0f} + FN × ${perdida_promedio:,.0f} "
                "por umbral. Ajustá los valores en la barra lateral."
            )


if __name__ == "__main__":
    main()