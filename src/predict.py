# -*- coding: utf-8 -*-
"""
Módulo de predicción.

Carga los artefactos generados por `src/train.py` y permite clasificar
transacciones nuevas:

    df_resultado = predecir(dataset_carga)
      -> agrega columnas: "Probabilidad_Fraude" y "Fraude_Predicho"

Funciona también si el archivo incluye la columna `Class` (etiqueta real):
en ese caso se agregan además las métricas de evaluación.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
MODELO_DIR = RAIZ / "models"
MODELO_PKL = MODELO_DIR / "rf_smote.pkl"
SCALER_PKL = MODELO_DIR / "scaler.pkl"
CONFIG_JSON = MODELO_DIR / "config.json"

_artefactos = None


def cargar_artefactos():
    """Carga (una sola vez) modelo, escalador y configuración."""
    global _artefactos
    if _artefactos is not None:
        return _artefactos

    if not MODELO_PKL.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODELO_PKL}. "
            "Ejecutá primero `python -m src.train` o `python src/train.py`."
        )

    with open(CONFIG_JSON, "r", encoding="utf-8") as f:
        config = json.load(f)

    _artefactos = {
        "modelo": joblib.load(MODELO_PKL),
        "scaler": joblib.load(SCALER_PKL),
        "umbral": config["umbral"],
        "columnas": config["columnas"],
        "columnas_escalar": config["columnas_escalar"],
    }
    return _artefactos


def validar_esquema(carga: pd.DataFrame) -> list:
    """Devuelve la lista de columnas obligatorias faltantes (vacía si todo está bien)."""
    col = cargar_artefactos()["columnas"]
    faltantes = [c for c in col if c not in carga.columns]
    return faltantes


def predecir(carga: pd.DataFrame) -> pd.DataFrame:
    """Clasifica cada transacción y devuelve un DataFrame con las predicciones."""
    artefactos = cargar_artefactos()
    umbral = artefactos["umbral"]

    faltantes = validar_esquema(carga)
    if faltantes:
        raise ValueError(
            "El archivo no tiene las columnas requeridas. Faltan: "
            + ", ".join(faltantes)
            + f" (se necesitan: {', '.join(artefactos['columnas'])})"
        )

    X = carga[artefactos["columnas"]].copy()
    X[artefactos["columnas_escalar"]] = artefactos["scaler"].transform(
        X[artefactos["columnas_escalar"]]
    )

    y_prob = artefactos["modelo"].predict_proba(X)[:, 1]

    resultado = carga.copy()
    resultado["Probabilidad_Fraude"] = y_prob
    resultado["Fraude_Predicho"] = (y_prob >= umbral).astype(int)
    return resultado


def metricas(carga: pd.DataFrame, resultado: pd.DataFrame) -> dict:
    """Calcula métricas solo si la data incluye la etiqueta real `Class`."""
    if "Class" not in resultado.columns:
        return {}

    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    y_real = resultado["Class"].astype(int)
    y_pred = resultado["Fraude_Predicho"]
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_real, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
    }


if __name__ == "__main__":
    # Demo rápida: predicción sobre un subconjunto del dataset original
    demo = pd.read_csv(RAIZ / "data" / "creditcard.csv").head(100)
    out = predecir(demo)
    print(out[["Time", "Amount", "Probabilidad_Fraude", "Fraude_Predicho"]].head(10))
    print(f"\nUmbral aplicado: {cargar_artefactos()['umbral']}")