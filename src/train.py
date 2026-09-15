# -*- coding: utf-8 -*-
"""
Entrenamiento del modelo final de detección de fraude.

Replica el flujo del notebook `notebooks/fraud_analysis.ipynb` evitando
fuga de datos (data leakage):

    dataset -> split estratificado -> RobustScaler (SOLO train) -> SMOTE (SOLO train)
    -> Random Forest -> guardar artefactos

Salida:
    models/rf_smote.pkl   : modelo entrenado (Random Forest + SMOTE)
    models/scaler.pkl     : escalador RobustScaler ajustado en train
    models/config.json    : columnas requeridas, umbral de decisión y metadata
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data" / "creditcard.csv"
MODELO_DIR = RAIZ / "models"
MODELO_PKL = MODELO_DIR / "rf_smote.pkl"
SCALER_PKL = MODELO_DIR / "scaler.pkl"
CONFIG_JSON = MODELO_DIR / "config.json"

TEST_SIZE = 0.3
UMBRAL_FINAL = 0.7
RANDOM_STATE = 42
COLUMNAS_ESCALAR = ["Time", "Amount"]


def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Carga el dataset y elimina filas duplicadas (misma decisión que el notebook)."""
    df = pd.read_csv(ruta)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def preparar_datos(df: pd.DataFrame):
    """Split estratificado + escalado sin fuga de datos + SMOTE sobre train."""
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # 1) Split PRIMERO, siempre antes de ajustar el scaler
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 2) Scaler ajustado SOLO con entrenamiento
    scaler = RobustScaler()
    scaler.fit(X_train[COLUMNAS_ESCALAR])

    # 3) Transformar ambos conjuntos con ese scaler
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[COLUMNAS_ESCALAR] = scaler.transform(X_train[COLUMNAS_ESCALAR])
    X_test[COLUMNAS_ESCALAR] = scaler.transform(X_test[COLUMNAS_ESCALAR])

    # 4) Balanceo sintético SOLO sobre train; el test queda 100% real
    X_smote, y_smote = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)

    return X_smote, y_smote, X_test, y_test, scaler


def entrenar(X_smote: pd.DataFrame, y_smote: pd.Series) -> RandomForestClassifier:
    modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
    modelo.fit(X_smote, y_smote)
    return modelo


def evaluar(modelo, X_test: pd.DataFrame, y_test: pd.Series, umbral: float) -> None:
    """Evalúa el modelo elegido con el umbral final y muestra resumen en consola."""
    y_prob = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= umbral).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("=" * 56)
    print("EVALUACIÓN DEL MODELO FINAL (umbral {:.1f})".format(umbral))
    print("=" * 56)
    print(f"Fraudes reales en test:      {y_test.sum()}")
    print(f"Fraudes detectados:          {tp} ({rec*100:.1f}% del fraude)  [Recall]")
    print(f"Fraudes no detectados (FN):  {fn}")
    print(f"Normales marcadas (FP):      {fp}")
    print(f"Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print()


def guardar_artefactos(modelo, scaler, columnas: list) -> None:
    MODELO_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, MODELO_PKL)
    joblib.dump(scaler, SCALER_PKL)

    config = {
        "columnas": list(columnas),
        "columnas_escalar": COLUMNAS_ESCALAR,
        "umbral": UMBRAL_FINAL,
        "modelo": "RandomForestClassifier (SMOTE, n_estimators=100)",
        "descripcion": (
            "Modelo final de detección de fraude. Probabilidad >= umbral se "
            "clasifica como fraude. Las columnas Time y Amount deben escalarse "
            "con scaler.pkl antes de predecir."
        ),
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main() -> None:
    if not DATOS.exists():
        raise FileNotFoundError(
            f"No se encontró {DATOS}. Descargá el dataset de Kaggle y colocálo en data/."
        )

    print(f"Cargando datos desde {DATOS} ...")
    df = cargar_datos(DATOS)
    print(f"Registros tras limpiar duplicados: {df.shape[0]:,} | "
          f"fraudes: {(df['Class'] == 1).sum()}")

    print("Preparando datos (split -> escalar -> SMOTE) ...")
    X_smote, y_smote, X_test, y_test, scaler = preparar_datos(df)
    print(f"Entrenamiento con SMOTE: {X_smote.shape[0]:,} filas")

    print("Entrenando Random Forest ...")
    modelo = entrenar(X_smote, y_smote)

    print("Guardando artefactos ...")
    guardar_artefactos(modelo, scaler, list(X_smote.columns))
    print(f"  -> {MODELO_PKL}")
    print(f"  -> {SCALER_PKL}")
    print(f"  -> {CONFIG_JSON}")

    evaluar(modelo, X_test, y_test, UMBRAL_FINAL)


if __name__ == "__main__":
    main()