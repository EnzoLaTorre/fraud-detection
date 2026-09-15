# -*- coding: utf-8 -*-
"""Tests unitarios para src/predict.py.

Requiere los artefactos de models/ (ya incluidos en el repositorio).
Ejecutar con:  pytest -q
"""

import pandas as pd
import pytest

from src import predict

U = 0.7


def _carga_minima(n=5):
    """DataFrame con las 30 columnas requeridas por el modelo."""
    cols = predict.cargar_artefactos()["columnas"]
    return pd.DataFrame({c: [0.0] * n for c in cols})


def test_cargar_artefactos_tiene_claves_ok():
    art = predict.cargar_artefactos()
    for clave in ("modelo", "scaler", "umbral", "columnas", "columnas_escalar"):
        assert clave in art
    assert art["umbral"] == pytest.approx(0.7)
    assert len(art["columnas"]) == 30


def test_validar_esquema_sin_faltantes():
    assert predict.validar_esquema(_carga_minima()) == []


def test_validar_esquema_detecta_faltantes():
    falta = predict.validar_esquema(pd.DataFrame({"Time": [1.0]}))
    assert len(falta) == 29
    assert "V1" in falta


def test_predecir_falla_sin_columnas():
    with pytest.raises(ValueError, match="Faltan"):
        predict.predecir(pd.DataFrame({"x": [1.0]}))


def test_predecir_agrega_probabilidad_y_clase():
    carga = _carga_minima(10)
    out = predict.predecir(carga)
    assert {"Probabilidad_Fraude", "Fraude_Predicho"} <= set(out.columns)
    assert out["Probabilidad_Fraude"].between(0, 1).all()
    assert out["Fraude_Predicho"].isin([0, 1]).all()
    assert (out["Probabilidad_Fraude"] >= 0.7).astype(int).eq(
        out["Fraude_Predicho"]
    ).all()
    assert len(out) == len(carga)


def test_clasificar_sin_probabilidad_lanza_error():
    with pytest.raises(ValueError, match="Probabilidad_Fraude"):
        predict.clasificar(pd.DataFrame({"Fraude_Predicho": [1]}), U)


def test_clasificar_aplica_umbral_y_no_muta_original():
    df = pd.DataFrame({"Probabilidad_Fraude": [0.2, 0.7, 0.9]})
    original = df.copy()
    out = predict.clasificar(df, 0.7)
    assert list(out["Fraude_Predicho"]) == [0, 1, 1]
    assert df["Probabilidad_Fraude"].equals(original["Probabilidad_Fraude"])
    assert "Fraude_Predicho" not in original.columns


def test_clasificar_umbral_menor_detecta_mas():
    df = pd.DataFrame({"Probabilidad_Fraude": [0.0, 0.5, 1.0]})
    assert list(predict.clasificar(df, 0.3)["Fraude_Predicho"]) == [0, 1, 1]
    assert list(predict.clasificar(df, 0.9)["Fraude_Predicho"]) == [0, 0, 1]


def test_metricas_sin_class_devuelve_vacio():
    df = pd.DataFrame({"Probabilidad_Fraude": [0.1], "Fraude_Predicho": [0]})
    assert predict.metricas(df, df) == {}


def test_metricas_calculo_manual():
    df = pd.DataFrame({
        "Class": [0, 0, 0, 1, 1, 1, 1],
        "Probabilidad_Fraude": [0.1, 0.2, 0.9, 0.3, 0.6, 0.8, 0.99],
        "Fraude_Predicho": [0, 0, 1, 0, 0, 1, 1],
    })
    m = predict.metricas(df, df)
    assert m["tn"] == 2 and m["fp"] == 1
    assert m["tp"] == 2 and m["fn"] == 2
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))


def test_importancia_modelo_valores_descendentes():
    variables, importancias = predict.importancia_modelo(5)
    assert len(variables) == len(importancias) == 5
    assert importancias.tolist() == sorted(importancias.tolist(), reverse=True)
    assert all(v in predict.cargar_artefactos()["columnas"] for v in variables)