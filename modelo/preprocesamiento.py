"""Limpieza, imputación, eliminación de atípicos y codificación del dataset de crédito."""

import numpy as np
import pandas as pd

from modelo.configuracion import (
    COLUMNA_CLIENTE,
    COLUMNA_OBJETIVO,
    COLUMNAS_CATEGORICAS,
    COLUMNAS_MODELO,
    COLUMNAS_NUMERICAS,
    MAPA_CREDIT_MIX,
    MAPA_OBJETIVO,
    MAPA_PAGO_MINIMO,
)

PATRON_HISTORIAL = r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?"


def cargar_datos(ruta) -> pd.DataFrame:
    df = pd.read_csv(ruta, low_memory=False)
    return df[[COLUMNA_CLIENTE] + COLUMNAS_MODELO + [COLUMNA_OBJETIVO]].copy()


def a_numero(serie: pd.Series) -> pd.Series:
    texto = serie.astype(str).str.replace("_", "", regex=False).str.strip()
    return pd.to_numeric(texto, errors="coerce")


def historial_a_meses(serie: pd.Series) -> pd.Series:
    partes = serie.astype(str).str.extract(PATRON_HISTORIAL).astype(float)
    return partes[0] * 12 + partes[1]


def corregir_formatos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Outstanding_Debt"] = a_numero(df["Outstanding_Debt"])
    df["Num_of_Delayed_Payment"] = a_numero(df["Num_of_Delayed_Payment"])
    df["Credit_History_Age"] = historial_a_meses(df["Credit_History_Age"])

    df.loc[df["Num_of_Delayed_Payment"] < 0, "Num_of_Delayed_Payment"] = np.nan
    df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", np.nan)
    return df


def _moda(serie: pd.Series):
    moda = serie.mode()
    return moda.iloc[0] if not moda.empty else np.nan


def imputar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena con la mediana / moda del mismo cliente y, si no existe, con la global."""
    df = df.copy()
    grupos = df.groupby(COLUMNA_CLIENTE)

    for c in COLUMNAS_NUMERICAS:
        df[c] = df[c].fillna(grupos[c].transform("median"))
        df[c] = df[c].fillna(df[c].median())

    for c in COLUMNAS_CATEGORICAS:
        df[c] = df[c].fillna(grupos[c].transform(_moda))
        df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df


def conteo_atipicos_iqr(df: pd.DataFrame, columnas) -> pd.DataFrame:
    filas = []
    for c in columnas:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        inf, sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = ((df[c] < inf) | (df[c] > sup)).sum()
        filas.append({"variable": c, "Q1": q1, "Q3": q3, "IQR": iqr,
                      "limite_inferior": inf, "limite_superior": sup, "atipicos": int(n)})
    return pd.DataFrame(filas).set_index("variable")


def eliminar_atipicos_iqr(df: pd.DataFrame, columnas) -> pd.DataFrame:
    for c in columnas:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[c] >= q1 - 1.5 * iqr) & (df[c] <= q3 + 1.5 * iqr)]
    return df


def codificar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Credit_Mix"] = df["Credit_Mix"].map(MAPA_CREDIT_MIX)
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].map(MAPA_PAGO_MINIMO)
    if COLUMNA_OBJETIVO in df.columns:
        df[COLUMNA_OBJETIVO] = df[COLUMNA_OBJETIVO].map(MAPA_OBJETIVO)
    return df


def preparar_dataset(ruta):
    """Ejecuta todo el preprocesamiento y devuelve X, y y los grupos por cliente."""
    df = cargar_datos(ruta)
    df = corregir_formatos(df)
    df = imputar_nulos(df)
    df = eliminar_atipicos_iqr(df, COLUMNAS_NUMERICAS)
    df = codificar(df)
    X = df[COLUMNAS_MODELO]
    y = df[COLUMNA_OBJETIVO]
    grupos = df[COLUMNA_CLIENTE]
    return X, y, grupos


def solicitud_a_dataframe(solicitud: dict) -> pd.DataFrame:
    """Convierte una solicitud individual (valores originales) al formato de entrada del modelo."""
    fila = dict(solicitud)
    fila["Credit_Mix"] = MAPA_CREDIT_MIX[fila["Credit_Mix"]]
    fila["Payment_of_Min_Amount"] = MAPA_PAGO_MINIMO[fila["Payment_of_Min_Amount"]]
    return pd.DataFrame([fila])[COLUMNAS_MODELO]
