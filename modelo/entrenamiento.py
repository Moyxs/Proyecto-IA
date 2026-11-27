"""Entrenamiento, comparación, evaluación y guardado del modelo de riesgo crediticio."""

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

from modelo.configuracion import (
    COLUMNAS_MODELO,
    DIR_ARTEFACTOS,
    ETIQUETAS_RIESGO,
    MAPA_CREDIT_MIX,
    MAPA_PAGO_MINIMO,
    NIVELES_RIESGO,
    RANDOM_STATE,
    RUTA_DATOS,
    RUTA_METRICAS,
    RUTA_MODELO,
    TEST_SIZE,
)
from modelo.preprocesamiento import preparar_dataset

VALORES_K = [5, 15, 31, 51]
PROFUNDIDADES = [3, 4, 5, 6, 7, 8, 10]
MUESTRA_SVC = 20000


def dividir(X, y, grupos):
    """División 80/20 por cliente: los meses de un mismo cliente quedan en un solo conjunto."""
    divisor = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    idx_train, idx_test = next(divisor.split(X, y, grupos))
    return (X.iloc[idx_train], X.iloc[idx_test],
            y.iloc[idx_train], y.iloc[idx_test],
            grupos.iloc[idx_train])


def crear_pipeline(clasificador) -> Pipeline:
    return Pipeline([("escalador", StandardScaler()), ("clasificador", clasificador)])


def buscar_hiperparametro(constructor, valores, X_train, y_train, grupos_train, nombre):
    cv = GroupKFold(n_splits=3)
    resultados = []
    for v in valores:
        f1 = cross_val_score(crear_pipeline(constructor(v)), X_train, y_train,
                             groups=grupos_train, cv=cv, scoring="f1_macro", n_jobs=-1).mean()
        resultados.append({nombre: v, "f1_macro_cv": round(float(f1), 4)})
    tabla = pd.DataFrame(resultados)
    mejor = tabla.loc[tabla["f1_macro_cv"].idxmax(), nombre]
    return mejor, tabla


def evaluar(modelo, X_test, y_test) -> dict:
    y_pred = modelo.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro")), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro")), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        "matriz_confusion": confusion_matrix(y_test, y_pred).tolist(),
        "reporte": classification_report(
            y_test, y_pred, target_names=[ETIQUETAS_RIESGO[i] for i in range(3)], output_dict=True),
    }


def entrenar_candidatos(X_train, y_train, grupos_train):
    k, tabla_k = buscar_hiperparametro(
        lambda v: KNeighborsClassifier(n_neighbors=v), VALORES_K,
        X_train, y_train, grupos_train, "k")
    profundidad, tabla_arbol = buscar_hiperparametro(
        lambda v: DecisionTreeClassifier(max_depth=v, random_state=RANDOM_STATE), PROFUNDIDADES,
        X_train, y_train, grupos_train, "max_depth")

    knn = crear_pipeline(KNeighborsClassifier(n_neighbors=int(k))).fit(X_train, y_train)
    arbol = crear_pipeline(
        DecisionTreeClassifier(max_depth=int(profundidad), random_state=RANDOM_STATE)).fit(X_train, y_train)

    muestra = X_train.sample(n=min(MUESTRA_SVC, len(X_train)), random_state=RANDOM_STATE).index
    svc = crear_pipeline(
        CalibratedClassifierCV(SVC(kernel="rbf", C=1.0), ensemble=False, cv=3)
    ).fit(X_train.loc[muestra], y_train.loc[muestra])

    candidatos = {
        f"KNN (k={int(k)})": knn,
        f"Árbol de decisión (max_depth={int(profundidad)})": arbol,
        f"SVC rbf (muestra {len(muestra)})": svc,
    }
    return candidatos, {"knn": tabla_k, "arbol": tabla_arbol}


def comparar(candidatos, X_test, y_test):
    resultados = {nombre: evaluar(m, X_test, y_test) for nombre, m in candidatos.items()}
    tabla = pd.DataFrame({
        n: {k: r[k] for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]}
        for n, r in resultados.items()
    }).T.sort_values("f1_macro", ascending=False)
    return tabla, resultados


def perfiles_por_clase(X, y) -> dict:
    medianas = X.assign(clase=y.values).groupby("clase").median()
    return {NIVELES_RIESGO[int(c)]: medianas.loc[c].round(2).to_dict() for c in medianas.index}


def rangos_entrenamiento(X) -> dict:
    return {c: {"min": float(X[c].min()), "max": float(X[c].max())} for c in X.columns}


def guardar_artefactos(nombre, modelo, metricas, tabla, X, y):
    DIR_ARTEFACTOS.mkdir(parents=True, exist_ok=True)
    paquete = {
        "modelo": modelo,
        "nombre_modelo": nombre,
        "columnas": COLUMNAS_MODELO,
        "niveles_riesgo": NIVELES_RIESGO,
        "etiquetas_riesgo": ETIQUETAS_RIESGO,
        "mapa_credit_mix": MAPA_CREDIT_MIX,
        "mapa_pago_minimo": MAPA_PAGO_MINIMO,
        "perfiles": perfiles_por_clase(X, y),
        "rangos": rangos_entrenamiento(X),
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(paquete, RUTA_MODELO)

    resumen = {
        "modelo_seleccionado": nombre,
        "metricas_prueba": metricas,
        "comparacion": tabla.to_dict(orient="index"),
        "registros_usados": int(len(X)),
        "fecha_entrenamiento": paquete["fecha_entrenamiento"],
    }
    RUTA_METRICAS.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    return paquete


def main():
    X, y, grupos = preparar_dataset(RUTA_DATOS)
    print(f"Registros después de la limpieza: {len(X)}")
    print("Distribución de clases:\n", y.map(ETIQUETAS_RIESGO).value_counts(normalize=True).round(3))

    X_train, X_test, y_train, y_test, grupos_train = dividir(X, y, grupos)
    print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

    candidatos, busquedas = entrenar_candidatos(X_train, y_train, grupos_train)
    print("\nBúsqueda de k (KNN):\n", busquedas["knn"].to_string(index=False))
    print("\nBúsqueda de max_depth (Árbol):\n", busquedas["arbol"].to_string(index=False))

    tabla, resultados = comparar(candidatos, X_test, y_test)
    print("\nComparación en el conjunto de prueba:\n", tabla.to_string())

    mejor = tabla.index[0]
    print(f"\nModelo seleccionado: {mejor}")
    print("Matriz de confusión:\n", np.array(resultados[mejor]["matriz_confusion"]))

    guardar_artefactos(mejor, candidatos[mejor], resultados[mejor], tabla, X, y)
    print(f"\nModelo guardado en {RUTA_MODELO}")
    print(f"Métricas guardadas en {RUTA_METRICAS}")


if __name__ == "__main__":
    main()
