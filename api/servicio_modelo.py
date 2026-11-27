import os
from pathlib import Path

import joblib

from modelo.configuracion import RUTA_MODELO
from modelo.preprocesamiento import solicitud_a_dataframe


class ServicioModelo:
    def __init__(self, ruta: Path | None = None):
        self.ruta = Path(ruta or os.getenv("RUTA_MODELO", RUTA_MODELO))
        self.paquete = None

    def cargar(self):
        if not self.ruta.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {self.ruta}")
        self.paquete = joblib.load(self.ruta)

    @property
    def cargado(self) -> bool:
        return self.paquete is not None

    @property
    def nombre(self) -> str | None:
        return self.paquete["nombre_modelo"] if self.cargado else None

    def informacion(self) -> dict:
        return {
            "model": self.paquete["nombre_modelo"],
            "features": self.paquete["columnas"],
            "risk_levels": self.paquete["niveles_riesgo"],
            "training_date": self.paquete["fecha_entrenamiento"],
            "class_profiles": self.paquete["perfiles"],
            "training_ranges": self.paquete["rangos"],
        }

    def fuera_de_rango(self, solicitud: dict) -> list[str]:
        rangos = self.paquete["rangos"]
        datos = solicitud_a_dataframe(solicitud).iloc[0]
        return [c for c, r in rangos.items() if not (r["min"] <= datos[c] <= r["max"])]

    def predecir(self, solicitud: dict) -> dict:
        modelo = self.paquete["modelo"]
        niveles = self.paquete["niveles_riesgo"]
        etiquetas = self.paquete["etiquetas_riesgo"]

        X = solicitud_a_dataframe(solicitud)
        probabilidades = modelo.predict_proba(X)[0]
        clases = [int(c) for c in modelo.classes_]
        clase = clases[int(probabilidades.argmax())]

        return {
            "risk_level": niveles[clase],
            "risk_class": clase,
            "risk_label": etiquetas[clase],
            "score": round(float(probabilidades.max()), 4),
            "probabilities": {niveles[c]: round(float(p), 4) for c, p in zip(clases, probabilidades)},
            "out_of_range": self.fuera_de_rango(solicitud),
            "model": self.nombre,
        }
