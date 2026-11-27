"""API de riesgo crediticio. Ejecutar con: uvicorn api.main:app --reload"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.esquemas import EstadoServicio, RespuestaPrediccion, SolicitudCredito
from api.servicio_modelo import ServicioModelo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api_riesgo")

servicio = ServicioModelo()


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    try:
        servicio.cargar()
        log.info("Modelo cargado: %s", servicio.nombre)
    except FileNotFoundError as error:
        log.error(str(error))
    yield


app = FastAPI(
    title="API de Riesgo Crediticio",
    description="Expone el modelo predictivo de riesgo crediticio (0 = bajo, 1 = medio, 2 = alto).",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


def _verificar_modelo():
    if not servicio.cargado:
        raise HTTPException(status_code=503, detail="El modelo no está cargado. Ejecute el entrenamiento primero.")


@app.get("/", tags=["General"])
def inicio():
    return {"mensaje": "API de Riesgo Crediticio", "documentacion": "/docs"}


@app.get("/health", response_model=EstadoServicio, tags=["General"])
def salud():
    return EstadoServicio(status="ok" if servicio.cargado else "sin_modelo",
                          model_loaded=servicio.cargado, model=servicio.nombre)


@app.get("/model-info", tags=["Modelo"])
def informacion_modelo():
    _verificar_modelo()
    return servicio.informacion()


@app.post("/predict", response_model=RespuestaPrediccion, tags=["Modelo"])
def predecir(solicitud: SolicitudCredito):
    _verificar_modelo()
    try:
        resultado = servicio.predecir(solicitud.model_dump())
    except Exception as error:
        log.exception("Error al predecir")
        raise HTTPException(status_code=500, detail=f"Error interno al predecir: {error}") from error
    log.info("Predicción: %s (score=%.3f)", resultado["risk_level"], resultado["score"])
    return resultado
