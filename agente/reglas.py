"""Reglas de decisión del agente según el nivel de riesgo predicho."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    nivel: str
    accion: str
    resultado: str
    color: str


REGLAS = {
    "high": Decision(
        nivel="Riesgo alto",
        accion="Rechazar solicitud y recomendar educación financiera.",
        resultado="RECHAZADA",
        color="#c0392b",
    ),
    "medium": Decision(
        nivel="Riesgo medio",
        accion="Solicitar documentación adicional y evaluar nuevamente.",
        resultado="EN REVISIÓN",
        color="#d68910",
    ),
    "low": Decision(
        nivel="Riesgo bajo",
        accion="Aprobar solicitud con condiciones estándar.",
        resultado="APROBADA",
        color="#1e8449",
    ),
}

UMBRAL_CONFIANZA = 0.50


def decidir(risk_level: str) -> Decision:
    if risk_level not in REGLAS:
        raise ValueError(f"Nivel de riesgo desconocido: {risk_level}")
    return REGLAS[risk_level]


def confianza_baja(score: float) -> bool:
    return score < UMBRAL_CONFIANZA
