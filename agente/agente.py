"""Agente de riesgo crediticio: consulta la API, interpreta la predicción y toma la decisión."""

import logging
from dataclasses import dataclass, field

from agente.analisis import analizar_factores
from agente.cliente_api import ClienteAPI, ErrorAPI
from agente.configuracion import NOMBRES_VARIABLES, USAR_LLM
from agente.explicador_llm import ExplicadorLLM, justificacion_por_reglas
from agente.reglas import Decision, confianza_baja, decidir

log = logging.getLogger("agente_riesgo")


@dataclass
class ResultadoAgente:
    exito: bool
    decision: Decision | None = None
    prediccion: dict = field(default_factory=dict)
    factores: dict = field(default_factory=dict)
    justificacion: str = ""
    origen_justificacion: str = ""
    advertencias: list[str] = field(default_factory=list)
    error: str = ""


class AgenteRiesgoCrediticio:
    def __init__(self, cliente: ClienteAPI | None = None, explicador: ExplicadorLLM | None = None,
                 usar_llm: bool = USAR_LLM):
        self.cliente = cliente or ClienteAPI()
        self.explicador = explicador or ExplicadorLLM()
        self.usar_llm = usar_llm
        self._perfiles = None

    def _obtener_perfiles(self) -> dict:
        if self._perfiles is None:
            self._perfiles = self.cliente.informacion_modelo()["class_profiles"]
        return self._perfiles

    def evaluar(self, solicitud: dict, usar_llm: bool | None = None) -> ResultadoAgente:
        try:
            prediccion = self.cliente.predecir(solicitud)
            perfiles = self._obtener_perfiles()
        except ErrorAPI as error:
            log.error("Error de la API: %s", error.mensaje)
            return ResultadoAgente(exito=False, error=error.mensaje)

        try:
            decision = decidir(prediccion["risk_level"])
        except (KeyError, ValueError) as error:
            return ResultadoAgente(exito=False, prediccion=prediccion,
                                   error=f"La respuesta de la API no se pudo interpretar: {error}")

        factores = analizar_factores(solicitud, perfiles)

        advertencias = []
        if confianza_baja(prediccion["score"]):
            advertencias.append(
                f"La confianza del modelo es baja ({prediccion['score']:.0%}); "
                "se sugiere revisión manual por un analista.")
        if prediccion.get("out_of_range"):
            nombres = ", ".join(NOMBRES_VARIABLES.get(v, v) for v in prediccion["out_of_range"])
            advertencias.append(f"Valores fuera del rango de entrenamiento: {nombres}.")

        if usar_llm if usar_llm is not None else self.usar_llm:
            justificacion, origen = self.explicador.redactar(prediccion, decision, factores)
        else:
            justificacion, origen = justificacion_por_reglas(prediccion, decision, factores), "reglas"

        return ResultadoAgente(
            exito=True,
            decision=decision,
            prediccion=prediccion,
            factores=factores,
            justificacion=justificacion,
            origen_justificacion=origen,
            advertencias=advertencias,
        )
