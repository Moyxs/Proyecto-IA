"""Redacción de la justificación con qwen2.5-coder a través de Ollama."""

import logging

from agente.analisis import describir_factor
from agente.configuracion import KEEP_ALIVE_LLM, MODELO_LLM, OLLAMA_HOST, TIEMPO_ESPERA_LLM

log = logging.getLogger("agente_riesgo")

INSTRUCCIONES_SISTEMA = (
    "Eres un analista de crédito de una institución financiera. "
    "Redactas justificaciones breves, formales y en español para decisiones de crédito ya tomadas. "
    "No cambias la decisión, no inventas datos y solo usas la información proporcionada."
)


def construir_prompt(prediccion: dict, decision, factores: dict) -> str:
    desfavorables = "\n".join(f"- {describir_factor(f)}" for f in factores["desfavorables"][:4]) or "- Ninguno"
    favorables = "\n".join(f"- {describir_factor(f)}" for f in factores["favorables"][:4]) or "- Ninguno"
    probabilidades = ", ".join(f"{k}: {v:.0%}" for k, v in prediccion["probabilities"].items())
    return (
        f"Nivel de riesgo predicho por el modelo: {decision.nivel} "
        f"(confianza {prediccion['score']:.0%}; probabilidades {probabilidades}).\n"
        f"Decisión del agente: {decision.resultado}. Acción: {decision.accion}\n\n"
        f"Factores que elevan el riesgo:\n{desfavorables}\n\n"
        f"Factores que reducen el riesgo:\n{favorables}\n\n"
        "Escribe un solo párrafo de 3 a 5 oraciones dirigido al comité de crédito que justifique "
        "la decisión, mencionando los factores más relevantes con sus valores y la acción a seguir."
    )


def justificacion_por_reglas(prediccion: dict, decision, factores: dict) -> str:
    partes = [f"El modelo clasificó la solicitud como {decision.nivel.lower()} "
              f"con una confianza de {prediccion['score']:.0%}."]
    if factores["desfavorables"]:
        lista = "; ".join(describir_factor(f) for f in factores["desfavorables"][:3])
        partes.append(f"Los factores que más elevan el riesgo son: {lista}.")
    if factores["favorables"]:
        lista = "; ".join(describir_factor(f) for f in factores["favorables"][:3])
        partes.append(f"Los factores a favor del solicitante son: {lista}.")
    partes.append(f"Acción: {decision.accion}")
    return " ".join(partes)


class ExplicadorLLM:
    def __init__(self, modelo: str = MODELO_LLM, host: str = OLLAMA_HOST, tiempo_espera: float = TIEMPO_ESPERA_LLM):
        self.modelo = modelo
        self.host = host
        self.tiempo_espera = tiempo_espera

    def _cliente(self):
        from ollama import Client

        return Client(host=self.host, timeout=self.tiempo_espera)

    def precalentar(self) -> bool:
        """Carga el modelo en memoria de Ollama para que la primera evaluación no espere la carga."""
        try:
            self._cliente().generate(model=self.modelo, prompt="", keep_alive=KEEP_ALIVE_LLM)
            log.info("Modelo %s cargado en Ollama", self.modelo)
            return True
        except Exception as error:
            log.warning("No se pudo precargar %s: %s", self.modelo, error)
            return False

    def redactar(self, prediccion: dict, decision, factores: dict) -> tuple[str, str]:
        """Devuelve (texto, origen). Si Ollama no está disponible usa la justificación por reglas."""
        try:
            respuesta = self._cliente().chat(
                model=self.modelo,
                messages=[
                    {"role": "system", "content": INSTRUCCIONES_SISTEMA},
                    {"role": "user", "content": construir_prompt(prediccion, decision, factores)},
                ],
                options={"temperature": 0.2},
                keep_alive=KEEP_ALIVE_LLM,
            )
            texto = respuesta["message"]["content"].strip()
            if texto:
                return texto, self.modelo
        except Exception as error:
            log.warning("No se pudo usar el LLM (%s): %s", self.modelo, error)
        return justificacion_por_reglas(prediccion, decision, factores), "reglas"
