"""Comparación de la solicitud con los perfiles típicos de cada nivel de riesgo."""

from agente.configuracion import NOMBRES_VARIABLES

MAPA_CREDIT_MIX = {"Bad": 0, "Standard": 1, "Good": 2}
MAPA_PAGO_MINIMO = {"No": 0, "Yes": 1}
SEPARACION_MINIMA = 0.10


def _a_numero(variable: str, valor):
    if variable == "Credit_Mix":
        return MAPA_CREDIT_MIX[valor]
    if variable == "Payment_of_Min_Amount":
        return MAPA_PAGO_MINIMO[valor]
    return float(valor)


def _a_texto(variable: str, mediana: float):
    inverso = {"Credit_Mix": MAPA_CREDIT_MIX, "Payment_of_Min_Amount": MAPA_PAGO_MINIMO}.get(variable)
    if inverso is None:
        return mediana
    return {v: k for k, v in inverso.items()}[round(mediana)]


def analizar_factores(solicitud: dict, perfiles: dict) -> dict:
    """Clasifica cada variable como favorable o desfavorable según su cercanía
    a la mediana del perfil de riesgo bajo o del perfil de riesgo alto."""
    bajo, alto = perfiles["low"], perfiles["high"]
    favorables, desfavorables = [], []

    for variable, valor in solicitud.items():
        x = _a_numero(variable, valor)
        m_bajo, m_alto = bajo[variable], alto[variable]
        separacion = abs(m_alto - m_bajo)
        if separacion / max(abs(m_bajo), abs(m_alto), 1) < SEPARACION_MINIMA:
            continue
        cercania_alto = (abs(x - m_bajo) - abs(x - m_alto)) / separacion
        factor = {
            "variable": variable,
            "nombre": NOMBRES_VARIABLES.get(variable, variable),
            "valor": valor,
            "mediana_riesgo_bajo": _a_texto(variable, m_bajo),
            "mediana_riesgo_alto": _a_texto(variable, m_alto),
            "peso": round(abs(cercania_alto), 3),
        }
        (desfavorables if cercania_alto > 0 else favorables).append(factor)

    favorables.sort(key=lambda f: f["peso"], reverse=True)
    desfavorables.sort(key=lambda f: f["peso"], reverse=True)
    return {"favorables": favorables, "desfavorables": desfavorables}


def describir_factor(f: dict) -> str:
    return (f"{f['nombre']}: {f['valor']} "
            f"(típico riesgo bajo: {f['mediana_riesgo_bajo']}, típico riesgo alto: {f['mediana_riesgo_alto']})")
