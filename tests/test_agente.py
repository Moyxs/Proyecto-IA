import pytest

from agente.agente import AgenteRiesgoCrediticio
from agente.analisis import analizar_factores
from agente.cliente_api import ClienteAPI, ErrorAPI
from agente.reglas import decidir

PERFILES = {
    "low": {"Credit_Utilization_Ratio": 32.6, "Outstanding_Debt": 730.0, "Interest_Rate": 7.0,
            "Num_Credit_Inquiries": 3.0, "Delay_from_due_date": 10.0, "Credit_Mix": 2.0,
            "Payment_of_Min_Amount": 0.0, "Credit_History_Age": 289.0, "Num_of_Delayed_Payment": 8.0,
            "Monthly_Inhand_Salary": 3671.0},
    "high": {"Credit_Utilization_Ratio": 32.0, "Outstanding_Debt": 1773.0, "Interest_Rate": 20.0,
             "Num_Credit_Inquiries": 8.0, "Delay_from_due_date": 24.0, "Credit_Mix": 1.0,
             "Payment_of_Min_Amount": 1.0, "Credit_History_Age": 173.0, "Num_of_Delayed_Payment": 16.0,
             "Monthly_Inhand_Salary": 2697.0},
}

SOLICITUD = {
    "Credit_Utilization_Ratio": 30.0, "Outstanding_Debt": 1900.0, "Interest_Rate": 22,
    "Num_Credit_Inquiries": 10, "Delay_from_due_date": 30, "Credit_Mix": "Bad",
    "Payment_of_Min_Amount": "Yes", "Credit_History_Age": 120, "Num_of_Delayed_Payment": 18,
    "Monthly_Inhand_Salary": 1500.0,
}


class ClienteFalso:
    def __init__(self, prediccion=None, error=None):
        self.prediccion = prediccion
        self.error = error

    def predecir(self, solicitud):
        if self.error:
            raise self.error
        return self.prediccion

    def informacion_modelo(self):
        return {"class_profiles": PERFILES}


@pytest.mark.parametrize("nivel,resultado,accion", [
    ("high", "RECHAZADA", "Rechazar solicitud y recomendar educación financiera."),
    ("medium", "EN REVISIÓN", "Solicitar documentación adicional y evaluar nuevamente."),
    ("low", "APROBADA", "Aprobar solicitud con condiciones estándar."),
])
def test_reglas(nivel, resultado, accion):
    d = decidir(nivel)
    assert d.resultado == resultado
    assert d.accion == accion


def test_regla_desconocida():
    with pytest.raises(ValueError):
        decidir("extremo")


def test_factores_desfavorables():
    f = analizar_factores(SOLICITUD, PERFILES)
    variables = {x["variable"] for x in f["desfavorables"]}
    assert {"Outstanding_Debt", "Interest_Rate", "Credit_Mix"} <= variables
    assert "Credit_Utilization_Ratio" not in variables | {x["variable"] for x in f["favorables"]}


def test_agente_decision_y_justificacion():
    prediccion = {"risk_level": "high", "score": 0.81,
                  "probabilities": {"low": 0.02, "medium": 0.17, "high": 0.81}, "out_of_range": []}
    agente = AgenteRiesgoCrediticio(cliente=ClienteFalso(prediccion), usar_llm=False)
    r = agente.evaluar(SOLICITUD)
    assert r.exito
    assert r.decision.resultado == "RECHAZADA"
    assert "riesgo alto" in r.justificacion
    assert r.origen_justificacion == "reglas"


def test_agente_confianza_baja():
    prediccion = {"risk_level": "low", "score": 0.45,
                  "probabilities": {"low": 0.45, "medium": 0.40, "high": 0.15}, "out_of_range": []}
    r = AgenteRiesgoCrediticio(cliente=ClienteFalso(prediccion), usar_llm=False).evaluar(SOLICITUD)
    assert any("confianza" in a for a in r.advertencias)


def test_agente_error_api():
    agente = AgenteRiesgoCrediticio(cliente=ClienteFalso(error=ErrorAPI("API caída")), usar_llm=False)
    r = agente.evaluar(SOLICITUD)
    assert not r.exito
    assert r.error == "API caída"


def test_cliente_sin_conexion():
    cliente = ClienteAPI("http://127.0.0.1:9", tiempo_espera=1, reintentos=0)
    with pytest.raises(ErrorAPI, match="No se pudo conectar"):
        cliente.predecir(SOLICITUD)
