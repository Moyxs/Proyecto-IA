import pytest
from fastapi.testclient import TestClient

from api.main import app

SOLICITUD = {
    "Credit_Utilization_Ratio": 25.63,
    "Outstanding_Debt": 1886.54,
    "Interest_Rate": 17,
    "Num_Credit_Inquiries": 12,
    "Delay_from_due_date": 13,
    "Credit_Mix": "Standard",
    "Payment_of_Min_Amount": "Yes",
    "Credit_History_Age": 225,
    "Num_of_Delayed_Payment": 16,
    "Monthly_Inhand_Salary": 971.16,
}


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:
        yield c


def test_salud(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_prediccion_valida(cliente):
    r = cliente.post("/predict", json=SOLICITUD)
    assert r.status_code == 200
    datos = r.json()
    assert datos["risk_level"] in {"low", "medium", "high"}
    assert 0 <= datos["score"] <= 1
    assert abs(sum(datos["probabilities"].values()) - 1) < 1e-3


def test_prediccion_riesgo_alto(cliente):
    assert cliente.post("/predict", json=SOLICITUD).json()["risk_level"] == "high"


@pytest.mark.parametrize("campo,valor", [
    ("Credit_Mix", "Excelente"),
    ("Payment_of_Min_Amount", "NM"),
    ("Credit_Utilization_Ratio", 150),
    ("Monthly_Inhand_Salary", -10),
])
def test_validacion_pydantic(cliente, campo, valor):
    r = cliente.post("/predict", json={**SOLICITUD, campo: valor})
    assert r.status_code == 422


def test_campo_faltante(cliente):
    datos = dict(SOLICITUD)
    datos.pop("Interest_Rate")
    assert cliente.post("/predict", json=datos).status_code == 422


def test_fuera_de_rango(cliente):
    r = cliente.post("/predict", json={**SOLICITUD, "Interest_Rate": 80})
    assert "Interest_Rate" in r.json()["out_of_range"]
