from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "train.csv"
DIR_ARTEFACTOS = RAIZ / "modelo" / "artefactos"
RUTA_MODELO = DIR_ARTEFACTOS / "modelo_riesgo.joblib"
RUTA_METRICAS = DIR_ARTEFACTOS / "metricas.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

COLUMNA_CLIENTE = "Customer_ID"
COLUMNA_OBJETIVO = "Credit_Score"

COLUMNAS_NUMERICAS = [
    "Credit_Utilization_Ratio",
    "Outstanding_Debt",
    "Interest_Rate",
    "Num_Credit_Inquiries",
    "Delay_from_due_date",
    "Credit_History_Age",
    "Num_of_Delayed_Payment",
    "Monthly_Inhand_Salary",
]
COLUMNAS_CATEGORICAS = ["Credit_Mix", "Payment_of_Min_Amount"]

COLUMNAS_MODELO = [
    "Credit_Utilization_Ratio",
    "Outstanding_Debt",
    "Interest_Rate",
    "Num_Credit_Inquiries",
    "Delay_from_due_date",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Credit_History_Age",
    "Num_of_Delayed_Payment",
    "Monthly_Inhand_Salary",
]

MAPA_OBJETIVO = {"Good": 0, "Standard": 1, "Poor": 2}
MAPA_CREDIT_MIX = {"Bad": 0, "Standard": 1, "Good": 2}
MAPA_PAGO_MINIMO = {"No": 0, "Yes": 1}

NIVELES_RIESGO = {0: "low", 1: "medium", 2: "high"}
ETIQUETAS_RIESGO = {0: "Riesgo bajo", 1: "Riesgo medio", 2: "Riesgo alto"}
