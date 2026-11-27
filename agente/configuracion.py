import os

URL_API = os.getenv("URL_API", "http://127.0.0.1:8000")
TIEMPO_ESPERA_API = float(os.getenv("TIEMPO_ESPERA_API", "10"))
REINTENTOS_API = int(os.getenv("REINTENTOS_API", "2"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODELO_LLM = os.getenv("MODELO_LLM", "qwen2.5-coder:7b")
USAR_LLM = os.getenv("USAR_LLM", "1") == "1"
TIEMPO_ESPERA_LLM = float(os.getenv("TIEMPO_ESPERA_LLM", "300"))
KEEP_ALIVE_LLM = os.getenv("KEEP_ALIVE_LLM", "30m")

PUERTO_GRADIO = int(os.getenv("PUERTO_GRADIO", "7860"))
HOST_GRADIO = os.getenv("HOST_GRADIO", "127.0.0.1")

NOMBRES_VARIABLES = {
    "Credit_Utilization_Ratio": "Utilización del crédito (%)",
    "Outstanding_Debt": "Deuda pendiente",
    "Interest_Rate": "Tasa de interés (%)",
    "Num_Credit_Inquiries": "Consultas de crédito",
    "Delay_from_due_date": "Días de atraso promedio",
    "Credit_Mix": "Mezcla de crédito",
    "Payment_of_Min_Amount": "Paga solo el mínimo",
    "Credit_History_Age": "Antigüedad del historial (meses)",
    "Num_of_Delayed_Payment": "Pagos atrasados",
    "Monthly_Inhand_Salary": "Salario mensual neto",
}
