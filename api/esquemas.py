from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SolicitudCredito(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Credit_Utilization_Ratio": 31.5,
                "Outstanding_Debt": 1200.5,
                "Interest_Rate": 12,
                "Num_Credit_Inquiries": 4,
                "Delay_from_due_date": 15,
                "Credit_Mix": "Standard",
                "Payment_of_Min_Amount": "Yes",
                "Credit_History_Age": 220,
                "Num_of_Delayed_Payment": 12,
                "Monthly_Inhand_Salary": 3500.0,
            }
        }
    )

    Credit_Utilization_Ratio: float = Field(..., ge=0, le=100, description="Porcentaje de utilización del crédito")
    Outstanding_Debt: float = Field(..., ge=0, description="Deuda pendiente")
    Interest_Rate: float = Field(..., ge=0, le=100, description="Tasa de interés (%)")
    Num_Credit_Inquiries: int = Field(..., ge=0, le=100, description="Número de consultas de crédito")
    Delay_from_due_date: int = Field(..., ge=-30, le=365, description="Días promedio de atraso respecto a la fecha de pago")
    Credit_Mix: Literal["Good", "Standard", "Bad"] = Field(..., description="Calidad de la mezcla de créditos")
    Payment_of_Min_Amount: Literal["Yes", "No"] = Field(..., description="Paga solo el monto mínimo")
    Credit_History_Age: int = Field(..., ge=0, le=1200, description="Antigüedad del historial crediticio en meses")
    Num_of_Delayed_Payment: int = Field(..., ge=0, le=100, description="Número de pagos atrasados")
    Monthly_Inhand_Salary: float = Field(..., gt=0, description="Salario mensual neto")


class RespuestaPrediccion(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    risk_class: int = Field(..., ge=0, le=2)
    risk_label: str
    score: float = Field(..., ge=0, le=1, description="Probabilidad de la clase predicha")
    probabilities: dict[str, float]
    out_of_range: list[str] = Field(default_factory=list, description="Variables fuera del rango visto en entrenamiento")
    model: str


class EstadoServicio(BaseModel):
    status: str
    model_loaded: bool
    model: str | None = None
