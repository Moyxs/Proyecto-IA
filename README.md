# Agente de Evaluación de Riesgo Crediticio

Proyecto final · IS-701 Inteligencia Artificial · UNAH · III PAC 2026

Sistema que evalúa solicitudes de crédito y toma decisiones automatizadas a partir de un modelo predictivo. Tiene tres componentes:

1. **Modelo predictivo** (scikit-learn): clasifica al solicitante en riesgo bajo (0), medio (1) o alto (2).
2. **API REST** (FastAPI): expone el modelo en `POST /predict` con validación Pydantic.
3. **Agente** (Gradio + Ollama `qwen2.5-coder:7b`): recibe la solicitud, consulta la API, aplica las reglas de decisión y redacta la justificación.

```
Usuario ──► Gradio (agente/app.py)
              │
              ▼
        AgenteRiesgoCrediticio ──HTTP──► FastAPI /predict ──► Pipeline (StandardScaler + Árbol)
              │
              ├─ reglas.py        → decisión (aprobar / revisar / rechazar)
              ├─ analisis.py      → factores favorables y desfavorables
              └─ explicador_llm.py → justificación con qwen2.5-coder (Ollama)
```

## Estructura

```
Proyecto IA/
├── data/train.csv                     Dataset Credit Score Classification
├── notebooks/
│   └── 01_modelo_riesgo_crediticio.ipynb   Proceso completo del modelo, con resultados
├── modelo/
│   ├── configuracion.py               Rutas, columnas y codificaciones
│   ├── preprocesamiento.py            Limpieza, imputación, IQR y codificación
│   ├── entrenamiento.py               Entrenamiento, comparación y guardado
│   └── artefactos/
│       ├── modelo_riesgo.joblib       Modelo entrenado
│       └── metricas.json              Métricas de evaluación
├── api/
│   ├── main.py                        Aplicación FastAPI
│   ├── esquemas.py                    Modelos Pydantic de entrada y salida
│   └── servicio_modelo.py             Carga del modelo y predicción
├── agente/
│   ├── app.py                         Interfaz Gradio
│   ├── agente.py                      Orquestación del agente
│   ├── cliente_api.py                 Cliente HTTP con manejo de errores
│   ├── reglas.py                      Reglas de decisión
│   ├── analisis.py                    Análisis de factores
│   ├── explicador_llm.py              Justificación con qwen2.5-coder
│   └── configuracion.py               Variables de entorno
├── tests/                             Pruebas de la API y del agente (pytest)
├── docker/                            Dockerfiles y dependencias por servicio
├── docker-compose.yml
├── docs/
│   ├── manual_usuario_instalacion.md
│   └── investigacion.md
└── requirements.txt
```

## Variables del modelo

| Variable | Descripción |
|---|---|
| `Credit_Utilization_Ratio` | Porcentaje de utilización del crédito |
| `Outstanding_Debt` | Deuda pendiente |
| `Interest_Rate` | Tasa de interés (%) |
| `Num_Credit_Inquiries` | Consultas de crédito |
| `Delay_from_due_date` | Días promedio de atraso |
| `Credit_Mix` | Mezcla de crédito: Good, Standard, Bad |
| `Payment_of_Min_Amount` | Paga solo el mínimo: Yes, No |
| `Credit_History_Age` | Antigüedad del historial en meses |
| `Num_of_Delayed_Payment` | Número de pagos atrasados |
| `Monthly_Inhand_Salary` | Salario mensual neto |

## Resultados del modelo

División 80/20 por cliente (ningún cliente aparece en entrenamiento y prueba a la vez). Hiperparámetros elegidos con validación cruzada GroupKFold (3 particiones) y F1 macro.

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---|---|---|---|
| **Árbol de decisión (max_depth=4)** | **0.7016** | **0.6796** | **0.7228** | **0.6870** |
| SVC rbf (muestra 20 000) | 0.6948 | 0.6778 | 0.7106 | 0.6768 |
| KNN (k=51) | 0.6888 | 0.6627 | 0.6717 | 0.6610 |

Modelo seleccionado: Árbol de decisión, por tener el mayor F1 macro.

## Inicio rápido

```powershell
pip install -r requirements.txt
ollama pull qwen2.5-coder:7b
python -m modelo.entrenamiento
python -m uvicorn api.main:app --port 8000
python -m agente.app
```

- API y documentación interactiva: http://127.0.0.1:8000/docs
- Interfaz del agente: http://127.0.0.1:7860

Instrucciones completas en [docs/manual_usuario_instalacion.md](docs/manual_usuario_instalacion.md).
