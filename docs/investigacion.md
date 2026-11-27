# Investigación: FastAPI, Gradio e integración del agente

## 1. FastAPI

FastAPI es un framework web de Python para construir APIs REST. Se basa en Starlette (servidor ASGI) y en Pydantic (validación de datos), y genera automáticamente la documentación OpenAPI.

### Características usadas en el proyecto

| Característica | Uso en el proyecto |
|---|---|
| Decoradores de ruta (`@app.get`, `@app.post`) | Endpoints `/`, `/health`, `/model-info` y `/predict` |
| Modelos Pydantic como cuerpo de la petición | `SolicitudCredito` valida tipos, rangos (`ge`, `le`, `gt`) y valores permitidos (`Literal`) |
| `response_model` | `RespuestaPrediccion` define y valida la estructura del JSON de salida |
| `HTTPException` | Respuestas 503 (modelo no cargado) y 500 (error al predecir) |
| `lifespan` | Carga el modelo una sola vez al iniciar el servidor |
| Documentación automática | Swagger UI en `/docs` y ReDoc en `/redoc` |
| `TestClient` | Pruebas automáticas de la API sin levantar el servidor |

### Validación con Pydantic

Cuando una petición no cumple el esquema, FastAPI responde automáticamente con **422 Unprocessable Entity** y una lista de errores con la ubicación (`loc`), el mensaje (`msg`) y el valor recibido. El agente usa esa lista para mostrar al usuario qué campo es inválido.

```python
class SolicitudCredito(BaseModel):
    Credit_Utilization_Ratio: float = Field(..., ge=0, le=100)
    Credit_Mix: Literal["Good", "Standard", "Bad"]
```

### Servidor

FastAPI se ejecuta con **Uvicorn**, un servidor ASGI:

```
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`api.main` es el módulo y `app` la instancia de `FastAPI`.

### Carga del modelo

El modelo se guarda con `joblib` como un `Pipeline` de scikit-learn (`StandardScaler` + clasificador). Así la API aplica la misma transformación que en el entrenamiento sin repetir código. Junto al pipeline se guardan las columnas en orden, los diccionarios de codificación, las medianas por clase y los rangos de entrenamiento.

---

## 2. Gradio

Gradio es una librería de Python para crear interfaces web para modelos de machine learning sin escribir HTML ni JavaScript.

### Conceptos

| Concepto | Descripción |
|---|---|
| `gr.Blocks` | Contenedor que permite organizar la interfaz con filas, columnas y grupos |
| Componentes de entrada | `gr.Number`, `gr.Slider`, `gr.Radio`, `gr.Dropdown`, `gr.Checkbox` |
| Componentes de salida | `gr.HTML`, `gr.Markdown`, `gr.JSON` |
| Eventos | `boton.click(funcion, inputs=[...], outputs=[...])` conecta un botón con una función de Python |
| `gr.Examples` | Carga valores de ejemplo en el formulario con un clic |
| `demo.load` | Ejecuta una función al abrir la página (estado de la API) |
| `launch()` | Inicia el servidor web en el puerto indicado |

### Flujo en el proyecto

1. El usuario llena el formulario.
2. `evaluar_solicitud` recibe los valores en el mismo orden de la lista `inputs`.
3. Se construye el diccionario de la solicitud (la antigüedad en años y meses se convierte a meses).
4. Se llama al agente y su resultado se devuelve en el orden de la lista `outputs`.

---

## 3. Agente de riesgo crediticio

Un agente es un programa que **percibe** su entorno, **razona** sobre lo percibido y **actúa** para cumplir un objetivo. En este proyecto:

| Etapa | Componente | Descripción |
|---|---|---|
| Percepción | `app.py` | Recibe la solicitud del usuario desde Gradio |
| Acción sobre una herramienta | `cliente_api.py` | Envía los datos a `POST /predict` |
| Interpretación | `agente.py` | Lee nivel de riesgo, probabilidad y valores fuera de rango |
| Decisión | `reglas.py` | Aplica las reglas fijas según el nivel de riesgo |
| Justificación | `analisis.py` + `explicador_llm.py` | Identifica los factores y redacta la explicación |

### Reglas de decisión

| Nivel | Decisión |
|---|---|
| Riesgo alto | Rechazar solicitud y recomendar educación financiera. |
| Riesgo medio | Solicitar documentación adicional y evaluar nuevamente. |
| Riesgo bajo | Aprobar solicitud con condiciones estándar. |

La decisión la toman siempre las reglas. El LLM no puede cambiarla; solo redacta la explicación.

### Manejo de errores de la API

| Situación | Tratamiento |
|---|---|
| API apagada (`ConnectionError`) | Reintenta 2 veces con espera creciente y luego informa que no hay conexión |
| Tiempo de espera agotado (`Timeout`) | Reintenta y luego informa el tiempo agotado |
| Error 422 | Muestra cada campo inválido con su mensaje |
| Error 503 | Informa que la API no tiene modelo cargado |
| Otro código HTTP o respuesta no JSON | Muestra el código y el detalle |
| Nivel de riesgo desconocido en la respuesta | Informa que la respuesta no se pudo interpretar |

### Justificación de la decisión

1. **Análisis de factores** (`analisis.py`): compara cada dato del solicitante con la mediana de los clientes de riesgo bajo y de riesgo alto obtenida del entrenamiento (`/model-info`). Si el valor está más cerca del perfil de riesgo alto, el factor eleva el riesgo; si está más cerca del de riesgo bajo, lo reduce. Las variables cuyas medianas casi no difieren entre perfiles (menos de 10 %) no se consideran.
2. **Redacción con LLM** (`explicador_llm.py`): se envía a `qwen2.5-coder:7b` por medio de Ollama un prompt con el nivel, la confianza, la decisión y los factores principales, con temperatura 0.2 para respuestas estables.
3. **Respaldo por reglas**: si Ollama no responde, la justificación se arma con una plantilla a partir de los mismos factores, de modo que el agente siempre entrega una justificación.

---

## 4. Ollama y qwen2.5-coder

Ollama ejecuta modelos de lenguaje de forma local y expone una API HTTP en el puerto 11434. Se usa la librería `ollama` de Python:

```python
from ollama import Client
cliente = Client(host="http://127.0.0.1:11434")
respuesta = cliente.chat(model="qwen2.5-coder:7b", messages=[...], options={"temperature": 0.2})
```

| Parámetro | Valor | Motivo |
|---|---|---|
| `model` | `qwen2.5-coder:7b` | Modelo definido para el proyecto |
| `temperature` | 0.2 | Respuestas consistentes entre ejecuciones |
| `keep_alive` | 30 min | Mantiene el modelo en memoria entre evaluaciones |
| Mensaje de sistema | Rol de analista de crédito | Limita la respuesta a justificar sin cambiar la decisión ni inventar datos |

Al iniciar la interfaz, el agente precarga el modelo en un hilo aparte para que la primera evaluación no espere la carga en memoria.

---

## 5. Docker

| Archivo | Contenido |
|---|---|
| `docker/Dockerfile.api` | Imagen `python:3.12-slim` con la API, el preprocesamiento y el modelo entrenado; incluye `HEALTHCHECK` sobre `/health` |
| `docker/Dockerfile.agente` | Imagen con el agente y la interfaz Gradio |
| `docker-compose.yml` | Levanta ambos servicios; el agente espera a que la API esté sana (`service_healthy`) |

Ollama se mantiene en el equipo anfitrión porque necesita acceso a la GPU y el modelo ocupa 4.7 GB; el contenedor del agente lo alcanza mediante `host.docker.internal`.

---

## 6. Referencias

- FastAPI: https://fastapi.tiangolo.com
- Pydantic: https://docs.pydantic.dev
- Uvicorn: https://www.uvicorn.org
- Gradio: https://www.gradio.app/docs
- Ollama: https://github.com/ollama/ollama
- Ollama Python: https://github.com/ollama/ollama-python
- scikit-learn: https://scikit-learn.org/stable/
- Dataset Credit Score Classification: https://www.kaggle.com/datasets/parisrohan/credit-score-classification
