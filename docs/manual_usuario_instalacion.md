# Manual de instalación y de usuario

Agente de Evaluación de Riesgo Crediticio · IS-701 Inteligencia Artificial

---

## Parte 1. Manual de instalación

### 1.1 Requisitos

| Componente | Versión usada | Uso |
|---|---|---|
| Python | 3.12 o superior (probado con 3.14) | Modelo, API y agente |
| Ollama | 0.40 o superior | Ejecuta el LLM local |
| Modelo `qwen2.5-coder:7b` | 4.7 GB | Redacta la justificación de la decisión |
| Docker Desktop (opcional) | — | Ejecución en contenedores |
| RAM | 8 GB mínimo | El LLM usa cerca de 5 GB |

### 1.2 Instalación local

1. Abrir una terminal en la carpeta del proyecto.

2. (Opcional) Crear y activar un entorno virtual:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Instalar dependencias:

   ```powershell
   pip install -r requirements.txt
   ```

4. Instalar Ollama desde https://ollama.com/download y descargar el modelo:

   ```powershell
   ollama pull qwen2.5-coder:7b
   ```

5. Verificar que el dataset esté en `data/train.csv`.

### 1.3 Entrenar el modelo

Opción A, script:

```powershell
python -m modelo.entrenamiento
```

Opción B, notebook con todo el proceso y las gráficas: abrir `notebooks/01_modelo_riesgo_crediticio.ipynb` en Jupyter y ejecutar todas las celdas.

```powershell
python -m jupyter lab
```

Ambas opciones generan:

- `modelo/artefactos/modelo_riesgo.joblib`: pipeline entrenado (escalador + clasificador) y metadatos.
- `modelo/artefactos/metricas.json`: métricas de evaluación y comparación de modelos.

El entrenamiento tarda entre 2 y 4 minutos.

### 1.4 Ejecutar la API

```powershell
python -m uvicorn api.main:app --port 8000
```

Comprobación: abrir http://127.0.0.1:8000/health. Debe responder `"model_loaded": true`.

### 1.5 Ejecutar el agente

En otra terminal, con la API en ejecución y Ollama abierto:

```powershell
python -m agente.app
```

Abrir http://127.0.0.1:7860.

### 1.6 Ejecutar las pruebas

```powershell
python -m pytest -q tests
```

### 1.7 Ejecución con Docker

Requiere Docker Desktop y que el modelo ya esté entrenado (`modelo/artefactos/modelo_riesgo.joblib`). Ollama se ejecuta en el equipo anfitrión.

```powershell
docker compose up --build
```

| Servicio | Contenedor | Puerto |
|---|---|---|
| API FastAPI | `riesgo-api` | 8000 |
| Agente Gradio | `riesgo-agente` | 7860 |

El agente se conecta a la API por la red interna de Docker (`http://api:8000`) y a Ollama por `http://host.docker.internal:11434`.

Para detener: `docker compose down`.

### 1.8 Variables de entorno

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `URL_API` | `http://127.0.0.1:8000` | Dirección de la API que consulta el agente |
| `TIEMPO_ESPERA_API` | `10` | Segundos de espera por respuesta de la API |
| `REINTENTOS_API` | `2` | Reintentos ante fallo de conexión |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Dirección de Ollama |
| `MODELO_LLM` | `qwen2.5-coder:7b` | Modelo usado para la justificación |
| `USAR_LLM` | `1` | `0` desactiva el LLM y usa la justificación por reglas |
| `TIEMPO_ESPERA_LLM` | `300` | Segundos de espera por la respuesta del LLM |
| `PUERTO_GRADIO` | `7860` | Puerto de la interfaz |
| `RUTA_MODELO` | `modelo/artefactos/modelo_riesgo.joblib` | Archivo del modelo que carga la API |

### 1.9 Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| La interfaz muestra "API: no disponible" | La API no está en ejecución | Iniciar la API (paso 1.4) |
| `/health` devuelve `"model_loaded": false` | No existe el archivo del modelo | Entrenar el modelo (paso 1.3) |
| La justificación dice "Generada por reglas" | Ollama cerrado o modelo no descargado | Abrir Ollama y ejecutar `ollama pull qwen2.5-coder:7b` |
| La primera evaluación tarda | Ollama carga el modelo en memoria | El agente precarga el modelo al iniciar; las siguientes evaluaciones tardan entre 10 y 30 s |
| Puerto ocupado | Otro proceso usa 8000 o 7860 | Usar `--port` en uvicorn o la variable `PUERTO_GRADIO` |

---

## Parte 2. Manual de usuario

### 2.1 Pantalla principal

La interfaz tiene dos columnas:

- **Izquierda:** formulario con los datos del solicitante, agrupados en *Ingresos y deuda*, *Comportamiento de pago* e *Historial crediticio*.
- **Derecha:** resultado de la evaluación.

En la parte superior se muestra el estado de la conexión con la API, el modelo cargado y el LLM configurado.

### 2.2 Campos del formulario

| Campo | Descripción | Rango |
|---|---|---|
| Salario mensual neto | Ingreso mensual después de deducciones | Mayor que 0 |
| Deuda pendiente | Saldo de deuda actual | 0 o más |
| Utilización del crédito (%) | Porcentaje del límite de crédito usado | 0 a 100 |
| Tasa de interés (%) | Tasa de interés de sus créditos | 0 a 100 |
| Días de atraso promedio | Días de atraso respecto a la fecha de pago | −30 a 365 |
| Número de pagos atrasados | Cantidad de pagos realizados tarde | 0 a 100 |
| ¿Paga solo el monto mínimo? | Sí / No | — |
| Mezcla de crédito | Buena, Estándar o Mala | — |
| Consultas de crédito recientes | Veces que se consultó su historial | 0 a 100 |
| Antigüedad (años y meses) | Tiempo desde su primer crédito | Meses de 0 a 11 |

### 2.3 Evaluar una solicitud

1. Completar el formulario, o seleccionar uno de los **ejemplos** de la parte inferior (riesgo bajo, medio y alto).
2. Marcar o desmarcar **Redactar justificación con qwen2.5-coder:7b**. Desmarcado, la justificación se genera por reglas y la respuesta es inmediata.
3. Presionar **Evaluar solicitud**.

### 2.4 Interpretar el resultado

**Decisión del agente**

| Nivel predicho | Decisión | Acción |
|---|---|---|
| Riesgo bajo | APROBADA | Aprobar solicitud con condiciones estándar. |
| Riesgo medio | EN REVISIÓN | Solicitar documentación adicional y evaluar nuevamente. |
| Riesgo alto | RECHAZADA | Rechazar solicitud y recomendar educación financiera. |

La tarjeta también muestra la **confianza** (probabilidad de la clase predicha). Pueden aparecer avisos:

- *Confianza baja*: la probabilidad es menor a 50 %; el caso requiere revisión manual.
- *Valores fuera del rango de entrenamiento*: algún dato está fuera de los valores con que se entrenó el modelo.

**Justificación:** párrafo que explica la decisión con los factores más relevantes. Indica si la redactó el LLM o si se generó por reglas.

**Factores analizados:** tabla que compara cada dato del solicitante con el valor típico (mediana) de los clientes de riesgo bajo y de riesgo alto. Cada variable se marca como *Eleva el riesgo* o *Reduce el riesgo* según a cuál de los dos perfiles se acerca más.

**Respuesta de la API (/predict):** sección plegable con el JSON devuelto por la API.

### 2.5 Mensajes de error

| Mensaje | Significado |
|---|---|
| No se pudo conectar con la API | La API no está en ejecución o la URL es incorrecta |
| La API no respondió en N segundos | La API tardó más del tiempo de espera |
| Datos de entrada inválidos | Algún campo no cumple la validación (se indica cuál) |
| El servicio no está disponible | La API está activa pero sin modelo cargado |

### 2.6 Uso directo de la API

Documentación interactiva (Swagger): http://127.0.0.1:8000/docs

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio y del modelo |
| GET | `/model-info` | Variables, niveles de riesgo, perfiles y rangos |
| POST | `/predict` | Predicción de riesgo |

Ejemplo de solicitud:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body '{"Credit_Utilization_Ratio":25.63,"Outstanding_Debt":1886.54,"Interest_Rate":17,"Num_Credit_Inquiries":12,"Delay_from_due_date":13,"Credit_Mix":"Standard","Payment_of_Min_Amount":"Yes","Credit_History_Age":225,"Num_of_Delayed_Payment":16,"Monthly_Inhand_Salary":971.16}'
```

Respuesta:

```json
{
  "risk_level": "high",
  "risk_class": 2,
  "risk_label": "Riesgo alto",
  "score": 0.7554,
  "probabilities": {"low": 0.0039, "medium": 0.2407, "high": 0.7554},
  "out_of_range": [],
  "model": "Árbol de decisión (max_depth=4)"
}
```

Si un campo no es válido la API responde con código **422** y el detalle de cada error.
