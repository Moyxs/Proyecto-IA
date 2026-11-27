"""Interfaz Gradio del agente de riesgo crediticio. Ejecutar con: python -m agente.app"""

import logging
import threading

import gradio as gr

from agente.agente import AgenteRiesgoCrediticio, ResultadoAgente
from agente.analisis import describir_factor
from agente.cliente_api import ErrorAPI
from agente.configuracion import HOST_GRADIO, MODELO_LLM, PUERTO_GRADIO, URL_API

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

agente = AgenteRiesgoCrediticio()

OPCIONES_CREDIT_MIX = [("Buena (Good)", "Good"), ("Estándar (Standard)", "Standard"), ("Mala (Bad)", "Bad")]
OPCIONES_PAGO_MINIMO = [("Sí", "Yes"), ("No", "No")]

EJEMPLOS = [
    [34.18, 597.31, 9, 2, 13, "Good", "No", 28, 4, 8, 8278.51, True],
    [37.08, 156.11, 15, 1, 27, "Standard", "No", 20, 4, 14, 7336.87, True],
    [25.63, 1886.54, 17, 12, 13, "Standard", "Yes", 18, 9, 16, 971.16, True],
]


def construir_solicitud(utilizacion, deuda, tasa, consultas, atraso, credit_mix, pago_minimo,
                        anios_historial, meses_historial, pagos_atrasados, salario) -> dict:
    return {
        "Credit_Utilization_Ratio": float(utilizacion),
        "Outstanding_Debt": float(deuda),
        "Interest_Rate": float(tasa),
        "Num_Credit_Inquiries": int(consultas),
        "Delay_from_due_date": int(atraso),
        "Credit_Mix": credit_mix,
        "Payment_of_Min_Amount": pago_minimo,
        "Credit_History_Age": int(anios_historial) * 12 + int(meses_historial),
        "Num_of_Delayed_Payment": int(pagos_atrasados),
        "Monthly_Inhand_Salary": float(salario),
    }


def tarjeta_decision(r: ResultadoAgente) -> str:
    d = r.decision
    p = r.prediccion
    avisos = "".join(f"<li>{a}</li>" for a in r.advertencias)
    avisos_html = f"<ul style='margin:8px 0 0;padding-left:18px'>{avisos}</ul>" if avisos else ""
    return (
        f"<div style='border-left:6px solid {d.color};padding:12px 16px;border-radius:6px;"
        f"background:rgba(127,127,127,.08)'>"
        f"<div style='font-size:13px;opacity:.75'>Decisión del agente</div>"
        f"<div style='font-size:26px;font-weight:700;color:{d.color}'>{d.resultado}</div>"
        f"<div style='font-size:17px;margin-top:4px'><b>{d.nivel}</b> · confianza {p['score']:.0%}</div>"
        f"<div style='margin-top:8px'>{d.accion}</div>{avisos_html}</div>"
    )


def _formato(valor) -> str:
    return f"{valor:,.2f}".rstrip("0").rstrip(".") if isinstance(valor, (int, float)) else str(valor)


def tabla_factores(r: ResultadoAgente) -> str:
    filas = []
    for tipo, lista in (("Eleva el riesgo", r.factores["desfavorables"]), ("Reduce el riesgo", r.factores["favorables"])):
        for f in lista:
            filas.append(f"| {tipo} | {f['nombre']} | {_formato(f['valor'])} | "
                         f"{_formato(f['mediana_riesgo_bajo'])} | {_formato(f['mediana_riesgo_alto'])} |")
    if not filas:
        return "Sin factores relevantes."
    encabezado = "| Efecto | Variable | Valor | Típico riesgo bajo | Típico riesgo alto |\n|---|---|---|---|---|\n"
    return encabezado + "\n".join(filas)


def evaluar_solicitud(utilizacion, deuda, tasa, consultas, atraso, credit_mix, pago_minimo,
                      anios_historial, meses_historial, pagos_atrasados, salario, usar_llm):
    try:
        solicitud = construir_solicitud(utilizacion, deuda, tasa, consultas, atraso, credit_mix, pago_minimo,
                                        anios_historial, meses_historial, pagos_atrasados, salario)
    except (TypeError, ValueError):
        error = "<div style='color:#c0392b'><b>Error:</b> complete todos los campos con valores numéricos.</div>"
        return error, "", "", {}

    r = agente.evaluar(solicitud, usar_llm=usar_llm)
    if not r.exito:
        error = f"<div style='color:#c0392b'><b>No se pudo evaluar la solicitud.</b><br>{r.error}</div>"
        return error, "", "", {}

    origen = f"Redactada por {r.origen_justificacion}" if r.origen_justificacion != "reglas" \
        else "Generada por reglas"
    justificacion = f"{r.justificacion}\n\n*{origen}.*"
    return tarjeta_decision(r), justificacion, tabla_factores(r), r.prediccion


def estado_servicios() -> str:
    try:
        salud = agente.cliente.salud()
        api = f"API: conectada ({salud.get('model')})" if salud.get("model_loaded") else "API: sin modelo cargado"
    except ErrorAPI as error:
        api = f"API: no disponible. {error.mensaje}"
    return f"{api} · LLM: {MODELO_LLM} · URL: {URL_API}"


def crear_interfaz() -> gr.Blocks:
    with gr.Blocks(title="Agente de Riesgo Crediticio") as demo:
        gr.Markdown("# Agente de Riesgo Crediticio\nIngrese los datos del solicitante para evaluar la solicitud de crédito.")
        estado = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Ingresos y deuda")
                    salario = gr.Number(label="Salario mensual neto", value=3500, minimum=1)
                    deuda = gr.Number(label="Deuda pendiente", value=1200, minimum=0)
                    utilizacion = gr.Slider(0, 100, value=32, step=0.1, label="Utilización del crédito (%)")
                    tasa = gr.Slider(0, 100, value=12, step=1, label="Tasa de interés (%)")
                with gr.Group():
                    gr.Markdown("### Comportamiento de pago")
                    atraso = gr.Slider(-30, 365, value=15, step=1, label="Días de atraso promedio")
                    pagos_atrasados = gr.Slider(0, 100, value=12, step=1, label="Número de pagos atrasados")
                    pago_minimo = gr.Radio(OPCIONES_PAGO_MINIMO, value="Yes", label="¿Paga solo el monto mínimo?")
                with gr.Group():
                    gr.Markdown("### Historial crediticio")
                    credit_mix = gr.Dropdown(OPCIONES_CREDIT_MIX, value="Standard", label="Mezcla de crédito")
                    consultas = gr.Slider(0, 100, value=4, step=1, label="Consultas de crédito recientes")
                    with gr.Row():
                        anios = gr.Number(label="Antigüedad (años)", value=18, minimum=0, precision=0)
                        meses = gr.Number(label="Antigüedad (meses)", value=4, minimum=0, maximum=11, precision=0)
                usar_llm = gr.Checkbox(value=True, label=f"Redactar justificación con {MODELO_LLM}")
                boton = gr.Button("Evaluar solicitud", variant="primary")

            with gr.Column(scale=1):
                decision = gr.HTML()
                gr.Markdown("### Justificación")
                justificacion = gr.Markdown()
                gr.Markdown("### Factores analizados")
                factores = gr.Markdown()
                with gr.Accordion("Respuesta de la API (/predict)", open=False):
                    respuesta = gr.JSON()

        entradas = [utilizacion, deuda, tasa, consultas, atraso, credit_mix, pago_minimo,
                    anios, meses, pagos_atrasados, salario, usar_llm]
        salidas = [decision, justificacion, factores, respuesta]

        gr.Examples(examples=EJEMPLOS, inputs=entradas, label="Ejemplos: riesgo bajo, medio y alto")
        boton.click(evaluar_solicitud, inputs=entradas, outputs=salidas)
        demo.load(estado_servicios, outputs=estado)
    return demo


if __name__ == "__main__":
    threading.Thread(target=agente.explicador.precalentar, daemon=True).start()
    crear_interfaz().launch(server_name=HOST_GRADIO, server_port=PUERTO_GRADIO)
