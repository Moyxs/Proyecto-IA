"""Cliente HTTP del agente hacia la API de riesgo crediticio."""

import time

import requests

from agente.configuracion import REINTENTOS_API, TIEMPO_ESPERA_API, URL_API


class ErrorAPI(Exception):
    def __init__(self, mensaje: str, codigo: int | None = None, detalle=None):
        super().__init__(mensaje)
        self.mensaje = mensaje
        self.codigo = codigo
        self.detalle = detalle


class ClienteAPI:
    def __init__(self, url_base: str = URL_API, tiempo_espera: float = TIEMPO_ESPERA_API,
                 reintentos: int = REINTENTOS_API):
        self.url_base = url_base.rstrip("/")
        self.tiempo_espera = tiempo_espera
        self.reintentos = reintentos

    def _solicitar(self, metodo: str, ruta: str, **kwargs) -> dict:
        url = f"{self.url_base}{ruta}"
        ultimo_error = None
        for intento in range(self.reintentos + 1):
            try:
                respuesta = requests.request(metodo, url, timeout=self.tiempo_espera, **kwargs)
            except requests.ConnectionError as error:
                ultimo_error = ErrorAPI(f"No se pudo conectar con la API en {self.url_base}. "
                                        "Verifique que el servidor esté en ejecución.")
            except requests.Timeout:
                ultimo_error = ErrorAPI(f"La API no respondió en {self.tiempo_espera} segundos.")
            else:
                return self._procesar(respuesta)
            if intento < self.reintentos:
                time.sleep(0.5 * (intento + 1))
        raise ultimo_error

    @staticmethod
    def _procesar(respuesta: requests.Response) -> dict:
        if respuesta.status_code == 200:
            try:
                return respuesta.json()
            except ValueError as error:
                raise ErrorAPI("La API devolvió una respuesta que no es JSON válido.") from error

        try:
            detalle = respuesta.json().get("detail")
        except ValueError:
            detalle = respuesta.text

        if respuesta.status_code == 422:
            campos = []
            if isinstance(detalle, list):
                for e in detalle:
                    campo = ".".join(str(p) for p in e.get("loc", [])[1:])
                    campos.append(f"{campo}: {e.get('msg')}")
            raise ErrorAPI("Datos de entrada inválidos: " + "; ".join(campos), 422, detalle)
        if respuesta.status_code == 503:
            raise ErrorAPI(f"El servicio no está disponible: {detalle}", 503, detalle)
        raise ErrorAPI(f"Error de la API (HTTP {respuesta.status_code}): {detalle}", respuesta.status_code, detalle)

    def salud(self) -> dict:
        return self._solicitar("GET", "/health")

    def informacion_modelo(self) -> dict:
        return self._solicitar("GET", "/model-info")

    def predecir(self, solicitud: dict) -> dict:
        return self._solicitar("POST", "/predict", json=solicitud)
