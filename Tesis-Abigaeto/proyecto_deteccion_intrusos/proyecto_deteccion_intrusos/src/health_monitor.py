"""
RE21: Servicio permanente de vigilancia del estado del entorno simulado
(procesos, servicios y recursos virtuales), emitiendo una notificación
inmediata ante una anomalía técnica.

Corre en un hilo (thread) separado del loop principal de inferencia, para
no afectar el rendimiento del procesamiento de video.
"""

import threading
import time

import psutil

try:
    # Importado como modulo de la GUI: src.health_monitor
    from src.utils import get_logger
except ImportError:
    # Importado como modulo suelto por 06_realtime_pipeline.py: health_monitor
    from utils import get_logger


class HealthMonitor(threading.Thread):
    def __init__(self, config, db):
        super().__init__(daemon=True)
        self.config = config["health_monitor"]
        self.db = db
        self.logger = get_logger("health_monitor", self.config["log_path"])
        self._stop_event = threading.Event()

    def run(self):
        self.logger.info("Monitor de salud del sistema iniciado.")
        while not self._stop_event.is_set():
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory().percent

            status = "OK"
            note = ""

            if cpu >= self.config["cpu_threshold_percent"]:
                status = "ANOMALIA"
                note += f"CPU alta ({cpu}%). "
            if ram >= self.config["ram_threshold_percent"]:
                status = "ANOMALIA"
                note += f"RAM alta ({ram}%). "

            self.db.insert_health_log(cpu, ram, status, note)

            if status == "ANOMALIA":
                self.logger.error(f"[SALUD DEL SISTEMA] Anomalía detectada: {note}")
            else:
                self.logger.info(f"[SALUD DEL SISTEMA] CPU {cpu}% | RAM {ram}% | OK")

            self._stop_event.wait(self.config["check_interval_seconds"])

    def stop(self):
        self._stop_event.set()
