"""
RE02: Programacion de horarios de monitoreo automatizado.

El abigeato ocurre predominantemente de noche, asi que el sistema opera
dentro de ventanas horarias configurables en lugar de correr 24/7. Esto evita
falsas alarmas por actividad diurna (operarios, visitas) y reduce el consumo
de la camara y del equipo.

Las ventanas se guardan en `configs/config.yaml` bajo `schedule.windows`, cada
una con `start` y `end` en formato HH:MM. Una ventana que cruza la medianoche
(por ejemplo 20:00 -> 06:00) se maneja correctamente: la ventana sigue activa
despues de medianoche cuando `end < start`.
"""

import time
from datetime import datetime

try:
    from src.utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger("schedule")


def _parse_hhmm(value):
    """Convierte 'HH:MM' a minutos desde medianoche. Acepta int o str."""
    if isinstance(value, int):
        return value
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Formato de hora invalido: {value!r} (se espera HH:MM)")
    return int(parts[0]) * 60 + int(parts[1])


class MonitoringSchedule:
    def __init__(self, config):
        """`config` es la seccion `schedule` del config.yaml."""
        self.config = config or {}
        self.windows = self.config.get("windows", [{"start": "00:00", "end": "23:59"}])
        self.enabled = self.config.get("enabled", True)
        self.on_enter_log = self.config.get("log_transitions", True)
        self._was_active = None
        self._last_evaluated_minute = None

    def is_active(self, dt=None):
        """True si el instante cae dentro de alguna ventana configurada.

        Ventana que cruza medianoche: si end < start, la ventana cubre desde
        `start` hasta medianoche y desde las 00:00 hasta `end`.
        """
        if not self.enabled:
            return True  # si elschedule esta deshabilitado, opera continuo

        if dt is None:
            now = datetime.now()
        else:
            now = dt
        current = now.hour * 60 + now.minute

        for window in self.windows:
            start = _parse_hhmm(window["start"])
            end = _parse_hhmm(window["end"])

            if start == end:
                # Ventana de 24 horas.
                return True
            if start < end:
                if start <= current < end:
                    return True
            else:
                # Cruzando medianoche.
                if current >= start or current < end:
                    return True
        return False

    def active_windows_now(self, dt=None):
        """Devuelve las ventanas horarias que abarcan el instante actual."""
        if dt is None:
            now = datetime.now()
        else:
            now = dt
        current = now.hour * 60 + now.minute
        active = []
        for window in self.windows:
            start = _parse_hhmm(window["start"])
            end = _parse_hhmm(window["end"])
            if start == end:
                active.append(window)
            elif start < end:
                if start <= current < end:
                    active.append(window)
            else:
                if current >= start or current < end:
                    active.append(window)
        return active

    def next_transition(self, dt=None):
        """Proximo cambio de estado (entrar o salir de una ventana).

        Se usa para mostrar en la GUI "la proxima activacion es a las 20:00".
        """
        if not self.enabled:
            return None
        if dt is None:
            now = datetime.now()
        else:
            now = dt
        current = now.hour * 60 + now.minute

        candidates = []
        for window in self.windows:
            for moment in (_parse_hhmm(window["start"]), _parse_hhmm(window["end"])):
                if moment == current:
                    continue
                delta = (moment - current) % (24 * 60)
                candidates.append(delta)
        if not candidates:
            return None
        delta = min(candidates)
        target = (current + delta) % (24 * 60)
        return f"{target // 60:02d}:{target % 60:02d}"

    def check_and_log_transition(self):
        """Detecta el cambio activo/inactivo y lo registra una sola vez.

        La GUI la llama en cada frame. El log de transiciones deja evidencia
        auditable de que el sistema opero dentro del horario pactado.
        """
        active = self.is_active()
        now_min = int(time.time() // 60)
        if self._last_evaluated_minute == now_min:
            return active
        self._last_evaluated_minute = now_min

        if self._was_active is not None and active != self._was_active:
            if self.on_enter_log:
                if active:
                    logger.info(f"[RE02] Monitoreo ACTIVADO. Ventanas: {self.active_windows_now()}")
                else:
                    logger.info("[RE02] Monitoreo FUERA DE HORARIO: deteccion suspendida.")
        self._was_active = active
        return active

    def describe(self):
        """Texto corto para el HUD."""
        if not self.enabled:
            return "CONTINUO (24/7)"
        if not self.windows:
            return "SIN VENTANAS"
        parts = [f"{w['start']}-{w['end']}" for w in self.windows]
        return " / ".join(parts)
