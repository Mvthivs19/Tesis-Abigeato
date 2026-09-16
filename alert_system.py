"""
Sistema de alertas del proyecto. Cubre:

  RE12 - Alerta local simulada en caseta de vigilancia virtual
  RE14 - Respuesta disuasoria simulada (luces/sirena virtuales)
  RE13 - Notificación remota con evidencia visual
  RE19 - Mantener operación local ante corte de Internet
         (las notificaciones remotas se encolan en la BD y se reintentan
          más tarde; el sistema NUNCA depende de tener Internet para
          seguir alertando localmente)

Como no hay hardware físico (luces/sirena/caseta real), estas acciones se
"simulan": quedan registradas en el log y en la base de datos como si se
hubiesen ejecutado. Si en el futuro se conecta hardware real, solo hay
que reemplazar el contenido de _activar_luces_sirena_virtual() por una
llamada real (ej. a un microcontrolador vía GPIO/MQTT).
"""

import time
import requests

from utils import get_logger

logger = get_logger("alertas")


class AlertSystem:
    def __init__(self, config, db):
        self.config = config
        self.db = db
        self._last_alert_time = {}

    def cooldown_ok(self, class_name):
        cooldown = self.config["alerts"]["cooldown_seconds"]
        last = self._last_alert_time.get(class_name, 0)
        return (time.time() - last) >= cooldown

    def mark_alerted(self, class_name):
        self._last_alert_time[class_name] = time.time()

    # ---------- RE12: alerta local ----------
    def trigger_local_alert(self, class_name, confidence):
        if not self.config["alerts"]["local"]["enabled"]:
            return
        logger.warning(
            f"[ALERTA LOCAL SIMULADA] Intrusión '{class_name}' detectada "
            f"(confianza {confidence:.2f}). Señal visual intermitente + "
            f"sonora activada en la caseta de vigilancia virtual."
        )

    # ---------- RE14: disuasión simulada ----------
    def trigger_deterrent(self):
        if not self.config["alerts"]["deterrent"]["enabled"]:
            return
        logger.warning(
            "[DISUASIÓN SIMULADA] Luces y sirena virtuales del perímetro: ENCENDIDAS."
        )
        # Punto de extensión: aquí iría la llamada real a hardware físico
        # (GPIO, MQTT, API de un controlador) si en el futuro se despliega
        # en campo. Por ahora queda como acción simulada/loggeada.

    # ---------- RE13 + RE19: notificación remota con reintentos ----------
    def send_remote_notification(self, event_id, class_name, confidence, snapshot_path):
        """Intenta notificar de inmediato. Si falla (sin Internet), la
        notificación ya quedó encolada en la BD (ver database.insert_event)
        y se reintentará en el próximo ciclo de retry_pending_notifications()."""
        if not self.config["alerts"]["remote"]["enabled"]:
            return

        self._try_send_one(
            queue_id=None,  # se resuelve dentro de retry_pending si aplica
            event_id=event_id,
            class_name=class_name,
            confidence=confidence,
            snapshot_path=snapshot_path,
        )

    def retry_pending_notifications(self):
        """Debe llamarse periódicamente desde el loop principal. Revisa la
        cola de notificaciones no entregadas (por caídas de Internet) y
        reintenta enviarlas. Esto es lo que garantiza RE19: el sistema no
        pierde eventos aunque se corte la conectividad."""
        if not self.config["alerts"]["remote"]["enabled"]:
            return

        pending = self.db.get_pending_notifications()
        max_retries = self.config["alerts"]["remote"]["max_retries_per_cycle"]

        for row in pending[:max_retries]:
            queue_id, event_id, timestamp, class_name, confidence, snapshot_path = row
            self._try_send_one(queue_id, event_id, class_name, confidence, snapshot_path)

    def _try_send_one(self, queue_id, event_id, class_name, confidence, snapshot_path):
        webhook_url = self.config["alerts"]["remote"]["webhook_url"]
        timeout = self.config["alerts"]["remote"]["timeout_seconds"]

        payload = {
            "event_id": event_id,
            "class_name": class_name,
            "confidence": confidence,
        }
        try:
            if snapshot_path:
                with open(snapshot_path, "rb") as f:
                    response = requests.post(
                        webhook_url, data=payload, files={"snapshot": f}, timeout=timeout
                    )
            else:
                response = requests.post(webhook_url, data=payload, timeout=timeout)

            response.raise_for_status()
            logger.info(f"[NOTIFICACIÓN REMOTA] Evento {event_id} enviado correctamente.")
            if queue_id is not None:
                self.db.mark_notification_delivered(queue_id)

        except requests.exceptions.RequestException as e:
            logger.error(f"[NOTIFICACIÓN REMOTA] Falló envío del evento {event_id} (sin Internet u otro error): {e}")
            if queue_id is not None:
                self.db.increment_notification_attempt(queue_id)
            # No se relanza la excepción: el sistema sigue operando localmente (RE19)
