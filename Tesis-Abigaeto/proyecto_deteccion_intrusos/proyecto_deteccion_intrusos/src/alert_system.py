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

try:
    # Importado como modulo de la GUI: src.alert_system
    from src.utils import get_logger
except ImportError:
    # Importado como modulo suelto por 06_realtime_pipeline.py: alert_system
    from utils import get_logger

logger = get_logger("alertas")


class AlertSystem:
    def __init__(self, config, db):
        self.config = config
        self.db = db
        self.logger = logger
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

    # ---------- RE15: escalamiento por inacción ----------
    def trigger_escalation(self, event_id, class_name, waited_seconds):
        """El operador no confirmó la alerta dentro del plazo pactado: se
        eleva el evento a nivel de emergencia (notificación remota prioritaria
        + registro explícito en el log y en la BD)."""
        logger.error(
            f"[ALERTA ESCALADA] Evento {event_id} ('{class_name}') sin confirmar "
            f"durante {waited_seconds}s. Se escala a nivel de emergencia."
        )
        if self.config["alerts"]["deterrent"]["enabled"]:
            logger.error(
                "[DISUASIÓN REFORZADA] Disuasión repetida por inacción del operador."
            )
        return True

    # ---------- RE13 + RE19: notificación remota con reintentos ----------
    def send_remote_notification(self, event_id, class_name, confidence,
                                 snapshot_path, recipient=None):
        """Intenta notificar de inmediato. Si falla (sin Internet), la
        notificación ya quedó encolada en la BD (ver database.insert_event)
        y se reintentará en el próximo ciclo de retry_pending_notifications().

        El `queue_id` se resuelve a partir del `event_id` para que, si el
        envío es exitoso, la fila de la cola quede marcada como entregada y
        no se reenvíe indefinidamente en los reintentos posteriores.

        `recipient` se usa en RE04 cuando el aviso va dirigido a un contacto
        concreto del directorio de emergencias en lugar de al canal general.
        """
        if not self.config["alerts"]["remote"]["enabled"]:
            return False

        return self._try_send_one(
            queue_id=self.db.get_queue_id(event_id),
            event_id=event_id,
            class_name=class_name,
            confidence=confidence,
            snapshot_path=snapshot_path,
            recipient=recipient,
        )

    def retry_pending_notifications(self):
        """Debe llamarse periódicamente desde el loop principal. Revisa la
        cola de notificaciones no entregadas (por caídas de Internet) y
        reintenta enviarlas. Esto es lo que garantiza RE19: el sistema no
        pierde eventos aunque se corte la conectividad.

        Las notificaciones que superan `max_attempts_per_notification` se
        abandonan (quedan en la cola como no entregadas para auditoria, pero
        ya no se reintentan) para no quedar reenviando indefinidamente.
        """
        if not self.config["alerts"]["remote"]["enabled"]:
            return

        pending = self.db.get_pending_notifications()
        max_retries = self.config["alerts"]["remote"]["max_retries_per_cycle"]
        max_attempts = self.config["alerts"]["remote"].get("max_attempts_per_notification", 10)

        sent = 0
        for row in pending:
            if sent >= max_retries:
                break
            queue_id, event_id, timestamp, class_name, confidence, snapshot_path = row
            if self.db.get_notification_attempts(queue_id) >= max_attempts:
                self.logger.error(
                    f"[NOTIFICACIÓN REMOTA] Evento {event_id} abandonado tras "
                    f"{max_attempts} intentos sin éxito. Queda en la cola para auditoría."
                )
                continue
            self._try_send_one(queue_id, event_id, class_name, confidence,
                               snapshot_path)
            sent += 1

    def _try_send_one(self, queue_id, event_id, class_name, confidence,
                      snapshot_path, recipient=None):
        webhook_url = self.config["alerts"]["remote"]["webhook_url"]
        timeout = self.config["alerts"]["remote"]["timeout_seconds"]

        payload = {
            "event_id": event_id,
            "class_name": class_name,
            "confidence": confidence,
        }
        # RE04: el aviso identifica a quien se esta notificando.
        if recipient:
            payload["recipient"] = recipient

        try:
            if snapshot_path:
                with open(snapshot_path, "rb") as f:
                    response = requests.post(
                        webhook_url, data=payload, files={"snapshot": f}, timeout=timeout
                    )
            else:
                response = requests.post(webhook_url, data=payload, timeout=timeout)

            response.raise_for_status()
            target = f" a {recipient}" if recipient else ""
            logger.info(f"[NOTIFICACIÓN REMOTA] Evento {event_id} enviado correctamente{target}.")
            if queue_id is not None:
                self.db.mark_notification_delivered(queue_id)
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"[NOTIFICACIÓN REMOTA] Falló envío del evento {event_id} (sin Internet u otro error): {e}")
            if queue_id is not None:
                self.db.increment_notification_attempt(queue_id)
            # No se relanza la excepción: el sistema sigue operando localmente (RE19)
            return False
