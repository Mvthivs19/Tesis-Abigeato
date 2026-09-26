"""
Orquestador de la cadena de alerta del sistema.

Este modulo concentra la logica de negocio de la alerta para que la GUI
(`src/gui/tab_detection.py`) y el pipeline de consola
(`src/06_realtime_pipeline.py`) ejecuten EXACTAMENTE el mismo flujo.
Antes cada consumidor hablaba directo con `AlertSystem` por su cuenta, lo
que hacia que la GUI se quedara sin alertas, sin clips y sin cola offline.

Requisitos cobertos aqui:
  RE12 - Alerta local en la caseta de vigilancia.
  RE13 - Notificacion remota con evidencia visual.
  RE14 - Respuesta disuasoria.
  RE15 - Escalamiento automatico ante inaccion del operador.
  RE17 - Clip de video probatorio (pre + post evento).
  RE19 - Operacion local ante corte de Internet.
  RE20 - Sincronizacion de la cola cuando vuelve la conectividad.

Uso tipico desde el loop de deteccion:

    coordinator = AlertCoordinator(config, db)
    ...
    while running:
        coordinator.push_frame(display_frame)   # en CADA frame
        if humano_en_zona:
            coordinator.register_intrusion(display_frame, conf, snapshot_path)
        coordinator.tick()                       # reintentos + escalamiento
    ...
    coordinator.close()
"""

import os
import threading
import time

import cv2

try:
    # Importado como modulo de la GUI: src.alert_coordinator
    from src.alert_system import AlertSystem
    from src.video_recorder import ClipRecorder
except ImportError:
    # Importado como modulo suelto: alert_coordinator
    from alert_system import AlertSystem
    from video_recorder import ClipRecorder


class AlertCoordinator:
    def __init__(self, config, db):
        self.config = config
        self.db = db
        self.alerts = AlertSystem(config, db)

        rec = config.get("recording", {})
        self.recorder = ClipRecorder(
            output_dir=rec.get("output_dir", "data/event_clips"),
            pre_seconds=rec.get("pre_event_seconds", 5),
            post_seconds=rec.get("post_event_seconds", 10),
            fps=config.get("camera", {}).get("fps_limit", 10),
        )

        alerts_cfg = config.get("alerts", {})
        self.escalate_after = alerts_cfg.get("escalate_after_seconds", 60)
        self.human_class = config.get("classes", {}).get("human_class", "humano")
        self.snapshot_dir = rec.get("output_dir", "data/event_clips")

        self._lock = threading.Lock()
        self._last_tick = 0.0
        self._tick_interval = max(alerts_cfg.get("cooldown_seconds", 30) / 3.0, 5.0)
        self._closed = False
        self.escalation_notice = None

        os.makedirs(self.snapshot_dir, exist_ok=True)

    # ---------- Buffer de video (RE17) ----------

    def push_frame(self, frame):
        """Alimenta el buffer circular. Debe llamarse en CADA frame, con el
        frame ya anotado (bounding boxes y zona) para que el clip probatorio
        muestre la deteccion."""
        if frame is not None and not self._closed:
            self.recorder.push_frame(frame)

    # ---------- Alerta de intrusion (RE12/13/14/17) ----------

    def register_intrusion(self, display_frame, confidence, human_class=None):
        """Ejecuta la cadena completa de alerta ante una intrusion humana.

        Devuelve el `event_id` creado, o None si el evento cae dentro del
        periodo de enfriamiento configurado.
        """
        human_class = human_class or self.human_class

        # RE12/14/13/17: evita duplicar la cadena completa dentro del cooldown.
        if not self.alerts.cooldown_ok(human_class):
            return None

        now = time.time()

        # RE17: evidencia visual inmediata (snapshot).
        snapshot_path = os.path.join(self.snapshot_dir, f"snapshot_{int(now)}.jpg")
        try:
            cv2.imwrite(snapshot_path, display_frame)
        except Exception:
            snapshot_path = None

        # RE16: registro en base de datos (encola la notificacion remota).
        event_id = self.db.insert_event(
            class_name=human_class,
            confidence=confidence,
            snapshot_path=snapshot_path,
        )

        # RE17: clip de video con segundos previos y posteriores.
        clip_path = self.recorder.trigger(event_id)
        if clip_path:
            self.db.update_event_clip(event_id, clip_path)

        # RE12: aviso en la caseta de vigilancia.
        self.alerts.trigger_local_alert(human_class, confidence)

        # RE14: respuesta disuasoria.
        self.alerts.trigger_deterrent()

        # RE13: notificacion remota. Si no hay Internet queda encolada (RE19).
        if snapshot_path:
            self.alerts.send_remote_notification(
                event_id, human_class, confidence, snapshot_path
            )

        self.alerts.mark_alerted(human_class)
        return event_id

    # ---------- Mantenimiento periodico (RE15/19/20) ----------

    def tick(self, force=False):
        """Reintenta notificaciones pendientes y escala alertas no
        confirmadas. Se llama periodicamente desde el loop de deteccion."""
        now = time.time()
        if not force and (now - self._last_tick) < self._tick_interval:
            return
        self._last_tick = now

        # RE19/RE20: reintento de la cola cuando vuelve la conectividad.
        self.alerts.retry_pending_notifications()

        # RE15: escalamiento por inaccion del operador.
        self._check_escalation()

    def _check_escalation(self):
        rows = self.db.get_unacknowledged_events(self.escalate_after, limit=5)
        for event_id, timestamp, class_name, confidence, snapshot_path in rows:
            self.db.mark_event_escalated(event_id)
            waited = int(self.escalate_after)

            # RE15: escalamiento explicito + aviso local reforzado.
            self.alerts.trigger_escalation(event_id, class_name, waited)
            self.escalation_notice = (
                f"ESCALADO e{event_id} sin confirmar {waited}s"
            )

            # RE13: reintenta el envio remoto al escalar.
            if snapshot_path:
                self.alerts.send_remote_notification(
                    event_id, class_name, confidence, snapshot_path
                )

    # ---------- Confirmacion del operador (RE15/RE18) ----------

    def acknowledge_last(self, is_false_positive=False):
        """Confirma (y opcionalmente marca como falso positivo) el ultimo
        evento pendiente. Devuelve el event_id confirmado o None."""
        row = self.db.get_last_unacknowledged_event()
        if not row:
            return None
        event_id = row[0]
        self.db.mark_event_acknowledged(event_id, is_false_positive)
        return event_id

    def pending_summary(self):
        """Resumen para la GUI: notificaciones encoladas y evento abierto."""
        with self._lock:
            pending_remote = self.db.get_pending_count()
        row = self.db.get_last_unacknowledged_event()
        return pending_remote, row

    def close(self):
        self._closed = True
        try:
            self.recorder._close_writer()
        except Exception:
            pass
