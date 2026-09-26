"""
Orquestador de la cadena de alerta del sistema.

Este modulo concentra la logica de negocio de la alerta para que la GUI
(`src/gui/tab_detection.py`) y el pipeline de consola
(`src/06_realtime_pipeline.py`) ejecuten EXACTAMENTE el mismo flujo.
Antes cada consumidor hablaba directo con `AlertSystem` por su cuenta, lo
que hacia que la GUI se quedara sin alertas, sin clips y sin cola offline.

Requisitos cobertos aqui:
  RE02 - Programacion de horarios de monitoreo (delega en schedule.py).
  RE04 - Aviso a contactos de emergencia (delega en contacts.py).
  RE11 - Filtrado de ruido y condiciones ambientales (delega en frame_filter.py).
  RE12 - Alerta local en la caseta de vigilancia.
  RE13 - Notificacion remota con evidencia visual.
  RE14 - Respuesta disuasoria.
  RE15 - Escalamiento automatico ante inaccion del operador.
  RE17 - Clip de video probatorio (pre + post evento).
  RE19 - Operacion local ante corte de Internet.
  RE20 - Sincronizacion de la cola cuando vuelve la conectividad.
  RE21 - Compromiso fisico de la camara (delega en integrity_monitor.py).

Uso tipico desde el loop de deteccion:

    coordinator = AlertCoordinator(config, db)
    ...
    while running:
        frame, quality, score, metrics = coordinator.push_frame(raw_frame)
        status, detail, compromised = coordinator.watch_integrity(frame, fps)
        if coordinator.schedule.is_active() and not compromised:
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
    from src.schedule import MonitoringSchedule
    from src.contacts import ContactDirectory
    from src.frame_filter import FrameQualityFilter
    from src.integrity_monitor import IntegrityMonitor
except ImportError:
    # Importado como modulo suelto: alert_coordinator
    from alert_system import AlertSystem
    from video_recorder import ClipRecorder
    from schedule import MonitoringSchedule
    from contacts import ContactDirectory
    from frame_filter import FrameQualityFilter
    from integrity_monitor import IntegrityMonitor

# Etiqueta logica para un falso positivo sin clase real declarada. No es una
# clase del modelo: en el dataset exportado equivale a "sin objeto".
BACKGROUND_LABEL = "background"


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
        # Techo de antiquity para escalar (RE15). Sin el, reiniciar el sistema
        # con un evento viejo sin confirmar vuelve a disparar la emergencia.
        self.max_pending_age = alerts_cfg.get("max_escalation_age_seconds",
                                              72 * 3600)
        self.human_class = config.get("classes", {}).get("human_class", "humano")
        self.snapshot_dir = rec.get("output_dir", "data/event_clips")

        self._lock = threading.Lock()
        self._last_tick = 0.0
        self._tick_interval = max(alerts_cfg.get("cooldown_seconds", 30) / 3.0, 5.0)
        self._closed = False
        self.escalation_notice = None

        # RE02: ventanas horarias de monitoreo. Con `enabled: false` el
        # sistema opera continuo.
        self.schedule = MonitoringSchedule(config.get("schedule", {}))

        # RE11: filtro de ruido y condiciones ambientales.
        self.quality_filter = FrameQualityFilter(
            config.get("frame_filter", {}), history_size=8
        )

        # RE21: vigilancia del compromiso fisico de la camara.
        self.integrity = IntegrityMonitor(config.get("integrity", {}), db=db)

        # RE04: directorio de contactos de emergencia. Comparte la misma base
        # de datos, por lo que crea su esquema si aun no existe.
        self.contacts = ContactDirectory(getattr(db, "db_path", "data/events.db"))

        # Estado operativo que la GUI muestra en el HUD.
        self.suspended_reason = None
        self.stats = {"events": 0, "suppressed_by_schedule": 0,
                      "suppressed_by_quality": 0, "contacts_notified": 0}

        os.makedirs(self.snapshot_dir, exist_ok=True)

    # ---------- Buffer de video (RE17) ----------

    def push_frame(self, frame):
        """Evalua la calidad del frame ORIGINAL (RE11) y devuelve
        (frame_util, quality, score, metrics).

        No escribe en el buffer de video: eso lo hace `push_annotated` con la
        imagen ya dibujada. Mantener una sola entrada por frame al buffer es
        indispensable, porque el clip probatorio (RE17) se arma con esta
        secuencia; si se empujaran el frame crudo y el anotado, el clip
        reproduciria a doble velocidad y desalineado respecto de la alerta.

        Cuando la imagen es inutil (negra) se sustituye por el ultimo frame
        limpio, de modo que la vigilancia no se queda ciega ante niebla o un
        taponamiento momentaneo.
        """
        if self._closed or frame is None:
            return frame, "ok", 1.0, {}
        return self.quality_filter.process(frame)

    def push_annotated(self, annotated_frame):
        """Agrega al buffer el frame YA anotado (bounding boxes + zona).

        Es la unica via de entrada al buffer circular, para mantener la
        correspondencia 1 frame capturado = 1 frame del clip.
        """
        if annotated_frame is not None and not self._closed:
            self.recorder.push_frame(annotated_frame)

    # ---------- RE02 / RE11 / RE21: condiciones de operacion ----------

    def watch_integrity(self, frame, fps=None):
        """Actualiza el monitor de compromiso fisico de la camara (RE21)."""
        return self.integrity.update(frame, quality_filter=self.quality_filter,
                                     fps=fps)

    def can_detect(self):
        """Indica si se permite inferencia en este instante.

        Tres condiciones, en orden de prioridad:
          1. RE02: fuera del horario programado, el sistema espera.
          2. RE21: camara comprometida, la deteccion seria enganosa.
          3. RE11: calidad insuficiente para una deteccion confiable.
        """
        if not self.schedule.check_and_log_transition():
            self.suspended_reason = "FUERA DE HORARIO (RE02)"
            return False

        if self.integrity.last_status == "comprometida":
            self.suspended_reason = f"CAMARA COMPROMETIDA (RE21): {self.integrity.last_detail}"
            return False

        quality = self.quality_filter.last_quality
        allowed, why = self.quality_filter.is_detection_allowed(quality,
                                                                self.quality_filter.last_score)
        if not allowed:
            self.suspended_reason = f"CALIDAD INSUFICIENTE (RE11): {why}"
            return False

        self.suspended_reason = None
        return True

    def hud_status(self):
        """Estado operativo compacto para el HUD de la GUI."""
        active = self.schedule.check_and_log_transition()
        return {
            "schedule_enabled": self.schedule.enabled,
            "schedule": self.schedule.describe(),
            "schedule_active": active,
            "quality": self.quality_filter.last_quality,
            "quality_score": round(self.quality_filter.last_score, 2),
            "degraded_seconds": self.quality_filter.degradation_seconds(),
            "integrity": self.integrity.last_status,
            "integrity_detail": self.integrity.last_detail,
            "integrity_alert": self.integrity.active_alert(),
            "suspended_reason": self.suspended_reason,
            "detection_allowed": self.can_detect(),
            "contacts": len(self.contacts.list_contacts()),
            "stats": dict(self.stats),
        }

    # ---------- Alerta de intrusion (RE12/13/14/17) ----------

    def register_intrusion(self, display_frame, confidence, human_class=None,
                           bbox=None):
        """Ejecuta la cadena completa de alerta ante una intrusion humana.

        `bbox` es la caja que produjo la deteccion ("x1,y1,x2,y2"). Se guarda en
        el evento para que RE18 pueda escribir una etiqueta YOLO con caja cuando
        el operador marque el disparo como falso positivo.

        Devuelve el `event_id` creado, o None si el evento cae dentro del
        periodo de enfriamiento configurado o si el sistema esta suspendido
        por RE02/RE11/RE21.
        """
        human_class = human_class or self.human_class

        # RE02/RE11/RE21: no se generan alertas cuando el sistema no puede
        # afirmar con confianza lo que ve.
        if not self.can_detect():
            if self.suspended_reason and "RE02" in self.suspended_reason:
                self.stats["suppressed_by_schedule"] += 1
            elif self.suspended_reason and "RE11" in self.suspended_reason:
                self.stats["suppressed_by_quality"] += 1
            return None

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
            bbox=bbox,
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

        # RE04: aviso a los contactos de emergencia registrados.
        self._notify_emergency_contacts(event_id, snapshot_path)

        self.alerts.mark_alerted(human_class)
        self.stats["events"] += 1
        return event_id

    def _notify_emergency_contacts(self, event_id, snapshot_path):
        """RE04: recorre el directorio de contactos en orden de prioridad."""
        def dispatcher(contact, _event_id):
            ok = True
            if contact["channel"] == "webhook" and snapshot_path:
                ok = self.alerts.send_remote_notification(
                    _event_id, contact["name"], 1.0, snapshot_path,
                    recipient=contact.get("email") or contact.get("phone"),
                )
            else:
                # SMS/email/telefono: se deja traza en el log porque el
                # despliegue de campo usa una pasarela SMS o un telefono
                # fijo en la caseta, no un envio HTTP directo.
                self.alerts.logger.info(
                    f"[RE04] Avisando por {contact['channel']} a "
                    f"{contact['name']} ({contact.get('phone') or contact.get('email')})"
                )
            try:
                contact = dict(contact)
                contact["delivered"] = ok
                self.db.insert_contact_log(_event_id, contact)
            except Exception as exc:  # noqa: BLE001
                self.alerts.logger.error(f"[RE04] Auditoria fallo: {exc}")
            return ok

        try:
            summary = self.contacts.notify_all(event_id, dispatcher=dispatcher)
            self.stats["contacts_notified"] += summary["notificados"]
        except Exception as exc:  # noqa: BLE001
            self.alerts.logger.error(f"[RE04] Directorio de contactos fallo: {exc}")

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
        rows = self.db.get_unacknowledged_events(
            self.escalate_after, limit=5, max_age_seconds=self.max_pending_age
        )
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

    def acknowledge_last(self, is_false_positive=False, access=None,
                         real_class=None):
        """Confirma (y opcionalmente marca como falso positivo) el ultimo
        evento pendiente. Devuelve (event_id, error) o (None, None) si no hay
        nada pendiente.

        RE18: marcar un falso positivo altera el dataset de entrenamiento, por
        lo que se exige el permiso `label_false_positive` que solo el
        Administrador posee. El Visualizador si puede confirmar la alarma.

        `real_class` es la clase que el operador vio realmente. El exportador la
        usa para la etiqueta YOLO: sin ella la muestra se exporta como negativa.
        """
        if is_false_positive and access is not None:
            try:
                access.require("label_false_positive")
            except Exception as exc:  # AccessDenied
                self.alerts.logger.error(f"[RE18] {exc}")
                return None, str(exc)

        row = self.db.get_last_unacknowledged_event()
        if not row:
            return None, None
        event_id = row[0]
        self.db.mark_event_acknowledged(event_id, is_false_positive, real_class)
        if is_false_positive:
            self.alerts.logger.info(
                f"[RE18] Evento {event_id} marcado como falso positivo "
                f"(clase real: {real_class or BACKGROUND_LABEL}); "
                "disponible para el dataset de reentrenamiento."
            )
        return event_id, None

    def remote_status(self):
        """Estado del canal remoto, para que la GUI no senale un problema
        inexistente (RE13/RE19/RE20).

        Con `alerts.remote.enabled = false` los eventos se encolan igual (la cola
        es tambien la traza de auditoria), pero NADA sale de ahi. Mostrar "COLA
        OFFLINE 12" invita al operador a buscar una falla de conectividad que no
        existe: lo correcto es decir que el canal esta desactivado y quantas
        alertas quedan sin envio.
        """
        enabled = bool(self.config.get("alerts", {})
                       .get("remote", {}).get("enabled", False))
        pending = self.db.get_pending_count()
        return {
            "enabled": enabled,
            "pending": pending,
            "note": ("" if enabled else
                     f"CANAL REMOTO DESACTIVADO: {pending} alerta(s) en cola "
                     "sin envio (traza de auditoria)"),
        }

    def pending_summary(self):
        """Resumen para la GUI: notificaciones encoladas y evento abierto.

        El evento abierto se limita a la misma ventana de antigüedad que el
        escalamiento: un evento de hace dias no se sigue mostrando como "la
        alerta activa" del sistema, porque induce al operador a creer que hay
        una intrusión en curso cuando en realidad es un registro histórico sin
        confirmar. El evento sigue en el historial y se puede confirmar desde
        ahí (`acknowledge_last` no aplica este filtro a propósito).
        """
        with self._lock:
            pending_remote = self.db.get_pending_count()
        row = self.db.get_last_unacknowledged_event(
            max_age_seconds=self.max_pending_age
        )
        return pending_remote, row

    # ---------- RE20: sincronizacion explicita ----------

    def sync_pending(self, force=False):
        """RE20: sincroniza la cola acumulada durante el corte de Internet.

        Devuelve cuantos eventos quedaron entregados. El pipeline de consola
        la invoca al detectar que volvio la conectividad; la GUI la llama con
        `force` desde el boton "Sincronizar ahora".
        """
        before = self.db.get_pending_count()
        self.alerts.retry_pending_notifications()
        after = self.db.get_pending_count()
        synced = max(before - after, 0)
        if synced or force:
            self.alerts.logger.info(
                f"[RE20] Sincronizacion: {synced} evento(s) entregados, "
                f"{after} aun en cola."
            )
        return synced

    def close(self):
        self._closed = True
        try:
            self.recorder._close_writer()
        except Exception:
            pass
