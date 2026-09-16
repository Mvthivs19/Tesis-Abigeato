"""
Paso 6 (pipeline principal): integra todo el sistema.

  RE06 - Procesa el flujo de video nocturno simulado, emulando las
         restricciones de un dispositivo Edge de bajo consumo (fps
         limitado, modelo liviano YOLOv8n).
  RE01 - Restringe la inferencia a la zona (polígono) definida con
         05_define_zone.py.
  RE12/RE14 - Dispara alertas locales y disuasión simulada.
  RE13/RE17 - Guarda snapshot + clip de evidencia y encola notificación.
  RE19 - Sigue operando localmente aunque falle el envío remoto.
  RE21 - Corre el monitor de salud en paralelo.

Uso:
  python src/06_realtime_pipeline.py
Presiona 'q' en la ventana de video para detener el sistema.
"""

import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

from utils import load_config, get_logger
from database import EventDatabase
from video_recorder import ClipRecorder
from alert_system import AlertSystem
from health_monitor import HealthMonitor

logger = get_logger("pipeline_principal")


def point_in_zone(cx, cy, polygon):
    if not polygon:
        return True  # si no se definió zona, se vigila el frame completo
    poly_np = np.array(polygon, dtype="int32")
    return cv2.pointPolygonTest(poly_np, (cx, cy), False) >= 0


def open_capture(source, logger, wait_seconds):
    """Abre la fuente de video con reintentos, para no morir si el
    'stream' simulado se corta momentáneamente (parte del espíritu de RE19:
    resiliencia operativa ante interrupciones)."""
    while True:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
        logger.error(f"No se pudo abrir la fuente de video '{source}'. Reintentando en {wait_seconds}s...")
        time.sleep(wait_seconds)


def main():
    config = load_config()

    weights_path = config["model"]["weights_path"]
    if not os.path.exists(weights_path):
        logger.error(f"No se encontró el modelo entrenado en: {weights_path}")
        logger.error("Corre primero src/04_train.py para generar 'best.pt'.")
        return

    model = YOLO(weights_path)
    class_names = config["classes"]["names"]
    human_class = config["classes"]["human_class"]
    conf_threshold = config["model"]["conf_threshold"]
    device = config["model"]["device"]
    imgsz = config["model"]["imgsz"]

    db = EventDatabase(config["database"]["path"])
    recorder = ClipRecorder(
        output_dir=config["recording"]["output_dir"],
        pre_seconds=config["recording"]["pre_event_seconds"],
        post_seconds=config["recording"]["post_event_seconds"],
        fps=config["camera"]["fps_limit"],
    )
    alerts = AlertSystem(config, db)

    health_monitor = HealthMonitor(config, db)
    health_monitor.start()

    polygon = config["zone"]["polygon"]
    if not polygon:
        logger.warning("No hay zona de vigilancia definida (RE01). Se vigilará el frame completo.")
        logger.warning("Corre src/05_define_zone.py para delimitar una zona específica.")

    source = config["camera"]["source"]
    fps_limit = config["camera"]["fps_limit"]
    frame_interval = 1.0 / fps_limit

    cap = open_capture(source, logger, config["camera"]["reconnect_wait_seconds"])
    logger.info("Pipeline principal iniciado. Presiona 'q' para detener.")

    last_notification_retry = 0
    notification_retry_interval = 30  # segundos

    try:
        while True:
            loop_start = time.time()

            ok, frame = cap.read()
            if not ok:
                logger.error("Se perdió la señal de video. Reintentando conexión (RE19)...")
                cap.release()
                cap = open_capture(source, logger, config["camera"]["reconnect_wait_seconds"])
                continue

            results = model.predict(
                frame, imgsz=imgsz, conf=conf_threshold, device=device, verbose=False
            )[0]

            display = frame.copy()
            if polygon:
                cv2.polylines(display, [np.array(polygon, dtype="int32")], True, (0, 255, 0), 2)

            human_detected_in_zone = None

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                class_name = class_names[cls_id]

                in_zone = point_in_zone(cx, cy, polygon)
                color = (0, 0, 255) if class_name == human_class else (255, 200, 0)
                cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(display, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if class_name == human_class and in_zone:
                    human_detected_in_zone = conf

            recorder.push_frame(display)

            if human_detected_in_zone is not None and alerts.cooldown_ok(human_class):
                alerts.mark_alerted(human_class)

                snapshot_path = os.path.join(
                    config["recording"]["output_dir"], f"snapshot_{int(time.time())}.jpg"
                )
                cv2.imwrite(snapshot_path, display)

                event_id = db.insert_event(
                    class_name=human_class,
                    confidence=human_detected_in_zone,
                    snapshot_path=snapshot_path,
                )
                clip_path = recorder.trigger(event_id)

                alerts.trigger_local_alert(human_class, human_detected_in_zone)
                alerts.trigger_deterrent()
                alerts.send_remote_notification(event_id, human_class, human_detected_in_zone, snapshot_path)

                logger.info(f"Evento {event_id} registrado. Clip: {clip_path}")

            if time.time() - last_notification_retry > notification_retry_interval:
                alerts.retry_pending_notifications()
                last_notification_retry = time.time()

            cv2.imshow("Sistema de deteccion (simulado) - Presiona 'q' para salir", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        health_monitor.stop()
        logger.info("Pipeline principal detenido.")


if __name__ == "__main__":
    main()
