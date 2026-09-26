"""
Paso 6 (pipeline principal): integra todo el sistema.

Este modulo es el MISMO flujo que la GUI (`src/gui/tab_detection.py`), no una
copia: ambos consumen `AlertCoordinator`. Antes esta version manejaba alertas,
clips, horarios y reintentos por su cuenta, y por eso se comportaba distinto
que la interfaz: la GUI no emitia clips, el CLI no respeta el horario de RE02 y
ninguno de los dos aplicaba el filtro de calidad de RE11.

  RE06 - Procesa el flujo de video nocturno simulado, emulando las
         restricciones de un dispositivo Edge de bajo consumo (fps
         limitado, modelo liviano YOLOv8n).
  RE01 - Restringe la inferencia a la zona (polígono) definida con
         05_define_zone.py.
  RE02 - Programa las ventanas horarias de monitoreo.
  RE04 - Avisa a los contactos de emergencia registrados.
  RE11 - Filtra ruido y condiciones ambientales (borroso, negro, ruido).
  RE12 - Alerta local en la caseta de vigilancia.
  RE13 - Notificacion remota con evidencia visual (encolada si no hay red).
  RE14 - Respuesta disuasoria.
  RE15 - Escala el evento si el operador no lo confirma, con tope de antigüedad.
  RE17 - Guarda snapshot + clip de evidencia (pre + post evento).
  RE19 - Sigue operando localmente aunque falle el envio remoto.
  RE20 - Sincroniza la cola cuando vuelve la conectividad.
  RE21 - Corre el monitor de integridad de la camara en paralelo.

Uso:
  python src/06_realtime_pipeline.py                 # deteccion con ventana
  python src/06_realtime_pipeline.py --headless      # deteccion sin ventana
  python src/06_realtime_pipeline.py --evaluate      # solo FPR + ROI y salir
  python src/06_realtime_pipeline.py --seconds 120   # corre 2 min y para

Teclas durante la ejecucion:
  q  detener    ESPACIO  pausar/reanudar
  a  confirmar la ultima alerta (RE15)
  f  marcar la ultima alerta como falso positivo (RE18)
  s  sincronizar la cola offline ahora (RE20)
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Permite ejecutar el archivo suelto (`python src/06_realtime_pipeline.py`)
# sin tener que instalar el paquete `src`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_logger
from src.database import EventDatabase
from src.alert_coordinator import AlertCoordinator
from src.model_loader import load_yolo
from src.health_monitor import HealthMonitor

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


def draw_overlay(frame, polygon, detections, human_class, text_lines):
    """Dibuja la zona, las cajas y el HUD sobre una COPIA del frame.

    Devuelve el frame anotado. Es el unico frame que entra al buffer del
    clip probatorio (RE17): un frame capturado debe corresponden a un frame
    del clip, o la evidencia queda desalineada con la hora de la alerta.
    """
    annotated = frame.copy()
    if polygon:
        cv2.polylines(annotated, [np.array(polygon, dtype="int32")], True, (0, 255, 0), 2)

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["xyxy"])
        color = (0, 0, 255) if det["class"] == human_class else (255, 200, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for i, line in enumerate(text_lines):
        y = 20 + i * 18
        cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
    return annotated


def run_evaluation(config, db, model):
    """Objetivo 4: mide FPR y ROI, escribe los informes y devuelve el codigo
    de salida. No entra en modo deteccion."""
    from src.fpr_evaluation import FPREvaluation
    from src.roi_analysis import ROIAnalysis

    evaluation_cfg = config.get("evaluation", {})
    clips = list(evaluation_cfg.get("negative_clips", []))

    if clips:
        logger.info(f"[CLI] Evaluacion FPR sobre {len(clips)} clip(s).")
        fpr = FPREvaluation(config, db=db, model=model)
        report = fpr.evaluate_dataset(
            clips,
            progress=lambda i, total, name: logger.info(f"[CLI]   ({i}/{total}) {name}"),
        )
        logger.info(f"[CLI] FPR: {fpr.format_report(report)}")
        logger.info(f"[CLI] Informe: {fpr.output_path}")
    else:
        logger.warning("[CLI] evaluation.negative_clips vacio: no se midio FPR.")

    logger.info("[CLI] Analisis ROI con los parametros economicos configurados.")
    roi = ROIAnalysis(config, db=db)
    roi_report = roi.compute()
    logger.info(f"[CLI] ROI: {roi.format_report(roi_report)}")
    logger.info(f"[CLI] Informe: {roi.output_path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pipeline de deteccion de intrusiones")
    parser.add_argument("--headless", action="store_true",
                        help="no abre ventana de video (util para pruebas)")
    parser.add_argument("--evaluate", action="store_true",
                        help="solo ejecuta FPR + ROI y termina")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="detiene el pipeline tras N segundos (0 = hasta 'q')")
    parser.add_argument("--config", default=None,
                        help="ruta alterna de configuracion YAML")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config()

    weights_path = config["model"]["weights_path"]
    if not os.path.exists(weights_path):
        logger.error(f"No se encontró el modelo entrenado en: {weights_path}")
        logger.error("Corre primero src/04_train.py para generar 'best.pt'.")
        return 1

    # Un unico punto de carga del modelo en todo el sistema: aplica la
    # allowlist de PyTorch >= 2.6. Antes este archivo parcheaba torch.load
    # con weights_only=False, lo que desactiva la proteccion contra
    # ejecucion de codigo al deserializar el checkpoint.
    model = load_yolo(weights_path, device=config["model"].get("device"))
    class_names = config["classes"]["names"]
    human_class = config["classes"]["human_class"]
    conf_threshold = config["model"]["conf_threshold"]
    device = config["model"]["device"]
    imgsz = config["model"]["imgsz"]

    db = EventDatabase(config["database"]["path"])

    if args.evaluate:
        return run_evaluation(config, db, model)

    # Todo el flujo de alerta vive en el coordinador (RE02/04/11/12-15/17/19-21).
    coordinator = AlertCoordinator(config, db)

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
    logger.info("Pipeline principal iniciado. q=salir, ESPACIO=pausa, a=confirmar, "
                "f=falso positivo, s=sincronizar.")
    if coordinator.schedule.enabled:
        logger.info(f"[RE02] Horario activo: {coordinator.schedule.describe()} | "
                    f"proxima transicion: {coordinator.schedule.next_transition()}")

    paused = False
    last = None
    started_at = time.time()
    fps_measured = 0.0
    frame_times = []
    hud = {}
    last_hud_refresh = 0.0

    try:
        while True:
            loop_start = time.time()

            if not paused:
                ok, frame = cap.read()
                if not ok:
                    # Si es un archivo de video (no una cámara en vivo), lo más
                    # probable es que simplemente llegó al final: lo reiniciamos
                    # desde el principio en vez de tratarlo como una falla de
                    # conexión (útil para revisar el sistema en loop).
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if total_frames > 0:
                        logger.info("Fin del video de prueba. Reiniciando desde el principio...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()

                    if not ok:
                        logger.error("Se perdió la señal de video. Reintentando conexión (RE19)...")
                        cap.release()
                        cap = open_capture(source, logger, config["camera"]["reconnect_wait_seconds"])
                        continue

                # RE11: calidad del frame ORIGINAL, antes de dibujar nada.
                frame, quality, quality_score, _metrics = coordinator.push_frame(frame)

                # RE21: compromiso físico de la cámara (giro, tápano, pérdida
                # de señal, FPS deprimido).
                status, detail, _compromised = coordinator.watch_integrity(frame, fps=fps_measured)

                # RE02/RE11/RE21: sin estas tres condiciones el sistema NO
                # puede afirmar con confianza lo que ve, asi que no infiere.
                allowed = coordinator.can_detect()

                detections = []
                best_human = None
                best_box = None
                if allowed:
                    results = model.predict(
                        frame, imgsz=imgsz, conf=conf_threshold, device=device, verbose=False
                    )[0]
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id >= len(class_names):
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        detections.append({
                            "xyxy": (x1, y1, x2, y2),
                            "class": class_names[cls_id],
                            "confidence": float(box.conf[0]),
                            "in_zone": point_in_zone(cx, cy, polygon),
                        })
                        if class_names[cls_id] == human_class and point_in_zone(cx, cy, polygon):
                            if best_human is None or float(box.conf[0]) > best_human:
                                best_human = float(box.conf[0])
                                best_box = (x1, y1, x2, y2)

                # RE02/RE11/RE21: FPS real para el monitor de integridad.
                frame_times.append(loop_start)
                frame_times = [t for t in frame_times if loop_start - t <= 5.0]
                fps_measured = (len(frame_times) - 1) / max(
                    frame_times[-1] - frame_times[0], 1e-6
                ) if len(frame_times) > 1 else 0.0

                now = time.time()
                if now - last_hud_refresh > 1.0:
                    last_hud_refresh = now
                    hud = coordinator.hud_status()

                lines = [
                    f"FPS {fps_measured:.1f} | calidad {quality} ({quality_score:.2f}) | "
                    f"integridad {status}",
                    f"RE02 horario: {hud.get('schedule', '-')} "
                    f"{'ACTIVO' if hud.get('schedule_active') else 'FUERA'}",
                ]
                if coordinator.suspended_reason:
                    lines.append(f"SUSPENDIDO: {coordinator.suspended_reason}")
                if coordinator.escalation_notice:
                    lines.append(coordinator.escalation_notice)
                if paused:
                    lines.append("PAUSADO (barra espaciadora para reanudar)")

                # RE17: una sola entrada al buffer, con el frame YA anotado.
                annotated = draw_overlay(frame, polygon, detections, human_class, lines)
                coordinator.push_annotated(annotated)

                # RE12/13/14/15/17/19/20: la cadena completa, con cooldown,
                # snapshot, clip, disuasión, cola offline y contactos (RE04).
                if best_human is not None:
                    bbox = ",".join(f"{v:.1f}" for v in best_box) if best_box else None
                    event_id = coordinator.register_intrusion(annotated, best_human,
                                                              bbox=bbox)
                    if event_id:
                        logger.info(f"Evento {event_id} registrado (RE12/13/14/17).")

                # RE15/19/20: reintentos de la cola y escalamiento por inacción.
                coordinator.tick()

                last = annotated

            if not args.headless and last is not None:
                cv2.imshow("Sistema de deteccion (simulado)", last)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    paused = not paused
                    logger.info("Pausado." if paused else "Reanudado.")
                elif key == ord("a"):
                    event_id, error = coordinator.acknowledge_last()
                    logger.info(f"[RE15] Confirmacion: {error or f'evento {event_id} confirmado'}")
                elif key == ord("f"):
                    # RE18: el permissionado vive en la GUI; en consola la
                    # operacion es explicita del operador.
                    event_id, error = coordinator.acknowledge_last(is_false_positive=True)
                    logger.info(f"[RE18] Falso positivo: {error or f'evento {event_id}'}")
                elif key == ord("s"):
                    coordinator.sync_pending(force=True)

            if args.seconds and (time.time() - started_at) >= args.seconds:
                logger.info(f"Tiempo de prueba cumplido ({args.seconds}s). Deteniendo.")
                break

            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario.")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        health_monitor.stop()
        coordinator.close()
        logger.info("Pipeline principal detenido.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
