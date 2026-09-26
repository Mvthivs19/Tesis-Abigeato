"""
Vista de Deteccion en Vivo: video feed con bounding boxes, HUD y alertas.
Incluye definicion de zona de vigilancia integrada.
"""

import os
import sys
import cv2
import numpy as np
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.gui.app import (
    SIDEBAR_W, WIN_W, WIN_H, HEADER_H, FOOTER_H,
    BG_CARD, BG_INPUT,
    TEXT_WHITE, TEXT_DIM, TEXT_MUTED,
    ACCENT, ACCENT_HOVER, GREEN, RED, CYAN, ORANGE, PURPLE, BLUE,
    BORDER, BORDER_HI,
    rounded_rect, badge, progress_bar, section_title,
)
from src.database import EventDatabase
from src.alert_coordinator import AlertCoordinator
from src.health_monitor import HealthMonitor

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


CLASS_COLORS = {
    "humano": (70, 70, 235),
    "bovino": (200, 200, 40),
    "equino": (40, 170, 245),
    "ovino": (90, 200, 110),
    "porcino": (60, 130, 240),
}


class DetectionView:
    def __init__(self, app):
        self.app = app
        self.cap = None
        self.model = None
        self.running = False
        self.thread = None
        self.current_frame = None
        self.display_frame = None
        self.paused = False

        self.zone_mode = False
        self.zone_points = []
        self.polygon = []

        self.detections = []
        self.alerts_log = []
        self.last_alert_time = 0
        self.frame_count = 0
        self.fps_measured = 0.0
        self.inference_ms = 0.0
        self._fps_mark = time.time()
        self._frame_mark = 0

        # Cadena de alerta completa (RE12/13/14/15/17/19/20) y monitor de
        # salud (RE21). Se crean al iniciar la deteccion, no aqui, porque el
        # ClipRecorder debe vivir durante toda la sesion de video.
        self.coordinator = None
        self.health = None
        self.pending_remote = 0
        self.open_event = None
        self.escalation_notice = None
        self._ack_buttons = []

        self._load_model()
        self._load_zone()

    def _load_model(self):
        weights = self.app.config["model"]["weights_path"]
        if os.path.exists(weights) and YOLO:
            try:
                self.model = YOLO(weights)
            except Exception:
                self.model = None

    def _load_zone(self):
        self.polygon = self.app.config.get("zone", {}).get("polygon", [])

    def _scaled_polygon(self, w, h):
        """Escala el polígono configurado al tamaño real del frame.

        La zona se guarda en coordenadas absolutas, pero el video puede tener
        otra resolucion. Se normaliza usando el bbox del propio poligono, de modo
        que la zona cubra la misma proporcion relativa sin importar el tamano.
        """
        if not self.polygon:
            return np.array([], dtype=np.int32)

        pts = np.array(self.polygon, dtype=np.float32)
        min_x, min_y = pts[:, 0].min(), pts[:, 1].min()
        max_x, max_y = pts[:, 0].max(), pts[:, 1].max()
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)

        norm = pts.copy()
        norm[:, 0] = (pts[:, 0] - min_x) / span_x
        norm[:, 1] = (pts[:, 1] - min_y) / span_y

        scaled = norm * np.array([w, h], dtype=np.float32)
        return scaled.astype(np.int32)

    def _zone_in_frame(self):
        """Poligono escalado al frame actual (o None si no hay zona)."""
        if not self.polygon or self.current_frame is None:
            return None
        h, w = self.current_frame.shape[:2]
        return self._scaled_polygon(w, h)

    def on_enter(self):
        self._load_zone()
        self._start_detection()

    def _start_detection(self):
        if self.running:
            return

        # Si el hilo anterior sigue cerrando (el usuario detuvo y reinicio
        # rapido), se espera para que no compitan dos loops de deteccion
        # sobre el mismo source.
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

        source = self.app.config["camera"]["source"]
        if isinstance(source, str) and not source.isdigit() and not os.path.exists(source):
            return

        if isinstance(source, str) and source.isdigit():
            self.cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            self.cap = None
            return

        for _ in range(5):
            self.cap.read()

        self.running = True
        self.frame_count = 0
        self._frame_mark = 0
        self._fps_mark = time.time()
        self.escalation_notice = None
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()

    def _detection_loop(self):
        fps_limit = self.app.config["camera"]["fps_limit"]
        frame_interval = 1.0 / max(fps_limit, 1)
        conf_threshold = self.app.config["model"]["conf_threshold"]
        imgsz = self.app.config["model"]["imgsz"]
        device = self.app.config["model"]["device"]
        class_names = self.app.config["classes"]["names"]
        human_class = self.app.config["classes"]["human_class"]

        db = EventDatabase(self.app.config["database"]["path"])

        # Cadena de alerta RE12/13/14/15/17/19/20 y monitor de salud RE21.
        # Antes la GUI solo escribia en la BD, por lo que no disparaba
        # alertas, no guardaba clips y no drenaba la cola offline.
        self.coordinator = AlertCoordinator(self.app.config, db)
        self._start_health_monitor(db)

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            loop_start = time.time()

            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            self.frame_count += 1
            self.current_frame = frame.copy()
            display = frame.copy()

            zone_pts = self._zone_in_frame()

            intrusion = False
            if self.model:
                infer_start = time.time()
                results = self.model.predict(
                    frame, imgsz=imgsz, conf=conf_threshold,
                    device=device, verbose=False
                )[0]
                self.inference_ms = (time.time() - infer_start) * 1000

                self.detections = []
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_name = class_names[cls_id] if cls_id < len(class_names) else "?"
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    in_zone = self._point_in_zone(cx, cy, zone_pts)

                    self.detections.append({
                        "class": class_name,
                        "confidence": conf,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "center": (cx, cy),
                        "in_zone": in_zone,
                    })

                    self._draw_detection(display, class_name, conf,
                                         (int(x1), int(y1), int(x2), int(y2)),
                                         (cx, cy), in_zone, human_class)

                    if class_name == human_class and in_zone:
                        intrusion = True

            if zone_pts is not None and len(zone_pts):
                overlay = display.copy()
                cv2.fillPoly(overlay, [zone_pts], (40, 90, 20))
                cv2.addWeighted(overlay, 0.12, display, 0.88, 0, display)
                cv2.polylines(display, [zone_pts], True, (90, 220, 110), 2, cv2.LINE_AA)

            for pt in self.zone_points:
                cv2.circle(display, tuple(pt), 5, (0, 220, 255), -1)
                cv2.circle(display, tuple(pt), 5, (30, 30, 30), 1)

            self._draw_zone_label(display, zone_pts)

            h, w = display.shape[:2]
            max_w = WIN_W - SIDEBAR_W - 260
            max_h = WIN_H - HEADER_H - FOOTER_H - 90
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            display = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # RE17: el frame YA anotado alimenta el buffer circular, para que
            # el clip probatorio muestre los bounding boxes y la zona.
            self.coordinator.push_frame(display)

            # RE12/13/14/16/17: cadena de alerta ante intrusion en la zona.
            if intrusion:
                event_id = self.coordinator.register_intrusion(
                    display, max(
                        (d["confidence"] for d in self.detections
                         if d["class"] == human_class and d["in_zone"]),
                        default=0.0,
                    ),
                    human_class,
                )
                if event_id is not None:
                    self._log_alert(f"INTRUSO e{event_id} conf ver HUD")

            # RE15/19/20: reintentos de la cola offline y escalamiento.
            self.coordinator.tick()
            self.escalation_notice = self.coordinator.escalation_notice
            if self.coordinator.escalation_notice:
                self._log_alert(self.coordinator.escalation_notice)
                self.coordinator.escalation_notice = None
            self.pending_remote, self.open_event = self.coordinator.pending_summary()

            self.display_frame = display

            if time.time() - self._fps_mark >= 1.0:
                self.fps_measured = (self.frame_count - self._frame_mark) / (time.time() - self._fps_mark)
                self._frame_mark = self.frame_count
                self._fps_mark = time.time()

            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

        # Cierre ordenado de los servicios auxiliares.
        if self.coordinator:
            self.coordinator.close()
        self._stop_health_monitor()

    def _start_health_monitor(self, db):
        """RE21: vigilancia de CPU/RAM en hilo aparte, sin afectar la inferencia."""
        try:
            self.health = HealthMonitor(self.app.config, db)
            self.health.start()
        except Exception:
            self.health = None

    def _stop_health_monitor(self):
        if self.health:
            try:
                self.health.stop()
                self.health.join(timeout=2)
            except Exception:
                pass
            self.health = None

    def _log_alert(self, msg):
        self.alerts_log.append((time.strftime("%H:%M:%S"), msg))
        if len(self.alerts_log) > 200:
            del self.alerts_log[:-200]

    def _draw_detection(self, frame, class_name, conf, bbox, center, in_zone, human_class):
        x1, y1, x2, y2 = bbox
        cx, cy = center
        color = CLASS_COLORS.get(class_name, (200, 200, 60))
        is_human = class_name == human_class

        thick = 3 if is_human else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        if is_human:
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (30, 30, 30), 1)

        label = f"{class_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx, ly = x1, max(y1 - 8, th + 6)
        cv2.rectangle(frame, (lx, ly - th - 6), (lx + tw + 12, ly + 4), color, -1)
        cv2.putText(frame, label, (lx + 6, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)

        if in_zone:
            cv2.circle(frame, (int(cx), int(cy)), 7, (0, 230, 255), 2)
            cv2.drawMarker(frame, (int(cx), int(cy)), (0, 230, 255),
                           cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)

    def _acknowledge_open_event(self, is_false_positive=False):
        """RE15/RE18: el operador confirma el evento abierto desde el panel de
        alertas. Al confirmar se detiene el escalamiento."""
        if not self.coordinator:
            return
        event_id = self.coordinator.acknowledge_last(is_false_positive)
        if event_id is None:
            return
        self.escalation_notice = None
        msg = "confirmado" if not is_false_positive else "marcado FALSO POSITIVO"
        self._log_alert(f"Evento e{event_id} {msg}")
        self.pending_remote, self.open_event = self.coordinator.pending_summary()

    def _draw_zone_label(self, frame, zone_pts):
        if zone_pts is None or not len(zone_pts):
            return
        x, y = zone_pts[:, 0].min(), zone_pts[:, 1].min()
        cv2.rectangle(frame, (x, y - 22), (x + 150, y - 4), (60, 180, 90), -1)
        cv2.putText(frame, "ZONA DE VIGILANCIA", (x + 6, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)

    def _point_in_zone(self, cx, cy, zone_pts):
        if zone_pts is None or not len(zone_pts):
            return True
        return cv2.pointPolygonTest(zone_pts, (cx, cy), False) >= 0

    # ---------- Render HUD ----------

    def render(self, canvas):
        x0 = SIDEBAR_W + 26
        y0 = HEADER_H + 22

        self._render_buttons(canvas, x0, y0)

        video_y = y0 + 56

        if self.display_frame is not None:
            fh, fw = self.display_frame.shape[:2]
            canvas[video_y:video_y + fh, x0:x0 + fw] = self.display_frame
            self._draw_video_overlay(canvas, x0, video_y, fw, fh)
            self._render_side_panel(canvas, x0 + fw + 18, video_y, fw)
        else:
            h = WIN_H - HEADER_H - FOOTER_H - 100
            rounded_rect(canvas, (x0, video_y), (x0 + 700, video_y + h), 10, BG_CARD)
            cv2.putText(canvas, "Sin senal de video", (x0 + 280, video_y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, "Presiona 'Iniciar' para comenzar la deteccion",
                        (x0 + 230, video_y + h // 2 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_MUTED, 1, cv2.LINE_AA)

    def _render_buttons(self, canvas, x0, y0):
        buttons = []
        bx = x0

        if not self.zone_mode:
            if self.running:
                buttons.append({
                    "label": "Reanudar" if self.paused else "Pausar",
                    "rect": (bx, y0, 100, 38),
                    "color": GREEN if self.paused else ACCENT,
                    "action": self._toggle_pause,
                })
                bx += 110
                buttons.append({
                    "label": "Detener",
                    "rect": (bx, y0, 90, 38),
                    "color": RED,
                    "action": self._stop_detection,
                })
                bx += 100
            else:
                buttons.append({
                    "label": "Iniciar",
                    "rect": (bx, y0, 100, 38),
                    "color": GREEN,
                    "action": self._start_detection,
                })
                bx += 110

            buttons.append({
                "label": "Definir zona",
                "rect": (bx, y0, 120, 38),
                "color": ACCENT,
                "action": self._enter_zone_mode,
            })
        else:
            cv2.putText(canvas, "MODO ZONA: click para agregar puntos | g=guardar  r=reiniciar  x=cancelar",
                        (x0, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ACCENT, 1, cv2.LINE_AA)

        self.app.set_custom_buttons(buttons)

    def _draw_video_overlay(self, canvas, x, y, w, h):
        """Barra de estado sobre el video."""
        # Chip superior izquierdo: estado
        if not self.running:
            chip_txt, chip_col = "DETENIDO", RED
        elif self.paused:
            chip_txt, chip_col = "PAUSADO", ACCENT
        else:
            chip_txt, chip_col = "EN VIVO", GREEN

        cv2.rectangle(canvas, (x, y), (x + w, y + 26), (18, 18, 18), -1)
        badge(canvas, x + 12, y + 18, chip_txt, chip_col, 78)

        info = f"FPS {self.fps_measured:.1f}   INF {self.inference_ms:.0f} ms   FRAME {self.frame_count}"
        tw = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
        cv2.putText(canvas, info, (x + w - tw - 12, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_WHITE, 1, cv2.LINE_AA)

        # Contador de detecciones
        if self.detections:
            humans = sum(1 for d in self.detections if d["class"] == "humano")
            txt = f"{len(self.detections)} objetos"
            tw2 = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
            cv2.rectangle(canvas, (x + 100, y + 4), (x + 112 + tw2, y + 26), (18, 18, 18), -1)
            cv2.putText(canvas, txt, (x + 106, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, CYAN, 1, cv2.LINE_AA)
            if humans:
                ht = f"{humans} intruso(s)"
                hw = cv2.getTextSize(ht, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
                hx = x + 120 + tw2
                cv2.rectangle(canvas, (hx, y + 4), (hx + 12 + hw, y + 26), (18, 18, 18), -1)
                cv2.putText(canvas, ht, (hx + 6, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, RED, 1, cv2.LINE_AA)

        self._draw_escalation_banner(canvas, x, y, w, h)

        # Marco
        cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_HI, 1)

    def _draw_escalation_banner(self, canvas, x, y, w, h):
        """RE15: banda de escalamiento mientras haya un evento sin confirmar."""
        if not self.open_event:
            return
        event_id, class_name, confidence = self.open_event
        escalate_after = self.app.config["alerts"].get("escalate_after_seconds", 60)

        bh = 30
        by = y + 34
        cv2.rectangle(canvas, (x + 8, by), (x + 8 + 320, by + bh), (35, 35, 120), -1)
        cv2.rectangle(canvas, (x + 8, by), (x + 8 + 320, by + bh), ORANGE, 1)

        txt = f"ALERTA ACTIVA e{event_id}  {class_name} {confidence:.0%}"
        cv2.putText(canvas, txt, (x + 18, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_WHITE, 1, cv2.LINE_AA)

        sub = f"escalara en {escalate_after}s si no confirmas"
        cv2.putText(canvas, sub, (x + 18, by + bh + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, ORANGE, 1, cv2.LINE_AA)

        # RE19/RE20: notificaciones esperando reconexion.
        if self.pending_remote:
            q = f"offline: {self.pending_remote} en cola"
            cv2.putText(canvas, q, (x + w - cv2.getTextSize(
                q, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0][0] - 12, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, ORANGE, 1, cv2.LINE_AA)

    def _render_side_panel(self, canvas, x, y, h):
        """Panel lateral: detecciones + alertas."""
        panel_w = WIN_W - x - 26
        if panel_w < 180:
            return

        # Detecciones
        ph = 200
        rounded_rect(canvas, (x, y), (x + panel_w, y + ph), 10, BG_CARD)
        section_title(canvas, x + 12, y + 22, "DETECCIONES", panel_w - 30)

        class_counts = {}
        for d in self.detections:
            cls = d["class"]
            if cls not in class_counts:
                class_counts[cls] = {"n": 0, "conf": 0.0, "zone": 0}
            class_counts[cls]["n"] += 1
            class_counts[cls]["conf"] = max(class_counts[cls]["conf"], d["confidence"])
            if d["in_zone"]:
                class_counts[cls]["zone"] += 1

        if not class_counts:
            cv2.putText(canvas, "Sin detecciones", (x + 14, y + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_MUTED, 1, cv2.LINE_AA)
        else:
            for i, (cls, info) in enumerate(list(class_counts.items())[:7]):
                ry = y + 46 + i * 21
                color = CLASS_COLORS.get(cls, ACCENT)
                cv2.rectangle(canvas, (x + 14, ry - 8), (x + 20, ry + 6), color, -1)
                cv2.putText(canvas, cls, (x + 28, ry + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_WHITE, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"x{info['n']}", (x + 130, ry + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{info['conf']:.0%}", (x + 165, ry + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
                if info["zone"]:
                    cv2.putText(canvas, "ZONA", (x + 200, ry + 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 220, 255), 1, cv2.LINE_AA)

        # Alertas
        ay = y + ph + 16
        ah = WIN_H - ay - FOOTER_H - 20
        rounded_rect(canvas, (x, ay), (x + panel_w, ay + ah), 10, BG_CARD)
        section_title(canvas, x + 12, ay + 22, "ALERTAS", panel_w - 30)

        # RE15/RE18: acciones de confirmacion sobre el evento abierto.
        list_y = ay + 46
        if self.open_event:
            self._ack_buttons = [
                {"rect": (x + 12, list_y, 0, 0, panel_w - 24, 26), "action": "ack"},
                {"rect": (x + 12, list_y + 32, 0, 0, panel_w - 24, 26), "action": "fp"},
            ]
            bw = panel_w - 24
            self._ack_buttons[0]["rect"] = (x + 12, list_y, x + 12 + bw, list_y + 26)
            self._ack_buttons[1]["rect"] = (x + 12, list_y + 32, x + 12 + bw, list_y + 58)

            rounded_rect(canvas, (x + 12, list_y), (x + 12 + bw, list_y + 26), 5, GREEN)
            cv2.putText(canvas, "Confirmar alerta", (x + 12 + bw // 2 - 52, list_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 20, 20), 1, cv2.LINE_AA)

            rounded_rect(canvas, (x + 12, list_y + 32), (x + 12 + bw, list_y + 58), 5, BG_INPUT)
            cv2.putText(canvas, "Marcar falso positivo", (x + 12 + bw // 2 - 72, list_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_DIM, 1, cv2.LINE_AA)

            list_y += 76
        else:
            self._ack_buttons = []

        recent = self.alerts_log[-8:]
        if not recent:
            cv2.putText(canvas, "Sin alertas registradas", (x + 14, list_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, "Se genera al detectar humano", (x + 14, list_y + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)
        else:
            for i, (ts, msg) in enumerate(reversed(recent)):
                ry = list_y + 12 + i * 26
                if ry > ay + ah - 14:
                    break
                rounded_rect(canvas, (x + 10, ry - 13), (x + panel_w - 10, ry + 11), 5, BG_INPUT)
                cv2.circle(canvas, (x + 20, ry - 1), 3, RED, -1)
                cv2.putText(canvas, ts, (x + 30, ry + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)
                cv2.putText(canvas, msg[:26], (x + 78, ry + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, RED, 1, cv2.LINE_AA)

    # ---------- Controles ----------

    def _toggle_pause(self):
        self.paused = not self.paused

    def _stop_detection(self):
        self.running = False
        self.paused = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.display_frame = None
        self.current_frame = None

    def _enter_zone_mode(self):
        self.zone_mode = True
        self.zone_points = []
        self.app.set_custom_buttons([{
            "label": "Guardar zona",
            "rect": (SIDEBAR_W + 26, WIN_H - 60, 120, 38),
            "color": GREEN,
            "action": self._save_zone,
        }, {
            "label": "Reiniciar",
            "rect": (SIDEBAR_W + 156, WIN_H - 60, 110, 38),
            "color": ACCENT,
            "action": self._reset_zone,
        }, {
            "label": "Cancelar",
            "rect": (SIDEBAR_W + 276, WIN_H - 60, 100, 38),
            "color": RED,
            "action": self._cancel_zone,
        }])

    def _save_zone(self):
        if len(self.zone_points) >= 3:
            self.polygon = list(self.zone_points)
            self.app.config["zone"]["polygon"] = self.polygon
            from src.utils import save_config
            save_config(self.app.config)
        self.zone_mode = False
        self.zone_points = []
        self.app.set_custom_buttons([])

    def _reset_zone(self):
        self.zone_points = []

    def _cancel_zone(self):
        self.zone_mode = False
        self.zone_points = []
        self.app.set_custom_buttons([])

    def on_click(self, mx, my):
        # Botones de confirmacion del panel de alertas (RE15/RE18).
        if not self.zone_mode:
            for btn in self._ack_buttons:
                x1, y1, x2, y2 = btn["rect"]
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    self._acknowledge_open_event(btn["action"] == "fp")
                    return
            return

        if self.current_frame is None:
            return

        fx = SIDEBAR_W + 26
        fy = HEADER_H + 22 + 56
        fh, fw = self.current_frame.shape[:2]

        if self.display_frame is not None:
            dh, dw = self.display_frame.shape[:2]
        else:
            return

        if not (fx <= mx <= fx + dw and fy <= my <= fy + dh):
            return

        rel_x = (mx - fx) / dw
        rel_y = (my - fy) / dh
        self.zone_points.append([int(rel_x * fw), int(rel_y * fh)])

    def on_key(self, key):
        if self.zone_mode:
            if key == ord("g"):
                self._save_zone()
            elif key == ord("r"):
                self._reset_zone()
            elif key == ord("x"):
                self._cancel_zone()
        elif key == ord(" "):
            self._toggle_pause()
        elif key == ord("a"):
            # RE15: confirmar atajo.
            self._acknowledge_open_event(False)
        elif key == ord("f"):
            # RE18: marcar falso positivo atajo.
            self._acknowledge_open_event(True)

    def cleanup(self):
        self._stop_detection()
        if self.coordinator:
            self.coordinator.close()
        self._stop_health_monitor()
