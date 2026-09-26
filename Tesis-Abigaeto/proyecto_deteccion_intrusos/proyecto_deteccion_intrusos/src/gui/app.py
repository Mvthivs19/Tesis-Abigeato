"""
Ventana principal de la aplicación con navegación por pestañas.

Interfaz profesional para presentación de tesis: tema oscuro moderno,
navegación lateral con iconos, header con estado y footer informativo.
"""

import cv2
import numpy as np
import time
import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils import load_config
from src.database import EventDatabase


WIN_W, WIN_H = 1360, 760
SIDEBAR_W = 230
HEADER_H = 62
FOOTER_H = 30

# --- Paleta de colores (BGR) ---
BG_ROOT = (22, 20, 18)
BG_SIDEBAR = (30, 27, 24)
BG_HEADER = (34, 31, 28)
BG_CONTENT = (26, 24, 22)
BG_CARD = (40, 37, 34)
BG_CARD_HI = (48, 45, 41)
BG_INPUT = (34, 32, 30)
BG_ELEVATED = (52, 48, 44)

ACCENT = (196, 137, 38)
ACCENT_HOVER = (222, 165, 60)
ACCENT_DIM = (110, 78, 24)

TEXT_WHITE = (238, 236, 233)
TEXT_DIM = (150, 146, 140)
TEXT_MUTED = (105, 102, 98)

GREEN = (96, 190, 104)
GREEN_DIM = (52, 96, 58)
RED = (86, 86, 216)
RED_DIM = (62, 52, 130)
BLUE = (196, 150, 70)
CYAN = (188, 168, 60)
ORANGE = (72, 140, 235)
PURPLE = (170, 96, 190)

BORDER = (62, 58, 54)
BORDER_HI = (88, 82, 76)


# --- Utilidades de dibujo ---

def rounded_rect(canvas, pt1, pt2, radius, color, thickness=-1, border_color=None, border_thick=1):
    """Rectángulo con esquinas redondeadas."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if r == 0:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        return
    cv2.rectangle(canvas, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(canvas, (x1, y1 + r), (x2, y2 - r), color, thickness)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(canvas, (cx, cy), r, color, thickness)
    if border_color is not None:
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.circle(canvas, (cx, cy), r, border_color, border_thick)
        cv2.line(canvas, (x1 + r, y1), (x2 - r, y1), border_color, border_thick)
        cv2.line(canvas, (x1 + r, y2), (x2 - r, y2), border_color, border_thick)
        cv2.line(canvas, (x1, y1 + r), (x1, y2 - r), border_color, border_thick)
        cv2.line(canvas, (x2, y1 + r), (x2, y2 - r), border_color, border_thick)


def card(canvas, x, y, w, h, title=None, accent_color=ACCENT):
    """Dibuja una tarjeta con borde superior de acento y título opcional."""
    rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
    cv2.rectangle(canvas, (x + 10, y), (x + w - 10, y + 3), accent_color, -1)
    if title:
        cv2.putText(canvas, title, (x + 14, y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, TEXT_DIM, 1, cv2.LINE_AA)


def badge(canvas, x, y, text, color, w=None):
    """Etiqueta tipo píldora con fondo de color."""
    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
    width = w if w else tw + 20
    rounded_rect(canvas, (x, y - 11), (x + width, y + 6), 8, color)
    cv2.putText(canvas, text, (x + (width - tw) // 2, y + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, BG_ROOT, 1, cv2.LINE_AA)
    return width


def progress_bar(canvas, x, y, w, h, pct, color, bg=BG_INPUT):
    """Barra de progreso horizontal."""
    rounded_rect(canvas, (x, y), (x + w, y + h), h // 2, bg)
    fill = int(w * max(0.0, min(1.0, pct / 100.0)))
    if fill > 0:
        rounded_rect(canvas, (x, y), (x + fill, y + h), h // 2, color)


def sparkline(canvas, x, y, w, h, values, color, max_val=100.0):
    """Mini gráfico de línea."""
    if not values:
        return
    rounded_rect(canvas, (x, y), (x + w, y + h), 8, BG_INPUT)
    n = len(values)
    if n < 2:
        return
    points = []
    for i, v in enumerate(values):
        px = x + int((i / (n - 1)) * (w - 8)) + 4
        py = y + h - 4 - int((min(v, max_val) / max_val) * (h - 12))
        points.append((px, py))
    for i in range(len(points) - 1):
        cv2.line(canvas, points[i], points[i + 1], color, 2, cv2.LINE_AA)
    cv2.circle(canvas, points[-1], 3, color, -1)


def kv_row(canvas, x, y, label, value, label_w=170, value_color=TEXT_WHITE, scale=0.42):
    """Fila etiqueta: valor."""
    cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, str(value), (x + label_w, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, value_color, 1, cv2.LINE_AA)


def section_title(canvas, x, y, text, w=180):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_WHITE, 1, cv2.LINE_AA)
    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    cv2.line(canvas, (x + tw + 12, y - 4), (x + w, y - 4), BORDER, 1)


# --- Iconos vectoriales para el sidebar ---

def draw_icon(canvas, kind, cx, cy, color, scale=1.0):
    """Dibuja iconos simples con primitivas de OpenCV."""
    s = int(10 * scale)
    if kind == "dashboard":
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            x1 = cx + (dx * s) - (s if dx < 0 else 0)
            y1 = cy + (dy * s) - (s if dy < 0 else 0)
            cv2.rectangle(canvas, (x1, y1), (x1 + s - 2, y1 + s - 2), color, -1)
    elif kind == "dataset":
        cv2.ellipse(canvas, (cx, cy - s // 2), (s, s // 3), 0, 0, 360, color, 2)
        cv2.line(canvas, (cx - s, cy - s // 2), (cx - s, cy + s), color, 2)
        cv2.line(canvas, (cx + s, cy - s // 2), (cx + s, cy + s), color, 2)
        cv2.ellipse(canvas, (cx, cy + s), (s, s // 3), 0, 0, 180, color, 2)
    elif kind == "training":
        cv2.circle(canvas, (cx, cy), s, color, 2)
        cv2.circle(canvas, (cx, cy), s // 3, color, -1)
        for ang in [0, 45, 90, 135, 180, 225, 270, 315]:
            import math
            r1 = s + 3
            r2 = s + 8
            a = math.radians(ang)
            cv2.line(canvas,
                     (int(cx + r1 * math.cos(a)), int(cy + r1 * math.sin(a))),
                     (int(cx + r2 * math.cos(a)), int(cy + r2 * math.sin(a))),
                     color, 2)
    elif kind == "detection":
        cv2.rectangle(canvas, (cx - s, cy - s), (cx + s, cy + s), color, 2)
        cv2.line(canvas, (cx - s + 3, cy), (cx + s - 3, cy), color, 1)
        cv2.line(canvas, (cx, cy - s + 3), (cx, cy + s - 3), color, 1)
    elif kind == "history":
        cv2.circle(canvas, (cx, cy), s, color, 2)
        cv2.line(canvas, (cx, cy), (cx, cy - s // 2), color, 2)
        cv2.line(canvas, (cx, cy), (cx + s // 2, cy + s // 3), color, 2)
    elif kind == "health":
        cv2.line(canvas, (cx - s, cy), (cx - s // 3, cy + s // 2), color, 3, cv2.LINE_AA)
        cv2.line(canvas, (cx - s // 3, cy + s // 2), (cx + s // 3, cy - s // 2), color, 3, cv2.LINE_AA)
        cv2.line(canvas, (cx + s // 3, cy - s // 2), (cx + s, cy), color, 3, cv2.LINE_AA)
    elif kind == "alert":
        cv2.circle(canvas, (cx, cy), s, color, 2)
        cv2.line(canvas, (cx, cy - s // 2), (cx, cy + s // 4), color, 3, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy + s // 2 - 1), 2, color, -1)
    elif kind == "model":
        cv2.rectangle(canvas, (cx - s, cy - s // 2), (cx + s, cy + s // 2), color, 2)
        cv2.line(canvas, (cx - s + 4, cy + s // 2), (cx - s + 8, cy + s), color, 2)
        cv2.line(canvas, (cx + s - 8, cy + s // 2), (cx + s - 4, cy + s), color, 2)
    elif kind == "camera":
        cv2.rectangle(canvas, (cx - s, cy - s // 3), (cx + s // 2, cy + s // 3), color, 2)
        import math
        a = math.radians(30)
        tipx = int(cx + s + (s // 2) * math.cos(a))
        tipy = int(cy - (s // 2) * math.sin(a) - s // 2)
        cv2.line(canvas, (cx + s // 2, cy), (tipx, tipy), color, 2)
        cv2.line(canvas, (cx + s // 2, cy), (tipx, tipy + s), color, 2)


class Application:
    def __init__(self):
        self.win_name = "Sistema de Deteccion de Intrusos vs. Ganado"
        self.running = True
        self.current_view = "dashboard"

        self.config = load_config()
        self.db = EventDatabase(self.config["database"]["path"])

        self.mouse_x, self.mouse_y = 0, 0
        self.mouse_clicked = False

        self.views = {}
        self.buttons = []
        self.custom_buttons = []
        self._last_status = {}

        self._init_views()
        self._init_sidebar()

        cv2.namedWindow(self.win_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win_name, self._mouse_cb)

    def _mouse_cb(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_clicked = True

    def _init_sidebar(self):
        nav_items = [
            ("dashboard", "Dashboard", "dashboard"),
            ("dataset", "Dataset", "dataset"),
            ("training", "Entrenamiento", "training"),
            ("detection", "Deteccion", "detection"),
            ("history", "Historial", "history"),
            ("health", "Salud", "health"),
        ]
        self.buttons = []
        y_start = HEADER_H + 34
        for i, (key, label, icon) in enumerate(nav_items):
            by = y_start + i * 46
            self.buttons.append({
                "key": key,
                "label": label,
                "icon": icon,
                "rect": (14, by, SIDEBAR_W - 28, 38),
            })

    def _init_views(self):
        from src.gui.tab_dashboard import DashboardView
        from src.gui.tab_dataset import DatasetView
        from src.gui.tab_training import TrainingView
        from src.gui.tab_detection import DetectionView
        from src.gui.tab_history import HistoryView
        from src.gui.tab_health import HealthView

        self.views = {
            "dashboard": DashboardView(self),
            "dataset": DatasetView(self),
            "training": TrainingView(self),
            "detection": DetectionView(self),
            "history": HistoryView(self),
            "health": HealthView(self),
        }

    # ---------- Dibujado de la estructura ----------

    def draw_sidebar(self, canvas):
        cv2.rectangle(canvas, (0, 0), (SIDEBAR_W, WIN_H), BG_SIDEBAR, -1)
        cv2.line(canvas, (SIDEBAR_W, 0), (SIDEBAR_W, WIN_H), BORDER, 1)

        # Logo
        logo_cx, logo_cy = 34, HEADER_H // 2 + 6
        cv2.circle(canvas, (logo_cx, logo_cy), 18, ACCENT_DIM, -1)
        draw_icon(canvas, "detection", logo_cx, logo_cy, ACCENT, 0.85)

        cv2.putText(canvas, "DETECTRON", (62, logo_cy + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_WHITE, 2, cv2.LINE_AA)
        cv2.putText(canvas, "v2.0  |  Tesis 2026", (62, logo_cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_MUTED, 1, cv2.LINE_AA)

        # Separador
        cv2.line(canvas, (14, HEADER_H + 16), (SIDEBAR_W - 14, HEADER_H + 16), BORDER, 1)
        cv2.putText(canvas, "MODULOS", (20, HEADER_H + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)

        for btn in self.buttons:
            x, y, w, h = btn["rect"]
            is_active = btn["key"] == self.current_view
            is_hover = (x <= self.mouse_x <= x + w and y <= self.mouse_y <= y + h)

            if is_active:
                rounded_rect(canvas, (x, y), (x + w, y + h), 8, BG_CARD_HI)
                cv2.rectangle(canvas, (x, y + 6), (x + 3, y + h - 6), ACCENT, -1)
                icon_color, text_color = ACCENT, TEXT_WHITE
            elif is_hover:
                rounded_rect(canvas, (x, y), (x + w, y + h), 8, BG_CARD)
                icon_color, text_color = TEXT_WHITE, TEXT_WHITE
            else:
                icon_color, text_color = TEXT_MUTED, TEXT_DIM

            draw_icon(canvas, btn["icon"], x + 22, y + h // 2, icon_color, 0.75)
            cv2.putText(canvas, btn["label"], (x + 42, y + h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_color, 1, cv2.LINE_AA)

        # Indicador de GPU al pie del sidebar
        self._draw_sidebar_footer(canvas)

    def _draw_sidebar_footer(self, canvas):
        y = WIN_H - 92
        cv2.line(canvas, (14, y - 14), (SIDEBAR_W - 14, y - 14), BORDER, 1)

        import torch
        gpu_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0).replace("NVIDIA ", "")[:16] if gpu_ok else "CPU"

        dot_color = GREEN if gpu_ok else RED
        cv2.circle(canvas, (24, y + 2), 4, dot_color, -1)
        cv2.putText(canvas, gpu_name, (34, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Dispositivo de inferencia", (24, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_MUTED, 1, cv2.LINE_AA)

    def draw_header(self, canvas):
        cv2.rectangle(canvas, (0, 0), (WIN_W, HEADER_H), BG_HEADER, -1)
        cv2.line(canvas, (0, HEADER_H), (WIN_W, HEADER_H), BORDER, 1)

        titles = {
            "dashboard": ("Dashboard", "Resumen del estado del sistema"),
            "dataset": ("Dataset", "Estadisticas del conjunto de datos"),
            "training": ("Entrenamiento", "Configuracion y metricas del modelo"),
            "detection": ("Deteccion en vivo", "Inferencia YOLOv8s en tiempo real"),
            "history": ("Historial", "Eventos de intrusion registrados"),
            "health": ("Salud del sistema", "Monitoreo de CPU, RAM y recursos"),
        }
        title, subtitle = titles.get(self.current_view, ("", ""))

        cv2.putText(canvas, title, (SIDEBAR_W + 26, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, TEXT_WHITE, 2, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (SIDEBAR_W + 26, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_MUTED, 1, cv2.LINE_AA)

        # Estado a la derecha
        rx = WIN_W - 26
        now = datetime.datetime.now()
        clock_text = now.strftime("%H:%M:%S")
        date_text = now.strftime("%d/%m/%Y")

        tw = cv2.getTextSize(clock_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        cv2.putText(canvas, clock_text, (rx - tw, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_WHITE, 2, cv2.LINE_AA)
        dw = cv2.getTextSize(date_text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
        cv2.putText(canvas, date_text, (rx - dw, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_DIM, 1, cv2.LINE_AA)

        # Estado de detections
        view = self.views.get(self.current_view)
        det_count = len(getattr(view, "detections", []) or [])
        alert_count = len(getattr(view, "alerts_log", []) or [])

        cx = rx - max(tw, dw) - 30
        if alert_count:
            badge(canvas, cx - 88, 26, f"{alert_count} alertas", RED)
        if self.current_view == "detection":
            badge(canvas, cx - 88, 46, f"{det_count} objetos", CYAN)

    def draw_footer(self, canvas):
        y = WIN_H - FOOTER_H
        cv2.rectangle(canvas, (0, y), (WIN_W, WIN_H), BG_HEADER, -1)
        cv2.line(canvas, (0, y), (WIN_W, y), BORDER, 1)

        view = self.views.get(self.current_view)
        model_path = self.config["model"]["weights_path"]
        model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))

        status_parts = [
            f"Modelo: {model_name}",
            f"Resolucion: {self.config['model']['imgsz']}px",
            f"Confianza: {self.config['model']['conf_threshold']}",
        ]

        tx = SIDEBAR_W + 26
        for part in status_parts:
            cv2.putText(canvas, part, (tx, WIN_H - 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_MUTED, 1, cv2.LINE_AA)
            tw = cv2.getTextSize(part, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0][0]
            tx += tw + 20

        hint = "Q: salir  |  H: dashboard"
        hw = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0][0]
        cv2.putText(canvas, hint, (WIN_W - 26 - hw, WIN_H - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_MUTED, 1, cv2.LINE_AA)

    def draw_custom_buttons(self, canvas):
        for btn in self.custom_buttons:
            x, y, w, h = btn["rect"]
            is_hover = (x <= self.mouse_x <= x + w and y <= self.mouse_y <= y + h)
            base = btn.get("color", ACCENT)
            color = btn.get("hover_color", ACCENT_HOVER) if is_hover else base
            rounded_rect(canvas, (x, y), (x + w, y + h), 7, color)

            label = btn["label"]
            tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
            text_color = btn.get("text_color", BG_ROOT)
            cv2.putText(canvas, label, (x + (w - tw) // 2, y + h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv2.LINE_AA)

    def set_custom_buttons(self, buttons):
        self.custom_buttons = buttons

    # ---------- Bucle principal ----------

    def run(self):
        last_frame_time = time.time()
        fps_limit = 20
        frame_interval = 1.0 / fps_limit

        while self.running:
            now = time.time()
            elapsed = now - last_frame_time

            if elapsed < frame_interval:
                cv2.waitKey(1)
                continue

            last_frame_time = now

            canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
            canvas[:] = BG_CONTENT

            self.draw_sidebar(canvas)
            self.draw_header(canvas)

            view = self.views.get(self.current_view)
            if view:
                view.render(canvas)

            self.draw_custom_buttons(canvas)
            self.draw_footer(canvas)

            if self.mouse_clicked:
                self.mouse_clicked = False
                self._handle_click()

            cv2.imshow(self.win_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.stop()
            elif key == ord("h"):
                self.switch_view("dashboard")
            elif view and hasattr(view, "on_key"):
                view.on_key(key)

        self.stop()

    def _handle_click(self):
        mx, my = self.mouse_x, self.mouse_y

        for btn in self.buttons:
            x, y, w, h = btn["rect"]
            if x <= mx <= x + w and y <= my <= y + h:
                self.switch_view(btn["key"])
                return

        for btn in self.custom_buttons:
            x, y, w, h = btn["rect"]
            if x <= mx <= x + w and y <= my <= y + h:
                if "action" in btn:
                    btn["action"]()
                return

        view = self.views.get(self.current_view)
        if view and hasattr(view, "on_click"):
            view.on_click(mx, my)

    def switch_view(self, key):
        if key in self.views:
            self.current_view = key
            self.custom_buttons = []
            view = self.views[key]
            if hasattr(view, "on_enter"):
                view.on_enter()

    def stop(self):
        for view in self.views.values():
            if hasattr(view, "cleanup"):
                view.cleanup()
        self.running = False
        cv2.destroyAllWindows()
