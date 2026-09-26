"""
Vista de Historial: tabla de eventos de intrusion registrados.

Cada fila muestra tambien el estado de confirmacion (RE15), si fue escalado y
si quedo marcado como falso positivo (RE18). Sin esas columnas el operador no
podia distinguir una intrusion real de un disparo ya descartado, y el historial
terminaba sirviendo solo como registro, no como herramienta de operacion.
"""

import cv2

from src.gui.app import (
    SIDEBAR_W, WIN_W, WIN_H, HEADER_H, FOOTER_H,
    BG_CARD, BG_INPUT,
    TEXT_WHITE, TEXT_DIM, TEXT_MUTED,
    ACCENT, GREEN, RED, CYAN, ORANGE,
    BORDER,
    rounded_rect, badge, section_title,
)
from src.utils import get_logger

logger = get_logger("tab_history")


CLASS_COLORS = {
    "humano": RED,
    "bovino": CYAN,
    "equino": ACCENT,
    "ovino": GREEN,
    "porcino": ORANGE,
}


class HistoryView:
    def __init__(self, app):
        self.app = app
        self.events = []
        self.selected_idx = -1
        self.filter_class = None

    def on_enter(self):
        self._load_events()

    def _load_events(self):
        """Carga el historial desde la base compartida.

        Ante un error se registra en el log. Antes la vista hacia
        `sqlite3.connect` por su cuenta y convertia cualquier fallo en "no hay
        eventos registrados", que es indistinguible de una finca tranquila: un
        fallo de esquema o de permisos se escondia como una tabla vacia.
        """
        try:
            self.events = self.app.db.get_all_events(100, class_name=self.filter_class)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"No se pudo leer el historial de eventos: {exc}")
            self.events = []

    def render(self, canvas):
        x0 = SIDEBAR_W + 26
        w = WIN_W - SIDEBAR_W - 52
        y = HEADER_H + 24

        self._render_header(canvas, x0, y, w)
        self._render_table(canvas, x0, y + 92, w)

    def _render_header(self, canvas, x, y, w):
        h = 74
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        cv2.rectangle(canvas, (x + 10, y), (x + w - 10, y + 3), RED, -1)

        total = len(self.events)
        with_snap = sum(1 for e in self.events if e[4])
        pending = sum(1 for e in self.events if not e[6])
        false_pos = sum(1 for e in self.events if e[7])

        items = [
            ("EVENTOS", str(total), TEXT_WHITE),
            ("CON EVIDENCIA", str(with_snap), CYAN),
            ("PENDIENTES (RE15)", str(pending), ORANGE),
            ("FALSOS POSITIVOS", str(false_pos), RED),
        ]

        cw = 190
        for i, (label, value, color) in enumerate(items):
            cx = x + 20 + i * cw
            cv2.putText(canvas, label, (cx, y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, value, (cx, y + 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
            if i < len(items) - 1:
                cv2.line(canvas, (cx + cw - 24, y + 16), (cx + cw - 24, y + h - 16), BORDER, 1)

        # Filtro
        names = ["TODAS"] + self.app.config["classes"]["names"]
        fx = x + w - len(names) * 76 - 20
        fy = y + 24
        for i, name in enumerate(names):
            active = (self.filter_class is None and i == 0) or self.filter_class == name
            color = RED if name == "humano" else (CYAN if name == "bovino" else ACCENT)
            rounded_rect(canvas, (fx + i * 76, fy), (fx + i * 76 + 70, fy + 26), 6,
                         color if active else BG_INPUT)
            cv2.putText(canvas, name.upper()[:7], (fx + i * 76 + 6, fy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (25, 25, 25) if active else TEXT_MUTED, 1, cv2.LINE_AA)

    def _render_table(self, canvas, x, y, w):
        h = WIN_H - y - FOOTER_H - 20
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        section_title(canvas, x + 16, y + 24, "REGISTRO DE EVENTOS", w - 40)

        headers = ["ID", "FECHA / HORA", "CLASE", "CONFIANZA", "EVIDENCIA",
                   "ESTADO", "ESCALADO", "F. POSITIVO"]
        col_x = [x + 20, x + 80, x + 300, x + 430, x + 560,
                 x + 660, x + 800, x + 920]
        header_y = y + 50

        for i, hdr in enumerate(headers):
            cv2.putText(canvas, hdr, (col_x[i], header_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_MUTED, 1, cv2.LINE_AA)

        cv2.line(canvas, (x + 20, header_y + 8), (x + w - 20, header_y + 8), BORDER, 1)

        if not self.events:
            cy = y + h // 2 - 20
            cv2.putText(canvas, "No hay eventos registrados", (x + 260, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, "Los eventos se generan al detectar un humano dentro de la zona",
                        (x + 180, cy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_MUTED, 1, cv2.LINE_AA)
            return

        row_h = 26
        max_rows = min(len(self.events), (h - 90) // row_h)

        for i in range(max_rows):
            ev = self.events[i]
            (ev_id, ts, class_name, conf, snap, clip,
             acknowledged, is_fp, escalated) = ev
            ry = y + 72 + i * row_h

            if i % 2 == 0:
                rounded_rect(canvas, (x + 14, ry - 6), (x + w - 14, ry + 18), 5, BG_INPUT)

            color = CLASS_COLORS.get(class_name, ACCENT)

            cv2.putText(canvas, str(ev_id), (col_x[0], ry + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_DIM, 1, cv2.LINE_AA)
            cv2.putText(canvas, (ts[:19] if ts else "?"), (col_x[1], ry + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)

            cv2.rectangle(canvas, (col_x[2], ry - 3), (col_x[2] + 6, ry + 11), color, -1)
            cv2.putText(canvas, class_name, (col_x[2] + 14, ry + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

            cv2.putText(canvas, f"{conf:.2f}" if conf is not None else "?", (col_x[3], ry + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_WHITE, 1, cv2.LINE_AA)

            ev_txt = "Snapshot" if snap else ("Clip" if clip else "-")
            ev_col = GREEN if (snap or clip) else TEXT_MUTED
            cv2.putText(canvas, ev_txt, (col_x[4], ry + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, ev_col, 1, cv2.LINE_AA)

            # RE15: un evento sin confirmar es el unico que exige una accion
            # del operador, asi que se destaca frente a los ya confirmados.
            if acknowledged:
                estado, estado_col = ("Confirmado" if not is_fp else "Descartado"), TEXT_MUTED
            else:
                estado, estado_col = "PENDIENTE", ORANGE
            cv2.putText(canvas, estado, (col_x[5], ry + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, estado_col, 1, cv2.LINE_AA)

            badge(canvas, col_x[6], ry + 11, "SI" if escalated else "NO",
                  RED if escalated else BORDER, 40)

            badge(canvas, col_x[7], ry + 11, "SI" if is_fp else "NO",
                  RED if is_fp else BORDER, 40)
