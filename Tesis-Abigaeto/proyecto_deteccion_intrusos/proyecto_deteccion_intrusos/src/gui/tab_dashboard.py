"""
Vista de Dashboard: resumen del estado del sistema.
"""

import os
import cv2
import numpy as np
import psutil
import sqlite3
import time
from pathlib import Path

from src.gui.app import (
    SIDEBAR_W, WIN_W, WIN_H, HEADER_H, FOOTER_H,
    BG_CARD, BG_INPUT, BG_CARD_HI,
    TEXT_WHITE, TEXT_DIM, TEXT_MUTED,
    ACCENT, ACCENT_DIM, GREEN, RED, BLUE, CYAN, ORANGE, PURPLE,
    GREEN_DIM, RED_DIM,
    BORDER, BORDER_HI,
    rounded_rect, card, badge, progress_bar, sparkline, kv_row, section_title,
)


CLASS_COLORS = {
    "humano": RED,
    "bovino": CYAN,
    "equino": ACCENT,
    "ovino": GREEN,
    "porcino": ORANGE,
}


class DashboardView:
    def __init__(self, app):
        self.app = app
        self.cpu_hist = []
        self.ram_hist = []
        self._last_sample = 0
        self._event_total = 0

    def on_enter(self):
        self._load_events()

    def _load_events(self):
        try:
            conn = sqlite3.connect(self.app.config["database"]["path"])
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM events")
            self._event_total = cur.fetchone()[0]
            conn.close()
        except Exception:
            self._event_total = 0

    def _sample(self):
        now = time.time()
        if now - self._last_sample < 1.5:
            return
        self._last_sample = now
        self.cpu_hist.append(psutil.cpu_percent(interval=0.05))
        self.ram_hist.append(psutil.virtual_memory().percent)
        if len(self.cpu_hist) > 40:
            self.cpu_hist = self.cpu_hist[-40:]
            self.ram_hist = self.ram_hist[-40:]

    def render(self, canvas):
        self._sample()
        self._load_events()

        x0 = SIDEBAR_W + 26
        content_w = WIN_W - SIDEBAR_W - 52
        y = HEADER_H + 24

        self._render_status_cards(canvas, x0, y, content_w)
        self._render_metrics_row(canvas, x0, y + 122, content_w)
        self._render_requirements(canvas, x0, y + 252, content_w)
        self._render_classes(canvas, x0, y + 452, content_w)

    # ---- Fila 1: estado del sistema ----

    def _render_status_cards(self, canvas, x, y, w):
        weights = self.app.config["model"]["weights_path"]
        model_ok = os.path.exists(weights)
        model_name = os.path.basename(os.path.dirname(os.path.dirname(weights)))

        source = self.app.config["camera"]["source"]
        is_file = isinstance(source, str) and not source.isdigit() and os.path.exists(source)
        cam_ok = is_file or str(source).isdigit()

        try:
            conn = sqlite3.connect(self.app.config["database"]["path"])
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM events")
            events = cur.fetchone()[0]
            conn.close()
            db_ok = True
        except Exception:
            events = 0
            db_ok = False

        cards = [
            ("MODELO YOLO", model_name if model_ok else "No encontrado",
             "Operativo" if model_ok else "Ausente", GREEN if model_ok else RED, "model"),
            ("FUENTE DE VIDEO", os.path.basename(str(source))[:24],
             "Conectada" if cam_ok else "Sin senal", GREEN if cam_ok else RED, "camera"),
            ("BASE DE DATOS", "SQLite",
             f"{events} eventos" if db_ok else "Sin conexion", GREEN if db_ok else RED, "history"),
            ("CLASES", "5 clases",
             "humano, bovino, equino, ovino, porcino", ACCENT, "dataset"),
        ]

        gap = 14
        cw = (w - gap * 3) // 4
        ch = 104

        for i, (title, value, detail, color, icon) in enumerate(cards):
            cx = x + i * (cw + gap)
            rounded_rect(canvas, (cx, y), (cx + cw, y + ch), 10, BG_CARD)
            cv2.rectangle(canvas, (cx + 10, y), (cx + cw - 10, y + 3), color, -1)

            cv2.putText(canvas, title, (cx + 14, y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_MUTED, 1, cv2.LINE_AA)

            val_scale = 0.5 if len(value) > 20 else 0.6
            cv2.putText(canvas, value, (cx + 14, y + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, val_scale, TEXT_WHITE, 1, cv2.LINE_AA)

            cv2.circle(canvas, (cx + 18, y + 78), 4, color, -1)
            cv2.putText(canvas, detail, (cx + 28, y + 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_DIM, 1, cv2.LINE_AA)

    # ---- Fila 2: métricas del modelo ----

    def _render_metrics_row(self, canvas, x, y, w):
        h = 112
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)

        cv2.putText(canvas, "METRICAS DEL MODELO", (x + 16, y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MUTED, 1, cv2.LINE_AA)

        metrics = self._read_metrics()

        if metrics:
            items = [
                ("Precision", metrics["precision"], 85.0, GREEN),
                ("Recall", metrics["recall"], 70.0, CYAN),
                ("mAP50", metrics["mAP50"], 70.0, ACCENT),
                ("mAP50-95", metrics["mAP50_95"], None, PURPLE),
            ]
            gap = 16
            cw = (w - 200 - gap * 3) // 4

            for i, (label, val, target, color) in enumerate(items):
                cx = x + 16 + i * (cw + gap)
                pct = val * 100
                cv2.putText(canvas, label, (cx, y + 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_DIM, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{pct:.1f}%", (cx, y + 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
                progress_bar(canvas, cx, y + 82, cw - 10, 7, pct, color)
                if target is not None:
                    met = "OK" if pct >= target else "BAJO"
                    cv2.putText(canvas, f"meta {target:.0f}%  {met}", (cx, y + 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                                GREEN if pct >= target else RED, 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "Sin metricas disponibles", (x + 16, y + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MUTED, 1, cv2.LINE_AA)

        # Resources panel a la derecha
        rx = x + w - 180
        cpu = self.cpu_hist[-1] if self.cpu_hist else 0
        ram = self.ram_hist[-1] if self.ram_hist else 0

        cv2.line(canvas, (rx - 18, y + 16), (rx - 18, y + h - 16), BORDER, 1)

        cv2.putText(canvas, "CPU", (rx, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{cpu:.0f}%", (rx + 150, y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN if cpu < 80 else RED, 1, cv2.LINE_AA)
        sparkline(canvas, rx, y + 38, 150, 26, self.cpu_hist, GREEN)

        cv2.putText(canvas, "RAM", (rx, y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{ram:.0f}%", (rx + 150, y + 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN if ram < 80 else RED, 1, cv2.LINE_AA)
        sparkline(canvas, rx, y + 90, 150, 14, self.ram_hist, BLUE)

    def _read_metrics(self):
        import csv
        import glob
        candidates = sorted(glob.glob("runs_detect/intrusos_custom_v*/results.csv"))
        for path in reversed(candidates):
            try:
                with open(path, "r") as f:
                    rows = list(csv.DictReader(f))
                if not rows:
                    continue
                last = {k.strip(): v for k, v in rows[-1].items()}

                def get(key):
                    try:
                        return float(last.get(key, 0))
                    except (TypeError, ValueError):
                        return 0.0

                return {
                    "epochs": len(rows),
                    "precision": get("metrics/precision(B)"),
                    "recall": get("metrics/recall(B)"),
                    "mAP50": get("metrics/mAP50(B)"),
                    "mAP50_95": get("metrics/mAP50-95(B)"),
                }
            except Exception:
                continue
        return None

    # ---- Fila 3: requerimientos ----

    def _render_requirements(self, canvas, x, y, w):
        h = 182
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)

        section_title(canvas, x + 16, y + 24, "REQUERIMIENTOS DE LA TESIS", w - 40)

        reqs = [
            ("RE01", "Zona de vigilancia poligonal"),
            ("RE06", "Emulacion Edge de bajo consumo"),
            ("RE12", "Alerta local simulada"),
            ("RE13", "Notificacion remota con evidencia"),
            ("RE14", "Respuesta disuasoria simulada"),
            ("RE17", "Clip de evidencia con buffer"),
            ("RE19", "Resiliencia ante corte de Internet"),
            ("RE21", "Monitoreo de salud del sistema"),
        ]

        col_w = (w - 40) // 2
        for i, (rid, desc) in enumerate(reqs):
            col = i // 4
            row = i % 4
            cx = x + 16 + col * col_w
            cy = y + 54 + row * 32

            rounded_rect(canvas, (cx, cy - 4), (cx + 46, cy + 18), 6, GREEN_DIM)
            cv2.putText(canvas, rid, (cx + 9, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, desc, (cx + 56, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv2.LINE_AA)
            cv2.circle(canvas, (cx + col_w - 40, cy + 7), 4, GREEN, -1)

    # ---- Fila 4: clases ----

    def _render_classes(self, canvas, x, y, w):
        h = 128
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)

        section_title(canvas, x + 16, y + 24, "CLASES DETECTADAS", w - 40)

        names = self.app.config["classes"]["names"]
        counts = self._class_counts()
        max_c = max(counts.values()) if counts else 1

        gap = 16
        cw = (w - 40 - gap * 4) // 5

        for i, name in enumerate(names):
            cx = x + 16 + i * (cw + gap)
            color = CLASS_COLORS.get(name, ACCENT)
            count = counts.get(name, 0)

            rounded_rect(canvas, (cx, y + 38), (cx + cw, y + h - 14), 8, BG_INPUT)
            cv2.rectangle(canvas, (cx + 8, y + 38), (cx + cw - 8, y + 41), color, -1)

            cv2.putText(canvas, name.upper(), (cx + 12, y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{count}", (cx + 12, y + 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
            progress_bar(canvas, cx + 12, y + h - 30, cw - 24, 6,
                         (count / max_c * 100) if max_c else 0, color)

    def _class_counts(self):
        """Cuenta anotaciones por clase (cacheado: leer ~28K labels es lento)."""
        import time as _t

        now = _t.time()
        if getattr(self, "_cc_cache_time", 0) and now - self._cc_cache_time < 30:
            return self._cc_cache

        counts = {name: 0 for name in self.app.config["classes"]["names"]}
        lbl_root = Path("data/custom_yolo/labels")
        if not lbl_root.exists():
            self._cc_cache, self._cc_cache_time = counts, now
            return counts

        names = list(counts.keys())
        for f in lbl_root.rglob("*.txt"):
            try:
                for line in f.read_text(errors="ignore").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ci = int(line.split()[0])
                    except (ValueError, IndexError):
                        continue
                    if 0 <= ci < len(names):
                        counts[names[ci]] += 1
            except Exception:
                continue

        self._cc_cache = counts
        self._cc_cache_time = now
        return counts
