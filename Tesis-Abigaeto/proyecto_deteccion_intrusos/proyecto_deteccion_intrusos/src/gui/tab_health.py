"""
Vista de Salud: monitoreo de CPU, RAM y logs del sistema.
"""

import os
import cv2
import numpy as np
import psutil
import sqlite3
import time

from src.gui.app import (
    SIDEBAR_W, WIN_W, WIN_H, HEADER_H,
    BG_CARD, BG_INPUT,
    TEXT_WHITE, TEXT_DIM, TEXT_MUTED,
    ACCENT, GREEN, RED, CYAN, ORANGE, PURPLE,
    BORDER,
    rounded_rect, badge, progress_bar, section_title,
)


class HealthView:
    def __init__(self, app):
        self.app = app
        self.cpu_history = []
        self.ram_history = []
        self.gpu_history = []
        self.last_check = 0
        self.health_log = []

    def on_enter(self):
        self._load_health_log()

    def _load_health_log(self):
        try:
            conn = sqlite3.connect(self.app.config["database"]["path"])
            cur = conn.cursor()
            cur.execute("""
                SELECT timestamp, cpu_percent, ram_percent, status, note
                FROM health_log ORDER BY id DESC LIMIT 30
            """)
            self.health_log = cur.fetchall()
            conn.close()
        except Exception:
            self.health_log = []

    def _sample(self):
        now = time.time()
        if now - self.last_check < 2:
            return
        self.last_check = now

        cpu = psutil.cpu_percent(interval=0.05)
        ram = psutil.virtual_memory().percent
        self.cpu_history.append(cpu)
        self.ram_history.append(ram)

        if len(self.cpu_history) > 60:
            self.cpu_history = self.cpu_history[-60:]
            self.ram_history = self.ram_history[-60:]

    def render(self, canvas):
        self._sample()
        self._load_health_log()

        x0 = SIDEBAR_W + 26
        w = WIN_W - SIDEBAR_W - 52
        y = HEADER_H + 24

        cpu = self.cpu_history[-1] if self.cpu_history else 0
        ram = self.ram_history[-1] if self.ram_history else 0

        self._render_gauges(canvas, x0, y, w, cpu, ram)
        self._render_graphs(canvas, x0, y + 158, w)
        self._render_log(canvas, x0, y + 400, w)

    # ---- Gauges ----

    def _render_gauges(self, canvas, x, y, w, cpu, ram):
        h = 140
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        section_title(canvas, x + 16, y + 24, "ESTADO DE RECURSOS", w - 40)

        cfg = self.app.config["health_monitor"]
        cpu_th = cfg["cpu_threshold_percent"]
        ram_th = cfg["ram_threshold_percent"]

        gauges = [
            ("CPU", cpu, cpu_th, GREEN, "load"),
            ("RAM", ram, ram_th, CYAN, "mem"),
        ]

        gap = 20
        cw = (w - 40 - gap) // 2

        for i, (label, val, th, color, _kind) in enumerate(gauges):
            cx = x + 20 + i * (cw + gap)
            rounded_rect(canvas, (cx, y + 42), (cx + cw, y + h - 14), 8, BG_INPUT)

            cv2.putText(canvas, label, (cx + 16, y + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)

            ok = val < th
            vcolor = color if ok else RED
            cv2.putText(canvas, f"{val:.1f}%", (cx + 16, y + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, vcolor, 2, cv2.LINE_AA)

            badge(canvas, cx + 130, y + 88, "NORMAL" if ok else "ALERTA",
                  GREEN if ok else RED, 76)

            progress_bar(canvas, cx + 16, y + 112, cw - 32, 8, val, vcolor)
            cv2.putText(canvas, f"umbral {th}%", (cx + cw - 90, y + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)

        # Info panel
        ix = x + 20 + 2 * (cw + gap)
        if ix + cw <= x + w:
            rounded_rect(canvas, (ix, y + 42), (x + w - 20, y + h - 14), 8, BG_INPUT)
            import torch
            gpu_ok = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if gpu_ok else "Sin GPU CUDA"
            vram = 0
            if gpu_ok:
                try:
                    free, total = torch.cuda.mem_get_info()
                    vram = total / (1024 ** 3)
                except Exception:
                    vram = 0

            cv2.putText(canvas, "DISPOSITIVO", (ix + 16, y + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_MUTED, 1, cv2.LINE_AA)
            name = gpu_name.replace("NVIDIA GeForce ", "")[:22]
            cv2.putText(canvas, name, (ix + 16, y + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_WHITE, 1, cv2.LINE_AA)
            detail = f"VRAM {vram:.1f} GB" if gpu_ok else "Inferencia en CPU"
            cv2.putText(canvas, detail, (ix + 16, y + 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, GREEN if gpu_ok else RED, 1, cv2.LINE_AA)
            badge(canvas, ix + 140, y + 82, "CUDA" if gpu_ok else "CPU",
                  GREEN if gpu_ok else RED, 58)

    # ---- Gráficas ----

    def _render_graphs(self, canvas, x, y, w):
        h = 224
        gap = 18
        cw = (w - gap) // 2

        cfg = self.app.config["health_monitor"]

        graphs = [
            ("CPU (%) - ultimos 120 s", self.cpu_history, GREEN, cfg["cpu_threshold_percent"]),
            ("RAM (%) - ultimos 120 s", self.ram_history, CYAN, cfg["ram_threshold_percent"]),
        ]

        for i, (title, data, color, th) in enumerate(graphs):
            cx = x + i * (cw + gap)
            rounded_rect(canvas, (cx, y), (cx + cw, y + h), 10, BG_CARD)
            cv2.putText(canvas, title, (cx + 16, y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv2.LINE_AA)

            gx, gy = cx + 46, y + 40
            gw, gh = cw - 66, h - 76

            rounded_rect(canvas, (gx, gy), (gx + gw, gy + gh), 6, BG_INPUT)

            for pct in [0, 25, 50, 75, 100]:
                ly = gy + gh - int((pct / 100) * gh)
                cv2.line(canvas, (gx, ly), (gx + gw, ly), BORDER, 1)
                cv2.putText(canvas, str(pct), (gx - 30, ly + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_MUTED, 1, cv2.LINE_AA)

            ty = gy + gh - int((th / 100) * gh)
            cv2.line(canvas, (gx, ty), (gx + gw, ty), (70, 70, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"umbral {th}", (gx + gw - 74, ty - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, (90, 90, 210), 1, cv2.LINE_AA)

            if len(data) < 2:
                continue

            points = []
            for j, v in enumerate(data):
                px = gx + int((j / (len(data) - 1)) * gw)
                py = gy + gh - int((min(v, 100) / 100) * gh)
                points.append((px, py))

            for j in range(len(points) - 1):
                cv2.line(canvas, points[j], points[j + 1], color, 2, cv2.LINE_AA)
            if points:
                cv2.circle(canvas, points[-1], 4, color, -1)
                cv2.circle(canvas, points[-1], 4, BG_CARD, 1)

    # ---- Log ----

    def _render_log(self, canvas, x, y, w):
        h = WIN_H - y - 30 - 20
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        section_title(canvas, x + 16, y + 24, "REGISTRO DE SALUD DEL SISTEMA", w - 40)

        if not self.health_log:
            cv2.putText(canvas, "Sin registros almacenados", (x + 20, y + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, "El monitor escribe en data/events.db cada intervalo configurado",
                        (x + 20, y + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_MUTED, 1, cv2.LINE_AA)
            return

        headers = ["TIMESTAMP", "CPU", "RAM", "ESTADO", "NOTA"]
        col_x = [x + 20, x + 280, x + 360, x + 450, x + 560]
        header_y = y + 50

        for i, hdr in enumerate(headers):
            cv2.putText(canvas, hdr, (col_x[i], header_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_MUTED, 1, cv2.LINE_AA)

        cv2.line(canvas, (x + 20, header_y + 8), (x + w - 20, header_y + 8), BORDER, 1)

        row_h = 22
        max_rows = min(len(self.health_log), max(1, (h - 84) // row_h))

        for i in range(max_rows):
            ts, cpu_v, ram_v, status, note = self.health_log[i]
            ry = y + 70 + i * row_h

            if i % 2 == 0:
                rounded_rect(canvas, (x + 14, ry - 4), (x + w - 14, ry + 15), 4, BG_INPUT)

            ok = status == "OK"
            cv2.circle(canvas, (col_x[0] + 4, ry + 4), 3, GREEN if ok else RED, -1)
            cv2.putText(canvas, (ts[:19] if ts else "?"), (col_x[0] + 14, ry + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_DIM, 1, cv2.LINE_AA)

            cv2.putText(canvas, f"{cpu_v:.0f}%" if cpu_v is not None else "-", (col_x[1], ry + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        GREEN if cpu_v is not None and cpu_v < 80 else RED, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{ram_v:.0f}%" if ram_v is not None else "-", (col_x[2], ry + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        GREEN if ram_v is not None and ram_v < 80 else RED, 1, cv2.LINE_AA)

            badge(canvas, col_x[3], ry + 11, status, GREEN if ok else RED, 42)
            cv2.putText(canvas, (note or "")[:60], (col_x[4], ry + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)
