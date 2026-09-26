"""
Vista de Entrenamiento: configuracion, ejecucion y metricas del modelo.
"""

import os
import sys
import csv
import glob
import cv2
import numpy as np
import threading
import subprocess

from src.gui.app import (
    SIDEBAR_W, WIN_W, WIN_H, HEADER_H,
    BG_CARD, BG_INPUT,
    TEXT_WHITE, TEXT_DIM, TEXT_MUTED,
    ACCENT, ACCENT_HOVER, GREEN, RED, CYAN, ORANGE, PURPLE,
    GREEN_DIM, RED_DIM,
    BORDER,
    rounded_rect, badge, progress_bar, section_title, sparkline,
)


class TrainingView:
    def __init__(self, app):
        self.app = app
        self.training = False
        self.training_thread = None
        self.training_log = []
        self.metrics = {}
        self.model_version = ""
        self._curves = []

    def on_enter(self):
        self._load_metrics()

    # ---- Carga de metricas ----

    def _load_metrics(self):
        self.metrics = {}
        self.model_version = ""
        self._curves = []

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

                self.metrics = {
                    "epochs": len(rows),
                    "precision": get("metrics/precision(B)"),
                    "recall": get("metrics/recall(B)"),
                    "mAP50": get("metrics/mAP50(B)"),
                    "mAP50_95": get("metrics/mAP50-95(B)"),
                }
                self.model_version = os.path.basename(os.path.dirname(path))
                self._curves = self._read_curves(path)
                break
            except Exception:
                continue

    def _read_curves(self, path):
        out = {"precision": [], "recall": [], "mAP50": [], "map": []}
        try:
            with open(path, "r") as f:
                for row in csv.DictReader(f):
                    row = {k.strip(): v for k, v in row.items()}
                    for key, col in [
                        ("precision", "metrics/precision(B)"),
                        ("recall", "metrics/recall(B)"),
                        ("mAP50", "metrics/mAP50(B)"),
                        ("map", "metrics/mAP50-95(B)"),
                    ]:
                        try:
                            out[key].append(float(row.get(col, 0)))
                        except (TypeError, ValueError):
                            pass
        except Exception:
            pass
        return out

    # ---- Render ----

    def render(self, canvas):
        x0 = SIDEBAR_W + 26
        w = WIN_W - SIDEBAR_W - 52
        y = HEADER_H + 24

        self._render_config(canvas, x0, y, w)
        self._render_metrics(canvas, x0, y + 190, w)
        self._render_curves(canvas, x0, y + 384, w)
        self._render_button(canvas, x0, y + 568, w)

    # ---- Configuración ----

    def _render_config(self, canvas, x, y, w):
        cfg = self.app.config
        h = 176
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        section_title(canvas, x + 16, y + 24, "CONFIGURACION DEL ENTRENAMIENTO", w - 40)

        left = [
            ("Modelo base", "YOLOv8s (COCO)"),
            ("Arquitectura", "Small - 11.2M params"),
            ("Dataset", "custom_yolo (5 clases)"),
            ("Transfer learning", "Si"),
        ]
        right = [
            ("Epocas", str(cfg.get("training", {}).get("epochs", 40))),
            ("Resolucion", f"{cfg['model']['imgsz']} px"),
            ("Batch size", str(cfg.get("training", {}).get("batch", 8))),
            ("Optimizer", "AdamW"),
            ("Precision mixta", "AMP (FP16)"),
        ]

        lx = x + 20
        rx = x + w // 2 + 10

        for i, (k, v) in enumerate(left):
            ly = y + 54 + i * 26
            cv2.putText(canvas, k, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, v, (lx + 160, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_WHITE, 1, cv2.LINE_AA)

        for i, (k, v) in enumerate(right):
            ry = y + 54 + i * 24
            cv2.putText(canvas, k, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, v, (rx + 150, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_WHITE, 1, cv2.LINE_AA)

        device = "CUDA" if cfg["model"].get("device") != "cpu" else "CPU"
        badge(canvas, x + w - 130, y + 24, device, GREEN if device == "CUDA" else RED)

    # ---- Métricas ----

    def _render_metrics(self, canvas, x, y, w):
        h = 180
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        section_title(canvas, x + 16, y + 24, "METRICAS DE EVALUACION", w - 40)

        weights_path = self.app.config["model"]["weights_path"]
        model_exists = os.path.exists(weights_path)

        if not self.metrics:
            msg = "Modelo disponible, sin CSV de metricas" if model_exists else "No hay modelo entrenado"
            color = ACCENT if model_exists else RED
            cv2.putText(canvas, msg, (x + 20, y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            if model_exists:
                cv2.putText(canvas, weights_path, (x + 20, y + 94),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_MUTED, 1, cv2.LINE_AA)
            return

        if self.model_version:
            badge(canvas, x + w - 210, y + 18, self.model_version, ACCENT)
            badge(canvas, x + w - 108, y + 18, f"{self.metrics['epochs']} epochs", CYAN)

        items = [
            ("Precision", self.metrics["precision"], 85.0, GREEN),
            ("Recall", self.metrics["recall"], 70.0, CYAN),
            ("mAP50", self.metrics["mAP50"], 70.0, ACCENT),
            ("mAP50-95", self.metrics["mAP50_95"], None, PURPLE),
        ]

        gap = 16
        cw = (w - 40 - gap * 3) // 4

        for i, (label, val, target, color) in enumerate(items):
            cx = x + 20 + i * (cw + gap)
            pct = val * 100

            rounded_rect(canvas, (cx, y + 42), (cx + cw, y + h - 16), 8, BG_INPUT)
            cv2.rectangle(canvas, (cx + 8, y + 42), (cx + cw - 8, y + 45), color, -1)

            cv2.putText(canvas, label.upper(), (cx + 14, y + 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{pct:.1f}%", (cx + 14, y + 104),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            progress_bar(canvas, cx + 14, y + 118, cw - 28, 7, pct, color)

            if target is not None:
                met = pct >= target
                cv2.putText(canvas, f"Meta {target:.0f}%", (cx + 14, y + 144),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)
                badge(canvas, cx + 14, y + 160, "CUMPLIDO" if met else "PENDIENTE",
                      GREEN_DIM if met else RED, 70)

    # ---- Curvas ----

    def _render_curves(self, canvas, x, y, w):
        h = 170
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        section_title(canvas, x + 16, y + 24, "CURVAS DE ENTRENAMIENTO", w - 40)

        if not self._curves or not self._curves.get("mAP50"):
            cv2.putText(canvas, "Sin curvas disponibles (requiere results.csv)",
                        (x + 20, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MUTED, 1, cv2.LINE_AA)
            return

        series = [
            ("mAP50", ACCENT),
            ("mAP50-95", PURPLE),
            ("Precision", GREEN),
            ("Recall", CYAN),
        ]

        gap = 16
        cw = (w - 40 - gap * 3) // 4

        for i, (key, color) in enumerate(series):
            cx = x + 20 + i * (cw + gap)
            data = self._curves.get(key, [])
            if not data:
                continue

            final = data[-1]
            cv2.putText(canvas, key, (cx, y + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{final * 100:.1f}%", (cx, y + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

            sparkline(canvas, cx, y + 76, cw, h - 94, [v * 100 for v in data], color, max_val=100.0)

    # ---- Botón ----

    def _render_button(self, canvas, x, y, w):
        if self.training:
            h = 76
            rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
            cv2.circle(canvas, (x + 26, y + 38), 6, ACCENT, -1)
            cv2.putText(canvas, "Entrenando en segundo plano...", (x + 44, y + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, "Revisa la terminal de VS Code para el progreso",
                        (x + 44, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_MUTED, 1, cv2.LINE_AA)
            return

        buttons = [{
            "label": "Re-entrenar modelo",
            "rect": (x, y, 220, 42),
            "color": ACCENT,
            "hover_color": ACCENT_HOVER,
            "action": self._start_training,
        }]
        self.app.set_custom_buttons(buttons)

    # ---- Ejecución ----

    def _start_training(self):
        if self.training:
            return
        self.training = True
        self.training_log = []
        self.training_thread = threading.Thread(target=self._train_worker, daemon=True)
        self.training_thread.start()

    def _train_worker(self):
        try:
            subprocess.run(
                [sys.executable, "src/04_train.py"],
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            )
        except Exception as e:
            self.training_log.append(f"Error: {e}")
        finally:
            self.training = False
            self._load_metrics()

    def cleanup(self):
        pass
