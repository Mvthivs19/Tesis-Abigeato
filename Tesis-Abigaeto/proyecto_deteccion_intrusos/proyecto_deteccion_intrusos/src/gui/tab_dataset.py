"""
Vista de Dataset: gestion y estadisticas de imagenes de entrenamiento.
"""

import os
import cv2
import numpy as np
from pathlib import Path

from src.gui.app import (
    SIDEBAR_W, WIN_W, WIN_H, HEADER_H,
    BG_CARD, BG_INPUT,
    TEXT_WHITE, TEXT_DIM, TEXT_MUTED,
    ACCENT, GREEN, RED, CYAN, ORANGE, PURPLE,
    BORDER,
    rounded_rect, badge, progress_bar, section_title,
)


CLASES = ["humano", "bovino", "equino", "ovino", "porcino"]

CLASS_COLORS = {
    "humano": RED,
    "bovino": CYAN,
    "equino": ACCENT,
    "ovino": GREEN,
    "porcino": ORANGE,
}

SPLIT_META = {
    "train": ("ENTRENAMIENTO", GREEN),
    "val": ("VALIDACION", ACCENT),
    "test": ("PRUEBA", PURPLE),
}


class DatasetView:
    def __init__(self, app):
        self.app = app
        self.stats = {}
        self._cache = None
        self._cache_time = 0
        self._load_stats()

    def on_enter(self):
        import time as _t
        # Recalcula como maximo cada 30 s: leer ~28K labels toma varios segundos
        if self._cache and _t.time() - self._cache_time < 30:
            self.stats = self._cache
        else:
            self._load_stats()
            self._cache = self.stats
            self._cache_time = _t.time()

    def _load_stats(self):
        self.stats = {}
        yolo_dir = Path("data/custom_yolo")

        for split in ["train", "val", "test"]:
            img_dir = yolo_dir / "images" / split
            lbl_dir = yolo_dir / "labels" / split

            n_images = 0
            if img_dir.exists():
                for ext in ("*.jpg", "*.png", "*.jpeg"):
                    n_images += len(list(img_dir.glob(ext)))

            class_counts = {name: 0 for name in CLASES}
            if lbl_dir.exists():
                for lbl_file in lbl_dir.glob("*.txt"):
                    try:
                        for line in lbl_file.read_text(errors="ignore").splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            ci = int(line.split()[0])
                            if 0 <= ci < len(CLASES):
                                class_counts[CLASES[ci]] += 1
                    except Exception:
                        continue

            self.stats[split] = {"images": n_images, "classes": class_counts}

    def render(self, canvas):
        x0 = SIDEBAR_W + 26
        w = WIN_W - SIDEBAR_W - 52
        y = HEADER_H + 24

        total_images = sum(s["images"] for s in self.stats.values())
        total_anns = sum(sum(s["classes"].values()) for s in self.stats.values())

        self._render_summary(canvas, x0, y, w, total_images, total_anns)
        self._render_splits(canvas, x0, y + 100, w)
        self._render_class_distribution(canvas, x0, y + 220, w)
        self._render_per_split_table(canvas, x0, y + 440, w)

    # ---- Resumen ----

    def _render_summary(self, canvas, x, y, w, total_images, total_anns):
        h = 82
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        cv2.rectangle(canvas, (x + 10, y), (x + w - 10, y + 3), ACCENT, -1)

        items = [
            ("IMAGENES TOTALES", f"{total_images:,}", ACCENT),
            ("ANOTACIONES", f"{total_anns:,}", CYAN),
            ("CLASES", str(len(CLASES)), GREEN),
            ("FORMATO", "YOLO txt", PURPLE),
        ]

        gap = 16
        cw = (w - 40 - gap * 3) // 4
        for i, (label, value, color) in enumerate(items):
            cx = x + 20 + i * (cw + gap)
            cv2.putText(canvas, label, (cx, y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, value, (cx, y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            if i < 3:
                cv2.line(canvas, (cx + cw + gap // 2, y + 18),
                         (cx + cw + gap // 2, y + h - 18), BORDER, 1)

    # ---- Splits ----

    def _render_splits(self, canvas, x, y, w):
        h = 100
        gap = 16
        cw = (w - gap * 2) // 3

        for i, (split, data) in enumerate(self.stats.items()):
            label, color = SPLIT_META.get(split, (split.upper(), ACCENT))
            cx = x + i * (cw + gap)

            rounded_rect(canvas, (cx, y), (cx + cw, y + h), 10, BG_CARD)
            cv2.rectangle(canvas, (cx + 10, y), (cx + cw - 10, y + 3), color, -1)

            cv2.putText(canvas, label, (cx + 16, y + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{data['images']:,}", (cx + 16, y + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT_WHITE, 2, cv2.LINE_AA)
            tw = cv2.getTextSize(f"{data['images']:,}", cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0]
            cv2.putText(canvas, "imagenes", (cx + 24 + tw, y + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_DIM, 1, cv2.LINE_AA)

    # ---- Distribución por clase ----

    def _render_class_distribution(self, canvas, x, y, w):
        h = 202
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        section_title(canvas, x + 16, y + 24, "DISTRIBUCION DE ANOTACIONES POR CLASE", w - 40)

        totals = {name: 0 for name in CLASES}
        for data in self.stats.values():
            for name in CLASES:
                totals[name] += data["classes"].get(name, 0)

        total_anns = sum(totals.values()) or 1
        max_c = max(totals.values()) or 1

        bar_x = x + 130
        bar_max_w = w - 270
        row_h = 30

        for i, name in enumerate(CLASES):
            by = y + 52 + i * row_h
            count = totals[name]
            color = CLASS_COLORS.get(name, ACCENT)
            pct_of_max = (count / max_c * 100) if max_c else 0

            cv2.rectangle(canvas, (x + 20, by + 2), (x + 26, by + 16), color, -1)
            cv2.putText(canvas, name, (x + 34, by + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1, cv2.LINE_AA)

            rounded_rect(canvas, (bar_x, by), (bar_x + bar_max_w, by + 18), 9, BG_INPUT)
            fill = int(bar_max_w * (count / max_c)) if max_c else 0
            if fill > 6:
                rounded_rect(canvas, (bar_x, by), (bar_x + fill, by + 18), 9, color)

            pct_total = count / total_anns * 100
            cv2.putText(canvas, f"{count:,}", (bar_x + bar_max_w + 12, by + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{pct_total:.1f}%", (bar_x + bar_max_w + 82, by + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)

    # ---- Tabla por split ----

    def _render_per_split_table(self, canvas, x, y, w):
        h = 168
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        section_title(canvas, x + 16, y + 24, "DETALLE POR CLASE Y DIVISION", w - 40)

        headers = ["Clase", "Entrenamiento", "Validacion", "Prueba", "Total"]
        col_x = [x + 20, x + 190, x + 330, x + 450, x + 580]
        col_w = [150, 130, 120, 120, 0]

        header_y = y + 48
        for i, hdr in enumerate(headers):
            cv2.putText(canvas, hdr.upper(), (col_x[i], header_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_MUTED, 1, cv2.LINE_AA)

        cv2.line(canvas, (x + 20, header_y + 8), (x + w - 20, header_y + 8), BORDER, 1)

        row_y = header_y + 28
        for i, name in enumerate(CLASES):
            ry = row_y + i * 22
            if i % 2 == 0:
                rounded_rect(canvas, (x + 14, ry - 12), (x + w - 14, ry + 10), 5, BG_INPUT)

            color = CLASS_COLORS.get(name, ACCENT)
            cv2.rectangle(canvas, (x + 20, ry - 8), (x + 26, ry + 6), color, -1)
            cv2.putText(canvas, name, (x + 34, ry + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_WHITE, 1, cv2.LINE_AA)

            total = 0
            for j, split in enumerate(["train", "val", "test"]):
                c = self.stats.get(split, {}).get("classes", {}).get(name, 0)
                total += c
                cv2.putText(canvas, f"{c:,}", (col_x[j + 1], ry + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)

            cv2.putText(canvas, f"{total:,}", (col_x[4], ry + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
