"""
Vista de Evaluacion: FPR (tasa de falsos positivos) y ROI (retorno).

Estas dos cifras son las que el evaluador pide para cerrar los objetivos, y
las dos que con mas facilidad se reportan ROTOS si se muestran sin contexto:

- El FPR solo es defendible sobre clips NEGATIVOS verificados. Medido sobre el
  set de validacion del proyecto (que contiene personas) cualquier disparo puede
  ser una deteccion correcta, y un "0%" asi obtenido no significa nada.
- El ROI sale de supuestos economicos editables, no de mediciones. Un ROI de
  500% no indica un buen sistema: indica un supuesto de horas ahorradas que hay
  que revisar antes de presentarlo.

Por eso la vista nunca muestra un numero solo: cada tarjeta dice de donde sale,
si el dato es verificable y, cuando no lo es, por que.
"""

import json
import os
import threading
import time

import cv2
import numpy as np

from src.gui.app import (
    SIDEBAR_W, WIN_W, WIN_H, HEADER_H,
    BG_CARD, BG_INPUT,
    TEXT_WHITE, TEXT_DIM, TEXT_MUTED,
    ACCENT, GREEN, RED, CYAN, ORANGE, PURPLE,
    BORDER,
    rounded_rect, badge, kv_row, section_title, card,
)
from src.fpr_evaluation import FPREvaluation, default_negative_clips
from src.roi_analysis import ROIAnalysis

# FPR objetivo del documento de requerimientos.
FPR_TARGET = 5.0


def _verdict_color(verdict):
    if verdict == "CUMPLE":
        return GREEN
    if verdict == "DATOS INSUFICIENTES":
        return ORANGE
    return TEXT_MUTED


class EvaluationView:
    def __init__(self, app):
        self.app = app
        self.fpr_report = None
        self.roi_report = None
        self.busy = None
        self.result = None
        self.error = None
        self._thread = None
        self.clips = []
        # Cache de los informes en disco: mtime visto y momento del chequeo.
        # Sin esto, `render` abriria y parsearia dos JSON en cada frame.
        self._mtimes = {"fpr": None, "roi": None}
        self._checked_at = 0.0

    # ---- Ciclo de vida ----

    def on_enter(self):
        self._load_reports(force=True)

    def _load_reports(self, force=False):
        """Lee los ultimo reportes generados en disco.

        Se leen del archivo y no se recomputan al abrir la pestana: abrir una
        pantalla no puede costar minutos de inferencia, ni mucho menos escribir
        un reporte nuevo como efecto secundario de mirar.

        `render` corre a ~20 FPS, asi que recargar dos JSON en cada frame serian
        ~40 lecturas de disco por segundo mientras la deteccion sigue activa. Se
        recarga solo si el archivo cambio (mtime) y, en cualquier caso, como
        mucho una vez por segundo.
        """
        now = time.time()
        if not force and now - self._checked_at < 1.0:
            return

        for attr, filename, box in (
            ("fpr_report", "fpr_report.json", "fpr"),
            ("roi_report", "roi_report.json", "roi"),
        ):
            path = self._report_path(filename)
            try:
                mtime = os.path.getmtime(path) if os.path.exists(path) else None
            except OSError:
                mtime = None
            if mtime == self._mtimes[box] and getattr(self, attr) is not None:
                continue
            report = None
            if mtime is not None:
                try:
                    with open(path, encoding="utf-8") as f:
                        report = json.load(f)
                except (OSError, ValueError):
                    report = None
            self._mtimes[box] = mtime
            setattr(self, attr, report)

        self._checked_at = now

    def _report_path(self, filename):
        """Mismo directorio que usan FPREvaluation y ROIAnalysis."""
        out = self.app.config.get("evaluation", {}).get(
            "output_dir", "data/evaluation")
        return os.path.join(out, filename)

    # ---- Acciones ----

    def run_fpr(self):
        """RE09: mide la tasa de falsos positivos sobre clips negativos."""
        if not self._go("FPR"):
            return
        # Se respeta la lista de `evaluation.negative_clips` de config.yaml, que
        # es la misma que usa el CLI. La funcion de ayuda solo se consulta si el
        # operador no configuro ninguna, para no divergir entre interfaces.
        clips = list(self.app.config.get("evaluation", {})
                     .get("negative_clips") or [])
        if not clips:
            clips = default_negative_clips(self.app.config)
        clips = [c for c in clips if os.path.exists(c)]
        if not clips:
            self._finish_error(
                "No hay clips negativos para evaluar. Graba escenas de la finca "
                "SIN intrusiones y anotalas en evaluation.negative_clips; sin "
                "ground truth el FPR no es verificable.")
            return
        self.clips = clips

        def work():
            try:
                evaluator = FPREvaluation(self.app.config, db=self.app.db)
                report = evaluator.evaluate_dataset(clips)
                self._finish_ok("FPR", FPREvaluation.format_report(report))
            except Exception as exc:  # noqa: BLE001
                self._finish_error(f"FPR fallo: {exc}")

        self._spawn(work)

    def run_roi(self):
        """RE10: retorno de la inversion con los parametros de config."""
        if not self._go("ROI"):
            return

        def work():
            try:
                report = ROIAnalysis(self.app.config, db=self.app.db).compute()
                self._finish_ok("ROI", ROIAnalysis.format_report(report))
            except Exception as exc:  # noqa: BLE001
                self._finish_error(f"ROI fallo: {exc}")

        self._spawn(work)

    def _go(self, label):
        """Evita dos evaluaciones simultaneas.

        Cada corrida escribe el mismo archivo de reporte y abre el modelo; dos
        a la vez se pisan entre si y el panel muestra el resultado de la otra.
        """
        if self.busy:
            self.app.notify(f"{self.busy} en ejecucion, espera a que termine",
                            ORANGE)
            return False
        self.busy = label
        self.result = None
        self.error = None
        self.app.notify(f"Ejecutando {label}...", ACCENT, seconds=2.0)
        return True

    def _spawn(self, work):
        def target():
            try:
                work()
            finally:
                self.busy = None

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def _finish_ok(self, label, summary):
        self.result = {"label": label, "summary": summary}
        self._load_reports(force=True)
        if hasattr(self.app, "notify"):
            self.app.notify(f"{label} completado", GREEN, seconds=3.0)

    def _finish_error(self, message):
        self.error = message
        self.busy = None
        if hasattr(self.app, "notify"):
            self.app.notify(message, RED, seconds=5.0)

    # ---- Render ----

    def render(self, canvas):
        self._load_reports()

        x = SIDEBAR_W + 26
        w = WIN_W - SIDEBAR_W - 52
        y = HEADER_H + 24
        col_w = (w - 20) // 2

        buttons = []
        if self.busy:
            buttons.append({"label": f"{self.busy}... (ejecutando)",
                            "rect": (x, y, 190, 38), "color": TEXT_MUTED})
        else:
            buttons.append({"label": "Ejecutar FPR", "rect": (x, y, 190, 38),
                            "color": ACCENT, "action": self.run_fpr})
            buttons.append({"label": "Ejecutar ROI", "rect": (x + 200, y, 190, 38),
                            "color": PURPLE, "action": self.run_roi})
        self.app.set_custom_buttons(buttons)

        if self.error:
            badge(canvas, x + 410, y + 19, self.error[:60], RED)

        top = y + 54
        self._render_fpr(canvas, x, top, col_w)
        self._render_roi(canvas, x + col_w + 20, top, col_w)

    def _render_fpr(self, canvas, x, y, w):
        """RE09. La tarjeta principal es el veredicto, no el porcentaje."""
        h = 430
        card(canvas, x, y, w, h, "FPR - Tasa de falsos positivos", ACCENT)
        r = self.fpr_report
        cy = y + 52

        if not r:
            cv2.putText(canvas, "Sin evaluacion ejecutada", (x + 18, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, "Objetivo del documento: FPR < 5%", (x + 18, cy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_DIM, 1, cv2.LINE_AA)
            cy += 60
        else:
            verdict = FPREvaluation.format_report(r)["verdict"]
            badge(canvas, x + w - 150, y + 20, verdict, _verdict_color(verdict))
            cv2.putText(canvas, f"FPR {r.get('fpr_percentage', 0)}%   "
                                f"(objetivo {r.get('target', '< 5%')})",
                        (x + 18, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        TEXT_WHITE, 1, cv2.LINE_AA)
            cy += 30
            kv_row(canvas, x + 18, cy, "Escenas con disparo",
                   f"{r.get('scenes_with_false_positives', 0)}/"
                   f"{r.get('scenes_total', 0)}", w - 40)
            cy += 26
            kv_row(canvas, x + 18, cy, "Falsos positivos totales",
                   str(r.get("total_false_positives", 0)), w - 40)
            cy += 26
            kv_row(canvas, x + 18, cy, "Por minuto",
                   str(r.get("false_positives_per_minute", 0)), w - 40)
            cy += 26
            kv_row(canvas, x + 18, cy, "Minutos evaluados",
                   f"{r.get('video_seconds', 0) / 60:.1f}", w - 40)
            cy += 26
            kv_row(canvas, x + 18, cy, "Ground truth verificado",
                   "SI" if r.get("ground_truth_verified") else "NO",
                   w - 40, value_color=GREEN if r.get("ground_truth_verified")
                   else ORANGE)
            cy += 30

        # La nota de veredicto es la parte que evita que el numero se lea como
        # un resultado. Sin esto, un 0% sobre clips con personas se presenta
        # como sistema excelente.
        note = (r or {}).get("verdict_note") or (
            "El FPR se mide sobre clips sin intrusiones reales. El set de "
            "validacion del proyecto contiene personas, asi que medido ahi un "
            "0% no significa que el sistema no falle.")
        self._wrap_text(canvas, note, x + 18, cy, w - 36, 0.34, TEXT_DIM, 16)

        if r:
            self._wrap_text(canvas, r.get("metric_note", ""), x + 18,
                            y + h - 46, w - 36, 0.32, TEXT_MUTED, 12)

    def _render_roi(self, canvas, x, y, w):
        """RE10."""
        h = 430
        card(canvas, x, y, w, h, "ROI - Retorno de la inversion", PURPLE)
        r = self.roi_report
        cy = y + 52

        if not r:
            cv2.putText(canvas, "Sin analisis ejecutado", (x + 18, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MUTED, 1, cv2.LINE_AA)
            cy += 26
        else:
            lat = r.get("latencia_deteccion", {}) or {}
            cv2.putText(canvas, f"ROI {r.get('roi_porcentaje', 0)}%", (x + 18, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, TEXT_WHITE, 1, cv2.LINE_AA)
            cy += 30
            payback = r.get("retorno_inversion_anos")
            kv_row(canvas, x + 18, cy, "Retorno de inversion",
                   f"{payback} anos" if payback else "sin retorno",
                   w - 40, value_color=GREEN if payback and payback <= 3 else ORANGE)
            cy += 26
            kv_row(canvas, x + 18, cy, "Inversion inicial",
                   f"{r.get('inversion_inicial_clp', 0):,} CLP", w - 40)
            cy += 26
            kv_row(canvas, x + 18, cy, "Beneficio anual",
                   f"{r.get('beneficio_anual_clp', 0):,} CLP", w - 40)
            cy += 26
            kv_row(canvas, x + 18, cy, "Latencia de deteccion",
                   f"{lat.get('total_local_s', 0)} s", w - 40)
            cy += 30

        # Las advertencias de plausibilidad van arriba del numero, no al pie:
        # un ROI de 584% con supuestos sin verificar no debe leerse como el
        # titular del sistema.
        for warning in (r or {}).get("advertencias", []):
            self._wrap_text(canvas, f"! {warning}", x + 18, cy, w - 36,
                            0.34, ORANGE, 16)
            cy += 40

        self._wrap_text(canvas, (r or {}).get("nota_metodologica", (
            "El ROI depende de supuestos economicos editables en "
            "config.yaml. Reemplaza los montos por los tuyos antes de "
            "presentarlo.")), x + 18, y + h - 66, w - 36, 0.32, TEXT_MUTED, 12)

    def _wrap_text(self, canvas, text, x, y, max_w, scale, color, max_lines):
        """Ajuste de linea por ancho de caracter, no por palabras.

        OpenCV no mide texto, asi que se estima el ancho por caracter. Es
        aproximado pero suficiente para textos cortos de estado; un ajuste
        exacto exigiria medir la fuente y no aporta a la decision.
        """
        if not text:
            return
        words = str(text).split()
        line = ""
        lines = []
        for word in words:
            candidate = f"{line} {word}".strip()
            if len(candidate) * 8.2 * scale > max_w and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        for i, text_line in enumerate(lines[:max_lines]):
            suffix = "..." if i == max_lines - 1 and len(lines) > max_lines else ""
            cv2.putText(canvas, text_line + suffix, (x, y + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
