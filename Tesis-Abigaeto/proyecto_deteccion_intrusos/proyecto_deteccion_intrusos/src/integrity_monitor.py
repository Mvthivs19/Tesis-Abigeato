"""
RE21 (complemento): Deteccion de compromiso fisico de la camara y degradacion.

El `health_monitor.py` vigila el equipo (CPU/RAM). Este modulo vigila que la
propia camara no este siendo saboteada: un intruso que busca evadir la
deteccion puede tapar la lente, mover el soporte, conectar un cable o cortar la
senal.

Sintomas monitorizados:

  - Lente tapada / objeto cercano: el frame se vuelve uniforme o muy
    desenfocado de forma sostenida.
  - Cambio de encuadre: desplazamiento brusco entre frames consecutivos
    (movimiento fisico del soporte).
  - Perdida de senal: el source deja de entregar frames.
  - Caida de tasa de frames: la camara sigue conectada pero degradada.

Cada sintoma se persiste en `integrity_log` y genera una alerta de prioridad
alta, porque un sistema de seguridad ciego es peor que uno que avisa que no
ve. La inferencia se pausa automaticamente mientras la camara esta comprometida
para no generar detecciones sobre imagenes no confiables.
"""

import time
from datetime import datetime

import cv2
import numpy as np

try:
    from src.utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger("integrity_monitor")

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"

TYPE_COVERED = "lente_tapada"
TYPE_SHIFTED = "encuadre_desplazado"
TYPE_SIGNAL_LOSS = "perdida_de_senal"
TYPE_DEGRADED = "degradacion"


class IntegrityMonitor:
    def __init__(self, config=None, db=None):
        config = config or {}
        self.config = config
        self.db = db
        self.enabled = config.get("enabled", True)

        # Tolerancia al desplazamiento de encuadre, en pixeles de shift
        # estimado entre frames. Calibrada para vibracion de viento menor.
        self.max_shift_pixels = config.get("max_shift_pixels", 18.0)
        # Frames sin senal consecutivos antes de declarar perdida.
        self.signal_loss_frames = config.get("signal_loss_frames", 30)
        # Frames por debajo del FPS minimo antes de declarar degradacion.
        self.degraded_frames = config.get("degraded_frames", 20)
        self.min_fps = config.get("min_fps", 5.0)
        # Duracion mínima del sintoma para evitar alarmas por un frame suelto.
        self.debounce_seconds = config.get("debounce_seconds", 8.0)

        self._prev_gray = None
        self._prev_gray_time = None
        self._frames = 0
        self._consecutive_bad = 0
        self._consecutive_low_fps = 0
        self._fps_window = []
        self._active_alert = None
        self._alert_started = None
        # Ultima escritura en `integrity_log` por tipo de sintoma. Evita que un
        # problema persistente que dura minutos grabe una fila por frame.
        self._last_logged = {}
        self.last_status = "ok"
        self.last_detail = ""

    # ---------- Deteccion ----------

    @staticmethod
    def _estimate_shift(gray_a, gray_b):
        """Desplazamiento (dx, dy) entre dos frames por correlacion de fases.

        Se usa `phaseCorrelate` porque es barato y subpíxel: un intruso que
        mueve la camara produce un desplazamiento coherente en todo el frame,
        mientras que el ruido produce estimaciones incoherentes.
        """
        if gray_a.shape != gray_b.shape:
            return 0.0, 0.0
        a = np.float32(gray_a)
        b = np.float32(gray_b)
        try:
            (dx, dy), _response = cv2.phaseCorrelate(a, b)
        except cv2.error:
            return 0.0, 0.0
        return float(dx), float(dy)

    def update(self, frame, quality_filter=None, fps=None):
        """Analiza el frame actual. Devuelve (status, detail, compromised).

        `quality_filter` es un FrameQualityFilter opcional; si el filtro ya
        determino que la lente esta tapada, el monitor lo propaga como
        compromiso critico sin duplicar el diagnostico.
        """
        if not self.enabled:
            return "ok", "Monitor de integridad desactivado", False

        self._frames += 1
        compromised = False
        problems = []

        # --- Sintoma 1: perdida de senal (source sin frames) ---
        if frame is None:
            self._consecutive_bad += 1
            if self._consecutive_bad >= self.signal_loss_frames:
                compromised = True
                problems.append((TYPE_SIGNAL_LOSS, SEVERITY_CRITICAL,
                                 "El source de video dejo de entregar frames"))
            self._record(compromised, problems)
            return self._finalize(compromised, problems)

        self._consecutive_bad = 0

        # --- Sintoma 2: degradacion de tasa de frames ---
        # Un FPS medido en cero significa "aun no hay medicion", no "la camara
        # se detuvo". Confundir ambos produce una alerta de degradacion en cada
        # arranque, y el operador termina ignorar las advertencias que si
        # importan. Un 0 explicito del llamador se trata como medicion ausente.
        now = time.time()
        if fps is not None and fps > 0:
            self._fps_window.append((now, fps))
            self._fps_window = [t for t in self._fps_window if now - t[0] < 10]
            if len(self._fps_window) >= 5:
                avg = sum(f for _, f in self._fps_window) / len(self._fps_window)
                if avg < self.min_fps:
                    self._consecutive_low_fps += 1
                    # `degraded_frames` exige que la caida sea SOSTENIDA. Sin
                    # este contador, los primeros segundos de cada arranque
                    # (donde el FPS climbs desde 0 hasta su regimen) se
                    # reportaban como degradacion y llenaban `integrity_log`
                    # de decenas de filas identicas.
                    if self._consecutive_low_fps >= self.degraded_frames:
                        problems.append((TYPE_DEGRADED, SEVERITY_WARNING,
                                         f"FPS promedio {avg:.1f} < {self.min_fps}"))
                else:
                    self._consecutive_low_fps = 0
        else:
            self._consecutive_low_fps = 0

        # --- Sintoma 3: lente tapada (delegado al filtro de calidad) ---
        if quality_filter is not None and quality_filter.enabled:
            if quality_filter.last_quality == "blackout":
                problems.append((TYPE_COVERED, SEVERITY_CRITICAL,
                                 "Lente tapada o sucia: imagen uniforme"))
                compromised = True
            elif quality_filter.last_quality == "blur" and \
                    quality_filter.degradation_seconds() > self.debounce_seconds:
                problems.append((TYPE_COVERED, SEVERITY_WARNING,
                                 "Lente desenfocada de forma sostenida "
                                 "(posible empañamiento o manipulacion)"))
                compromised = True

        # --- Sintoma 4: desplazamiento de encuadre ---
        if quality_filter is not None:
            gray = quality_filter._to_gray(frame)
        elif frame.ndim == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is not None:
            dx, dy = self._estimate_shift(self._prev_gray, gray)
            shift = (dx ** 2 + dy ** 2) ** 0.5
            # Ignora el desplazamiento natural por movimiento de la escena:
            # solo nos importa un salto coherente grande.
            if shift > self.max_shift_pixels:
                problems.append((TYPE_SHIFTED, SEVERITY_CRITICAL,
                                 f"Encuadre desplazado {shift:.1f}px "
                                 f"(movimiento fisico de la camara)"))
                compromised = True
        self._prev_gray = gray

        self._record(compromised, problems)
        return self._finalize(compromised, problems)

    def _finalize(self, compromised, problems):
        if problems:
            worst = SEVERITY_CRITICAL if any(
                p[1] == SEVERITY_CRITICAL for p in problems) else SEVERITY_WARNING
            self.last_status = "comprometida" if compromised else "degradada"
            self.last_detail = "; ".join(p[2] for p in problems)
            if self._active_alert != worst:
                self._active_alert = worst
                self._alert_started = time.time()
                logger.error(f"[RE21] Integridad {worst.upper()}: {self.last_detail}")
        else:
            if self._active_alert is not None:
                logger.info("[RE21] Integridad restaurada tras alerta previa")
            self.last_status = "ok"
            self.last_detail = ""
            self._active_alert = None
            self._alert_started = None
        return self.last_status, self.last_detail, compromised

    def _record(self, compromised, problems):
        """Persiste los sintomas en `integrity_log`, con antirrebote.

        `update()` se llama en cada frame, asi que escribir sin filtro llenaba la
        tabla de cientos de filas repetidas. aqui solo entra: (a) el sintoma que
        aparece por primera vez, o (b) el que sigue vigente pasado el
        `debounce_seconds`, como recordatorio de que continua abierto.
        """
        if not problems or self.db is None:
            return
        now = time.time()
        active_types = {p[0] for p in problems}
        for ptype, severity, detail in problems:
            last = self._last_logged.get(ptype)
            if last is not None and (now - last) < self.debounce_seconds:
                continue
            self._last_logged[ptype] = now
            try:
                self.db.insert_integrity_log(
                    datetime.now().isoformat(), ptype, severity, detail
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[RE21] No se pudo registrar integridad: {exc}")
        # Sintomas que ya se resolvieron no deben arrastrar su marca de tiempo.
        for ptype in list(self._last_logged):
            if ptype not in active_types:
                del self._last_logged[ptype]

    # ---------- Reporte ----------

    def active_alert(self):
        """Alerta de integridad vigente, o None. La GUI la muestra en el HUD."""
        if self._active_alert is None:
            return None
        return {
            "severity": self._active_alert,
            "status": self.last_status,
            "detail": self.last_detail,
            "seconds": int(time.time() - (self._alert_started or time.time())),
        }

    def describe(self):
        return {
            "enabled": self.enabled,
            "status": self.last_status,
            "detail": self.last_detail,
            "frames": self._frames,
        }
