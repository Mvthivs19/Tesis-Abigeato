"""
RE11: Filtrado de ruido visual y condiciones ambientales adversas.

Una camara NIR nocturna captura fenomenos que no son intrusiones y que el
modelo puede confundir con una persona:

  - Niebla, lluvia o neblina: baja el contraste global.
  - Lente empañada, sucia o con escarcha: imagen desenfocada.
  - Viento fuerte o follaje moviendose: ráfagas de ruido de alta frecuencia.
  - Cambio de iluminacion (relampago, focos de un vehiculo): saturacion puntual.
  - Perder el senal: frame negro o gris uniforme.

En lugar de descartar la imagen, el filtro clasifica la calidad del frame y
sustituye los pixels deficitarios por un frame limpio anterior cuando es
posible. Esto mantiene la deteccion operativa en vez de dejarla ciega, y deja
traza en el log de cada evento de calidad.
"""

import time
from collections import deque

import cv2
import numpy as np

try:
    from src.utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger("frame_filter")

QUALITY_OK = "ok"
QUALITY_LOW_CONTRAST = "low_contrast"
QUALITY_BLUR = "blur"
QUALITY_NOISE = "noise"
QUALITY_OVEREXPOSED = "overexposed"
QUALITY_BLACKOUT = "blackout"


class FrameQualityFilter:
    def __init__(self, config=None, history_size=5):
        config = config or {}
        self.enabled = config.get("enabled", True)

        # Umbrales calibrados sobre el dataset de la finca (video NIR 1080x1920
        # y capturas nocturnas). Son configurables por el Administrador.
        self.min_contrast = config.get("min_contrast", 18.0)
        self.max_brightness = config.get("max_brightness", 235.0)
        self.min_brightness = config.get("min_brightness", 8.0)
        self.blur_threshold = config.get("blur_threshold", 28.0)
        self.noise_threshold = config.get("noise_threshold", 12.0)
        self.smooth_ratio = config.get("smooth_ratio", 0.25)
        self.history_size = config.get("history_size", history_size)

        self._history = deque(maxlen=self.history_size)
        self._last_good = None
        self._degraded_since = None
        self.last_quality = QUALITY_OK
        self.last_score = 1.0
        self.last_metrics = {}

    # ---------- Metricas de calidad ----------

    @staticmethod
    def _to_gray(frame):
        """Convierte a escala de grises tolerando 1, 3 o 4 canales.

        El source de video puede entregar BGR (lo habitual) o ya venir en
        gris si el sensor NIR exporta un solo canal. Sin esta normalizacion
        `cv2.cvtColor` lanza y el filtro mataria el loop de deteccion.
        """
        if frame is None:
            raise ValueError("frame None")
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3:
            channels = frame.shape[2]
            if channels == 1:
                return frame[:, :, 0]
            if channels == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if channels == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"Frame con {frame.ndim} dimensiones no soportado")

    @staticmethod
    def _contrast(gray):
        """Desviacion estandar de la luminancia. proxy de contraste global."""
        return float(gray.std())

    @staticmethod
    def _sharpness(gray):
        """Varianza del Laplaciano. Baja cuando la imagen esta desenfocada."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _noise_level(gray):
        """Residuo de un filtro de mediana: separa el ruido de la senal."""
        median = cv2.medianBlur(gray, 3)
        return float(np.abs(gray.astype(np.float32) - median.astype(np.float32)).mean())

    @staticmethod
    def _brightness(gray):
        return float(gray.mean())

    def assess(self, frame):
        """Clasifica la calidad del frame. Devuelve (quality, score, metrics)."""
        if not self.enabled:
            return QUALITY_OK, 1.0, {"enabled": False}

        gray = self._to_gray(frame)
        brightness = self._brightness(gray)
        contrast = self._contrast(gray)
        sharpness = self._sharpness(gray)
        noise = self._noise_level(gray)

        metrics = {
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "sharpness": round(sharpness, 2),
            "noise": round(noise, 2),
        }

        # 1. Perdida total de senal: negro o blanco uniforme.
        if brightness < self.min_brightness or brightness > self.max_brightness:
            if contrast < 5.0:
                return QUALITY_BLACKOUT, 0.0, metrics

        # 2. Niebla / lente empañada: contraste bajo pero bordes presentes.
        if contrast < self.min_contrast:
            return QUALITY_LOW_CONTRAST, contrast / max(self.min_contrast, 1e-6), metrics

        # 3. Lente sucia o fuera de foco.
        if sharpness < self.blur_threshold:
            return QUALITY_BLUR, sharpness / max(self.blur_threshold, 1e-6), metrics

        # 4. Viento, lluvia o interferencia: mucho residuo de mediana.
        if noise > self.noise_threshold:
            return QUALITY_NOISE, 1.0 - (noise - self.noise_threshold) / 100.0, metrics

        return QUALITY_OK, 1.0, metrics

    # ---------- Recuperacion ----------

    def process(self, frame):
        """Devuelve (frame_util, quality, score, metrics).

        Si el frame es inutil (blackout) y hay historial limpio, se devuelve
        el ultimo frame conocido bueno para no perder la vigilancia. Si no hay
        historial, se devuelve el frame original con score 0 para que el
        pipeline decida no inferir.

        `last_quality`/`last_score`/`last_metrics` se actualizan SIEMPRE, con
        independencia del frame devuelto: `AlertCoordinator.can_detect()` los
        consulta para decidir si permite inferencia, asi que dejarlos con el
        valor del frame anterior permitiria deteccionar con una imagen en
        negro o desenfocada creyendo que la calidad es buena.
        """
        quality, score, metrics = self.assess(frame)

        usable = frame
        if quality == QUALITY_OK:
            self._history.append(frame.copy())
            self._last_good = frame
            self._degraded_since = None
        else:
            if self._degraded_since is None:
                self._degraded_since = time.time()
                logger.warning(f"[RE11] Calidad degradada: {quality} "
                               f"(score {score:.2f}) {metrics}")
            # Suavizado: solo tiene sentido si el problema es niebla o ruido.
            if quality in (QUALITY_LOW_CONTRAST, QUALITY_NOISE):
                if self._history:
                    try:
                        usable = cv2.fastNlMeansDenoising(
                            frame, None, h=7, templateWindowSize=7,
                            searchWindowSize=21
                        )
                    except cv2.error:
                        # Un frame de un solo canal no admite este filtro.
                        usable = frame
            elif quality in (QUALITY_BLACKOUT, QUALITY_BLUR) and self._history:
                # Reutiliza el ultimo frame limpio: es preferible una imagen
                # levemente desactualizada a perder la deteccion. Si la camara
                # cambio de resolucion, un frame de otra geometria no sirve
                # y se descarta la historia para no propagar el error.
                last = self._history[-1]
                if last.shape == frame.shape:
                    usable = last.copy()
                else:
                    self._history.clear()
                    self._last_good = None

        self.last_quality = quality
        self.last_score = score
        self.last_metrics = metrics
        return usable, quality, score, metrics

    def is_detection_allowed(self, quality, score):
        """RE11: la inferencia se omite cuando la imagen no es confiable.

        Una imagen con score muy bajo puede generar mas falsos positivos que
        useful deteccion, que es exactamente lo que el filtro busca evitar.
        """
        if not self.enabled:
            return True, ""
        if quality == QUALITY_BLACKOUT:
            return False, "Senal perdida o lente tapada"
        if score < self.smooth_ratio:
            return False, f"Calidad insuficiente ({quality}, score {score:.2f})"
        return True, ""

    def degradation_seconds(self):
        if self._degraded_since is None:
            return 0
        return int(time.time() - self._degraded_since)

    def describe(self):
        if not self.enabled:
            return "Filtrado desactivado"
        return {
            "quality": self.last_quality,
            "score": round(self.last_score, 2),
            "degraded_s": self.degradation_seconds(),
        }
