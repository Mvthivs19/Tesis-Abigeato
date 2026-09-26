"""Smoke test del loop de deteccion con video real (sin ventana)."""
import os, sys, time, copy
import numpy as np

# Este archivo vive en <proyecto>/tests, pero la configuracion y los datos
# estan en la raiz del proyecto: el path debe subir un nivel.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.gui.app as appmod
appmod.cv2.namedWindow = lambda *a, **k: None
appmod.cv2.setMouseCallback = lambda *a, **k: None

from src.utils import load_config
from src.gui.tab_detection import DetectionView


class FakeApp:
    def __init__(self, config):
        self.config = config
        self.user = {"username": "admin", "role": "administrador"}
        self.access = None
        self.custom_buttons = []
        self.notices = []
        self._buttons = []

    def set_custom_buttons(self, buttons):
        self._buttons = buttons
        self.custom_buttons = buttons

    def require(self, perm):
        return True

    def notify(self, text, color=None, seconds=4.0):
        self.notices.append(text)


cfg = load_config("configs/config.yaml")
# Horario 24/7 para que la inferencia corra aunque sea de dia, y clip largo
# para que el buffer circular se llene sin disparar alertas.
cfg["schedule"]["enabled"] = False
cfg["alerts"]["clip_seconds"] = 4
cfg["alerts"]["cooldown_seconds"] = 999
cfg["camera"]["source"] = "data/sample_video.mp4"

app = FakeApp(cfg)
view = DetectionView(app)
print("modelo cargado:", view.model is not None)

# Instrumentacion: contar llamadas al coordinator.
calls = {"push_frame": 0, "push_annotated": 0, "integrity": 0, "can_detect": 0,
         "register": 0, "tick": 0}
orig_start = None

view._start_detection()
t0 = time.time()
# La primera inferencia inicializa el contexto de la GPU y puede tardar
# varios segundos, asi que la ventana debe ser holgada.
while view.running and time.time() - t0 < 35.0:
    time.sleep(0.5)

co = view.coordinator
print("frames procesados:", view.frame_count)
print("fps medido: %.2f" % view.fps_measured)
print("inferencia ms: %.1f" % view.inference_ms)
print("detecciones en el ultimo frame:", len(view.detections))
print("clases detectadas:", sorted({d["class"] for d in view.detections}))
print("calidad:", co.quality_filter.last_quality, "score %.2f" % co.quality_filter.last_score)
print("integridad:", co.integrity.last_status, "|", co.integrity.last_detail)
print("suspendido:", co.suspended_reason)
print("frames en el buffer del recorder:", len(co.recorder.frames)
      if hasattr(co.recorder, "frames") else
      len(getattr(co.recorder, "buffer", [])))
print("pendientes remotos:", view.pending_remote)
print("evento abierto:", view.open_event)
print("ultimas alertas:", view.alerts_log[-6:])
print("botones en header:", [b["label"] for b in app._buttons])
print("model_info:", view.model_info, "| model_error:", view.model_error)

df = view.display_frame
print("display_frame:", None if df is None else df.shape)

view._stop_detection()
time.sleep(0.6)
print("hilo detenido:", not view.running)
