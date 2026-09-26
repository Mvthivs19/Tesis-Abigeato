"""
RE18: el dialogo de falso positivo, rama por rama, hasta la etiqueta en disco.

Por que este archivo existe: la suite unitaria probaba el exportador y la base
de datos por separado, pero nadie habia recorrido el camino completo
`dialogo -> confirmar -> etiqueta YOLO`. Un error en el cable entre ambos
podia dejar al operador creyendo que corrigio el modelo cuando en realidad se
exporto un fondo (o al reves).

Se ejercita la vista real de deteccion. `tkinter` se sustituye por un doble:
no se abre ninguna ventana, pero cada rama devuelve un valor distinto, que es
justo lo que permite comprobar que todas estan cubiertas y no solo la que
devuelve el camino feliz.
"""

import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock as mock

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.database import EventDatabase
from src.retraining import FalsePositiveExporter
from src.utils import load_config

CLASSES = ["humano", "bovino", "equino", "ovino", "porcino"]


def _frame():
    img = np.full((240, 320, 3), 70, np.uint8)
    cv2.rectangle(img, (40, 60), (180, 200), (150, 120, 90), -1)
    return img


class Re18DialogTest(unittest.TestCase):
    """Cada rama del dialogo, con su efecto observable en la BD y el disco."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        self.db_path = os.path.join(self.tmp, "events.db")
        self.db = EventDatabase(self.db_path)
        self.snapdir = os.path.join(self.tmp, "snaps")
        os.makedirs(self.snapdir, exist_ok=True)
        self.snap = os.path.join(self.snapdir, "s.jpg")
        cv2.imwrite(self.snap, _frame())

        self.out = os.path.join(self.tmp, "fp")
        self.config = load_config()
        self.config["database"]["path"] = self.db_path
        self.config["retraining"]["output_dir"] = self.out
        self.config["retraining"]["copy_snapshots"] = True
        self.config["retraining"]["copy_clips"] = True
        self.config["classes"]["names"] = CLASSES
        # La vista lee las clases de `model.class_names`; se rellena la misma
        # lista para que el indice del dialogo sea predecible.
        self.config["model"]["class_names"] = CLASSES

        self.event_id = self.db.insert_event(
            "humano", 0.82, self.snap, bbox="40,60,180,200")
        self.view = self._make_view()

    def _make_view(self):
        """Vista de deteccion real, con un coordinator real sobre la BD."""
        from src.gui import tab_detection

        app = mock.MagicMock()
        app.config = self.config
        app.user = {"username": "admin", "role": "administrador"}
        app.access.can.return_value = True
        app.mouse_x = -1
        app.mouse_y = -1

        view = tab_detection.DetectionView.__new__(tab_detection.DetectionView)
        view.app = app
        view.model_info = None
        view.running = False
        view.thread = None
        view._coordinator_errors = 0
        view.current_frame = None
        view.display_frame = None
        view.paused = False
        view.zone_mode = False
        view.zone_points = []
        view.polygon = []
        view.detections = []
        view.alerts_log = []
        view.escalation_notice = None
        view.pending_remote = {}
        view.open_event = None
        view.remote_enabled = False
        view.last_suspended_reason = None

        from src.alert_coordinator import AlertCoordinator
        view.coordinator = AlertCoordinator(self.config, self.db)
        return view

    def _ask(self, returned):
        """Ejecuta el dialogo con `simpledialog.askstring` devolviendo
        `returned`, y devuelve lo que la vista respondio."""
        with mock.patch("src.gui.tab_detection.tk.Tk"), \
                mock.patch("src.gui.tab_detection.simpledialog.askstring",
                           return_value=returned):
            self.view._ask_false_positive_reason()
        return self.view

    def _event(self):
        """Estado del evento. Se consulta por SQL porque la clase no expone un
        getter de un evento concreto y aqui importa mas el estado crudo que la
        forma de obtenerlo. `acknowledged` e `is_false_positive` son enteros
        con 0 por defecto, no NULL: se normalizan a bool para que las
        aserciones se lean como lo que significan."""
        conn = self.db._connect()
        try:
            cur = conn.execute(
                "SELECT acknowledged, is_false_positive, real_class "
                "FROM events WHERE event_id = ?", (self.event_id,))
            row = cur.fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row, "el evento de prueba no existe")
        return {"acknowledged": bool(row[0]),
                "is_false_positive": bool(row[1]),
                "real_class": row[2]}

    def _log(self):
        """Mensajes del log de alertas. Las entradas son tuplas (hora, texto)."""
        return [text for _time, text in self.view.alerts_log]

    def _labels(self):
        """Devuelve el contenido de las etiquetas YOLO exportadas.

        El exportador escribe en `output/<sello>/labels`, no en `output/labels`:
        cada exportacion es una carpeta fechada para no pisar la anterior."""
        base = os.path.join(self.out, "labels")
        if not os.path.isdir(base):
            candidates = []
            for root, dirs, _files in os.walk(self.out):
                if os.path.basename(root) == "labels":
                    candidates.append(root)
            self.assertEqual(len(candidates), 1,
                             f"se esperaba una carpeta de etiquetas: {candidates}")
            base = candidates[0]
        found = {}
        for name in sorted(os.listdir(base)):
            with open(os.path.join(base, name), "r", encoding="utf-8") as handle:
                found[name] = [ln for ln in handle.read().splitlines() if ln]
        return found

    def _export(self):
        return FalsePositiveExporter(self.config, self.db).export()

    # ---------- ramas del dialogo ----------

    def test_cancel_leaves_the_event_open_and_untouched(self):
        """Cancelar significa "no marcar nada". Si se confirmara aqui, el
        operador cerraria una alerta sin haber decidido nada."""
        self._ask(None)
        event = self._event()
        self.assertFalse(event["acknowledged"])
        self.assertFalse(event["is_false_positive"])
        self.assertIsNone(event["real_class"])

    def test_zero_declares_background_and_exports_an_empty_label(self):
        """'0' = no habia objeto real. La etiqueta YOLO debe quedar VACIA:
        es un fondo, no una caja con la clase que el modelo se habia
        equivocado."""
        self._ask("0")
        event = self._event()
        self.assertTrue(event["acknowledged"])
        self.assertTrue(event["is_false_positive"])
        self.assertIsNone(event["real_class"])
        self._export()
        labels = self._labels()
        self.assertEqual(len(labels), 1, f"se esperaba 1 etiqueta: {labels}")
        self.assertEqual(list(labels.values())[0], [],
                         "un fondo debe exportarse como etiqueta vacia")

    def test_valid_option_records_the_real_class_and_a_real_box(self):
        """Elegir la clase real debe producir una etiqueta POSITIVA con la
        caja detectada: eso es lo que ensena al modelo a corregir la
        confusion, y no un fondo."""
        self._ask("2")                      # 2 = bovino
        event = self._event()
        self.assertTrue(event["is_false_positive"])
        self.assertEqual(event["real_class"], "bovino")
        self._export()
        labels = self._labels()
        self.assertEqual(len(labels), 1, f"se esperaba 1 etiqueta: {labels}")
        line = list(labels.values())[0][0]
        parts = line.split()
        self.assertEqual(len(parts), 5,
                         f"una etiqueta positiva YOLO tiene 5 campos: {line!r}")
        class_id = int(parts[0])
        self.assertEqual(CLASSES[class_id], "bovino",
                         "la clase del id YOLO no corresponde a 'bovino'")
        x, y, w, h = (float(v) for v in parts[1:])
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    def test_empty_answer_is_treated_as_background(self):
        """El dialogo viene con '0' por defecto. Aceptar sin escribir se
        confirma como pulsacion implicita: el operador acepta el fondo."""
        self._ask("")
        event = self._event()
        self.assertTrue(event["is_false_positive"])
        self.assertIsNone(event["real_class"])

    def test_invalid_option_changes_nothing(self):
        """Una entrada invalida no puede marcar el evento: se avisaria al
        operador que se guardo una correccion que no existe."""
        for bad in ("9", "abc", "-1", "1.5"):
            with self.subTest(opcion=bad):
                self.setUp()
                self._ask(bad)
                event = self._event()
                self.assertFalse(event["acknowledged"])
                self.assertFalse(event["is_false_positive"])
                self.assertTrue(
                    any("invalida" in m for m in self._log()),
                    f"no se aviso de la opcion invalida: {self._log()}")

    def test_tkinter_failure_falls_back_to_background(self):
        """Sin display, o con tkinter roto, el operador debe poder confirmar
        la alerta igualmente. El fallback es fondo, que es una etiqueta
        valida."""
        with mock.patch("src.gui.tab_detection.tk.Tk",
                        side_effect=Exception("sin display")):
            self.view._ask_false_positive_reason()
        event = self._event()
        self.assertTrue(event["is_false_positive"])
        self.assertIsNone(event["real_class"])
        self.assertTrue(
            any("No se pudo abrir el dialogo" in m for m in self._log()),
            f"no se aviso del fallo del dialogo: {self._log()}")
        self._export()
        self.assertEqual(len(self._labels()), 1)

    def test_viewer_role_cannot_label_false_positives(self):
        """RE05: el Visualizador confirma alertas, pero etiquetar para
        reentrenar es del Administrador. El permiso se delega en el
        coordinator con la sesion de acceso."""
        self.view.app.access.can.return_value = False
        self.view.app.user = {"username": "operador", "role": "visualizador"}
        with mock.patch("src.gui.tab_detection.tk.Tk"), \
                mock.patch("src.gui.tab_detection.simpledialog.askstring",
                           return_value="0"):
            self.view._ask_false_positive_reason()
        event = self._event()
        self.assertFalse(event["acknowledged"],
                         "un Visualizador no deberia poder etiquetar")
        self.assertTrue(any("Permiso denegado" in m for m in self._log()),
                        f"no se aviso del permiso denegado: {self._log()}")

    def test_real_class_survives_a_later_plain_acknowledgement(self):
        """Confirmar despues, sin reetiquetar, no debe borrar la clase real
        ya declarada: el exportador la necesita para la caja."""
        self._ask("3")                      # 3 = equino
        self.assertEqual(self._event()["real_class"], "equino")
        self.db.mark_event_acknowledged(self.event_id, is_false_positive=True)
        self.assertEqual(self._event()["real_class"], "equino")
        self._export()
        labels = self._labels()
        line = list(labels.values())[0][0]
        self.assertEqual(CLASSES[int(line.split()[0])], "equino")


if __name__ == "__main__":
    unittest.main(verbosity=2)
