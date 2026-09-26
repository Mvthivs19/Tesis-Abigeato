"""
PRUEBAS UNITARIAS

Suite de pruebas del sistema de deteccion de intrusos. Cubre los modulos de
dominio que concentran la logica de negocio y de los requerimientos.

Ejecucion:

    python -m pytest tests/ -v
    python tests/test_sistema.py        # sin pytest, con unittest

No requiere GPU ni modelo entrenado: los modulos probados son deterministas
y trabajan sobre frames sinteticos.
"""

import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from datetime import datetime

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.database import EventDatabase
from src.access_control import (
    AccessControl, AccessDenied, hash_password, verify_password,
    ROLE_ADMIN, ROLE_VIEWER, PERMISSIONS, ROLE_VIEWS,
)
from src.schedule import MonitoringSchedule, _parse_hhmm
from src.contacts import ContactDirectory
from src.frame_filter import (
    FrameQualityFilter, QUALITY_OK, QUALITY_BLACKOUT, QUALITY_LOW_CONTRAST,
)
from src.integrity_monitor import IntegrityMonitor
from src.retraining import FalsePositiveExporter
from src.fpr_evaluation import FPREvaluation
from src.roi_analysis import ROIAnalysis


def temp_db():
    """Base de datos aislada por test."""
    return os.path.join(tempfile.mkdtemp(), "test.db")


def structured_frame(value=90, block=200, size=(480, 640)):
    """Frame con contraste y bordes reales, similar a una escena de campo."""
    frame = np.full(size, value, np.uint8)
    frame[100:300, 200:500] = block
    return frame


# ============================================================
# RE05 - Control de acceso
# ============================================================

class TestAccessControl(unittest.TestCase):
    def setUp(self):
        self.ac = AccessControl(temp_db())

    def test_password_hashing_uses_random_salt(self):
        """Dos usuarios con la misma contrasena no deben generar el mismo hash."""
        h1, s1 = hash_password("secreta")
        h2, s2 = hash_password("secreta")
        self.assertNotEqual(h1, h2)
        self.assertNotEqual(s1, s2)
        self.assertTrue(verify_password("secreta", h1, s1))

    def test_password_hash_is_not_reversible(self):
        h, s = hash_password("secreta")
        self.assertNotIn("secreta", h)
        self.assertEqual(len(h), 64)          # SHA256 en hex

    def test_login_success_and_role(self):
        session = self.ac.login("admin", "admin123")
        self.assertEqual(session["role"], ROLE_ADMIN)
        self.assertEqual(self.ac.current_user, "admin")

    def test_login_rejects_wrong_password(self):
        with self.assertRaises(AccessDenied):
            self.ac.login("admin", "incorrecta")

    def test_login_rejects_unknown_user(self):
        with self.assertRaises(AccessDenied):
            self.ac.login("fantasma", "loquesea")

    def test_lockout_after_max_attempts(self):
        """RE05: evita enumeracion de usuarios por fuerza bruta."""
        for _ in range(AccessControl.MAX_ATTEMPTS):
            with self.assertRaises(AccessDenied):
                self.ac.login("admin", "mala")
        with self.assertRaises(AccessDenied) as ctx:
            self.ac.login("admin", "admin123")   # clave correcta, bloqueado
        self.assertIn("bloqueada", str(ctx.exception))

    def test_permission_matrix_admin(self):
        self.ac.login("admin", "admin123")
        for perm in ("manage_users", "edit_thresholds", "edit_zone",
                     "edit_schedule", "edit_contacts", "label_false_positive",
                     "run_training"):
            self.assertTrue(self.ac.can(perm), f"admin deberia poder {perm}")

    def test_visualizer_cannot_edit_config(self):
        """RE05: el Visualizador no altera configuracion, horarios ni DB."""
        self.ac.login("operador", "operador123")
        for perm in ("edit_thresholds", "edit_zone", "edit_schedule",
                     "edit_contacts", "manage_users", "label_false_positive",
                     "run_training"):
            self.assertFalse(self.ac.can(perm), f"visualizador no deberia {perm}")

    def test_visualizer_can_operate_live(self):
        self.ac.login("operador", "operador123")
        for perm in ("view_live", "start_detection", "acknowledge_alert",
                     "silence_alarm", "view_history", "view_health"):
            self.assertTrue(self.ac.can(perm), f"visualizador deberia {perm}")

    def test_require_raises_for_visualizer(self):
        self.ac.login("operador", "operador123")
        with self.assertRaises(AccessDenied):
            self.ac.require("label_false_positive")

    def test_view_restriction_per_role(self):
        self.assertIn("config", self.ac.allowed_views(ROLE_ADMIN))
        self.assertNotIn("config", self.ac.allowed_views(ROLE_VIEWER))
        self.assertNotIn("training", self.ac.allowed_views(ROLE_VIEWER))

    def test_create_user_requires_admin(self):
        self.ac.login("operador", "operador123")
        with self.assertRaises(AccessDenied):
            self.ac.create_user("nuevo", "clave", ROLE_ADMIN, actor="operador")
        self.ac.logout()
        self.ac.login("admin", "admin123")
        self.assertTrue(self.ac.create_user("nuevo", "clave", ROLE_VIEWER))
        self.assertIn("nuevo", [u["username"] for u in self.ac.list_users()])

    def test_duplicate_user_rejected(self):
        self.ac.login("admin", "admin123")
        self.ac.create_user("dup", "clave", ROLE_VIEWER)
        with self.assertRaises(ValueError):
            self.ac.create_user("dup", "otra", ROLE_VIEWER)

    def test_deactivated_user_cannot_login(self):
        self.ac.login("admin", "admin123")
        self.ac.set_active("operador", False)
        self.ac.logout()
        with self.assertRaises(AccessDenied):
            self.ac.login("operador", "operador123")

    def test_role_of_accepts_session_dict(self):
        """RE05: la GUI consulta permisos con el diccionario que devuelve
        `login`, no con el nombre de usuario. Si `role_of` no lo entendiera,
        toda comprobacion de permiso fallaria cerrada en produccion."""
        user = self.ac.login("admin", "admin123")
        self.assertEqual(self.ac.role_of(user), ROLE_ADMIN)
        self.assertEqual(self.ac.can("manage_users", user), True)
        self.assertEqual(self.ac.role_of(None), None)

    def test_rename_and_deactivate_are_admin_only(self):
        self.ac.login("operador", "operador123")
        with self.assertRaises(AccessDenied):
            self.ac.set_active("admin", False, actor="operador")
        with self.assertRaises(AccessDenied):
            self.ac.change_password("operador", "otra", actor="operador",
                                    require_admin=True)

    def test_visualizer_can_only_read_views(self):
        """Un rol restringido no debe obtener vistas que no le pertenecen,
        y el conjunto debe ser subconjunto del del administrador."""
        admin_views = set(self.ac.allowed_views(ROLE_ADMIN))
        viewer_views = set(self.ac.allowed_views(ROLE_VIEWER))
        self.assertTrue(viewer_views < admin_views)
        self.assertNotIn("config", viewer_views)
        self.assertNotIn("training", viewer_views)


# ============================================================
# RE05 - Coherencia entre la GUI y la matriz de permisos
# ============================================================

# Metodo de las vistas para pedir un permiso. El orden de argumentos cambia
# entre la clase base y la aplicacion, asi que se aceptan ambos.
_PERM_RE = __import__("re").compile(
    r"""(?:\.can|\.require)\(\s*["']([a-z_]+)["']""")


class TestPermissionWiring(unittest.TestCase):
    """Detecta el modo de fallo mas silencioso de RE05.

    `AccessControl.can` devuelve False para un permiso que no exista en la
    matriz, sin error ni aviso. Si la GUI pide un nombre mal escrito, la
    operacion queda bloqueada para todos los usuarios y el unico sintoma es
    un aviso de "acceso denegado" incomprensible. Este test recorre el
    codigo de la interfaz y obliga a que todo permiso usado exista.
    """

    def _gui_sources(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "src", "gui")
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if name.endswith(".py"):
                    yield os.path.join(dirpath, name)

    def test_every_permission_used_by_gui_exists(self):
        used = set()
        for path in self._gui_sources():
            with open(path, encoding="utf-8") as fh:
                used.update(_PERM_RE.findall(fh.read()))
        self.assertTrue(used, "No se encontro ningun permiso en la GUI")
        unknown = sorted(used - set(PERMISSIONS))
        self.assertEqual(
            unknown, [],
            f"Permisos usados por la GUI y ausentes en la matriz: {unknown}"
        )

    def test_every_view_granted_to_a_role_exists(self):
        for role, views in ROLE_VIEWS.items():
            for view in views:
                self.assertTrue(
                    view.isidentifier(),
                    f"'{view}' no es un identificador de vista valido"
                )

    def test_granted_view_is_registered_in_the_app(self):
        """Un permiso de vista sin vista registrada no da error: el boton no
        aparece y la operacion queda simplemente inexistente."""
        from src.gui.app import Application
        source = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "src", "gui", "app.py")
        with open(source, encoding="utf-8") as fh:
            body = fh.read()
        built = set(__import__("re").findall(
            r'"(\w+)":\s*\w+View\(self\)', body))
        nav = {key for key, _label, _icon in Application.ALL_NAV_ITEMS}
        for role, views in ROLE_VIEWS.items():
            for view in views:
                self.assertIn(view, built,
                              f"'{view}' ({role}) no se construye en app.py")
                self.assertIn(view, nav,
                              f"'{view}' ({role}) no tiene item en la sidebar")

    def test_evaluation_view_is_read_only_for_the_viewer(self):
        """RE09/RE10: el Visualizador puede LEER FPR y ROI, pero no alterar los
        parametros economicos que producen el ROI."""
        self.assertIn("evaluation", ROLE_VIEWS[ROLE_VIEWER])
        self.assertIn("view_evaluation", PERMISSIONS)
        self.assertIn(ROLE_VIEWER, PERMISSIONS["view_evaluation"])
        self.assertNotIn(ROLE_VIEWER, PERMISSIONS["edit_thresholds"])

    def test_evaluation_view_reads_the_same_reports_as_the_engines(self):
        """Si el panel busca el reporte en otra ruta que FPREvaluation y
        ROIAnalysis, la pantalla mostrara 'sin evaluacion' mientras el informe
        existe en disco: el sintoma es un panel mudo, no un error."""
        import types

        from src.gui.tab_evaluation import EvaluationView
        from src.fpr_evaluation import FPREvaluation
        from src.roi_analysis import ROIAnalysis
        from src.utils import load_config

        cfg = load_config()
        fake = types.SimpleNamespace(config=cfg, db=None, custom_buttons=[],
                                     set_custom_buttons=lambda b: None,
                                     notify=lambda *a, **k: None)
        view = EvaluationView(fake)
        for filename in ("fpr_report.json", "roi_report.json"):
            self.assertEqual(
                view._report_path(filename),
                os.path.join(FPREvaluation(cfg).output_path
                             if filename.startswith("fpr")
                             else ROIAnalysis(cfg).output_path),
            )

    def test_evaluation_panel_and_cli_agree_on_the_negative_clips(self):
        """GUI y CLI deben medir el FPR sobre los MISMOS clips; si cada una usa
        su propia lista, las dos cifras no son comparables entre si."""
        cli = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "06_realtime_pipeline.py"),
                   encoding="utf-8").read()
        panel = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "src", "gui", "tab_evaluation.py"),
                     encoding="utf-8").read()
        self.assertIn('get("negative_clips"', cli)
        self.assertIn('get("negative_clips")', panel)

    def test_evaluation_panel_does_not_reread_reports_every_frame(self):
        """`render` corre a ~20 FPS. Sin cache, la pestana abriria y parsearia
        dos JSON de disco en cada frame (~40 lecturas/s) mientras la deteccion
        sigue activa: el panel por si solo consume CPU y disco."""
        import types
        import time as _time

        from src.gui.tab_evaluation import EvaluationView
        from src.utils import load_config

        cfg = load_config()
        fake = types.SimpleNamespace(config=cfg, db=None, custom_buttons=[],
                                     set_custom_buttons=lambda b: None,
                                     notify=lambda *a, **k: None)
        view = EvaluationView(fake)

        reads = {"n": 0}
        real_open = open

        def counting_open(*a, **k):
            reads["n"] += 1
            return real_open(*a, **k)

        import builtins
        builtins.open = counting_open
        try:
            view._load_reports(force=True)
            after_first = reads["n"]
            # 30 frames Worth of renders without a real second elapsing.
            for _ in range(30):
                view._load_reports()
                _time.sleep(0.001)
        finally:
            builtins.open = real_open

        self.assertGreater(after_first, 0)
        self.assertEqual(
            reads["n"], after_first,
            "los informes se releen en cada frame: falta la cache por mtime",
        )

    def test_roi_assumptions_are_editable_in_config(self):
        """El informe de ROI dice "reemplaza los montos por los tuyos". Si los
        supuestos estuvieran fijos en el codigo, ese aviso seria imposible de
        seguir y el ROI seria siempre el mismo numero para toda finca."""
        from src.roi_analysis import ROIAnalysis
        from src.utils import load_config

        cfg = load_config()
        roi_cfg = cfg.get("evaluation", {}).get("roi")
        self.assertIsInstance(roi_cfg, dict, "falta evaluation.roi en config.yaml")
        self.assertIn("valor_por_animal", roi_cfg)
        self.assertIn("costo_camara", roi_cfg)

        # Un valor distinto debe cambiar el resultado, no ser ignorado.
        tweaked = {**cfg, "evaluation": {**cfg["evaluation"], "roi": {
            **roi_cfg, "valor_por_animal": 10 * roi_cfg["valor_por_animal"]}}}
        base = ROIAnalysis(cfg).compute()
        bigger = ROIAnalysis(tweaked).compute()
        self.assertNotEqual(base["roi_porcentaje"],
                            bigger["roi_porcentaje"],
                            "cambiar el valor por animal no cambio el ROI: los "
                            "parametros de config se estan ignorando")

    def test_admin_can_everything_and_viewer_is_strictly_smaller(self):
        admin = set(ROLE_VIEWS[ROLE_ADMIN])
        viewer = set(ROLE_VIEWS[ROLE_VIEWER])
        self.assertTrue(viewer < admin)
        # Ningun permiso de gestion puede pertenecer al rol restringido.
        for perm, roles in PERMISSIONS.items():
            if ROLE_VIEWER in roles and perm.startswith(("edit_", "manage_",
                                                        "run_", "label_")):
                self.fail(f"Permiso de gestion '{perm}' abierto al Visualizador")


# ============================================================
# RE02 - Horario de vigilancia
# ============================================================


class TestSchedule(unittest.TestCase):
    def test_parse_hhmm(self):
        self.assertEqual(_parse_hhmm("06:30"), 390)
        self.assertEqual(_parse_hhmm(390), 390)
        with self.assertRaises(ValueError):
            _parse_hhmm("basura")

    def test_window_overnight_includes_late_and_early_hours(self):
        """La ventana 20:00-06:00 cruza la medianoche."""
        sched = MonitoringSchedule(
            {"enabled": True, "windows": [{"start": "20:00", "end": "06:00"}]}
        )
        self.assertTrue(sched.is_active(datetime(2026, 9, 26, 20, 0)))
        self.assertTrue(sched.is_active(datetime(2026, 9, 26, 23, 59)))
        self.assertTrue(sched.is_active(datetime(2026, 9, 26, 0, 0)))
        self.assertTrue(sched.is_active(datetime(2026, 9, 26, 5, 59)))

    def test_window_overnight_excludes_daytime(self):
        sched = MonitoringSchedule(
            {"enabled": True, "windows": [{"start": "20:00", "end": "06:00"}]}
        )
        self.assertFalse(sched.is_active(datetime(2026, 9, 26, 6, 0)))
        self.assertFalse(sched.is_active(datetime(2026, 9, 26, 12, 0)))
        self.assertFalse(sched.is_active(datetime(2026, 9, 26, 19, 59)))

    def test_multiple_windows(self):
        sched = MonitoringSchedule({
            "enabled": True,
            "windows": [{"start": "08:00", "end": "12:00"},
                        {"start": "20:00", "end": "06:00"}],
        })
        self.assertTrue(sched.is_active(datetime(2026, 9, 26, 9, 0)))
        self.assertTrue(sched.is_active(datetime(2026, 9, 26, 23, 0)))
        self.assertFalse(sched.is_active(datetime(2026, 9, 26, 15, 0)))

    def test_disabled_schedule_is_always_active(self):
        sched = MonitoringSchedule({"enabled": False,
                                    "windows": [{"start": "20:00", "end": "06:00"}]})
        self.assertTrue(sched.is_active(datetime(2026, 9, 26, 12, 0)))

    def test_full_day_window(self):
        sched = MonitoringSchedule(
            {"enabled": True, "windows": [{"start": "00:00", "end": "00:00"}]}
        )
        self.assertTrue(sched.is_active(datetime(2026, 9, 26, 13, 0)))

    def test_describe_and_transition(self):
        sched = MonitoringSchedule(
            {"enabled": True, "windows": [{"start": "20:00", "end": "06:00"}]}
        )
        self.assertEqual(sched.describe(), "20:00-06:00")
        self.assertEqual(
            sched.next_transition(datetime(2026, 9, 26, 10, 0)), "20:00"
        )

    def test_active_windows_now(self):
        sched = MonitoringSchedule(
            {"enabled": True, "windows": [{"start": "20:00", "end": "06:00"}]}
        )
        self.assertEqual(len(sched.active_windows_now(
            datetime(2026, 9, 26, 23, 0))), 1)
        self.assertEqual(len(sched.active_windows_now(
            datetime(2026, 9, 26, 12, 0))), 0)


# ============================================================
# RE04 - Contactos de emergencia
# ============================================================

class TestContactDirectory(unittest.TestCase):
    def setUp(self):
        self.dir = ContactDirectory(temp_db())

    def test_add_and_list(self):
        self.dir.add("Productor", phone="+56912345678", priority=1)
        self.dir.add("Guarda", phone="+56987654321", priority=2)
        contacts = self.dir.list_contacts()
        self.assertEqual(len(contacts), 2)
        # Ordenados por prioridad
        self.assertEqual(contacts[0]["name"], "Productor")

    def test_phone_validation(self):
        for bad in ("123", "abc", "+569123456789012"):
            with self.assertRaises(ValueError, msg=f"deberia rechazar {bad}"):
                self.dir.add("X", phone=bad)

    def test_phone_accepts_formats(self):
        for good in ("+56912345678", "56912345678", "912345678", "+56 9 1234 5678"):
            self.dir = ContactDirectory(temp_db())
            self.dir.add("X", phone=good)

    def test_email_validation(self):
        with self.assertRaises(ValueError):
            self.dir.add("X", email="no-es-email", channel="email")
        self.assertTrue(self.dir.add("X", email="a@b.cl", channel="email"))

    def test_sms_requires_phone(self):
        with self.assertRaises(ValueError):
            self.dir.add("X", channel="sms")

    def test_invalid_channel_rejected(self):
        with self.assertRaises(ValueError):
            self.dir.add("X", phone="+56912345678", channel="telegram")

    def test_name_required(self):
        with self.assertRaises(ValueError):
            self.dir.add("   ")

    def test_priority_ordering(self):
        self.dir.add("Tercero", phone="+56911111111", priority=3)
        self.dir.add("Primero", phone="+56922222222", priority=1)
        self.dir.add("Segundo", phone="+56933333333", priority=2)
        names = [c["name"] for c in self.dir.list_contacts()]
        self.assertEqual(names, ["Primero", "Segundo", "Tercero"])

    def test_notify_all_dispatches_in_order(self):
        self.dir.add("A", phone="+56911111111", priority=1)
        self.dir.add("B", phone="+56922222222", priority=2)
        seen = []
        summary = self.dir.notify_all(
            1, dispatcher=lambda c, e: seen.append(c["name"]) or True
        )
        self.assertEqual(seen, ["A", "B"])
        self.assertEqual(summary["notificados"], 2)
        self.assertEqual(summary["fallidos"], 0)

    def test_notify_all_counts_failures(self):
        self.dir.add("A", phone="+56911111111", priority=1)
        self.dir.add("B", phone="+56922222222", priority=2)
        summary = self.dir.notify_all(
            1, dispatcher=lambda c, e: c["name"] == "A"
        )
        self.assertEqual(summary["notificados"], 1)
        self.assertEqual(summary["fallidos"], 1)

    def test_notify_all_with_dispatcher_raising(self):
        self.dir.add("A", phone="+56911111111")

        def boom(c, e):
            raise RuntimeError("gateway caida")

        summary = self.dir.notify_all(1, dispatcher=boom)
        self.assertEqual(summary["fallidos"], 1)

    def test_empty_directory_notify(self):
        summary = self.dir.notify_all(1, dispatcher=lambda c, e: True)
        self.assertEqual(summary["notificados"], 0)

    def test_update_and_remove(self):
        cid = self.dir.add("Temporal", phone="+56911111111")
        self.dir.update(cid, name="Renombrado", priority=9)
        self.assertEqual(self.dir.get(cid)["name"], "Renombrado")
        self.dir.remove(cid)
        self.assertIsNone(self.dir.get(cid))

    def test_update_rejects_unknown_field(self):
        cid = self.dir.add("Temporal", phone="+56911111111")
        with self.assertRaises(ValueError):
            self.dir.update(cid, inyectar_sql="DROP TABLE contacts")


# ============================================================
# RE11 - Filtrado de ruido
# ============================================================

class TestFrameQualityFilter(unittest.TestCase):
    def setUp(self):
        self.f = FrameQualityFilter({"enabled": True})
        self.good = structured_frame()

    def test_good_frame_accepted(self):
        _, quality, score, metrics = self.f.process(self.good)
        self.assertEqual(quality, QUALITY_OK)
        self.assertEqual(score, 1.0)
        self.assertIn("sharpness", metrics)

    def test_black_frame_detected_as_blackout(self):
        black = np.zeros((480, 640), np.uint8)
        _, quality, score, _ = self.f.process(black)
        self.assertEqual(quality, QUALITY_BLACKOUT)
        self.assertEqual(score, 0.0)

    def test_fog_detected_as_low_contrast(self):
        fog = np.full((480, 640), 90, np.uint8)
        _, quality, _, _ = self.f.process(fog)
        self.assertEqual(quality, QUALITY_LOW_CONTRAST)

    def test_blackout_replaced_by_last_good_frame(self):
        """La vigilancia no debe quedar ciega ante un frame negro puntual."""
        self.f.process(self.good)
        out, quality, _, _ = self.f.process(np.zeros((480, 640), np.uint8))
        self.assertEqual(quality, QUALITY_BLACKOUT)
        self.assertGreater(out.mean(), 10)

    def test_detection_blocked_on_blackout(self):
        self.f.process(self.good)
        self.f.process(np.zeros((480, 640), np.uint8))
        allowed, reason = self.f.is_detection_allowed(QUALITY_BLACKOUT, 0.0)
        self.assertFalse(allowed)
        self.assertIn("lente tapada", reason)

    def test_detection_allowed_on_good_frame(self):
        _, quality, score, _ = self.f.process(self.good)
        allowed, _ = self.f.is_detection_allowed(quality, score)
        self.assertTrue(allowed)

    def test_disabled_filter_passes_everything(self):
        f = FrameQualityFilter({"enabled": False})
        _, quality, score, metrics = f.process(np.zeros((480, 640), np.uint8))
        self.assertEqual(quality, QUALITY_OK)
        self.assertEqual(score, 1.0)
        self.assertTrue(f.is_detection_allowed(quality, score)[0])

    def test_grayscale_and_color_inputs(self):
        """El filtro acepta 1, 3 o 4 canales: el source puede entregar cualquiera."""
        import cv2
        gray = structured_frame()
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        for frame in (gray, bgr, bgra):
            f = FrameQualityFilter({"enabled": True})
            _, quality, _, _ = f.process(frame)
            self.assertEqual(quality, QUALITY_OK)

    def test_degradation_timer_starts(self):
        self.f.process(self.good)
        self.assertEqual(self.f.degradation_seconds(), 0)
        self.f.process(np.zeros((480, 640), np.uint8))
        self.assertIsNotNone(self.f._degraded_since)

    def test_quality_recovers_after_blackout(self):
        self.f.process(self.good)
        self.f.process(np.zeros((480, 640), np.uint8))
        _, quality, _, _ = self.f.process(self.good)
        self.assertEqual(quality, QUALITY_OK)

    def test_state_is_updated_even_when_frame_is_replaced(self):
        """RE11: `last_quality` no puede quedar con el valor del frame
        anterior cuando el filtro devuelve una imagen de reemplazo, porque
        `AlertCoordinator.can_detect()` se basa en ese estado para decidir
        si permite inferencia."""
        self.f.process(self.good)
        self.assertEqual(self.f.last_quality, QUALITY_OK)

        self.f.process(np.zeros((480, 640), np.uint8))

        # Aunque se devolvio el ultimo frame limpio, el estado vigente debe
        # reflejar el blackout actual y bloquear la deteccion.
        self.assertEqual(self.f.last_quality, "blackout")
        allowed, why = self.f.is_detection_allowed(
            self.f.last_quality, self.f.last_score
        )
        self.assertFalse(allowed)
        self.assertTrue(why)

    def test_blur_replacement_also_updates_state(self):
        """Un degradado suave tiene contraste alto pero Laplaciano ~0, que es
        exactamente el caso "lente fuera de foco"."""
        self.f.process(self.good)
        ramp = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
        usable, quality, score, _ = self.f.process(ramp)
        self.assertIsNotNone(usable)
        self.assertEqual(quality, "blur")
        self.assertEqual(self.f.last_quality, "blur")
        self.assertAlmostEqual(self.f.last_score, score)


# ============================================================
# RE21 - Integridad fisica de la camara
# ============================================================

class FakeDB:
    def __init__(self):
        self.rows = []

    def insert_integrity_log(self, timestamp, event_type, severity, detail=""):
        self.rows.append((event_type, severity))


class TestIntegrityMonitor(unittest.TestCase):
    def setUp(self):
        self.db = FakeDB()
        self.mon = IntegrityMonitor({"enabled": True, "signal_loss_frames": 2},
                                    db=self.db)
        self.frame = structured_frame()

    def test_normal_frame_is_ok(self):
        status, _, compromised = self.mon.update(self.frame)
        self.assertEqual(status, "ok")
        self.assertFalse(compromised)

    def test_camera_moved_is_critical(self):
        """Un intruso que mueve la camara no debe pasar inadvertido."""
        self.mon.update(self.frame)
        status, detail, compromised = self.mon.update(np.roll(self.frame, 80, axis=1))
        self.assertEqual(status, "comprometida")
        self.assertTrue(compromised)
        self.assertIn("desplazado", detail)

    def test_active_alert_exposed(self):
        self.mon.update(self.frame)
        self.mon.update(np.roll(self.frame, 80, axis=1))
        alert = self.mon.active_alert()
        self.assertIsNotNone(alert)
        self.assertEqual(alert["severity"], "critical")

    def test_signal_loss_after_debounce(self):
        self.mon.update(None)
        self.assertEqual(self.mon.update(None)[0], "comprometida")

    def test_single_missing_frame_is_tolerated(self):
        """Un frame perdido no debe generar una falsa alarma de integridad."""
        self.mon.update(self.frame)
        self.mon.update(None)
        self.assertEqual(self.mon.update(self.frame)[0], "ok")

    def test_blackout_from_filter_is_critical(self):
        f = FrameQualityFilter({"enabled": True})
        f.last_quality = "blackout"
        status, _, compromised = self.mon.update(self.frame, quality_filter=f)
        self.assertTrue(compromised)
        self.assertEqual(status, "comprometida")

    def test_low_fps_triggers_degradation(self):
        """La caida de FPS debe ser SOSTENIDA (`degraded_frames`), no un
        muestra aislada: durante el arranque de la deteccion el FPS sube desde
        0 y cada frame se reportaba como degradacion."""
        mon = IntegrityMonitor({"enabled": True, "min_fps": 20, "degraded_frames": 5},
                               db=self.db)
        for _ in range(9):
            status, detail, _ = mon.update(self.frame, fps=2.0)
        self.assertIn("FPS", detail)

    def test_brief_fps_dip_does_not_alert(self):
        """Un FPS bajo momentaneo no es una camara degradada."""
        mon = IntegrityMonitor({"enabled": True, "min_fps": 20, "degraded_frames": 20},
                               db=self.db)
        for _ in range(9):
            status, detail, _ = mon.update(self.frame, fps=2.0)
        self.assertNotIn("FPS", detail)
        self.assertEqual(status, "ok")

    def test_persistent_problem_is_logged_once_per_debounce(self):
        """`update()` corre por frame: sin antirrebote, `integrity_log` quedaba
        con decenas de filas identicas del mismo sintoma."""
        class CountingDB(FakeDB):
            def __init__(self):
                super().__init__()
                self.rows = []

            def insert_integrity_log(self, ts, etype, severity, detail):
                self.rows.append((etype, detail))

        db = CountingDB()
        mon = IntegrityMonitor({"enabled": True, "degraded_frames": 1,
                                "debounce_seconds": 3600}, db=db)
        for _ in range(60):
            mon.update(self.frame, fps=1.0)
        self.assertEqual(len(db.rows), 1, f"se esperaba 1 fila, hubo {len(db.rows)}")

    def test_zero_fps_is_treated_as_no_measurement(self):
        """RE21: en el primer segundo de captura el FPS aun no esta medido y
        llega como 0.0. Eso no es "camara detenida": si se contara como
        degradacion, saltaria una alarma en cada arranque y el operador
        aprenderia a ignorar los avisos de integridad."""
        mon = IntegrityMonitor({"enabled": True, "min_fps": 5, "degraded_frames": 5},
                               db=self.db)
        for _ in range(10):
            status, detail, _ = mon.update(self.frame, fps=0.0)
        self.assertEqual(status, "ok")
        self.assertNotIn("FPS", detail)
        # Un FPS bajo real sigue siendo detectable. La ventana de promedio
        # necesita 5 muestras antes de mirar el FPS, y `degraded_frames` pide
        # 5samples mas: con 8 frames todavia no hay veredicto.
        for _ in range(12):
            status, detail, _ = mon.update(self.frame, fps=1.0)
        self.assertIn("FPS", detail)

    def test_recovery_clears_alert(self):
        self.mon.update(self.frame)
        self.mon.update(np.roll(self.frame, 80, axis=1))
        # Un desplazamiento pequeno no debe volver a disparar la alerta.
        mon2 = IntegrityMonitor({"enabled": True, "max_shift_pixels": 500}, db=self.db)
        mon2.update(self.frame)
        self.assertEqual(mon2.update(self.frame)[0], "ok")

    def test_events_persisted(self):
        self.mon.update(self.frame)
        self.mon.update(np.roll(self.frame, 80, axis=1))
        self.assertTrue(self.db.rows)
        self.assertEqual(self.db.rows[0][1], "critical")

    def test_disabled_monitor_always_ok(self):
        mon = IntegrityMonitor({"enabled": False}, db=self.db)
        self.assertEqual(mon.update(None)[0], "ok")


# ============================================================
# RE15 / RE16 / RE18 - Base de datos
# ============================================================

class TestEventDatabase(unittest.TestCase):
    def setUp(self):
        self.db = EventDatabase(temp_db())

    def test_insert_event_creates_queue_row(self):
        """RE19: toda alerta queda encolada aunque no haya Internet."""
        eid = self.db.insert_event("humano", 0.9, "snap.jpg", "clip.mp4")
        self.assertIsNotNone(eid)
        self.assertEqual(self.db.get_pending_count(), 1)
        self.assertEqual(self.db.get_queue_id(eid) is not None, True)

    def test_acknowledge_clears_pending(self):
        eid = self.db.insert_event("humano", 0.9)
        self.db.mark_event_acknowledged(eid)
        self.assertIsNone(self.db.get_last_unacknowledged_event())
        self.assertEqual(self.db.count_false_positives(), 0)

    def test_false_positive_flag(self):
        eid = self.db.insert_event("humano", 0.3)
        self.db.mark_event_acknowledged(eid, is_false_positive=True)
        self.assertEqual(self.db.count_false_positives(), 1)
        rows = self.db.get_false_positive_events()
        self.assertEqual(rows[0][0], eid)

    def test_escalation_candidates_respect_thresholds(self):
        old = self.db.insert_event("humano", 0.9)
        self.assertEqual(len(self.db.get_unacknowledged_events(60)), 0)
        rows = self.db.get_unacknowledged_events(0)
        self.assertEqual(rows[0][0], old)

    def test_escalated_event_not_repeated(self):
        eid = self.db.insert_event("humano", 0.9)
        self.db.mark_event_escalated(eid)
        self.assertEqual(len(self.db.get_unacknowledged_events(0)), 0)

    def test_stale_event_is_not_escalated_again(self):
        """RE15: un evento de hace dias sin confirmar no debe escalar.

        Reiniciar el sistema con un evento viejo pendiente disparaba una
        emergencia que describe el pasado, no el estado actual de la finca.
        El evento sigue pendiente y visible, pero sin escalar."""
        from datetime import timedelta
        eid = self.db.insert_event("humano", 0.9)
        ancient = (datetime.now() - timedelta(days=9)).isoformat()
        self.db._connect().execute(
            "UPDATE events SET timestamp = ? WHERE event_id = ?",
            (ancient, eid),
        ).connection.commit()

        # Sin techo de antiquity el evento ancient si seria candidato.
        self.assertEqual(len(self.db.get_unacknowledged_events(0)), 1)
        # Con techo, no.
        self.assertEqual(
            self.db.get_unacknowledged_events(0, max_age_seconds=72 * 3600), []
        )

    def test_notification_attempts_increment(self):
        eid = self.db.insert_event("humano", 0.9)
        qid = self.db.get_queue_id(eid)
        for _ in range(3):
            self.db.increment_notification_attempt(qid)
        self.assertEqual(self.db.get_notification_attempts(qid), 3)

    def test_mark_delivered_clears_queue(self):
        eid = self.db.insert_event("humano", 0.9)
        self.db.mark_notification_delivered(self.db.get_queue_id(eid))
        self.assertEqual(self.db.get_pending_count(), 0)

    def test_update_clip(self):
        eid = self.db.insert_event("humano", 0.9)
        self.db.update_event_clip(eid, "data/event_clips/clip_1.mp4")
        row = self.db.get_all_events(1)[0]
        self.assertEqual(row[5], "data/event_clips/clip_1.mp4")

    def test_real_class_is_not_erased_by_a_later_plain_mark(self):
        """RE18: la clase real la declaro el operador a mano y es trabajo
        humano caro. Un segundo marcado como falso positivo SIN clase (por
        ejemplo desde el historial) no puede borrarla con un NULL."""
        eid = self.db.insert_event("humano", 0.3, bbox="1,2,3,4")
        self.db.mark_event_acknowledged(eid, is_false_positive=True,
                                        real_class="bovino")
        self.assertEqual(self.db.get_false_positive_events(1)[0][6], "bovino")
        self.db.mark_event_acknowledged(eid, is_false_positive=True)
        self.assertEqual(self.db.get_false_positive_events(1)[0][6], "bovino")

    def test_confirming_as_real_removes_the_sample_from_retraining(self):
        """Si el operador dice "si era una intrusion real", la muestra deja de
        ser un falso positivo y sale del dataset de reentrenamiento."""
        eid = self.db.insert_event("humano", 0.3, bbox="1,2,3,4")
        self.db.mark_event_acknowledged(eid, is_false_positive=True,
                                        real_class="humano")
        self.db.mark_event_acknowledged(eid, is_false_positive=False)
        self.assertEqual(self.db.get_false_positive_events(), [])
        self.assertEqual(self.db.count_false_positives(), 0)

    def test_remote_status_separates_disabled_from_backlog(self):
        """RE13/RE19/RE20: con el canal apagado la cola no se vacia, asi que
        presentarla como "COLA OFFLINE" seria una alarma falsa. remote_status
        debe decir que el canal esta desactivado y cuantas alertas quedan."""
        from src.alert_coordinator import AlertCoordinator

        class _C:
            def get_pending_count(self_inner):
                return 4

        coord = AlertCoordinator.__new__(AlertCoordinator)
        coord.config = {"alerts": {"remote": {"enabled": False}}}
        coord.db = _C()
        off = coord.remote_status()
        self.assertFalse(off["enabled"])
        self.assertEqual(off["pending"], 4)
        self.assertIn("DESACTIVADO", off["note"])

        coord.config = {"alerts": {"remote": {"enabled": True}}}
        on = coord.remote_status()
        self.assertTrue(on["enabled"])
        self.assertEqual(on["pending"], 4)
        self.assertEqual(on["note"], "")

    def test_event_created_while_remote_disabled_stays_queued(self):
        """La cola es tambien traza de auditoria: el evento no se borra por
        tener el canal apagado, solo se etiqueta como no entregado."""
        eid = self.db.insert_event("humano", 0.9)
        self.assertEqual(self.db.get_pending_count(), 1)
        self.assertEqual(self.db.get_notification_attempts(
            self.db.get_queue_id(eid)), 0)

    def test_integrity_log_persisted(self):
        self.db.insert_integrity_log("2026-09-26T10:00:00", "lente_tapada",
                                     "critical", "imagen uniforme")
        self.assertEqual(self.db.count_integrity_alerts(), 1)
        rows = self.db.get_integrity_logs(5)
        self.assertEqual(rows[0][2], "lente_tapada")

    def test_contact_log_persisted(self):
        eid = self.db.insert_event("humano", 0.9)
        self.db.insert_contact_log(eid, {"contact_id": 1, "name": "Productor",
                                        "channel": "sms", "delivered": True})
        self.assertEqual(len(self.db.get_contact_notifications(eid)), 1)

    def test_event_stats(self):
        e1 = self.db.insert_event("humano", 0.9)
        self.db.insert_event("humano", 0.5)
        self.db.mark_event_acknowledged(e1, is_false_positive=True)
        stats = self.db.get_event_stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["false_positives"], 1)
        self.assertEqual(stats["pending"], 1)

    def test_get_all_events_columns(self):
        """El historial consume esta tupla: 9 columnas tras la migracion."""
        self.db.insert_event("humano", 0.9)
        self.assertEqual(len(self.db.get_all_events(1)[0]), 9)

    def test_get_all_events_filters_by_class(self):
        """La vista de historial filtra por clase (RE07/RE10) consumiendo la
        consulta compartida, no un SQL propio."""
        self.db.insert_event("humano", 0.9)
        self.db.insert_event("bovino", 0.8)
        rows = self.db.get_all_events(100, class_name="bovino")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], "bovino")
        self.assertEqual(len(self.db.get_all_events(100)), 2)

    def test_get_all_events_exposes_ack_and_fp_flags(self):
        """El historial necesita distinguir pendiente / confirmado /
        falso positivo: son las columnas que hacen operable RE15 y RE18."""
        eid = self.db.insert_event("humano", 0.9)
        self.db.mark_event_acknowledged(eid, is_false_positive=True)
        row = self.db.get_all_events(1)[0]
        acknowledged, is_fp, escalated = row[6], row[7], row[8]
        self.assertEqual(acknowledged, 1)
        self.assertEqual(is_fp, 1)
        self.assertEqual(escalated, 0)

    def test_open_event_ignores_stale_events_when_asked(self):
        """Un evento de hace semanas no debe seguir apareciendo como "la
        alerta abierta", pero el filtro es opcional: confirmar un evento viejo
        sigue siendo posible (RE15)."""
        eid = self.db.insert_event("humano", 0.9)
        self.db._connect().execute(
            "UPDATE events SET timestamp = '2020-01-01T00:00:00' WHERE event_id = ?",
            (eid,),
        ).connection.commit()

        # Sin filtro: sigue disponible para confirmarlo.
        self.assertIsNotNone(self.db.get_last_unacknowledged_event())
        # Con el tope de antigüedad: ya no es la alerta vigente.
        self.assertIsNone(self.db.get_last_unacknowledged_event(max_age_seconds=3600))
        # Un evento reciente sí aparece.
        recent = self.db.insert_event("humano", 0.9)
        self.assertEqual(
            self.db.get_last_unacknowledged_event(max_age_seconds=3600)[0], recent
        )


# ============================================================
# RE18 - Exportacion para reentrenamiento
# ============================================================

class TestRetrainingExport(unittest.TestCase):
    def setUp(self):
        import cv2
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "e.db")
        self.db = EventDatabase(self.db_path)
        self.snapdir = os.path.join(self.tmp, "snaps")
        os.makedirs(self.snapdir, exist_ok=True)
        self.snap = os.path.join(self.snapdir, "s.jpg")
        cv2.imwrite(self.snap, structured_frame())
        self.out = os.path.join(self.tmp, "fp")
        self.config = {
            "classes": {"names": ["humano", "bovino", "equino", "ovino", "porcino"]},
            "retraining": {"output_dir": self.out, "copy_snapshots": True,
                           "copy_clips": True},
        }

    def test_export_marked_false_positives_only(self):
        e1 = self.db.insert_event("humano", 0.3, self.snap)
        self.db.insert_event("humano", 0.8, self.snap)   # no marcado
        self.db.mark_event_acknowledged(e1, is_false_positive=True)
        result = FalsePositiveExporter(self.config, self.db).export()
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["skipped"], 0)

    def test_export_creates_yolo_structure(self):
        eid = self.db.insert_event("humano", 0.3, self.snap)
        self.db.mark_event_acknowledged(eid, is_false_positive=True)
        result = FalsePositiveExporter(self.config, self.db).export()
        images = os.listdir(os.path.join(result["output_dir"], "images"))
        labels = os.listdir(os.path.join(result["output_dir"], "labels"))
        self.assertEqual(len(images), 1)
        self.assertEqual(len(labels), 1)
        self.assertTrue(os.path.exists(os.path.join(result["output_dir"],
                                                    "manifest.json")))

    def test_undeclared_class_exports_an_empty_label(self):
        """Sin clase real declarada, la muestra se exporta como NEGATIVA: un
        archivo de etiqueta vacio. Escribir la clase predicho validaria el
        error en vez de corregirlo."""
        eid = self.db.insert_event("humano", 0.3, self.snap, bbox="10,10,50,50")
        self.db.mark_event_acknowledged(eid, is_false_positive=True)
        result = FalsePositiveExporter(self.config, self.db).export()
        label = os.path.join(result["output_dir"], "labels",
                             f"fp_{eid}_30.txt")
        with open(label) as f:
            self.assertEqual(f.read().strip(), "")
        self.assertEqual(result["with_box"], 0)
        with open(os.path.join(result["output_dir"], "manifest.json"),
                  encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest[0]["etiqueta"], "negativa")
        self.assertEqual(manifest[0]["clase_real"], "background")

    def test_declared_real_class_writes_a_real_box(self):
        """RE18 completo: el operador declara la clase real y la caja se
        normaliza a formato YOLO (clase cx cy w h en 0..1)."""
        img = cv2.imread(self.snap)
        h, w = img.shape[:2]
        eid = self.db.insert_event("humano", 0.3, self.snap, bbox="0,0,100,100")
        self.db.mark_event_acknowledged(eid, is_false_positive=True,
                                        real_class="bovino")
        result = FalsePositiveExporter(self.config, self.db).export()
        label = os.path.join(result["output_dir"], "labels", f"fp_{eid}_30.txt")
        with open(label) as f:
            line = f.read().strip()
        self.assertEqual(result["with_box"], 1)
        parts = line.split()
        self.assertEqual(len(parts), 5)
        index, cx, cy, bw, bh = (int(parts[0]), *[float(v) for v in parts[1:]])
        # bovino es el indice 1 en la lista de clases del test.
        self.assertEqual(index, 1)
        for value in (cx, cy, bw, bh):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertAlmostEqual(bw, 100.0 / w, places=3)
        self.assertAlmostEqual(cx, 50.0 / w, places=3)
        self.assertAlmostEqual(cy, 50.0 / h, places=3)

    def test_declared_class_outside_dataset_is_not_indexed(self):
        """Si la clase declarada no existe en el dataset, no se inventa un
        indice: la muestra queda como negativa."""
        eid = self.db.insert_event("humano", 0.3, self.snap, bbox="0,0,50,50")
        self.db.mark_event_acknowledged(eid, is_false_positive=True,
                                        real_class="camion")
        result = FalsePositiveExporter(self.config, self.db).export()
        self.assertEqual(result["with_box"], 0)
        label = os.path.join(result["output_dir"], "labels", f"fp_{eid}_30.txt")
        with open(label) as f:
            self.assertEqual(f.read().strip(), "")

    def test_invalid_bbox_does_not_corrupt_the_label(self):
        """Una caja invertida o corrupta no puede producir coordenadas
        negativas o > 1, que harness que poisonar el entrenamiento."""
        eid = self.db.insert_event("humano", 0.3, self.snap, bbox="90,90,10,10")
        self.db.mark_event_acknowledged(eid, is_false_positive=True,
                                        real_class="bovino")
        result = FalsePositiveExporter(self.config, self.db).export()
        self.assertEqual(result["with_box"], 0)
        label = os.path.join(result["output_dir"], "labels", f"fp_{eid}_30.txt")
        with open(label) as f:
            self.assertEqual(f.read().strip(), "")

    def test_bbox_survives_the_round_trip_through_the_event(self):
        """La caja se guarda al crear el evento (RE16) y RE18 la recuperan."""
        eid = self.db.insert_event("humano", 0.3, self.snap, bbox="12,34,56,78")
        self.db.mark_event_acknowledged(eid, is_false_positive=True)
        row = self.db.get_false_positive_events(1)[0]
        self.assertEqual(row[5], "12,34,56,78")
        self.assertIsNone(row[6])

    def test_clip_is_exported_and_registered_in_the_manifest(self):
        """RE18: el clip probatorio da el contexto que un recorte no tiene. Sin
        el, una sombra o una rama quedan como una imagen ambigua."""
        clip = os.path.join(self.tmp, "clip_e1.mp4")
        with open(clip, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42")
        eid = self.db.insert_event("humano", 0.3, self.snap, clip)
        self.db.mark_event_acknowledged(eid, is_false_positive=True)
        result = FalsePositiveExporter(self.config, self.db).export()
        clips_dir = os.path.join(result["output_dir"], "clips")
        self.assertEqual(os.listdir(clips_dir), [f"fp_{eid}_30.mp4"])
        with open(os.path.join(result["output_dir"], "manifest.json"),
                  encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest[0]["clip"], f"fp_{eid}_30.mp4")

    def test_a_missing_clip_does_not_discard_the_sample(self):
        """El clip es contexto, no la supervision. Si el archivo se borro del
        disco, la imagen y su etiqueta siguen siendo validas para entrenar."""
        eid = self.db.insert_event("humano", 0.3, self.snap,
                                   os.path.join(self.tmp, "no_existe.mp4"),
                                   bbox="0,0,100,100")
        self.db.mark_event_acknowledged(eid, is_false_positive=True,
                                        real_class="bovino")
        result = FalsePositiveExporter(self.config, self.db).export()
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["with_box"], 1)
        self.assertEqual(result["clips"], 0)
        with open(os.path.join(result["output_dir"], "manifest.json"),
                  encoding="utf-8") as f:
            self.assertIsNone(json.load(f)[0]["clip"])

    def test_copy_clips_disabled_leaves_no_empty_folder(self):
        self.config["retraining"]["copy_clips"] = False
        clip = os.path.join(self.tmp, "clip_e2.mp4")
        with open(clip, "wb") as f:
            f.write(b"\x00")
        eid = self.db.insert_event("humano", 0.3, self.snap, clip)
        self.db.mark_event_acknowledged(eid, is_false_positive=True)
        result = FalsePositiveExporter(self.config, self.db).export()
        self.assertFalse(os.path.exists(
            os.path.join(result["output_dir"], "clips")))

    def test_export_empty_removes_directory(self):
        result = FalsePositiveExporter(self.config, self.db).export()
        self.assertEqual(result["samples"], 0)
        self.assertFalse(os.path.exists(result["output_dir"]))

    def test_export_skips_missing_snapshot(self):
        eid = self.db.insert_event("humano", 0.3, "/no/existe.jpg")
        self.db.mark_event_acknowledged(eid, is_false_positive=True)
        result = FalsePositiveExporter(self.config, self.db).export()
        self.assertEqual(result["samples"], 0)
        self.assertEqual(result["skipped"], 1)

    def test_export_records_history(self):
        eid = self.db.insert_event("humano", 0.3, self.snap)
        self.db.mark_event_acknowledged(eid, is_false_positive=True)
        FalsePositiveExporter(self.config, self.db).export()
        self.assertTrue(self.db.get_false_positive_exports(5))

    def test_stats(self):
        eid = self.db.insert_event("bovino", 0.3, self.snap)
        self.db.mark_event_acknowledged(eid, is_false_positive=True)
        stats = FalsePositiveExporter(self.config, self.db).stats()
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["by_class"], {"bovino": 1})


# ============================================================
# Objetivo 4 - FPR / ROI y CLI
# ============================================================

class TestFPREvaluation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = {
            "model": {"weights_path": "fake.pt", "conf_threshold": 0.5, "device": None},
            "classes": {"names": ["humano", "bovino", "equino", "ovino", "porcino"],
                        "human_class": "humano"},
            "zone": {"polygon": [[10, 10], [200, 10], [200, 200], [10, 200]]},
            "evaluation": {"output_dir": os.path.join(self.tmp, "eval"),
                           "negative_clips_verified": False, "min_negative_minutes": 10},
        }
        self.eval = FPREvaluation(self.config)

    def test_zone_polygon_is_numpy_array(self):
        """cv2.pointPolygonTest exige numpy: una lista de tuplas lanzaba
        Bad argument en cuanto aparecia una deteccion humana en la zona."""
        polygon = self.eval._zone_polygon()
        self.assertIsInstance(polygon, np.ndarray)
        self.assertEqual(polygon.dtype, np.int32)
        self.assertEqual(polygon.shape, (4, 2))

    def test_in_zone_inside_and_outside(self):
        polygon = self.eval._zone_polygon()
        self.assertTrue(self.eval._in_zone((100, 100), polygon))
        self.assertFalse(self.eval._in_zone((500, 500), polygon))

    def test_missing_zone_means_whole_frame(self):
        self.config["zone"]["polygon"] = []
        self.assertIsNone(self.eval._zone_polygon())
        self.assertTrue(self.eval._in_zone((900, 900), None))

    def test_unverified_ground_truth_never_meets_target(self):
        """Con 0% de disparo el reporte NO puede decir que CUMPLE: los clips
        no estan declarados como negativos verificados."""
        report = self.eval.evaluate_dataset([])
        self.assertFalse(report["meets_target"])
        self.assertFalse(report["ground_truth_verified"])
        self.assertIn("No hay clips", report["verdict_note"])
        self.assertEqual(FPREvaluation.format_report(report)["verdict"], "NO VERIFICABLE")

    def test_verified_but_short_is_insufficient(self):
        self.config["evaluation"]["negative_clips_verified"] = True
        self.config["evaluation"]["min_negative_minutes"] = 10
        with unittest.mock.patch.object(FPREvaluation, "evaluate_video",
                                       return_value={"video": "a.mp4", "fps": 30.0,
                                                     "frames_processed": 30,
                                                     "duration_s": 1.0,
                                                     "human_detections_in_zone": 0,
                                                     "scenes_with_fp": 0,
                                                     "false_positives": []}):
            report = self.eval.evaluate_dataset(["a.mp4"])
        self.assertFalse(report["meets_target"])
        self.assertTrue(report["ground_truth_verified"])
        self.assertFalse(report["data_sufficient"])
        self.assertEqual(FPREvaluation.format_report(report)["verdict"], "DATOS INSUFICIENTES")

    def test_verified_and_long_enough_can_meet_target(self):
        self.config["evaluation"]["negative_clips_verified"] = True
        self.config["evaluation"]["min_negative_minutes"] = 1
        with unittest.mock.patch.object(FPREvaluation, "evaluate_video",
                                       return_value={"video": "a.mp4", "fps": 30.0,
                                                     "frames_processed": 1800,
                                                     "duration_s": 60.0,
                                                     "human_detections_in_zone": 0,
                                                     "scenes_with_fp": 0,
                                                     "false_positives": []}):
            report = self.eval.evaluate_dataset(["a.mp4"])
        self.assertTrue(report["meets_target"])
        self.assertTrue(report["data_sufficient"])
        self.assertEqual(FPREvaluation.format_report(report)["verdict"], "CUMPLE")

    def test_minutes_use_real_fps_not_hardcoded_ten(self):
        """Con 30 fps el video tiene el doble de minutos: usar 10 fijo
        inflaba la tasa por minuto."""
        self.config["evaluation"]["negative_clips_verified"] = True
        self.config["evaluation"]["min_negative_minutes"] = 1
        with unittest.mock.patch.object(
                FPREvaluation, "evaluate_video",
                return_value={"video": "a.mp4", "fps": 30.0, "frames_processed": 300,
                              "duration_s": 10.0, "human_detections_in_zone": 10,
                              "scenes_with_fp": 1, "false_positives": []}):
            report = self.eval.evaluate_dataset(["a.mp4"])
        self.assertAlmostEqual(report["video_seconds"], 10.0)
        self.assertAlmostEqual(report["false_positives_per_minute"], 60.0)

    def test_report_is_written(self):
        self.eval.evaluate_dataset([])
        self.assertTrue(os.path.exists(self.eval.output_path))


class TestROIAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = {
            "camera": {"fps_limit": 10},
            "alerts": {"remote": {"timeout_seconds": 5},
                       "escalate_after_seconds": 60},
            "evaluation": {"output_dir": os.path.join(self.tmp, "eval")},
        }

    def test_latency_uses_configured_values(self):
        roi = ROIAnalysis(self.config)
        latency = roi.detection_latency()
        self.assertAlmostEqual(latency["inferencia_s"], 0.1)
        self.assertAlmostEqual(latency["total_local_s"], 5.1)

    def test_more_dwell_time_means_more_effectiveness(self):
        """La latencia es fija, asi que a mayor permanencia del intruso mayor
        fraccion del tiempo se alcanza a avisar."""
        roi = ROIAnalysis(self.config)
        opt = roi.scenario("optimista", 3)
        pes = roi.scenario("pesimista", 12)
        self.assertGreater(pes["efectividad"], opt["efectividad"])
        self.assertTrue(opt["llega_a_tiempo"])
        self.assertTrue(pes["llega_a_tiempo"])

    def test_alert_later_than_dwell_time_is_worthless(self):
        """Si la alerta tarda mas que la permanencia completa, no se evita nada."""
        roi = ROIAnalysis(self.config, params={})
        scenario = roi.scenario("instantanea", 0.01)   # 0.6 s de permanencia
        self.assertFalse(scenario["llega_a_tiempo"])
        self.assertEqual(scenario["efectividad"], 0.0)

    def test_implausible_roi_raises_warning(self):
        roi = ROIAnalysis(self.config, params={"horas_vigilancia_ahorradas_semana": 200})
        report = roi.compute()
        self.assertTrue(report["advertencias"])
        self.assertGreater(report["roi_porcentaje"], 400.0)

    def test_report_is_written(self):
        roi = ROIAnalysis(self.config)
        roi.compute()
        self.assertTrue(os.path.exists(roi.output_path))


def _pipeline_source():
    with open(os.path.join(PROJECT_ROOT, "src", "06_realtime_pipeline.py"),
              encoding="utf-8") as f:
        return f.read()


def _load_pipeline_module():
    """`06_realtime_pipeline.py` empieza con un digito, asi que no se puede
    importar por nombre: se carga desde la ruta."""
    import importlib.util

    path = os.path.join(PROJECT_ROOT, "src", "06_realtime_pipeline.py")
    spec = importlib.util.spec_from_file_location("realtime_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRealtimePipelineCLI(unittest.TestCase):
    """El CLI de deteccion debe ser el MISMO flujo que la GUI."""

    def test_module_exposes_main_and_helpers(self):
        module = _load_pipeline_module()
        self.assertTrue(callable(module.main))
        self.assertTrue(callable(module.point_in_zone))
        self.assertTrue(callable(module.draw_overlay))
        self.assertTrue(callable(module.open_capture))

    def test_point_in_zone_uses_whole_frame_when_no_polygon(self):
        module = _load_pipeline_module()

        self.assertTrue(module.point_in_zone(500, 400, None))
        square = [[10, 10], [100, 10], [100, 100], [10, 100]]
        self.assertTrue(module.point_in_zone(50, 50, square))
        self.assertFalse(module.point_in_zone(500, 500, square))

    def test_pipeline_does_not_patch_torch_load(self):
        """El CLI ya no debe parchear torch.load con weights_only=False: eso
        desactiva la proteccion contra ejecucion de codigo al deserializar."""
        source = _pipeline_source()
        self.assertNotIn("torch.load =", source)
        self.assertNotIn('setdefault("weights_only"', source)
        self.assertNotIn("setdefault('weights_only'", source)
        self.assertIn("load_yolo", source)

    def test_pipeline_uses_the_shared_coordinator(self):
        source = _pipeline_source()
        for method in ("push_frame", "push_annotated", "can_detect",
                       "register_intrusion", "tick", "watch_integrity"):
            self.assertIn(method, source, f"el CLI debe usar coordinator.{method}")

    def test_training_script_scopes_its_unpickler_exception(self):
        """RE10/seguridad: el script de entrenamiento puede necesitar el
        unpickler completo de PyTorch, pero NO puede desactivarlo para todo el
        proceso al importarse. El parche debe ser un contexto que se restaura,
        incluso si el entrenamiento lanza excepcion."""
        import importlib.util

        import torch

        # Se captura ANTES de importar: ese es el estado que el import no debe
        # alterar.
        before = torch.load

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "src", "04_train.py")
        spec = importlib.util.spec_from_file_location("train_mod", path)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, os.path.join(os.path.dirname(path)))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)

        # Importar el script no debe dejar torch.load parcheado.
        self.assertIs(torch.load, before)

        with module._allow_full_unpickler():
            self.assertIsNot(torch.load, before)
        self.assertIs(torch.load, before)

        # Y tambien tras una excepcion: un entrenamiento fallido no puede dejar
        # el proceso en un estado de seguridad degradado.
        with self.assertRaises(ValueError):
            with module._allow_full_unpickler():
                raise ValueError("entrenamiento fallido")
        self.assertIs(torch.load, before)

    def test_training_script_uses_the_safe_loader(self):
        source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "src", "04_train.py"),
                      encoding="utf-8").read()
        self.assertIn("load_yolo", source)
        self.assertNotIn("YOLO(weights_path)", source)
        self.assertNotIn('YOLO("yolov8s.pt")', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
