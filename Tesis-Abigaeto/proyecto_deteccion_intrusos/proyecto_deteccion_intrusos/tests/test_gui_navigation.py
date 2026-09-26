"""
Verificacion de navegacion y render de la GUI.

Por que este archivo existe: la suite unitaria comprobaba metodos sueltos con
dobles, pero nadie habia recorrido las vistas de verdad. Ese hueco escondia un
NameError real en la vista de Configuracion (`BG_ROOT` sin importar) que
hundia la aplicacion completa cuando el Administrador abria su pestana.

Dos niveles, porque no son lo mismo:

1. Render directo y determinista de cada vista. Es el que encuentra fallos de
   render y vistas que dibujan el mismo contenido.
2. Recorrido del bucle real `Application.run()` con `cv2` doble. Es el que
   comprueba que la aplicacion arranca, navega y sale sin excepciones.

Aislamiento: la configuracion se copia a un directorio temporal y la base de
datos es propia, asi que `data/events.db` no se toca. La deteccion apunta a un
video minimo generado aqui, no al video de la finca. `cv2.imshow` no se llama
realmente: no se abre ninguna ventana.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import time
import unittest
import unittest.mock as mock

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.utils import load_config

WIN_W, WIN_H, BG_ROOT = 1400, 900, (22, 20, 18)
SIDEBAR_CUT = 260          # el sidebar es comun a todas las vistas


def _tiny_video(path, frames=24):
    """Video minimo con textura: el filtro de calidad (RE11) no debe confundir
    un frame plano con una camara tapada durante la prueba."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (160, 120))
    for i in range(frames):
        img = np.full((120, 160, 3), 60, np.uint8)
        cv2.circle(img, (30 + i * 4, 60), 18, (200, 180, 120), -1)
        cv2.rectangle(img, (5, 5), (155, 115), (140, 140, 140), 2)
        writer.write(img)
    writer.release()
    return path


class GuiNavigationTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        self.config = load_config()
        self.config["database"]["path"] = os.path.join(self.tmp, "nav.db")
        self.config["retraining"]["output_dir"] = os.path.join(self.tmp, "fp")
        self.config["evaluation"]["output_dir"] = os.path.join(self.tmp, "eval")
        self.config["camera"]["source"] = _tiny_video(
            os.path.join(self.tmp, "tiny.mp4"))
        # Sin horarios: la vista de Configuracion no debe depender de la hora
        # local para poder pintarse.
        self.config["schedule"]["enabled"] = False

    # ---------- utilidades ----------

    def _enter(self, user, password):
        """Instancia la aplicacion real con la configuracion temporal y deja
        la sesion iniciada. `login` devuelve el dict que consume
        `complete_login`; se entra por API porque lo que se prueba aqui es la
        navegacion, no el tecleo del formulario."""
        from src.gui.app import Application
        with mock.patch("src.gui.app.load_config", return_value=self.config), \
                mock.patch("src.gui.app.cv2.imshow", lambda *a: None), \
                mock.patch("src.gui.app.cv2.waitKey", lambda *a: -1), \
                mock.patch("src.gui.app.cv2.destroyAllWindows", lambda *a: None):
            app = Application()
            info = app.access.login(user, password)
            self.assertIsNotNone(info, f"login fallo para {user}")
            app.complete_login(info)
        return app

    def _render_view(self, app, key):
        """Pinta una vista como lo hace `run()` y devuelve su huella de
        contenido. Lanza si la vista revienta: ese es el objetivo."""
        app.switch_view(key)
        canvas = np.full((WIN_H, WIN_W, 3), BG_ROOT, np.uint8)
        app.draw_sidebar(canvas)
        app.draw_header(canvas)
        view = app.views.get(app.current_view)
        if view is not None and hasattr(view, "render"):
            view.render(canvas)
        content = canvas[:, SIDEBAR_CUT:, :]
        return hashlib.sha1(content.tobytes()).hexdigest(), content

    def _close(self, app):
        """Cierra las vistas y comprueba que no queda ningun hilo vivo.

        El temporal de cada prueba se borra al terminar, asi que un hilo que
        sobreviviera al cierre se toparia con una base de datos borrada. En
        produccion el sintoma equivalente es un hilo que sigue leyendo la
        camara y la BD despues de cerrar la ventana, asi que se verifica de
        forma explicita en vez de tolerarlo como ruido de log."""
        for view in app.views.values():
            if hasattr(view, "cleanup"):
                try:
                    view.cleanup()
                except Exception:  # noqa: BLE001
                    pass
        time.sleep(0.3)
        leaks = [type(v).__name__ for v in app.views.values()
                 if getattr(v, "thread", None) is not None
                 and v.thread.is_alive()]
        self.assertEqual(leaks, [],
                         f"hilos vivos tras el cierre: {leaks}")

    # ---------- 1. render determinista de cada vista ----------

    def test_admin_renders_every_view_without_raising(self):
        app = self._enter("admin", "admin123")
        try:
            for key in app.views:
                try:
                    self._render_view(app, key)
                except Exception as exc:  # noqa: BLE001
                    self.fail(f"la vista '{key}' lanzo al renderizar: "
                              f"{type(exc).__name__}: {exc}")
                self.assertEqual(app.current_view, key,
                                 f"el Administrador no pudo abrir '{key}'")
        finally:
            self._close(app)

    def test_every_view_draws_its_own_content(self):
        """Si dos vistas pintaran el mismo frame, la navegacion "funcionaria"
        pero el usuario veria siempre la misma pantalla. Se comparan huellas
        del area de contenido y no el conteo de pixeles, que esta dominado por
        el marco comun (sidebar y header) y no distinguiria nada."""
        app = self._enter("admin", "admin123")
        try:
            seen = {}
            for key in app.views:
                digest, content = self._render_view(app, key)
                self.assertGreater(
                    int((content > 45).any(axis=2).sum()), 200,
                    f"la vista '{key}' no pinto practicamente nada")
                self.assertNotIn(
                    digest, seen,
                    f"las vistas '{seen.get(digest)}' y '{key}'-produjeron un "
                    "frame IDENTICO: probable vista rota o no conectada")
                seen[digest] = key
            self.assertGreaterEqual(len(seen), 6)
        finally:
            self._close(app)

    def test_viewer_cannot_open_restricted_modules(self):
        """RE05: el Visualizador opera y consulta, pero no entra a dataset,
        training ni config. Se comprueba a nivel de la propia navegacion, que
        es donde un fallo de permisos se traduciria en fuga de datos."""
        app = self._enter("operador", "operador123")
        try:
            allowed = app.access.allowed_views()
            for restricted in ("dataset", "training", "config"):
                self.assertNotIn(restricted, allowed)
                app.switch_view(restricted)
                self.assertNotEqual(
                    app.current_view, restricted,
                    f"el Visualizador entro a '{restricted}'")
            for permitted in ("detection", "history", "evaluation"):
                self.assertIn(permitted, allowed)
                app.switch_view(permitted)
                self.assertEqual(app.current_view, permitted)
        finally:
            self._close(app)

    # ---------- 2. bucle real de la aplicacion ----------

    def _run_loop(self, user, password, frames=110):
        from src.gui.app import Application

        state = {"n": 0, "app": None}
        visited, painted, errors = [], {}, []

        def fake_imshow(_win, canvas):
            n = state["n"]
            state["n"] = n + 1
            app = state["app"]
            try:
                painted[app.current_view] = int(
                    (np.asarray(canvas) > 45).any(axis=2).sum())
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))
            # Se recorre TODO el ciclo de vistas, empezando en el indice 0,
            # para que la prueba abarque la lista completa y no una parte.
            if app.user is not None and n % 12 == 0:
                allowed = app.access.allowed_views()
                if allowed:
                    nxt = allowed[(n // 12) % len(allowed)]
                    if nxt != app.current_view:
                        app.switch_view(nxt)
                        if nxt not in visited:
                            visited.append(nxt)
            if n > frames:
                app.stop()

        with mock.patch("src.gui.app.load_config", return_value=self.config), \
                mock.patch("src.gui.app.cv2.imshow", fake_imshow), \
                mock.patch("src.gui.app.cv2.waitKey", lambda *a: -1), \
                mock.patch("src.gui.app.cv2.destroyAllWindows",
                           lambda *a: None):
            app = Application()
            state["app"] = app
            info = app.access.login(user, password)
            app.complete_login(info)
            app.run()
            self.assertIsNotNone(app.user)
            app.logout()
            self.assertIsNone(app.user)
        # `run()` debe haber cerrado todos los hilos al salir; si no, el
        # temporal de la prueba se borra con un hilo todavia leyendo la BD.
        leaks = [type(v).__name__ for v in app.views.values()
                 if getattr(v, "thread", None) is not None
                 and v.thread.is_alive()]
        self.assertEqual(leaks, [], f"hilos vivos tras run(): {leaks}")
        return visited, painted, errors

    def test_run_loop_navigates_and_closes_session(self):
        visited, painted, errors = self._run_loop("admin", "admin123")
        self.assertEqual(errors, [])
        for expected in ("dashboard", "detection", "history", "health",
                         "evaluation", "config"):
            self.assertIn(expected, visited,
                          f"el bucle real no alcanzo '{expected}'")
        self.assertGreater(painted.get("dashboard", 0), 800)

    def test_run_loop_for_viewer_stays_inside_its_role(self):
        visited, _painted, errors = self._run_loop("operador", "operador123")
        self.assertEqual(errors, [])
        for forbidden in ("dataset", "training", "config"):
            self.assertNotIn(forbidden, visited,
                             f"el Visualizador alcanzo '{forbidden}'")

    def test_navigation_does_not_touch_the_real_database(self):
        """La prueba debe ser inocua: si escribiera en `data/events.db`
        cualquier evento de mas, contaminaria la evidencia de la tesis."""
        real_db = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data",
            "events.db")
        before = (os.path.getsize(real_db), os.path.getmtime(real_db)) \
            if os.path.exists(real_db) else None
        self._run_loop("admin", "admin123", frames=30)
        after = (os.path.getsize(real_db), os.path.getmtime(real_db)) \
            if os.path.exists(real_db) else None
        self.assertEqual(before, after,
                         "la navegacion modifico la base de datos real")


if __name__ == "__main__":
    unittest.main(verbosity=2)
