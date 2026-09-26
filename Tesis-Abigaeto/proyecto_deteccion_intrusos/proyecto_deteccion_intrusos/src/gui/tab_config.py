"""
Vista de Configuracion: RE02 (horarios), RE04 (contactos de emergencia),
RE05 (roles y permisos), RE11 (filtro de ruido) y RE18 (falsos positivos).

Todas las acciones de escritura exigen el permiso del rol con sesion. Cuando
el Visualizador abre esta vista ve los mismos datos en modo lectura, con los
controles deshabilitados, en vez de esconderlos: asi se entiende por que el
sistema no deja modificarlos.
"""

import os

import cv2

from src.gui.app import (
    SIDEBAR_W, WIN_W, WIN_H, HEADER_H,
    BG_CARD, BG_INPUT, BG_CARD_HI,
    TEXT_WHITE, TEXT_DIM, TEXT_MUTED,
    ACCENT, ACCENT_DIM, GREEN, RED, CYAN, ORANGE,
    BORDER, BORDER_HI,
    rounded_rect, badge, section_title, kv_row,
)
from src.utils import save_config
from src.contacts import ContactDirectory
from src.schedule import MonitoringSchedule


class ConfigView:
    # Botones de la vista: (id, etiqueta, x_rel, y, w, h, permiso requerido)
    BTN_SCHEDULE_TOGGLE = "sched_toggle"
    BTN_SCHEDULE_ADD = "sched_add"
    BTN_SCHEDULE_CLEAR = "sched_clear"
    BTN_CONTACT_ADD = "contact_add"
    BTN_FP_EXPORT = "fp_export"

    def __init__(self, app):
        self.app = app
        self.contacts = ContactDirectory(app.config["database"]["path"])
        self._buttons = []
        self._selected_contact = None
        self._fp_exports = []

    # ---------- Ciclo de vida ----------

    def on_enter(self):
        self.refresh()

    def refresh(self):
        self._fp_exports = self.app.db.get_false_positive_exports(5)

    def _can(self, permission):
        return self.app.access.can(permission)

    def _readonly_banner(self, canvas, x, y, w, permission):
        """Aviso de solo lectura para el Visualizador."""
        if self._can(permission):
            return y
        rounded_rect(canvas, (x, y), (x + w, y + 30), 7, (52, 40, 34))
        cv2.putText(canvas,
                    f"ROL VISUALIZADOR: solo lectura ({permission} requiere Administrador)",
                    (x + 14, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.34, ORANGE, 1, cv2.LINE_AA)
        return y + 42

    def _button(self, canvas, btn_id, label, x, y, w, h, permission,
                enabled=True, color=ACCENT):
        """Dibuja un boton y lo registra para el manejo de clics."""
        allowed = enabled and self._can(permission)
        is_hover = (x <= self.app.mouse_x <= x + w and y <= self.app.mouse_y <= y + h)
        fill = color if allowed else BG_INPUT
        if allowed and is_hover:
            fill = tuple(min(255, c + 26) for c in color)
        rounded_rect(canvas, (x, y), (x + w, y + h), 7, fill,
                     border_color=BORDER if not allowed else None)
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        tcolor = BG_ROOT if allowed else TEXT_MUTED
        cv2.putText(canvas, label, (x + (w - tw) // 2, y + h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, tcolor, 1, cv2.LINE_AA)
        self._buttons.append({"id": btn_id, "rect": (x, y, w, h), "enabled": allowed})
        return x + w + 12

    # ---------- Render principal ----------

    def render(self, canvas):
        self._buttons = []
        self.refresh()

        x0 = SIDEBAR_W + 26
        w = WIN_W - SIDEBAR_W - 52
        y = HEADER_H + 20

        left_w = int(w * 0.52)
        right_x = x0 + left_w + 20
        right_w = w - left_w - 20

        y_left = self._render_schedule(canvas, x0, y, left_w)
        self._render_filter(canvas, x0, y_left, left_w)
        self._render_contacts(canvas, right_x, y, right_w)
        self._render_access(canvas, right_x, y + 250, right_w)
        self._render_retraining(canvas, right_x, y + 470, right_w)

    # ---- RE02: horarios ----

    def _render_schedule(self, canvas, x, y, w):
        h = 196
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        section_title(canvas, x + 16, y + 24, "RE02  HORARIOS DE MONITOREO", w - 40)

        cfg = self.app.config.get("schedule", {})
        sched = MonitoringSchedule(cfg)
        active = sched.is_active()

        y2 = y + 56
        enabled = cfg.get("enabled", True)
        kv_row(canvas, x + 18, y2, "Programacion activa",
               "SI" if enabled else "NO (continuo 24/7)",
               190, GREEN if enabled else TEXT_DIM)
        badge(canvas, x + w - 118, y2 + 4,
              "DENTRO" if active else "FUERA", GREEN if active else ORANGE, 72)

        windows = cfg.get("windows", [])
        cv2.putText(canvas, "Ventanas configuradas:", (x + 18, y2 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)
        if windows:
            for i, win in enumerate(windows[:3]):
                text = f"  {win['start']}  ->  {win['end']}"
                covers = "  (cruza medianoche)" if win["end"] < win["start"] else ""
                cv2.putText(canvas, text + covers, (x + 30, y2 + 48 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_WHITE, 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "  Sin ventanas (operacion continua)",
                        (x + 30, y2 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_MUTED, 1, cv2.LINE_AA)

        by = y + h - 44
        bx = self._button(canvas, self.BTN_SCHEDULE_TOGGLE,
                          "DESACTIVAR" if enabled else "ACTIVAR",
                          x + 18, by, 132, 30, "edit_schedule")
        bx = self._button(canvas, self.BTN_SCHEDULE_ADD, "NOCTURNO 20-06",
                          bx, by, 176, 30, "edit_schedule")
        self._button(canvas, self.BTN_SCHEDULE_CLEAR, "CONTINUO 24/7",
                     bx, by, 140, 30, "edit_schedule")

        nxt = sched.next_transition()
        if nxt:
            cv2.putText(canvas, f"Proxima transicion: {nxt}", (x + w - 190, y + h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_MUTED, 1, cv2.LINE_AA)
        return y + h + 18

    # ---- RE11: filtro ----

    def _render_filter(self, canvas, x, y, w):
        h = 148
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        section_title(canvas, x + 16, y + 24, "RE11  FILTRADO DE RUIDO Y AMBIENTE", w - 40)

        cfg = self.app.config.get("frame_filter", {})
        integ = self.app.config.get("integrity", {})
        coord = getattr(self.app, "coordinator", None)
        qf = coord.quality_filter if coord else None
        im = coord.integrity if coord else None

        rows = [
            ("Filtro de calidad", "ACTIVO" if cfg.get("enabled", True) else "INACTIVO",
             GREEN if cfg.get("enabled", True) else TEXT_MUTED),
            ("Integridad de camara (RE21)", "ACTIVO" if integ.get("enabled", True) else "INACTIVO",
             GREEN if integ.get("enabled", True) else TEXT_MUTED),
        ]
        if qf is not None:
            color = GREEN if qf.last_quality == "ok" else ORANGE
            rows.append(("Calidad del frame en vivo", qf.last_quality.upper(), color))
        if im is not None:
            color = GREEN if im.last_status == "ok" else RED
            rows.append(("Estado de integridad", im.last_status.upper(), color))

        for i, (label, value, color) in enumerate(rows):
            ry = y + 54 + i * 24
            cv2.putText(canvas, label, (x + 18, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_DIM, 1, cv2.LINE_AA)
            cv2.putText(canvas, value, (x + 250, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

        th = cfg.get("min_contrast", 18.0)
        bl = cfg.get("blur_threshold", 28.0)
        cv2.putText(canvas,
                    f"umbrales: contraste>{th}  nitidez>{bl}  ratio>{cfg.get('smooth_ratio', 0.25)}",
                    (x + 18, y + h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)

    # ---- RE04: contactos ----

    def _render_contacts(self, canvas, x, y, w):
        h = 234
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        section_title(canvas, x + 16, y + 24, "RE04  CONTACTOS DE EMERGENCIA", w - 40)

        y2 = self._readonly_banner(canvas, x, y + 34, w, "edit_contacts") - y

        contacts = self.contacts.list_contacts()
        self._selected_contact = contacts[0]["contact_id"] if contacts else None

        if not contacts:
            cv2.putText(canvas, "Sin contactos registrados", (x + 18, y + y2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, ORANGE, 1, cv2.LINE_AA)
            cv2.putText(canvas, "La cadena de alerta no puede avisar a nadie sin esta lista",
                        (x + 18, y + y2 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)
        else:
            headers = ["CONTACTO", "CANAL", "PRIO", "TELEFONO"]
            col_x = [x + 18, x + 200, x + 268, x + 320]
            for i, hdr in enumerate(headers):
                cv2.putText(canvas, hdr, (col_x[i], y + y2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.line(canvas, (x + 14, y + y2 + 2), (x + w - 14, y + y2 + 2), BORDER, 1)

            for i, c in enumerate(contacts[:4]):
                ry = y + y2 + 24 + i * 26
                rounded_rect(canvas, (x + 14, ry - 6), (x + w - 14, ry + 16), 5, BG_INPUT)
                cv2.putText(canvas, c["name"][:20], (col_x[0], ry + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_WHITE, 1, cv2.LINE_AA)
                cv2.putText(canvas, c["channel"], (col_x[1], ry + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, CYAN, 1, cv2.LINE_AA)
                cv2.putText(canvas, str(c["priority"]), (col_x[2], ry + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, ACCENT, 1, cv2.LINE_AA)
                cv2.putText(canvas, (c["phone"] or c["email"] or "-")[:18], (col_x[3], ry + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_DIM, 1, cv2.LINE_AA)

        by = y + h - 42
        self._button(canvas, self.BTN_CONTACT_ADD, "+ AGREGAR CONTACTO",
                     x + 18, by, 190, 30, "edit_contacts")

    # ---- RE05: accesos ----

    def _render_access(self, canvas, x, y, w):
        h = 206
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        section_title(canvas, x + 16, y + 24, "RE05  USUARIOS Y ROLES", w - 40)

        users = self.app.access.list_users()
        for i, u in enumerate(users[:4]):
            ry = y + 54 + i * 24
            rounded_rect(canvas, (x + 14, ry - 6), (x + w - 14, ry + 14), 5, BG_INPUT)
            draw_letter = u["display_name"][:1].upper()
            cv2.circle(canvas, (x + 32, ry + 4), 9, ACCENT_DIM if u["role"] == "administrador" else BG_CARD_HI, -1)
            cv2.putText(canvas, draw_letter, (x + 28, ry + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, u["username"][:14], (x + 50, ry + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_WHITE, 1, cv2.LINE_AA)
            role = "ADMINISTRADOR" if u["role"] == "administrador" else "VISUALIZADOR"
            rcolor = ACCENT if u["role"] == "administrador" else CYAN
            cv2.putText(canvas, role, (x + 160, ry + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, rcolor, 1, cv2.LINE_AA)
            state = "ACTIVO" if u["active"] else "INACTIVO"
            cv2.putText(canvas, state, (x + w - 84, ry + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        GREEN if u["active"] else RED, 1, cv2.LINE_AA)

        cur = self.app.user
        if cur:
            who = self.app.access.name_of(cur["username"])
            note = f"Sesion activa: {who} ({cur['role']})"
            cv2.putText(canvas, note, (x + 18, y + h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_MUTED, 1, cv2.LINE_AA)

    # ---- RE18: reentrenamiento ----

    def _render_retraining(self, canvas, x, y, w):
        h = 130
        rounded_rect(canvas, (x, y), (x + w, y + h), 10, BG_CARD)
        section_title(canvas, x + 16, y + 24, "RE18  FALSOS POSITIVOS / REENTRENAMIENTO", w - 40)

        fp = self.app.db.count_false_positives()
        cv2.putText(canvas, "Eventos etiquetados como falso positivo:", (x + 18, y + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(fp), (x + 268, y + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, ACCENT if fp else TEXT_DIM, 2, cv2.LINE_AA)

        if self._fp_exports:
            last = self._fp_exports[0]
            cv2.putText(canvas, f"Ultima exportacion: {last[1]} ({last[2]} muestras)",
                        (x + 18, y + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "Sin exportaciones previas", (x + 18, y + 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)

        self._button(canvas, self.BTN_FP_EXPORT, "EXPORTAR PARA REENTRENAR",
                     x + 18, y + h - 42, 250, 30, "label_false_positive")

    # ---------- Clic ----------

    def on_click(self, mx, my):
        for btn in self._buttons:
            x, y, w, h = btn["rect"]
            if not (x <= mx <= x + w and y <= my <= y + h):
                continue
            if not btn["enabled"]:
                self.app.notify("Requiere rol Administrador", ORANGE)
                return
            self._action(btn["id"])
            return

    def _action(self, action_id):
        from src.utils import load_config

        if action_id in (self.BTN_SCHEDULE_TOGGLE, self.BTN_SCHEDULE_ADD,
                         self.BTN_SCHEDULE_CLEAR):
            if not self.app.require("edit_schedule"):
                return
            cfg = self.app.config
            if action_id == self.BTN_SCHEDULE_TOGGLE:
                cfg["schedule"]["enabled"] = not cfg.get("schedule", {}).get("enabled", True)
                msg = f"Programacion {'activada' if cfg['schedule']['enabled'] else 'desactivada'}"
            elif action_id == self.BTN_SCHEDULE_ADD:
                cfg["schedule"]["enabled"] = True
                cfg["schedule"]["windows"] = [{"start": "20:00", "end": "06:00"}]
                msg = "Horario nocturno 20:00-06:00 aplicado"
            else:
                cfg["schedule"]["enabled"] = True
                cfg["schedule"]["windows"] = [{"start": "00:00", "end": "23:59"}]
                msg = "Monitoreo continuo 24/7"
            save_config(cfg, "configs/config.yaml")
            # El coordinator en vivo debe recoger el cambio sin reiniciar.
            coord = getattr(self.app, "coordinator", None)
            if coord is not None:
                coord.schedule = MonitoringSchedule(cfg.get("schedule", {}))
            self.app.config = load_config("configs/config.yaml")
            self.app.notify(msg, GREEN)

        elif action_id == self.BTN_CONTACT_ADD:
            if not self.app.require("edit_contacts"):
                return
            self._add_demo_contact()

        elif action_id == self.BTN_FP_EXPORT:
            if not self.app.require("label_false_positive"):
                return
            self._export_false_positives()

    def _add_demo_contact(self):
        """Agrega el contacto de ejemplo del documento de requerimientos.

        En un despliegue real este formulario se reemplaza por el alta de
        contactos desde teclado; aqui se crean los contactos de referencia
        que el documento usa como ejemplo, para que la cadena RE04 pueda
        demostrarse de extremo a extremo.
        """
        existing = {c["name"] for c in self.contacts.list_contacts()}
        presets = [
            ("Productor", "+56912345678", "sms", 1, "propietario"),
            ("Guarda caseta", "+56987654321", "sms", 2, "vigilancia"),
            ("Carabineros sector", "+56955555555", "sms", 3, "emergencia"),
        ]
        added = []
        for name, phone, channel, priority, relation in presets:
            if name in existing:
                continue
            try:
                self.contacts.add(name, phone=phone, channel=channel,
                                  priority=priority, relation=relation)
                added.append(name)
            except ValueError:
                continue
        if added:
            self.app.notify(f"Contactos agregados: {', '.join(added)}", GREEN)
        else:
            self.app.notify("Los contactos de referencia ya estaban cargados", CYAN)
        self.refresh()

    def _export_false_positives(self):
        try:
            from src.retraining import FalsePositiveExporter
            exporter = FalsePositiveExporter(
                self.app.config, self.app.db
            )
            result = exporter.export()
            if result["samples"] == 0:
                self.app.notify(
                    "No hay falsos positivos etiquetados todavia (marcalos con F)",
                    ORANGE,
                )
            else:
                self.app.notify(
                    f"RE18: {result['samples']} muestras exportadas a "
                    f"{result['output_dir']}", GREEN, 6.0,
                )
        except Exception as exc:  # noqa: BLE001
            self.app.notify(f"Exportacion fallo: {exc}", RED, 6.0)
        self.refresh()
