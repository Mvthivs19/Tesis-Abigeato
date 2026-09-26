"""
RE05: Pantalla de inicio de sesion y control de acceso en la GUI.

El documento de requerimientos exige separar el rol Administrador del rol
Visualizador. La forma mas simple de garantizarlo es que TODA accion sensible
pase por `AccessControl.can()` y que la interfaz solo muestre al usuario lo que
su rol puede ejecutar, en vez de mostrar botones que despues se rechazan.

OpenCV no ofrece widgets de entrada de texto, asi que el login se implementa
con un bucle de teclado: se escriben los caracteres, `Tab` cambia de campo,
`Backspace` borra y `Enter` intenta validar. La contrasena se muestra
asteriscada.
"""

import cv2

from src.gui.app import (
    WIN_W, WIN_H, BG_ROOT, BG_CARD, BG_CARD_HI, BG_INPUT, BORDER, BORDER_HI,
    ACCENT, ACCENT_DIM, ACCENT_HOVER, TEXT_WHITE, TEXT_DIM, TEXT_MUTED,
    GREEN, RED, ORANGE, rounded_rect, badge, draw_icon,
)


class LoginView:
    """Formulario de acceso. Devuelve True cuando la sesion queda abierta."""

    def __init__(self, access):
        self.access = access
        self.username = ""
        self.password = ""
        self.field = 0            # 0 = usuario, 1 = contrasena
        self.message = ""
        self.message_color = TEXT_DIM
        self.attempts = 0
        self.done = False
        self.user = None
        # Lista de Suggestion/autocompletado de cuentas existentes.
        self.known_users = [u["username"] for u in access.list_users()]

    # ---------- Entrada ----------

    def handle_key(self, key):
        """Procesa una tecla. `key` es el codigo de cv2.waitKey."""
        if key in (13, 10):                       # Enter
            self._submit()
            return
        if key == 9:                              # Tab
            self.field = 1 - self.field
            self.message = ""
        elif key == 27:                           # Esc: salir
            self.done = True
            self.user = None
        elif key in (8, 127):                     # Backspace
            if self.field == 0:
                self.username = self.username[:-1]
            else:
                self.password = self.password[:-1]
        elif key == 3:                            # Ctrl+C
            self.done = True
            self.user = None
        elif 32 <= key <= 126:                    # caracter imprimible
            if self.field == 0:
                self.username += chr(key)
            else:
                self.password += chr(key)
            self.message = ""

    def _submit(self):
        if not self.username or not self.password:
            self.message = "Complete usuario y contrasena"
            self.message_color = ORANGE
            return
        try:
            self.user = self.access.login(self.username, self.password)
            self.done = True
        except Exception as exc:                  # AccessDenied
            self.attempts += 1
            self.message = str(exc)
            self.message_color = RED
            self.password = ""

    # ---------- Dibujado ----------

    def render(self, canvas):
        canvas[:] = BG_ROOT

        # Panel central
        pw, ph = 520, 380
        px, py = (WIN_W - pw) // 2, (WIN_H - ph) // 2
        rounded_rect(canvas, (px, py), (px + pw, py + ph), 14, BG_CARD, border_color=BORDER, border_thick=1)
        cv2.rectangle(canvas, (px + 20, py), (px + pw - 20, py + 3), ACCENT, -1)

        # Logo y titulo
        cv2.circle(canvas, (px + pw // 2, py + 62), 26, ACCENT_DIM, -1)
        draw_icon(canvas, "detection", px + pw // 2, py + 62, ACCENT, 1.2)
        cv2.putText(canvas, "DETECTRON", (px + pw // 2 - 92, py + 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, TEXT_WHITE, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Control de acceso al sistema", (px + pw // 2 - 118, py + 138),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA)

        # Campos
        self._field(canvas, px + 60, py + 176, pw - 120, 44, "Usuario", self.username, self.field == 0)
        self._field(canvas, px + 60, py + 232, pw - 120, 44, "Contrasena",
                    "*" * len(self.password), self.field == 1)

        # Mensaje
        if self.message:
            tw = cv2.getTextSize(self.message, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
            cv2.putText(canvas, self.message, (px + pw // 2 - tw // 2, py + 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.message_color, 1, cv2.LINE_AA)

        # Boton de ingreso
        bx, by, bw, bh = px + pw // 2 - 90, py + 312, 180, 40
        is_hover = (bx <= self.mouse_x <= bx + bw and by <= self.mouse_y <= by + bh)
        color = ACCENT_HOVER if is_hover else ACCENT
        rounded_rect(canvas, (bx, by), (bx + bw, by + bh), 8, color)
        label = "INGRESAR"
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0]
        cv2.putText(canvas, label, (bx + (bw - tw) // 2, by + bh // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, BG_ROOT, 1, cv2.LINE_AA)
        self._login_button = (bx, by, bw, bh)

        # Ayuda de navegacion
        hint = "TAB cambia de campo  |  ESC salir"
        tw = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0][0]
        cv2.putText(canvas, hint, (px + pw // 2 - tw // 2, py + ph - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_MUTED, 1, cv2.LINE_AA)

        # Cuentas de la instalacion (facilita la demostracion de la tesis)
        if self.known_users and self.attempts == 0:
            listing = "Cuentas: " + ", ".join(
                f"{u['username']}" for u in self.access.list_users()
            )
            tw = cv2.getTextSize(listing, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0][0]
            cv2.putText(canvas, listing, (px + pw // 2 - tw // 2, py + ph - 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_MUTED, 1, cv2.LINE_AA)

    def _field(self, canvas, x, y, w, h, label, value, focused):
        cv2.putText(canvas, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_DIM, 1, cv2.LINE_AA)
        border = ACCENT if focused else BORDER
        bg = BG_INPUT if focused else BG_CARD_HI
        rounded_rect(canvas, (x, y), (x + w, y + h), 8, bg, border_color=border, border_thick=2 if focused else 1)
        cv2.putText(canvas, value, (x + 12, y + h // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, TEXT_WHITE, 1, cv2.LINE_AA)
        if focused:
            # Cursor parpadeante
            import time
            if int(time.time() * 2) % 2 == 0:
                cx = x + 14 + cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0][0]
                cv2.line(canvas, (cx, y + 10), (cx, y + h - 10), ACCENT, 1)

    def mouse_clicked(self, mx, my):
        """Clic en el boton Ingressar. Los campos no necesitan clic: el foco
        se cambia con TAB, que es lo natural en una interfaz de teclado."""
        bx, by, bw, bh = getattr(self, "_login_button", (0, 0, 0, 0))
        if bx <= mx <= bx + bw and by <= my <= by + bh:
            self._submit()
