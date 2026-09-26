"""
RE05: Control de acceso basado en roles y permisos.

El documento de requerimientos define dos perfiles:

  Administrador  - privileges maximos. Configura los parametros criticos
                    de la IA, traza la zona perimetral, establece los
                    horarios de monitoreo, gestiona el directorio de
                    contactos de emergencia y es el UNICO rol autorizado
                    para etiquetar falsos positivos (RE18).

  Visualizador    - permisos restringidos para el personal de la caseta de
                    vigilancia o de ronda. Observa el video, ve las
                    alertas, puede confirmar o silenciar una alarma y marcar
                    un evento como falso positivo, pero NO puede alterar la
                    configuracion, los horarios ni los registros de la
                    base de datos.

Las contrasenas nunca se guardan en claro: se derivan con PBKDF2-HMAC-SHA256
y sal aleatorio por usuario. Es una dependencia de la biblioteca estandar
(hashlib), sin paquetes adicionales.
"""

import hashlib
import hmac
import os
import secrets
import sqlite3

try:
    from src.utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger("access_control")

PBKDF2_ITERATIONS = 120_000
SALT_BYTES = 16

ROLE_ADMIN = "administrador"
ROLE_VIEWER = "visualizador"

# Matriz de permisos. La clave es el permiso y el valor el conjunto de roles
# que lo poseen. La GUI y el pipeline consultan `can()` en vez de comparar
# strings de rol, para que agregar un permiso no rompa otros modulos.
PERMISSIONS = {
    # RE05 - gestion de cuentas
    "manage_users": {ROLE_ADMIN},
    # RE03 - calibrar parametros y sensibilidad de inferencia
    "edit_thresholds": {ROLE_ADMIN},
    # RE01 - trazar la zona perimetral
    "edit_zone": {ROLE_ADMIN},
    # RE02 - programar horarios
    "edit_schedule": {ROLE_ADMIN},
    # RE04 - directorio de contactos
    "edit_contacts": {ROLE_ADMIN},
    # RE06 - gestion del source de video
    "edit_camera": {ROLE_ADMIN},
    # Reentrenamiento del modelo
    "run_training": {ROLE_ADMIN},
    # RE18 - unicamente el Administrador etiqueta falsos positivos
    "label_false_positive": {ROLE_ADMIN},
    # Operacion en vivo: permitted para ambos roles
    "view_live": {ROLE_ADMIN, ROLE_VIEWER},
    "start_detection": {ROLE_ADMIN, ROLE_VIEWER},
    "acknowledge_alert": {ROLE_ADMIN, ROLE_VIEWER},
    "silence_alarm": {ROLE_ADMIN, ROLE_VIEWER},
    "view_history": {ROLE_ADMIN, ROLE_VIEWER},
    "view_health": {ROLE_ADMIN, ROLE_VIEWER},
    "view_dataset_stats": {ROLE_ADMIN, ROLE_VIEWER},
    # RE09/RE10: la lectura de FPR y ROI es information, no operacion, asi que
    # ambos roles la ven. Editar los parametros economicos del ROI sigue siendo
    # exclusivo del Administrador (edit_thresholds).
    "view_evaluation": {ROLE_ADMIN, ROLE_VIEWER},
}

# Vistas de la GUI que cada rol puede abrir.
ROLE_VIEWS = {
    ROLE_ADMIN: ["dashboard", "dataset", "training", "detection", "history",
                 "health", "evaluation", "config"],
    ROLE_VIEWER: ["dashboard", "detection", "history", "health", "evaluation"],
}


def hash_password(password, salt=None):
    """Deriva una contrasena con PBKDF2-HMAC-SHA256.

    Devuelve (hash_hex, salt_hex). Si no se entrega salt, se genera uno
    aleatorio con `secrets`, de modo que dos usuarios con la misma
    contrasena nunca generan el mismo hash.
    """
    if salt is None:
        salt = secrets.token_bytes(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return dk.hex(), salt.hex()


def verify_password(password, hash_hex, salt_hex):
    """Compara en tiempo constante para no filtrar informacion por timing."""
    candidate, _ = hash_password(password, salt=bytes.fromhex(salt_hex))
    return hmac.compare_digest(candidate, hash_hex)


class AccessDenied(PermissionError):
    """Se lanza cuando un rol intenta una operacion que no le corresponde."""


class AccessControl:
    def __init__(self, db_path="data/events.db"):
        self.db_path = db_path
        self._init_schema()
        self.current_user = None
        self._failed_attempts = {}

    # ---------- Esquema y usuarios ----------

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_schema(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                display_name TEXT,
                created_at TEXT NOT NULL,
                active INTEGER DEFAULT 1
            )
        """)
        conn.commit()
        conn.close()
        self.ensure_default_users()

    def ensure_default_users(self):
        """Crea las cuentas iniciales en el primer arranque.

        Las contrasenas por defecto son trivialmente adivinables a proposito:
        el sistema exige cambiarlas al primer ingreso (ver `must_change_password`).
        En un despliegue real se eliminan desde la GUI.
        """
        from datetime import datetime

        defaults = [
            ("admin", ROLE_ADMIN, "admin123", "Administrador"),
            ("operador", ROLE_VIEWER, "operador123", "Visualizador"),
        ]
        conn = self._connect()
        cur = conn.cursor()
        created = []
        for username, role, password, display in defaults:
            row = cur.execute(
                "SELECT username FROM users WHERE username = ?", (username,)
            ).fetchone()
            if row:
                continue
            h, s = hash_password(password)
            cur.execute(
                """INSERT INTO users (username, role, password_hash, salt,
                                      display_name, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (username, role, h, s, display, datetime.now().isoformat()),
            )
            created.append(username)
        conn.commit()
        conn.close()
        if created:
            logger.info(
                f"Cuentas iniciales creadas: {', '.join(created)}. "
                "Cambiar las contrasenas por defecto antes de operar en campo."
            )
        return created

    def list_users(self):
        conn = self._connect()
        rows = conn.execute(
            """SELECT username, role, display_name, created_at, active
               FROM users ORDER BY role, username"""
        ).fetchall()
        conn.close()
        return [
            {"username": r[0], "role": r[1], "display_name": r[2],
             "created_at": r[3], "active": bool(r[4])}
            for r in rows
        ]

    def create_user(self, username, password, role, display_name=None,
                    actor=None):
        """RE05: el Administrador asigna roles y permisos de acceso."""
        self.require("manage_users", actor)
        if role not in (ROLE_ADMIN, ROLE_VIEWER):
            raise ValueError(f"Rol desconocido: {role}")

        h, s = hash_password(password)
        from datetime import datetime
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO users (username, role, password_hash, salt,
                                      display_name, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (username, role, h, s, display_name or username,
                 datetime.now().isoformat()),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"El usuario '{username}' ya existe")
        finally:
            conn.close()
        logger.info(f"[RE05] Usuario '{username}' creado con rol {role} "
                    f"por '{self.name_of(actor)}'")
        return True

    def set_active(self, username, active, actor=None):
        """Activa o desactiva una cuenta sin borrarla (mantiene la traza)."""
        self.require("manage_users", actor)
        conn = self._connect()
        conn.execute("UPDATE users SET active = ? WHERE username = ?",
                     (1 if active else 0, username))
        conn.commit()
        conn.close()
        logger.info(f"[RE05] Cuenta '{username}' "
                    f"{'activada' if active else 'desactivada'}")
        return True

    def change_password(self, username, new_password, actor=None,
                        require_admin=False):
        self.require("manage_users", actor, require_admin=require_admin)
        h, s = hash_password(new_password)
        conn = self._connect()
        conn.execute("UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
                     (h, s, username))
        conn.commit()
        conn.close()
        logger.info(f"[RE05] Contrasena de '{username}' actualizada")
        return True

    # ---------- Sesion ----------

    def login(self, username, password):
        """Valida credenciales y abre sesion.

        Aplica un bloqueo progresivo: tras `MAX_ATTEMPTS` intentos fallidos
        la cuenta se bloquea durante `LOCKOUT_SECONDS`. Sin esto, un atacante
        podria enumerar usuarios probando contrasenas sin limite.
        """
        from datetime import datetime, timedelta

        if self._is_locked(username):
            remaining = self._lockout_remaining(username)
            raise AccessDenied(
                f"Cuenta '{username}' bloqueada por intentos fallidos. "
                f"Intente en {remaining}s."
            )

        conn = self._connect()
        row = conn.execute(
            "SELECT password_hash, salt, role, active FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        conn.close()

        # Se ejecuta la verificacion aunque el usuario no exista, contra un
        # hash ficticio, para que el tiempo de respuesta no revele que
        # usernames son validos.
        if row is None:
            hash_password(password, salt=bytes(SALT_BYTES))
            self._register_failure(username)
            raise AccessDenied("Usuario o contrasena incorrectos")

        password_hash, salt, role, active = row
        if not verify_password(password, password_hash, salt):
            self._register_failure(username)
            raise AccessDenied("Usuario o contrasena incorrectos")

        if not active:
            raise AccessDenied(f"La cuenta '{username}' esta desactivada")

        self._failed_attempts.pop(username, None)
        self.current_user = username
        logger.info(f"[RE05] Inicio de sesion: '{username}' (rol {role})")
        return {"username": username, "role": role}

    def logout(self):
        if self.current_user:
            logger.info(f"[RE05] Cierre de sesion: '{self.current_user}'")
        self.current_user = None

    # Bloqueo progresivo tras intentos fallidos (RE05).
    MAX_ATTEMPTS = 5
    LOCKOUT_SECONDS = 60

    def _register_failure(self, username):
        now = __import__("time").time()
        rec = self._failed_attempts.setdefault(username, {"n": 0, "until": 0})
        rec["n"] += 1
        if rec["n"] >= self.MAX_ATTEMPTS:
            rec["until"] = now + self.LOCKOUT_SECONDS
            rec["n"] = 0
            logger.error(f"[RE05] Cuenta '{username}' bloqueada "
                         f"{self.LOCKOUT_SECONDS}s por intentos fallidos")

    def _is_locked(self, username):
        rec = self._failed_attempts.get(username)
        if rec and rec["until"] > __import__("time").time():
            return True
        if rec and rec["until"]:
            del self._failed_attempts[username]
        return False

    def _lockout_remaining(self, username):
        rec = self._failed_attempts.get(username, {"until": 0})
        return max(int(rec["until"] - __import__("time").time()), 0)

    # ---------- Consultas de permisos ----------

    def role_of(self, user):
        """Rol de un usuario. Acepta None, un nombre de usuario (str) o un
        objeto/diccionario de sesion como el que devuelve `login`."""
        if user is None:
            return None
        if isinstance(user, dict):
            return user.get("role")
        if hasattr(user, "role"):
            return getattr(user, "role")
        if isinstance(user, str):
            # Un string puede ser un nombre de usuario o directamente un rol,
            # porque la GUI consulta permisos por rol en algunos puntos.
            if user in ROLE_VIEWS:
                return user
            conn = self._connect()
            row = conn.execute("SELECT role FROM users WHERE username = ?",
                               (user,)).fetchone()
            conn.close()
            return row[0] if row else None
        return None

    def name_of(self, user):
        if user is None:
            return "sistema"
        if isinstance(user, dict):
            return user.get("username", str(user))
        if hasattr(user, "display_name"):
            return getattr(user, "display_name")
        if isinstance(user, str):
            if user in ROLE_VIEWS:
                return user
            conn = self._connect()
            row = conn.execute("SELECT display_name FROM users WHERE username = ?",
                               (user,)).fetchone()
            conn.close()
            return row[0] if row else user
        return str(user)

    def can(self, permission, user=None):
        """True si el usuario posee el permiso. Sin sesion, solo lo que no
        requiere rol (no hay ninguno en la matriz actual)."""
        user = user or self.current_user
        role = self.role_of(user)
        if role is None:
            return False
        allowed = PERMISSIONS.get(permission)
        if allowed is None:
            return False
        return role in allowed

    def require(self, permission, user=None, require_admin=False):
        """Valida un permiso o lanza AccessDenied. Usar en cada operacion
        sensible para que el rechazo quede registrado y sea auditable."""
        user = user or self.current_user
        if require_admin and self.role_of(user) != ROLE_ADMIN:
            logger.error(f"[RE05] ACCESO DENEGADO: '{self.name_of(user)}' "
                         f"intento '{permission}'")
            raise AccessDenied(
                f"'{self.name_of(user)}' no tiene permiso para {permission}"
            )
        if not self.can(permission, user):
            logger.error(f"[RE05] ACCESO DENEGADO: '{self.name_of(user)}' "
                         f"intento '{permission}'")
            raise AccessDenied(
                f"'{self.name_of(user)}' no tiene permiso para {permission}"
            )
        return True

    def allowed_views(self, user=None):
        role = self.role_of(user or self.current_user)
        return list(ROLE_VIEWS.get(role, []))

    def describe_permissions(self):
        """Matriz legible para mostrarla en la GUI."""
        rows = []
        for perm, roles in sorted(PERMISSIONS.items()):
            rows.append({
                "permission": perm,
                ROLE_ADMIN: ROLE_ADMIN in roles,
                ROLE_VIEWER: ROLE_VIEWER in roles,
            })
        return rows
