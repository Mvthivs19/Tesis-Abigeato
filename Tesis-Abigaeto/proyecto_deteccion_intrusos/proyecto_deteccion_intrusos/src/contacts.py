"""
RE04: Directorio de contactos de emergencia.

El Administrador mantiene una lista de personas a quienes notificar cuando el
sistema detecta una intrusion: el productor, un familiar, el guarda de la
caseta, carabineros del sector, etc.

Cada contacto tiene un prioridad. Cuando se dispara una alerta, se notifican
los contactos en orden de prioridad ascendente y el sistema deja registro de
a quienes se aviso y a quienes no, para poder auditar que la cadena de
notificacion funciono.

No se guardan contrasenas ni datos sensibles: solo nombre, relacion, telefono
y canal preferido.
"""

import re
import sqlite3
from datetime import datetime

try:
    from src.utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger("contacts")

CHANNELS = ("sms", "email", "webhook", "telefono")

# Validacion de telefono chileno: 9 digitos que empiezan por 9 (movil),
# con prefijo de pais opcional (+56 / 56) y separadores opcionales.
_PHONE_RE = re.compile(r"^(?:\+?56)?9\d{8}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ContactDirectory:
    def __init__(self, db_path="data/events.db"):
        self.db_path = db_path
        self._init_schema()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_schema(self):
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                contact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                relation TEXT,
                phone TEXT,
                email TEXT,
                channel TEXT NOT NULL DEFAULT 'sms',
                priority INTEGER NOT NULL DEFAULT 10,
                active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def add(self, name, phone=None, email=None, channel="sms", priority=10,
            relation=None, actor_can=None):
        """RE04: el Administrador registra un contacto de emergencia."""
        if not name or not str(name).strip():
            raise ValueError("El nombre del contacto es obligatorio")
        if channel not in CHANNELS:
            raise ValueError(f"Canal invalido: {channel}. Use uno de {CHANNELS}")
        if phone:
            digits = re.sub(r"[\s\-]", "", str(phone))
            if not _PHONE_RE.match(digits):
                raise ValueError(f"Telefono chileno invalido: {phone}")
        if email and not _EMAIL_RE.match(str(email)):
            raise ValueError(f"Email invalido: {email}")
        if channel == "sms" and not phone:
            raise ValueError("Un contacto SMS requiere telefono")
        if channel == "email" and not email:
            raise ValueError("Un contacto de email requiere direccion")

        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO contacts (name, relation, phone, email, channel,
                                     priority, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name.strip(), relation, phone, email, channel, priority,
             datetime.now().isoformat()),
        )
        contact_id = cur.lastrowid
        conn.commit()
        conn.close()
        logger.info(f"[RE04] Contacto '{name}' agregado (canal {channel}, "
                    f"prioridad {priority})")
        return contact_id

    def update(self, contact_id, actor_can=None, **fields):
        allowed = {"name", "relation", "phone", "email", "channel",
                   "priority", "active"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Campos no editables: {bad}")
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        conn = self._connect()
        conn.execute(f"UPDATE contacts SET {sets} WHERE contact_id = ?",
                     list(fields.values()) + [contact_id])
        conn.commit()
        conn.close()
        logger.info(f"[RE04] Contacto {contact_id} actualizado: {fields}")
        return True

    def remove(self, contact_id, actor_can=None):
        conn = self._connect()
        conn.execute("DELETE FROM contacts WHERE contact_id = ?", (contact_id,))
        conn.commit()
        conn.close()
        logger.info(f"[RE04] Contacto {contact_id} eliminado")
        return True

    def list_contacts(self, only_active=True):
        sql = "SELECT contact_id, name, relation, phone, email, channel, priority, active FROM contacts"
        if only_active:
            sql += " WHERE active = 1"
        sql += " ORDER BY priority ASC, name ASC"
        conn = self._connect()
        rows = conn.execute(sql).fetchall()
        conn.close()
        return [
            {"contact_id": r[0], "name": r[1], "relation": r[2], "phone": r[3],
             "email": r[4], "channel": r[5], "priority": r[6],
             "active": bool(r[7])}
            for r in rows
        ]

    def get(self, contact_id):
        conn = self._connect()
        row = conn.execute(
            """SELECT contact_id, name, relation, phone, email, channel,
                      priority, active
               FROM contacts WHERE contact_id = ?""",
            (contact_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return {"contact_id": row[0], "name": row[1], "relation": row[2],
                "phone": row[3], "email": row[4], "channel": row[5],
                "priority": row[6], "active": bool(row[7])}

    def notify_all(self, event_id, dispatcher=None):
        """Notifica a todos los contactos activos en orden de prioridad.

        `dispatcher` es una funcion `dispatcher(contact, event_id) -> bool`
        que intenta entregar la notificacion por el canal que corresponda.
        Si no se entrega, se reintenta en el siguiente ciclo, igual que la
        cola de notificaciones remotas (RE19).

        Devuelve un resumen {"notificados": n, "fallidos": n, "detalle": [...]}.
        """
        contacts = self.list_contacts(only_active=True)
        if not contacts:
            logger.warning("[RE04] No hay contactos de emergencia registrados")
            return {"notificados": 0, "fallidos": 0, "detalle": []}

        summary = {"notificados": 0, "fallidos": 0, "detalle": []}
        for contact in contacts:
            ok = False
            if dispatcher is not None:
                try:
                    ok = bool(dispatcher(contact, event_id))
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"[RE04] Error notificando a "
                                 f"{contact['name']}: {exc}")
                    ok = False
            summary["detalle"].append({
                "contact_id": contact["contact_id"],
                "name": contact["name"],
                "channel": contact["channel"],
                "priority": contact["priority"],
                "delivered": ok,
            })
            if ok:
                summary["notificados"] += 1
            else:
                summary["fallidos"] += 1

        logger.info(
            f"[RE04] Evento {event_id}: {summary['notificados']} contactos "
            f"notificados, {summary['fallidos']} fallidos de {len(contacts)}"
        )
        return summary
