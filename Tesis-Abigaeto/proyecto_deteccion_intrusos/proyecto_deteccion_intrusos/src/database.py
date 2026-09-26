"""
Base de datos de eventos (SQLite: liviana, sin servidor, ideal para
un despliegue Edge de bajo consumo).

Tabla `events`: cada detección de intrusión confirmada (RE13, RE17).
  - acknowledged / acknowledged_at : el operador confirmó o silenció el evento.
    Si un evento sigue sin confirmar pasado `escalate_after_seconds`, el
    sistema lo escala automáticamente (RE15).
  - is_false_positive : el operador lo marcó como falso positivo para
    poder reentrenar el modelo con esos casos (RE18).
  - escalated        : ya se emitió la alerta de escalamiento (RE15).
  - clip_path        : clip de video probatorio (RE17).
Tabla `health_log`: histórico de estado del sistema (RE21).
Tabla `notification_queue`: cola de notificaciones remotas pendientes,
usada cuando no hay Internet disponible (RE19, RE20).
"""

import sqlite3
from datetime import datetime, timedelta

# Columnas de `events` que gestionan la confirmacion y el escalamiento
# del operador. Se agrega cada una por separado para tolerar bases de datos
# creadas por versiones anteriores del sistema.
_EVENT_MIGRATIONS = [
    ("acknowledged", "INTEGER DEFAULT 0"),
    ("acknowledged_at", "TEXT"),
    ("is_false_positive", "INTEGER DEFAULT 0"),
    ("escalated", "INTEGER DEFAULT 0"),
]


class EventDatabase:
    def __init__(self, db_path="data/events.db"):
        self.db_path = db_path
        self._init_schema()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_schema(self):
        conn = self._connect()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                snapshot_path TEXT,
                clip_path TEXT,
                notified INTEGER DEFAULT 0
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS health_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_percent REAL,
                ram_percent REAL,
                status TEXT,
                note TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS notification_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                attempts INTEGER DEFAULT 0,
                delivered INTEGER DEFAULT 0
            )
        """)

        # Primero la migracion (agrega las columnas faltantes) y despues el
        # indice, porque el indice referencia columns que pueden no existir
        # todavia en bases de datos creadas por versiones anteriores.
        self._migrate_events(cur)

        # Indice para las consultas de escalamiento (RE15), que siempre
        # filtran por confirmado + antiguedad.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_ack
            ON events (acknowledged, escalated, timestamp)
        """)

        conn.commit()
        conn.close()

    def _migrate_events(self, cur):
        """Agrega a `events` las columnas que falten, sin perder datos."""
        existing = {row[1] for row in cur.execute("PRAGMA table_info(events)")}
        for column, ddl in _EVENT_MIGRATIONS:
            if column not in existing:
                cur.execute(f"ALTER TABLE events ADD COLUMN {column} {ddl}")

    def insert_event(self, class_name, confidence, snapshot_path=None, clip_path=None):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO events (timestamp, class_name, confidence, snapshot_path, clip_path)
               VALUES (?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), class_name, confidence, snapshot_path, clip_path),
        )
        event_id = cur.lastrowid

        cur.execute(
            "INSERT INTO notification_queue (event_id, created_at) VALUES (?, ?)",
            (event_id, datetime.now().isoformat()),
        )

        conn.commit()
        conn.close()
        return event_id

    def get_pending_notifications(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT nq.id, e.event_id, e.timestamp, e.class_name, e.confidence, e.snapshot_path
            FROM notification_queue nq
            JOIN events e ON e.event_id = nq.event_id
            WHERE nq.delivered = 0
            ORDER BY nq.created_at ASC
        """)
        rows = cur.fetchall()
        conn.close()
        return rows

    def mark_notification_delivered(self, queue_id):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("UPDATE notification_queue SET delivered = 1 WHERE id = ?", (queue_id,))
        cur.execute(
            "UPDATE events SET notified = 1 WHERE event_id = (SELECT event_id FROM notification_queue WHERE id = ?)",
            (queue_id,),
        )
        conn.commit()
        conn.close()

    def increment_notification_attempt(self, queue_id):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("UPDATE notification_queue SET attempts = attempts + 1 WHERE id = ?", (queue_id,))
        conn.commit()
        conn.close()

    def get_notification_attempts(self, queue_id):
        conn = self._connect()
        row = conn.execute(
            "SELECT attempts FROM notification_queue WHERE id = ?", (queue_id,)
        ).fetchone()
        conn.close()
        return row[0] if row else 0

    def insert_health_log(self, cpu_percent, ram_percent, status, note=""):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO health_log (timestamp, cpu_percent, ram_percent, status, note)
               VALUES (?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), cpu_percent, ram_percent, status, note),
        )
        conn.commit()
        conn.close()

    # ---------- RE17: clip de video probatorio ----------

    def update_event_clip(self, event_id, clip_path):
        conn = self._connect()
        conn.execute("UPDATE events SET clip_path = ? WHERE event_id = ?",
                     (clip_path, event_id))
        conn.commit()
        conn.close()

    # ---------- RE13/RE19: resolucion de la cola ----------

    def get_queue_id(self, event_id):
        """Devuelve el id de la fila encolada para un evento, o None."""
        conn = self._connect()
        row = conn.execute(
            "SELECT id FROM notification_queue WHERE event_id = ? ORDER BY id DESC LIMIT 1",
            (event_id,),
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def get_pending_count(self):
        conn = self._connect()
        row = conn.execute(
            "SELECT COUNT(*) FROM notification_queue WHERE delivered = 0"
        ).fetchone()
        conn.close()
        return row[0] if row else 0

    # ---------- RE15: escalamiento por inaccion ----------

    def get_unacknowledged_events(self, older_than_seconds, limit=5):
        """Eventos de intrusion que el operador aun no ha confirmado y que
        llevan mas de `older_than_seconds` en espera (candidatos a RE15)."""
        cutoff = (datetime.now() - timedelta(seconds=older_than_seconds)).isoformat()
        conn = self._connect()
        rows = conn.execute(
            """SELECT event_id, timestamp, class_name, confidence, snapshot_path
               FROM events
               WHERE acknowledged = 0 AND escalated = 0 AND timestamp <= ?
               ORDER BY timestamp ASC
               LIMIT ?""",
            (cutoff, limit),
        ).fetchall()
        conn.close()
        return rows

    def mark_event_escalated(self, event_id):
        conn = self._connect()
        conn.execute("UPDATE events SET escalated = 1 WHERE event_id = ?", (event_id,))
        conn.commit()
        conn.close()

    # ---------- RE15/RE18: confirmacion del operador ----------

    def mark_event_acknowledged(self, event_id, is_false_positive=False):
        conn = self._connect()
        conn.execute(
            """UPDATE events
               SET acknowledged = 1, acknowledged_at = ?, is_false_positive = ?
               WHERE event_id = ?""",
            (datetime.now().isoformat(), 1 if is_false_positive else 0, event_id),
        )
        conn.commit()
        conn.close()

    def get_last_unacknowledged_event(self):
        """Ultimo evento de intrusion aun sin confirmar, o None."""
        conn = self._connect()
        row = conn.execute(
            """SELECT event_id, class_name, confidence
               FROM events WHERE acknowledged = 0
               ORDER BY event_id DESC LIMIT 1"""
        ).fetchone()
        conn.close()
        return row

    # ---------- Tablas de apoyo para la GUI ----------

    def get_all_events(self, limit=500):
        conn = self._connect()
        rows = conn.execute(
            """SELECT event_id, timestamp, class_name, confidence,
                      snapshot_path, clip_path, acknowledged, is_false_positive,
                      escalated
               FROM events ORDER BY event_id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        return rows

    def count_false_positives(self):
        conn = self._connect()
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE is_false_positive = 1"
        ).fetchone()
        conn.close()
        return row[0] if row else 0
