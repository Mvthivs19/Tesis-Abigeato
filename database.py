"""
Base de datos de eventos (SQLite: liviana, sin servidor, ideal para
un despliegue Edge de bajo consumo).

Tabla `events`: cada detección de intrusión confirmada (RE13, RE17).
Tabla `health_log`: histórico de estado del sistema (RE21).
Tabla `notification_queue`: cola de notificaciones remotas pendientes,
usada cuando no hay Internet disponible (RE19).
"""

import sqlite3
from datetime import datetime


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

        conn.commit()
        conn.close()

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
