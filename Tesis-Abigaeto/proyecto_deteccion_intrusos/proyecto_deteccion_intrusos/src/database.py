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
Tabla `integrity_log`: histórico de compromiso fisico o degradacion de la
  camara: lente tapada, encuadre desplazado, perdida de senal (RE21).
Tabla `notification_queue`: cola de notificaciones remotas pendientes,
  usada cuando no hay Internet disponible (RE19, RE20).
Tabla `contact_log`: auditoria de a quien se notifico en cada evento (RE04).
Tabla `false_positive_export`: trazabilidad de las exportaciones del dataset
  de falsos positivos para reentrenamiento (RE18).
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
    # RE18: caja detectada (JSON "x1,y1,x2,y2") y clase REAL declarada por el
    # operador. Sin la caja, el exportador de falsos positivos no puede escribir
    # una etiqueta YOLO de objeto y la muestra degrada a imagen sin etiqueta.
    ("bbox", "TEXT"),
    ("real_class", "TEXT"),
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

        # RE21: compromiso fisico o degradacion de la camara.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS integrity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                detail TEXT
            )
        """)

        # RE04: auditoria de notificaciones a contactos de emergencia.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS contact_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                contact_id INTEGER,
                contact_name TEXT,
                channel TEXT,
                delivered INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

        # RE18: control de exportaciones del dataset de falsos positivos.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS false_positive_export (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                export_path TEXT NOT NULL,
                samples INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                note TEXT
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

    def insert_event(self, class_name, confidence, snapshot_path=None, clip_path=None,
                     bbox=None):
        """Registra un evento y encola su notificacion remota (RE16/RE19).

        `bbox` es la caja que produjo la deteccion, en pixeles del frame
        ("x1,y1,x2,y2"). La guarda el exportador de RE18 para poder escribir una
        etiqueta YOLO con caja.
        """
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO events (timestamp, class_name, confidence, snapshot_path,
                                  clip_path, bbox)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), class_name, confidence, snapshot_path,
             clip_path, bbox),
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

    def get_unacknowledged_events(self, older_than_seconds, limit=5,
                                  max_age_seconds=None):
        """Eventos de intrusion que el operador aun no ha confirmado y que
        llevan mas de `older_than_seconds` en espera (candidatos a RE15).

        `max_age_seconds` acota la antiquity del candidato. Sin ese techo, un
        evento de hace semanas sin confirmar dispara una escalacion de
        emergencia en cada arranque del sistema: una alerta de hace un mes no
        describe el estado actual de la finca y entrena al operador a ignorar
        las escalaciones. Los eventos mas viejos siguen visibles en el
        historial como pendientes de confirmacion, que es lo que corresponde.
        """
        cutoff = (datetime.now() - timedelta(seconds=older_than_seconds)).isoformat()
        sql = (
            """SELECT event_id, timestamp, class_name, confidence, snapshot_path
               FROM events
               WHERE acknowledged = 0 AND escalated = 0 AND timestamp <= ?"""
        )
        args = [cutoff]
        if max_age_seconds:
            sql += " AND timestamp >= ?"
            args.append(
                (datetime.now()
                 - timedelta(seconds=max_age_seconds)).isoformat()
            )
        sql += " ORDER BY timestamp ASC LIMIT ?"
        args.append(limit)
        conn = self._connect()
        rows = conn.execute(sql, args).fetchall()
        conn.close()
        return rows

    def mark_event_escalated(self, event_id):
        conn = self._connect()
        conn.execute("UPDATE events SET escalated = 1 WHERE event_id = ?", (event_id,))
        conn.commit()
        conn.close()

    # ---------- RE15/RE18: confirmacion del operador ----------

    def mark_event_acknowledged(self, event_id, is_false_positive=False,
                                real_class=None):
        """Confirma el evento (RE15) y, si es falso positivo, guarda la clase
        REAL (RE18).

        `real_class` es lo que el operador vio realmente en la imagen. No se
        deduce de la prediccion: si el modelo dijo "humano" sobre una sombra y el
        operador no declara nada, la muestra se exporta como `background`, que es
        la unica etiqueta honesta.
        """
        conn = self._connect()
        conn.execute(
            """UPDATE events
               SET acknowledged = 1, acknowledged_at = ?, is_false_positive = ?,
                   real_class = COALESCE(?, real_class)
               WHERE event_id = ?""",
            (datetime.now().isoformat(), 1 if is_false_positive else 0,
             real_class, event_id),
        )
        conn.commit()
        conn.close()

    def get_last_unacknowledged_event(self, max_age_seconds=None):
        """Ultimo evento de intrusion aun sin confirmar, o None.

        `max_age_seconds` acota la busqueda por antiguedad. Sin ese tope, un
        evento viejo que nadie confirmo (porque el equipo estaba apagado, o
        porque nadie vio la alerta) seguia apareciendo para siempre como "la
        alerta abierta" en el HUD, semanas despues de ocurrido. Confirmar un
        evento viejo sigue siendo posible: quien llama decide si aplica el
        filtro.
        """
        conn = self._connect()
        sql = ("SELECT event_id, class_name, confidence "
               "FROM events WHERE acknowledged = 0")
        params = []
        if max_age_seconds is not None:
            # El corte se compara como texto ISO-8601, igual que en
            # `get_unacknowledged_events`. `julianday(timestamp)` NO sirve
            # aqui: `datetime.isoformat()` escribe microsegundos (6 digitos) y
            # las funciones de fecha de SQLite solo admiten milisegundos, por
            # lo que devuelven NULL y el filtro terminaba descartando todos los
            # eventos, incluidos los recien creados.
            cutoff = (datetime.now() - timedelta(seconds=max_age_seconds)).isoformat()
            sql += " AND timestamp >= ?"
            params.append(cutoff)
        sql += " ORDER BY event_id DESC LIMIT 1"
        row = conn.execute(sql, params).fetchone()
        conn.close()
        return row

    # ---------- Tablas de apoyo para la GUI ----------

    def get_all_events(self, limit=500, class_name=None):
        """Historial de eventos para la vista de historial.

        `class_name` filtra por clase. Antes cada vista montaba su propio SQL
        sobre `sqlite3`, y si el esquema cambiaba la vista se quedaba vacia en
        silencio; consuming esta consulta, una migracion se rompe una sola vez
        y en un lugar visible.
        """
        conn = self._connect()
        sql = (
            """SELECT event_id, timestamp, class_name, confidence,
                      snapshot_path, clip_path, acknowledged, is_false_positive,
                      escalated
               FROM events"""
        )
        params = []
        if class_name:
            sql += " WHERE class_name = ?"
            params.append(class_name)
        sql += " ORDER BY event_id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return rows

    def count_false_positives(self):
        conn = self._connect()
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE is_false_positive = 1"
        ).fetchone()
        conn.close()
        return row[0] if row else 0

    def get_false_positive_events(self, limit=1000):
        """Eventos etiquetados como falso positivo, con su snapshot (RE18).

        Son las muestras que el Administrador puede exportar para reentrenar
        el modelo, de modo que aprenda a no disparar ante esos casos. Incluye
        `bbox` (caja detectada), `real_class` (clase declarada por el
        operador) y `clip_path` para que el exportador pueda escribir una
        etiqueta YOLO util y copia el clip probatorio.
        """
        conn = self._connect()
        rows = conn.execute(
            """SELECT event_id, timestamp, class_name, confidence, snapshot_path,
                      bbox, real_class, clip_path
               FROM events
               WHERE is_false_positive = 1
               ORDER BY event_id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        return rows

    def record_false_positive_export(self, export_path, samples, note=""):
        conn = self._connect()
        conn.execute(
            """INSERT INTO false_positive_export (export_path, samples, created_at, note)
               VALUES (?, ?, ?, ?)""",
            (export_path, samples, datetime.now().isoformat(), note),
        )
        conn.commit()
        conn.close()

    def get_false_positive_exports(self, limit=20):
        conn = self._connect()
        rows = conn.execute(
            """SELECT id, export_path, samples, created_at, note
               FROM false_positive_export ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        return rows

    # ---------- RE21: integridad fisica de la camara ----------

    def insert_integrity_log(self, timestamp, event_type, severity, detail=""):
        conn = self._connect()
        conn.execute(
            """INSERT INTO integrity_log (timestamp, event_type, severity, detail)
               VALUES (?, ?, ?, ?)""",
            (timestamp, event_type, severity, detail),
        )
        conn.commit()
        conn.close()

    def get_integrity_logs(self, limit=200):
        conn = self._connect()
        rows = conn.execute(
            """SELECT id, timestamp, event_type, severity, detail
               FROM integrity_log ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        return rows

    def count_integrity_alerts(self, since_minutes=1440):
        cutoff = (datetime.now() - timedelta(minutes=since_minutes)).isoformat()
        conn = self._connect()
        row = conn.execute(
            "SELECT COUNT(*) FROM integrity_log WHERE timestamp >= ? AND severity = 'critical'",
            (cutoff,),
        ).fetchone()
        conn.close()
        return row[0] if row else 0

    # ---------- RE04: auditoria de contactos ----------

    def insert_contact_log(self, event_id, contact):
        conn = self._connect()
        conn.execute(
            """INSERT INTO contact_log (event_id, contact_id, contact_name, channel,
                                        delivered, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (event_id, contact.get("contact_id"), contact.get("name"),
             contact.get("channel"), 1 if contact.get("delivered") else 0,
             datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def get_contact_notifications(self, event_id):
        conn = self._connect()
        rows = conn.execute(
            """SELECT contact_name, channel, delivered, created_at
               FROM contact_log WHERE event_id = ? ORDER BY id ASC""",
            (event_id,),
        ).fetchall()
        conn.close()
        return rows

    # ---------- Resumen para el dashboard ----------

    def _table_exists(self, cur, name):
        row = cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None

    def get_event_stats(self, since_minutes=None):
        """Cifras agregadas para el dashboard. `since_minutes` acota la ventana."""
        conn = self._connect()
        cur = conn.cursor()
        if since_minutes:
            cutoff = (datetime.now() - timedelta(minutes=since_minutes)).isoformat()
            time_clause, args = " AND timestamp >= ?", (cutoff,)
        else:
            time_clause, args = "", ()

        def scalar(sql):
            return cur.execute(sql + time_clause, args).fetchone()[0]

        # `contacts` la crea ContactDirectory, que puede no haberse
        # instanciado todavia. Se comprueba en vez de asumir, porque el
        # dashboard se dibuja desde el arranque.
        active_contacts = 0
        if self._table_exists(cur, "contacts"):
            active_contacts = cur.execute(
                "SELECT COUNT(*) FROM contacts WHERE active = 1"
            ).fetchone()[0]

        stats = {
            "total": scalar("SELECT COUNT(*) FROM events WHERE 1=1"),
            "pending": scalar("SELECT COUNT(*) FROM events WHERE acknowledged = 0"),
            "confirmed": scalar(
                "SELECT COUNT(*) FROM events WHERE acknowledged = 1 AND is_false_positive = 0"
            ),
            "false_positives": scalar(
                "SELECT COUNT(*) FROM events WHERE is_false_positive = 1"
            ),
            "escalated": scalar("SELECT COUNT(*) FROM events WHERE escalated = 1"),
            "avg_confidence": round(
                scalar("SELECT AVG(confidence) FROM events WHERE 1=1") or 0.0, 4
            ),
            "active_contacts": active_contacts,
            "pending_notifications": cur.execute(
                "SELECT COUNT(*) FROM notification_queue WHERE delivered = 0"
            ).fetchone()[0],
        }
        conn.close()
        return stats
