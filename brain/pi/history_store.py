"""Durable sensor history, UTC timestamps, filtered export and resumable replication."""

from contextlib import contextmanager, closing
import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sqlite3
import threading
import uuid


COLUMNS = ("seq", "kind", "timestamp", "sensor_id", "sensor_name", "type", "source",
           "level", "status", "detail", "value", "unit", "state", "min_c", "max_c", "mean_c", "center_c")
DEFAULT_SETTINGS = {"record_interval_seconds": 5, "retention_days": 0}


def utc_timestamp(value=None):
    if value is None:
        value = datetime.now(timezone.utc)
    elif isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # Legacy CSV timestamps are in the Pi's local timezone.
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def csv_line(values):
    stream = io.StringIO(newline="")
    # Text names may be opened in spreadsheet software; keep them literal.
    safe = ["'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value for value in values]
    csv.writer(stream).writerow(safe)
    return stream.getvalue()


class HistoryStore:
    def __init__(self, path, legacy_csv=None):
        self.path = Path(path)
        self.legacy_csv = Path(legacy_csv) if legacy_csv else None
        self._lock = threading.RLock()
        self._ready = False

    def _connect(self):
        con = sqlite3.connect(str(self.path), timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def _initialize(self):
        with self._lock:
            if self._ready:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as con:
                con.execute("PRAGMA journal_mode=WAL")
                con.executescript("""
                    CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS records (
                        seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        kind TEXT NOT NULL, timestamp TEXT NOT NULL,
                        sensor_id TEXT NOT NULL, sensor_name TEXT, type TEXT, source TEXT,
                        level TEXT, status TEXT, detail TEXT, value REAL, unit TEXT, state TEXT,
                        min_c REAL, max_c REAL, mean_c REAL, center_c REAL);
                    CREATE INDEX IF NOT EXISTS records_time ON records(timestamp);
                    CREATE INDEX IF NOT EXISTS records_filter ON records(kind, sensor_id, seq);
                    CREATE INDEX IF NOT EXISTS records_sensor ON records(sensor_id, seq);
                    CREATE TABLE IF NOT EXISTS sensor_state (sensor_id TEXT PRIMARY KEY, reading TEXT NOT NULL);
                """)
                with con:
                    con.execute("INSERT OR IGNORE INTO metadata VALUES ('dataset_id', ?)", (str(uuid.uuid4()),))
                    con.execute("INSERT OR IGNORE INTO metadata VALUES ('settings', ?)", (json.dumps(DEFAULT_SETTINGS),))
                    con.execute("INSERT OR IGNORE INTO metadata VALUES ('pruned_through', '0')")
                    if not con.execute("SELECT 1 FROM metadata WHERE key='legacy_imported'").fetchone():
                        skipped = 0
                        if self.legacy_csv and self.legacy_csv.is_file():
                            with self.legacy_csv.open(newline="", encoding="utf-8-sig") as source:
                                for row in csv.DictReader(source):
                                    try:
                                        if not row.get("sensor_id"):
                                            raise ValueError("Missing legacy sensor ID")
                                        row["timestamp"] = utc_timestamp(row["timestamp"])
                                        self._insert(con, {**row, "kind": "events", "source": "legacy_csv"})
                                    except (ValueError, KeyError, TypeError):
                                        skipped += 1
                        con.execute("INSERT INTO metadata VALUES ('legacy_imported', '1')")
                        con.execute("INSERT OR REPLACE INTO metadata VALUES ('legacy_skipped', ?)", (str(skipped),))
            self._ready = True

    @contextmanager
    def connection(self):
        self._initialize()
        con = self._connect()
        try:
            with con:
                yield con
        finally:
            con.close()

    @staticmethod
    def _insert(con, record):
        columns = COLUMNS[1:]
        cursor = con.execute(
            "INSERT INTO records (" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")",
            [record.get(key) for key in columns],
        )
        return {**record, "seq": cursor.lastrowid}

    def settings(self):
        with self.connection() as con:
            return json.loads(con.execute("SELECT value FROM metadata WHERE key='settings'").fetchone()[0])

    def save_settings(self, values):
        if not isinstance(values, dict) or set(values) != set(DEFAULT_SETTINGS):
            raise ValueError("Provide record_interval_seconds and retention_days")
        for key, low, high in (("record_interval_seconds", 1, 3600), ("retention_days", 0, 3650)):
            if type(values[key]) is not int or not low <= values[key] <= high:
                raise ValueError(f"{key} must be an integer from {low} to {high}")
        with self.connection() as con:
            con.execute("UPDATE metadata SET value=? WHERE key='settings'", (json.dumps(values),))
        return values

    def add_system_event(self, detail):
        with self.connection() as con:
            return self._insert(con, {"kind": "events", "timestamp": utc_timestamp(), "sensor_id": "sentinel",
                                     "sensor_name": "Sentinel", "type": "system", "source": "pi",
                                     "level": "ok", "status": "ok", "detail": detail})

    def capture(self, readings, timestamp=None):
        """Save a full sampling interval and transitions in the same transaction."""
        timestamp = utc_timestamp(timestamp)
        events = []
        with self.connection() as con:
            previous = {row[0]: json.loads(row[1]) for row in con.execute("SELECT * FROM sensor_state")}
            current = {reading["sensor_id"]: dict(reading) for reading in readings}
            for sensor_id in previous.keys() - current.keys():
                current[sensor_id] = {**previous[sensor_id], "level": "offline", "status": "offline", "detail": "Offline",
                                      **{key: None for key in ("value", "min_c", "max_c", "mean_c", "center_c")}}
            for sensor_id, reading in current.items():
                reading.update(kind="readings", timestamp=timestamp)
                self._insert(con, reading)
                old = previous.get(sensor_id)
                if old is None or (old.get("level"), old.get("state")) != (reading.get("level"), reading.get("state")):
                    event = {**reading, "kind": "events"}
                    if old is None:
                        event["detail"] = "Sensor detected · " + (reading.get("detail") or reading.get("status", ""))
                    events.append(self._insert(con, event))
                con.execute("INSERT OR REPLACE INTO sensor_state VALUES (?, ?)", (sensor_id, json.dumps(reading, allow_nan=False)))
        return events

    def prune(self):
        days = self.settings()["retention_days"]
        if not days:
            return
        cutoff = utc_timestamp(datetime.now(timezone.utc) - timedelta(days=days))
        with self.connection() as con:
            highest = con.execute("SELECT MAX(seq) FROM records WHERE timestamp < ?", (cutoff,)).fetchone()[0]
            if highest is not None:
                con.execute("DELETE FROM records WHERE timestamp < ?", (cutoff,))
                con.execute("UPDATE metadata SET value=CAST(MAX(CAST(value AS INTEGER), ?) AS TEXT) WHERE key='pruned_through'", (highest,))

    def status(self):
        with self.connection() as con:
            metadata = dict(con.execute("SELECT key,value FROM metadata"))
            count, last = con.execute("SELECT COUNT(*), MAX(timestamp) FROM records").fetchone()
            latest = con.execute("SELECT seq FROM sqlite_sequence WHERE name='records'").fetchone()
            sensors = [{"id": row[0], "name": json.loads(row[1]).get("sensor_name") or row[0]}
                       for row in con.execute("SELECT * FROM sensor_state ORDER BY sensor_id")]
            known = {sensor["id"] for sensor in sensors}
            sensors.extend({"id": row[0], "name": row[1] or row[0]} for row in con.execute(
                "SELECT sensor_id, sensor_name FROM records WHERE seq IN (SELECT MAX(seq) FROM records GROUP BY sensor_id)") if row[0] not in known)
        return {"schema_version": 1, "dataset_id": metadata["dataset_id"], "record_count": count,
                "last_recorded_at": last, "latest_seq": latest[0] if latest else 0,
                "legacy_skipped": int(metadata.get("legacy_skipped", 0)),
                "settings": json.loads(metadata["settings"]), "sensors": sensors}

    @staticmethod
    def filters(kind="events", sensor_id="", start="", end="", before_seq=None):
        if kind not in ("events", "readings"):
            raise ValueError("kind must be events or readings")
        clauses, values = ["kind=?"], [kind]
        if sensor_id:
            clauses.append("sensor_id=?")
            values.append(sensor_id)
        for name, operator, value in (("start", ">=", start), ("end", "<", end)):
            if value:
                try:
                    normalized = utc_timestamp(value)
                except (ValueError, TypeError):
                    raise ValueError(name + " must be an ISO timestamp") from None
                clauses.append("timestamp" + operator + "?")
                values.append(normalized)
        if start and end and utc_timestamp(start) >= utc_timestamp(end):
            raise ValueError("End must be after start")
        if before_seq is not None:
            if type(before_seq) is not int or before_seq < 1:
                raise ValueError("before_seq must be a positive integer")
            clauses.append("seq < ?")
            values.append(before_seq)
        return " AND ".join(clauses), values

    def query(self, *, limit=200, **filters):
        where, values = self.filters(**filters)
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be from 1 to 1000")
        with self.connection() as con:
            rows = [dict(row) for row in con.execute("SELECT * FROM records WHERE " + where + " ORDER BY seq DESC LIMIT ?", [*values, limit + 1])]
        return {"records": rows[:limit], "has_more": len(rows) > limit,
                "before_seq": rows[limit - 1]["seq"] if len(rows) > limit else None}

    def export_csv(self, **filters):
        where, values = self.filters(**filters)
        # Validation happens before returning a generator so invalid requests return HTTP 400.
        def generate():
            yield "\ufeff" + csv_line(COLUMNS)
            with self.connection() as con:
                for row in con.execute("SELECT * FROM records WHERE " + where + " ORDER BY seq", values):
                    yield csv_line([row[key] for key in COLUMNS])
        return generate()

    def sync_batch(self, after=0, limit=500):
        if type(after) is not int or after < 0 or not 1 <= limit <= 1000:
            raise ValueError("Invalid history cursor or limit")
        with self.connection() as con:
            con.execute("BEGIN")
            metadata = dict(con.execute("SELECT key,value FROM metadata"))
            result = con.execute("SELECT seq FROM sqlite_sequence WHERE name='records'").fetchone()
            latest = result[0] if result else 0
            if after > latest:
                raise ValueError("Pi history is older than the PC archive; restore the current Pi database")
            rows = [dict(row) for row in con.execute("SELECT * FROM records WHERE seq > ? ORDER BY seq LIMIT ?", (after, limit))]
            cursor = rows[-1]["seq"] if rows else latest
            return {"schema_version": 1, "dataset_id": metadata["dataset_id"], "records": rows,
                    "next_cursor": cursor, "latest_seq": latest, "has_more": cursor < latest,
                    "pruned_through": int(metadata.get("pruned_through", 0))}
