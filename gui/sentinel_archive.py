"""PC log replication and cloud-folder copies; no Qt or third-party dependencies."""

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import urllib.error
import urllib.request
import uuid

COLUMNS = ("seq", "kind", "timestamp", "sensor_id", "sensor_name", "type", "source",
           "level", "status", "detail", "value", "unit", "state", "min_c", "max_c", "mean_c", "center_c")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def fetch_json(base_url, path, timeout=4):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(base_url + path, headers={"Accept": "application/json"})
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(8 * 1024 * 1024 + 1)
        if len(body) > 8 * 1024 * 1024:
            raise ValueError("History response is too large")
        result = json.loads(body)
        if not isinstance(result, dict) or result.get("schema_version") != 1:
            raise ValueError("Update the Pi files to enable log backup")
        return result
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise ValueError("Update the Pi files to enable log backup") from None
        try:
            detail = json.loads(error.read(8192)).get("error", str(error))
        except (ValueError, AttributeError):
            detail = str(error)
        raise ValueError(detail) from None


def dataset_name(value):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError("Invalid Pi dataset ID")
    return value


def safe_row(row):
    if not isinstance(row, dict) or set(row) != set(COLUMNS):
        raise ValueError("Invalid history record schema")
    if type(row["seq"]) is not int or row["seq"] < 1 or row["kind"] not in ("events", "readings"):
        raise ValueError("Invalid history record ID or kind")
    if not isinstance(row["timestamp"], str) or not isinstance(row["sensor_id"], str):
        raise ValueError("Invalid history timestamp or sensor ID")
    stamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
    if stamp.tzinfo is None or not row["sensor_id"]:
        raise ValueError("History needs UTC timestamps and sensor IDs")
    for key in ("value", "min_c", "max_c", "mean_c", "center_c"):
        value = row[key]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("Invalid history reading")
    for key in set(COLUMNS) - {"seq", "value", "min_c", "max_c", "mean_c", "center_c"}:
        if row[key] is not None and not isinstance(row[key], str):
            raise ValueError("Invalid history text")
    return row


@contextmanager
def atomic_file(destination, *, binary=False):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="wb" if binary else "w", prefix=".sentinel-", suffix=".tmp",
                                         dir=destination.parent, delete=False,
                                         **({} if binary else {"encoding": "utf-8-sig", "newline": ""}))
    temporary = Path(handle.name)
    try:
        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class ArchiveReplica:
    def __init__(self, root, dataset_id):
        self.dataset_id = dataset_name(dataset_id)
        self.directory = Path(root) / self.dataset_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "archive.sqlite3"
        with self.connection() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS records (seq INTEGER PRIMARY KEY, day TEXT NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS records_day ON records(day, kind, seq);
                CREATE TABLE IF NOT EXISTS dirty_days (day TEXT PRIMARY KEY);
            """)
            con.execute("INSERT OR IGNORE INTO metadata VALUES ('cursor','0')")
            con.execute("INSERT OR IGNORE INTO metadata VALUES ('dataset_id', ?)", (self.dataset_id,))

    @contextmanager
    def connection(self):
        con = sqlite3.connect(str(self.path), timeout=10)
        try:
            with con:
                yield con
        finally:
            con.close()

    def cursor(self):
        with self.connection() as con:
            return int(con.execute("SELECT value FROM metadata WHERE key='cursor'").fetchone()[0])

    def ingest(self, batch):
        if dataset_name(batch.get("dataset_id")) != self.dataset_id:
            raise ValueError("The Pi history database changed; reconnect to start its new archive")
        rows = batch.get("records")
        cursor, latest = batch.get("next_cursor"), batch.get("latest_seq")
        if not isinstance(rows, list) or len(rows) > 1000 or type(cursor) is not int or type(latest) is not int or not 0 <= cursor <= latest:
            raise ValueError("Invalid history cursor")
        if type(batch.get("has_more")) is not bool or batch["has_more"] != (cursor < latest):
            raise ValueError("Invalid history continuation")
        rows = [safe_row(row) for row in rows]
        sequences = [row["seq"] for row in rows]
        if sequences != sorted(set(sequences)) or (rows and cursor != sequences[-1]):
            raise ValueError("History records are out of order")
        with self.connection() as con:
            old_cursor = int(con.execute("SELECT value FROM metadata WHERE key='cursor'").fetchone()[0])
            if not rows and cursor < latest:
                raise ValueError("History cursor made no progress")
            for row in rows:
                data = json.dumps(row, sort_keys=True, allow_nan=False)
                old = con.execute("SELECT data FROM records WHERE seq=?", (row["seq"],)).fetchone()
                if old:
                    if old[0] != data:
                        raise ValueError("A saved record changed on the Pi; keep the current database")
                    continue
                if row["seq"] <= old_cursor:
                    raise ValueError("A history record is missing below the saved cursor")
                day = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).astimezone(timezone.utc).date().isoformat()
                con.execute("INSERT INTO records VALUES (?, ?, ?, ?)", (row["seq"], day, row["kind"], data))
                con.execute("INSERT OR IGNORE INTO dirty_days VALUES (?)", (day,))
            con.execute("UPDATE metadata SET value=? WHERE key='cursor'", (str(max(cursor, old_cursor)),))
            if batch.get("pruned_through", 0) > old_cursor:
                con.execute("INSERT OR REPLACE INTO metadata VALUES ('warning', ?)",
                            ("Older Pi records expired before they reached this PC",))

    def warning(self):
        with self.connection() as con:
            row = con.execute("SELECT value FROM metadata WHERE key='warning'").fetchone()
            return row[0] if row else ""

    def _export_day(self, day, destination):
        for kind in ("events", "readings"):
            with atomic_file(Path(destination) / (day + "-" + kind + ".csv")) as stream, self.connection() as con:
                writer = csv.writer(stream)
                writer.writerow(COLUMNS)
                for (data,) in con.execute("SELECT data FROM records WHERE day=? AND kind=? ORDER BY seq", (day, kind)):
                    row = json.loads(data)
                    values = [row.get(key) for key in COLUMNS]
                    writer.writerow(["'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value for value in values])

    def export_pending(self):
        with self.connection() as con:
            days = [row[0] for row in con.execute("SELECT day FROM dirty_days ORDER BY day")]
        for day in days:
            self._export_day(day, self.directory)
            with self.connection() as con:
                con.execute("DELETE FROM dirty_days WHERE day=?", (day,))

    def save_cloud_folder(self, root):
        """Write consistent snapshots into an existing cloud client's sync directory.

        This confirms local folder writes only, never a remote provider upload.
        """
        if not Path(root).is_dir():
            raise OSError("The cloud sync folder is unavailable; choose an existing folder")
        target = Path(root) / "Sentinel" / self.dataset_id
        key = "cloud_cursor:" + os.path.normcase(str(target.resolve()))
        with self.connection() as con:
            old = con.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
            after = int(old[0]) if old and (target / "history.sqlite3").is_file() else -1
            days = [row[0] for row in con.execute("SELECT DISTINCT day FROM records WHERE seq > ?", (after,))]
        for day in days:
            self._export_day(day, target)
        target.mkdir(parents=True, exist_ok=True)
        # SQLite backup copies committed WAL pages as well; copying the live file would not.
        handle, temporary = tempfile.mkstemp(prefix=".sentinel-", suffix=".sqlite3", dir=target)
        os.close(handle)
        try:
            with self.connection() as source:
                destination = sqlite3.connect(temporary)
                try:
                    source.backup(destination)
                finally:
                    destination.close()
            os.replace(temporary, target / "history.sqlite3")
        finally:
            Path(temporary).unlink(missing_ok=True)
        cursor = self.cursor()
        with atomic_file(target / "manifest.json") as stream:
            json.dump({"dataset_id": self.dataset_id, "through_seq": cursor,
                       "saved_at": datetime.now(timezone.utc).isoformat(),
                       "status": "Saved to local cloud folder; remote upload is managed by the sync client"}, stream, indent=2)
        with self.connection() as con:
            con.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", (key, str(cursor)))
        return str(target)


def replicate(base_url, pc_directory, cloud_directory="", *, cloud_due=False, cancelled=lambda: False, progress=lambda value: None):
    metadata = fetch_json(base_url, "/api/history/status")
    replica = ArchiveReplica(pc_directory, metadata.get("dataset_id"))
    result = {"pc": "Saving…", "cloud": "Waiting" if cloud_directory else "Off", "error": "", "cloud_saved": False}
    more = False
    for _ in range(10):
        if cancelled():
            return {**result, "pc": "Paused"}
        cursor = replica.cursor()
        batch = fetch_json(base_url, f"/api/history/sync?after={cursor}&limit=500")
        if cancelled():
            return {**result, "pc": "Paused"}
        if batch.get("has_more") and batch.get("next_cursor", 0) <= cursor:
            raise ValueError("History cursor made no progress")
        replica.ingest(batch)
        more = batch["has_more"]
        progress({**result, "pc": f"Saving · {replica.cursor():,} records processed"})
        if not more:
            break
    replica.export_pending()
    result.update(pc="Catching up…" if more else "Saved", more=more, cursor=replica.cursor(),
                  directory=str(replica.directory), warning=replica.warning())
    if cloud_directory and cloud_due and not cancelled():
        try:
            result["cloud_directory"] = replica.save_cloud_folder(cloud_directory)
            result.update(cloud="Folder saved · upload not verified", cloud_saved=True)
        except (OSError, sqlite3.Error) as error:
            result.update(cloud="Folder write failed", error=str(error))
    return result
