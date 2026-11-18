"""Continuous local recording and a durable SQL synchronization outbox."""
import csv
from contextlib import closing, nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from queue import Queue, Full
from queue import Empty
import time
import sqlite3
from threading import Event, Thread
from uuid import uuid4

logger = logging.getLogger("orchestrator.recording")
_synchronizers = {}
_active_recorders = set()


def pending_csv_recorders():
    return [recorder for recorder in _active_recorders.copy() if recorder.pending_csv and recorder.error is None]


def validate_settings(settings):
    settings = dict(settings)
    settings["enabled"] = True  # CSV recording is mandatory; SQL remains optional.
    settings.setdefault("sql_enabled", False)
    # Old settings remain readable, but retired definitions are not propagated.
    settings.pop("unit_operations", None)
    settings.pop("device_units", None)
    if not settings.get("output_folder"):
        raise ValueError("Choose a recording folder")
    url = settings.get("remote_sql_url", "")
    if url and not url.startswith(("sqlite:///", "postgresql+psycopg://")):
        raise ValueError("Remote SQL supports sqlite:/// paths or postgresql+psycopg:// URLs")
    return settings


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class CsvSpool:
    """Durable pending rows; retry locked CSVs without blocking acquisition."""
    def __init__(self, directory):
        self.path = directory / "execution.csv"
        self.pending = directory / "samples.pending.jsonl"
        self.state_path = directory / "csv_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"cursor": 0, "offset": 0}
        self.waiting = False

    def append(self, row):
        with self.pending.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def flush(self):
        if not self.pending.exists() or self.pending.stat().st_size <= self.state["cursor"]:
            return True
        with self.pending.open("r", encoding="utf-8") as stream:
            stream.seek(self.state["cursor"])
            lines = [stream.readline() for _ in range(256)]
            lines = [line for line in lines if line]
            cursor = stream.tell()
        rows = [json.loads(line) for line in lines]
        try:
            # Roll back an interrupted append before retrying. Commit the byte
            # offset only after the complete batch has been flushed.
            with self.path.open("r+" if self.path.exists() else "w+", newline="", encoding="utf-8") as target:
                target.seek(self.state["offset"])
                target.truncate()
                writer = csv.DictWriter(target, fieldnames=list(rows[0]))
                if self.state["offset"] == 0:
                    writer.writeheader()
                writer.writerows(rows)
                target.flush()
                offset = target.tell()
            atomic_json(self.state_path, {"cursor": cursor, "offset": offset})
            self.state = {"cursor": cursor, "offset": offset}
            if self.waiting:
                logger.info("CSV recording resumed: %s", self.path)
            self.waiting = False
            return cursor >= self.pending.stat().st_size
        except PermissionError:
            if not self.waiting:
                logger.warning("CSV is locked. Close it in Excel so saving can resume: %s. Pending rows are retained on disk.", self.path)
            self.waiting = True
            return False

    def clean(self):
        self.pending.unlink(missing_ok=True)
        self.state_path.unlink(missing_ok=True)


class RunRecorder:
    def __init__(self, root, campaign, configs, settings=None):
        self.folder = Path(root) / f"campaign_{campaign.campaign_id}_{uuid4().hex[:8]}"
        self.folder.mkdir(parents=True)
        from experiment.recovery import WriterLease
        self.lease = WriterLease(self.folder)
        if not self.lease.acquire():
            raise RuntimeError("Campaign recording is already owned by another process")
        self.database = self.folder.parent / "recordings.sqlite3"
        self.error = None
        self.queue = Queue(maxsize=10000)
        self.ready = Event()
        self.drained = Event()
        self.complete = Event()
        self.waiting_for_csv = False
        self.pending_csv = False
        self.thread = Thread(target=self._write, daemon=True)
        self.configs = configs
        self.sql_enabled = (settings or {}).get("sql_enabled", False)
        _active_recorders.add(self)
        self.thread.start()
        if not self.ready.wait(5):
            raise RuntimeError("Recording initialization timed out")
        self.check()
        self.submit("campaign", "", {"campaign": asdict(campaign), "devices": configs, "settings": settings or {}})

    def check(self):
        if self.error:
            raise RuntimeError(f"Local recording failed: {self.error}")

    def submit(self, kind, run_id, payload):
        self.check()
        if kind == "sample":
            # Apply before the shared queue: CSV, journal and SQL all receive
            # the same schema, with torque only for stirrers.
            payload = {key: value for key, value in payload.items()
                       if not key.rsplit(".", 1)[-1].lower().startswith("torque")
                       or self.configs.get(key.rsplit(".", 1)[0], {}).get("type") == "jkem"}
        try:
            self.queue.put_nowait((uuid4().hex, datetime.now(timezone.utc).isoformat(), kind, run_id, payload))
        except Full as error:
            raise RuntimeError("Recording queue is full; execution must stop") from error

    def _write(self):
        spools, runs = {}, {}
        published_metadata = {}
        waiting_metadata = set()

        def publish_changed(path, payload):
            if published_metadata.get(path) == payload:
                return True
            try:
                atomic_json(path, payload)
            except PermissionError:
                if path not in waiting_metadata:
                    logger.warning("Recording metadata is locked; retrying automatically: %s. Close it in other applications if necessary.", path)
                    waiting_metadata.add(path)
                return False
            published_metadata[path] = payload
            if path in waiting_metadata:
                waiting_metadata.discard(path)
                logger.info("Recording metadata saving resumed: %s", path)
            return True

        summary = None
        input_closed = False
        last_retry = 0
        try:
            try:
                from app.campaign_roots import register_root
                register_root(self.folder.parent)
            except OSError as error:
                logger.warning("Could not remember campaign folder for startup recovery: %s", error)
            connection = closing(sqlite3.connect(self.database, timeout=5)) if self.sql_enabled else nullcontext(None)
            with connection as db:
                if db is not None:
                    db.execute("PRAGMA journal_mode=WAL")
                    db.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, utc TEXT, kind TEXT, run_id TEXT, payload TEXT, uploaded INTEGER DEFAULT 0)")
                    db.commit()
                self.ready.set()
                while True:
                    try:
                        item = self.queue.get(timeout=.25) if not input_closed else None
                        if item is None:
                            input_closed = True
                        else:
                            event_id, utc, kind, run_id, payload = item
                            encoded = json.dumps(payload, allow_nan=False)
                            if kind == "campaign":
                                atomic_json(self.folder / "definition.json", payload)
                            elif kind == "run_start":
                                directory = self.folder / run_id
                                directory.mkdir()
                                import pandas as pd
                                pd.DataFrame(payload["planned"]).to_csv(directory / "planned.csv", index=False)
                                spools[run_id] = CsvSpool(directory)
                                publish_changed(directory / "run.json", {"run_id": run_id,
                                    "experiment_id": payload.get("experiment_id"),
                                    "experiment_number": payload.get("experiment_number"), "batch_name": payload.get("batch_name", run_id),
                                    "status": "running", "recording_complete": False})
                            elif kind == "sample":
                                spools[run_id].append({"utc": utc, **payload})
                            elif kind == "run_end":
                                runs[run_id] = payload
                            elif kind == "summary":
                                summary = payload
                            with (self.folder / "events.jsonl").open("a", encoding="utf-8") as log:
                                log.write(json.dumps({"id": event_id, "utc": utc, "kind": kind, "run_id": run_id, "payload": payload}) + "\n")
                            if db is not None:
                                db.execute("INSERT INTO events(id,utc,kind,run_id,payload) VALUES(?,?,?,?,?)", (event_id, utc, kind, run_id, encoded))
                                db.commit()
                    except Empty:
                        pass
                    if time.monotonic() - last_retry >= 1 or input_closed:
                        done = {key: spool.flush() for key, spool in spools.items()}
                        self.pending_csv = not all(done.values())
                        metadata_done = True
                        for key, result in runs.items():
                            metadata_done &= publish_changed(self.folder / key / "run.json", {**result, "recording_complete": done.get(key, True)})
                        self.pending_csv |= not metadata_done
                        if summary is not None:
                            self.pending_csv |= not publish_changed(self.folder / "summary.json", {**summary, "recording_complete": not self.pending_csv})
                        self.waiting_for_csv = any(spool.waiting for spool in spools.values()) or bool(waiting_metadata)
                        last_retry = time.monotonic()
                    if input_closed:
                        self.drained.set()
                        if not self.pending_csv:
                            for spool in spools.values():
                                spool.clean()
                            self.complete.set()
                            break
                        time.sleep(1)
        except Exception as error:
            self.error = error
            logger.error("Local recorder failed: %s", error)
        finally:
            self.lease.release()
            self.ready.set()
            self.drained.set()
            _active_recorders.discard(self)

    def close(self):
        self.queue.put(None, timeout=5)
        if not self.drained.wait(10):
            raise RuntimeError("Recording queue has not drained")
        if not self.waiting_for_csv:
            self.thread.join(5)
        self.check()


def sync_sql(database, destination):
    """Idempotently upload committed events. Failure leaves the local outbox intact.

    sqlite:///path supports offline cloud-like demos. Other SQLAlchemy URLs use
    optional SQLAlchemy + a database driver installed by the operator.
    """
    with closing(sqlite3.connect(database, timeout=5)) as local:
        rows = local.execute("SELECT id,utc,kind,run_id,payload FROM events WHERE uploaded=0 LIMIT 500").fetchall()
        if not rows:
            return 0
        if destination.startswith("sqlite:///"):
            target = Path(destination[len("sqlite:///"):]).resolve()
            if target == Path(database).resolve():
                raise ValueError("Remote SQL destination must differ from the local outbox")
            with closing(sqlite3.connect(target, timeout=2)) as remote:
                remote.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, utc TEXT, kind TEXT, run_id TEXT, payload TEXT)")
                remote.executemany("INSERT OR IGNORE INTO events VALUES(?,?,?,?,?)", rows)
                remote.commit()
        else:
            from sqlalchemy import create_engine, text
            engine = create_engine(destination, pool_pre_ping=True, connect_args={"connect_timeout": 3})
            try:
                with engine.begin() as remote:
                    remote.execute(text("CREATE TABLE IF NOT EXISTS events (id VARCHAR(64) PRIMARY KEY, utc TEXT, kind TEXT, run_id TEXT, payload TEXT)"))
                    for row in rows:
                        data = dict(zip(("id", "utc", "kind", "run_id", "payload"), row))
                        remote.execute(text("INSERT INTO events(id,utc,kind,run_id,payload) VALUES(:id,:utc,:kind,:run_id,:payload) ON CONFLICT (id) DO NOTHING"), data)
            finally:
                engine.dispose()
        local.executemany("UPDATE events SET uploaded=1 WHERE id=?", [(row[0],) for row in rows])
        local.commit()
        return len(rows)


class SqlSynchronizer:
    def __init__(self, database, destination):
        self.stop_event = Event()
        self.database, self.destination = database, destination
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        previous_error = None
        while not self.stop_event.is_set():
            try:
                count = sync_sql(self.database, self.destination)
                if previous_error:
                    logger.info("SQL synchronization recovered")
                previous_error = None
                if count:
                    continue
            except Exception as error:
                message = type(error).__name__  # Do not log credentials in SQL URLs.
                if message != previous_error:
                    logger.warning("Remote SQL unavailable (%s); records remain queued locally", message)
                previous_error = message
            self.stop_event.wait(5)

    def stop(self):
        self.stop_event.set()


def resume_sql(settings):
    destination = settings.get("remote_sql_url")
    if not settings.get("sql_enabled", True):
        destination = None
    database = Path(settings["output_folder"]) / "recordings.sqlite3"
    key = (str(database.resolve()), destination)
    for old_key, synchronizer in _synchronizers.items():
        if old_key != key or not destination or not settings.get("enabled", True):
            synchronizer.stop()
    if not destination or not settings.get("enabled", True) or not database.exists():
        return None
    current = _synchronizers.get(key)
    if current is None or current.stop_event.is_set():
        current = _synchronizers[key] = SqlSynchronizer(database, destination)
    return current


def stop_synchronizers():
    for synchronizer in _synchronizers.values():
        synchronizer.stop()
