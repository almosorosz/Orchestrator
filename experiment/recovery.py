"""Recover durable recordings after a crash, without restarting any hardware."""
from datetime import datetime, timezone
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys
import time


class WriterLease:
    """OS-owned lock: automatically released if the recording process dies."""
    def __init__(self, folder):
        self.path = Path(folder) / ".writer.lock"
        self.stream = None

    def acquire(self):
        if self.path.is_symlink():
            raise OSError("Recording lock must not be a symbolic link")
        stream = self.path.open("a+b")
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            return False
        self.stream = stream
        return True

    def release(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def writer_is_active(folder):
    lease = WriterLease(folder)
    try:
        if not lease.path.exists():
            return False
        if not lease.acquire():
            return True
        lease.release()
        return False
    except OSError:
        return True  # An inaccessible lock is not permission to recover a run.


def _repair_tail(path):
    """Keep a truncated crash tail for inspection; replay complete samples only."""
    if not path.exists() or not path.stat().st_size:
        return
    with path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n":
            return
        end = stream.tell()
        start = end
        while start:
            chunk_start = max(0, start - 65536)
            stream.seek(chunk_start)
            chunk = stream.read(start - chunk_start)
            at = chunk.rfind(b"\n")
            if at >= 0:
                start = chunk_start + at + 1
                break
            start = chunk_start
        stream.seek(start)
        tail = stream.read()
        path.with_suffix(path.suffix + ".partial").write_bytes(tail)
        stream.truncate(start)


def recover_campaign(folder, allow_legacy=False):
    from experiment.archive import journal_events, read_json
    from experiment.recording import CsvSpool, atomic_json
    folder = Path(folder).resolve()
    lease = WriterLease(folder)
    if not lease.path.exists() and not allow_legacy:
        summary = read_json(folder / "summary.json", {})
        if summary.get("recording_complete"):
            return {"folder": str(folder), "status": "unchanged"}
        return {"folder": str(folder), "status": "waiting", "message": "Legacy recording has no ownership lock. Close older app instances and enable legacy recovery to retry."}
    if not lease.acquire():
        return {"folder": str(folder), "status": "active", "message": "Recording is owned by a running process."}
    try:
        definition = read_json(folder / "definition.json")
        if not definition:
            raise ValueError("Recording definition is unavailable; retry after synchronization finishes.")
        existing = read_json(folder / "summary.json", {})
        pending = [p for p in folder.glob("*/samples.pending.jsonl") if p.parent.is_dir()]
        if any(path.parent.resolve().parent != folder or path.is_symlink() for path in pending):
            raise ValueError("Pending recording path is outside its campaign")
        if existing.get("recording_complete") and not pending:
            return {"folder": str(folder), "status": "unchanged"}
        starts, ends, summary = {}, {}, existing
        events = list(journal_events(folder))
        for event in events:
            run_id = event.get("run_id", "")
            if run_id and (Path(run_id).name != run_id or run_id in {".", ".."}):
                raise ValueError("Invalid run ID in recording journal")
            if event["kind"] == "run_start":
                starts[run_id] = event["payload"]
            elif event["kind"] == "run_end":
                ends[run_id] = event["payload"]
            elif event["kind"] == "summary":
                summary = event["payload"]
        for directory in folder.iterdir():
            if directory.is_dir() and not directory.is_symlink():
                marker = read_json(directory / "run.json", {})
                if marker.get("experiment_id") and marker.get("run_id") == directory.name:
                    starts.setdefault(directory.name, {"experiment_id": marker["experiment_id"],
                        "experiment_number": marker.get("experiment_number"), "batch_name": marker.get("batch_name", directory.name)})
                    if marker.get("status") in {"completed", "skipped", "stopped", "failed", "interrupted"}:
                        ends.setdefault(directory.name, marker)
        for path in pending:
            _repair_tail(path)
            spool = CsvSpool(path.parent)
            if spool.state["cursor"] > path.stat().st_size:
                raise ValueError("Pending CSV cursor exceeds available samples; original files retained.")
            csv_path = path.parent / "execution.csv"
            if spool.state["offset"] > (csv_path.stat().st_size if csv_path.exists() else 0):
                raise ValueError("CSV is shorter than its committed offset; original files retained.")
            while not spool.flush():
                if spool.waiting:
                    return {"folder": str(folder), "status": "waiting", "message": "Close locked CSV/metadata files, then retry recovery."}
        recovered_at = datetime.now(timezone.utc).isoformat()
        for run_id, start in starts.items():
            directory = folder / run_id
            if not directory.is_dir():
                continue
            result = ends.get(run_id) or {"run_id": run_id, "experiment_id": start["experiment_id"], "status": "interrupted"}
            result = {"experiment_number": start.get("experiment_number"), "batch_name": start.get("batch_name", run_id), **result}
            atomic_json(directory / "run.json", {**result, "recording_complete": True, "recovered_at": recovered_at})
        status = summary.get("status", "interrupted")
        # Preserve event IDs and uploaded flags when repairing an optional SQL outbox.
        settings = definition.get("settings", {})
        if settings.get("sql_enabled"):
            with closing(sqlite3.connect(folder.parent / "recordings.sqlite3", timeout=1)) as db:
                db.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, utc TEXT, kind TEXT, run_id TEXT, payload TEXT, uploaded INTEGER DEFAULT 0)")
                # Stream all events here (including samples); never duplicate a committed ID.
                with (folder / "events.jsonl").open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.endswith("\n"):
                            break
                        event = json.loads(line)
                        db.execute("INSERT OR IGNORE INTO events(id,utc,kind,run_id,payload) VALUES(?,?,?,?,?)",
                                   (event["id"], event["utc"], event["kind"], event["run_id"], json.dumps(event["payload"])))
                db.commit()
        atomic_json(folder / "summary.json", {**summary, "status": status,
                    "recording_complete": True, "recovered_at": recovered_at,
                    "recovery_note": "Recorded data recovered. Hardware and queued experiments were not restarted."})
        for path in pending:
            CsvSpool(path.parent).clean()
        return {"folder": str(folder), "status": "recovered", "message": f"Recording recovered ({status}). Hardware remains stopped."}
    finally:
        lease.release()


def recover_roots(roots, allow_legacy=False):
    from experiment.archive import campaign_folders
    results = []
    for folder in campaign_folders(roots):
        try:
            results.append(recover_campaign(folder, allow_legacy=allow_legacy))
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
            results.append({"folder": str(folder), "status": "waiting", "message": str(error)})
    return results
