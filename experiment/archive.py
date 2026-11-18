"""Read-only campaign discovery and bounded snapshots for recorded plots."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def campaign_folders(roots):
    found = set()
    for root in roots:
        path = Path(root).expanduser().resolve()
        try:
            candidates = [path] if (path / "definition.json").is_file() else path.glob("campaign_*")
            for folder in candidates:
                if folder.is_dir():
                    found.add(folder)
        except OSError:
            continue
    return sorted(found, key=str)


def journal_events(folder):
    """Samples are not needed for indexing or metadata recovery."""
    path = Path(folder) / "events.jsonl"
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break  # A crash can leave an incomplete final event.
                if '"kind": "sample"' in line:
                    continue
                yield json.loads(line)
    except FileNotFoundError:
        return


def campaign_details(folder):
    from experiment.recovery import writer_is_active
    folder = Path(folder).resolve()
    definition = read_json(folder / "definition.json")
    if not isinstance(definition, dict) or "campaign" not in definition:
        raise ValueError("Campaign metadata is temporarily unavailable. Refresh to retry.")
    active = writer_is_active(folder)
    campaign = definition["campaign"]
    summary = read_json(folder / "summary.json", {})
    by_experiment = {}
    try:
        directories = list(folder.iterdir())
    except OSError:
        directories = []
    for directory in directories:
        if not directory.is_dir() or directory.is_symlink():
            continue
        run = read_json(directory / "run.json")
        if isinstance(run, dict) and run.get("run_id") == directory.name:
            by_experiment[run.get("experiment_id")] = {**run, "folder": str(directory)}
    # Compatibility with campaigns recorded before running markers existed.
    if any(directory.is_dir() and (directory / "planned.csv").exists()
           and not read_json(directory / "run.json") for directory in directories):
        for event in journal_events(folder):
            if event.get("kind") == "run_start":
                payload = event["payload"]
                run_id = event["run_id"]
                if Path(run_id).name != run_id:
                    continue
                by_experiment.setdefault(payload.get("experiment_id"), {
                    "run_id": run_id, "experiment_id": payload.get("experiment_id"),
                    "status": "running", "recording_complete": False, "folder": str(folder / run_id)})
    runs = []
    for index, experiment in enumerate(campaign.get("experiments", []), 1):
        run = by_experiment.get(experiment["experiment_id"], {})
        status = run.get("status", "queued" if active else "not executed")
        if status == "running" and not active:
            status = "interrupted"
        if run.get("status") == "completed" and not run.get("recording_complete"):
            status = "saving"
        runs.append({**run, "index": index, "name": experiment["name"],
                     "experiment_id": experiment["experiment_id"], "status": status})
    status = summary.get("status", "running" if active else "interrupted")
    if summary and not summary.get("recording_complete"):
        status = "saving" if active else "recovery needed"
    try:
        created = (folder / "definition.json").stat().st_mtime
    except OSError:
        created = 0
    return {"folder": str(folder), "name": campaign.get("name", folder.name),
            "created_epoch": created, "recorded_at": datetime.fromtimestamp(created, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "campaign_id": campaign["campaign_id"], "status": status, "active": active,
            "completed": sum(r["status"] == "completed" for r in runs), "total": len(runs),
            "runs": runs, "devices": definition.get("devices", {}), "summary": summary}


def scan_campaigns(roots):
    rows, errors = [], []
    for folder in campaign_folders(roots):
        try:
            rows.append(campaign_details(folder))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{folder.name}: {error}")
    rows.sort(key=lambda row: row["created_epoch"], reverse=True)
    return {"campaigns": rows, "errors": errors}


def csv_snapshot(path):
    """Read only the complete lines present at open time, even during a run."""
    with Path(path).open("rb") as stream:
        remaining = Path(path).stat().st_size
        def lines():
            nonlocal remaining
            while remaining:
                line = stream.readline(remaining)
                remaining -= len(line)
                if not line or not line.endswith(b"\n"):
                    break
                yield line.decode("utf-8-sig")
        yield from csv.DictReader(lines())


def plot_traces(run, device_id=None, max_points=4000, device_configs=None, time_range=(None, None)):
    """Bound memory independently of recording length; CSV stays full resolution."""
    traces, batches = {}, {}
    start, end = time_range
    def reduce(values):
        if len(values) <= max_points:
            return values
        # Keep extrema in time buckets so short excursions remain visible.
        result = [values[0]]
        for bucket in np.array_split(np.asarray(values, dtype=float), max_points // 2 - 1):
            if len(bucket):
                for i in sorted({int(np.argmin(bucket[:, 1])), int(np.argmax(bucket[:, 1]))}):
                    result.append(tuple(bucket[i]))
        result.append(values[-1])
        return result
    for row in csv_snapshot(Path(run["folder"]) / "execution.csv"):
        try:
            time_s = float(row["elapsed_s"])
        except (ValueError, TypeError, KeyError):
            continue
        if not np.isfinite(time_s):
            continue
        if (start is not None and time_s < start) or (end is not None and time_s > end):
            continue
        for column, value in row.items():
            if not column or "." not in column or (device_id and column.rsplit(".", 1)[0] != device_id):
                continue
            if column.rsplit(".", 1)[-1] in {"connected", "error"}:
                continue
            if (column.rsplit(".", 1)[-1].lower().startswith("torque")
                    and (device_configs or {}).get(column.rsplit(".", 1)[0], {}).get("type") != "jkem"):
                continue
            try:
                number = float(value)
            except (ValueError, TypeError):
                continue
            if np.isfinite(number):
                batches.setdefault(column, []).append((time_s, number))
                if len(batches[column]) >= 8000:
                    traces[column] = reduce(traces.get(column, []) + batches.pop(column))
    for column, values in batches.items():
        traces[column] = reduce(traces.get(column, []) + values)
    return traces


def export_runs(details, run_ids, destination):
    selected = [r for r in details["runs"] if r.get("run_id") in run_ids]
    if not selected or any(r["status"] != "completed" or not r.get("recording_complete") for r in selected):
        raise ValueError("Select fully saved, completed experiments to export.")
    output = Path(destination) / f"experiment_export_{uuid4().hex[:10]}"
    output.mkdir(parents=True, exist_ok=False)
    definition = read_json(Path(details["folder"]) / "definition.json")
    if definition is None:
        raise ValueError("Campaign metadata is temporarily unavailable; retry export.")
    # CSV exports need profile/device metadata, not private SQL URLs or device
    # connection credentials from the recording configuration.
    metadata = {"campaign": definition["campaign"], "devices": {
        key: {field: config[field] for field in ("name", "type", "target_limits", "plot_index", "simulated") if field in config}
        for key, config in definition.get("devices", {}).items()}, "exported_run_ids": list(run_ids)}
    (output / "definition.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for run in selected:
        target = output / run["run_id"]
        target.mkdir()
        for name in ("execution.csv", "planned.csv", "run.json"):
            shutil.copy2(Path(run["folder"]) / name, target / name)
    return str(output)
