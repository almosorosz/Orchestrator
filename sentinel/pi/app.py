#!/usr/bin/env python3
"""
Fume Hood Monitor - backend for Raspberry Pi 5.

Serves the dashboard and exposes:
  GET  /                 -> dashboard UI (fumehood_dashboard.html)
  GET  /api/sensors      -> [{"id","temp_c","status"}, ...]
  GET  /api/userinfo     -> saved profile (JSON)
  POST /api/userinfo     -> save profile (JSON body)

Run:
  pip install flask          # inside your miniforge env
  python app.py
  # then open http://<pi-ip>:8000  (or on the Pi itself: http://localhost:8000)
"""

import csv
import atexit
import glob
import json
import os
import math
import signal
import threading
import time
from flask import Flask, jsonify, request, send_from_directory, Response

from pico_serial_reader import start_pico_reader, get_pico_sensors, get_pico_doors
from tc001_reader import TC001Reader, video_devices
from history_store import HistoryStore

BASE_DIR = "/sys/bus/w1/devices/"
HERE = os.path.dirname(os.path.abspath(__file__))
USERINFO_PATH = os.path.join(HERE, "userinfo.json")
PROBECONFIG_PATH = os.path.join(HERE, "probeconfig.json")
EVENTLOG_PATH = os.path.join(HERE, "event_log.csv")
EVENTLOG_COLUMNS = ["timestamp", "sensor_id", "sensor_name", "type", "level", "detail"]
history_store = HistoryStore(os.path.join(HERE, "sentinel_history.sqlite3"), EVENTLOG_PATH)
_history_stop = threading.Event()
_history_thread = None
_history_error = ""

# Default temperature thresholds, matching the dashboard's CONFIG.
DEFAULT_WARN_LEVEL = 26
DEFAULT_ALERT_LEVEL = 30

# Background event logger polling interval in seconds.
EVENT_LOG_POLL_SECONDS = 3

_eventlog_lock = threading.Lock()

# Camera (e.g. GC0307). Must appear as a V4L2 device: check `ls /dev/video*`.
# Override the index with:  CAMERA_INDEX=1 python app.py
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
CAMERA_DEVICE = os.environ.get("CAMERA_DEVICE", "").strip()
CAMERA_W, CAMERA_H = 640, 480          # GC0307 is VGA
_camera = None
_camera_lock = threading.Lock()
_camera_device = None

# Detect TC001 automatically; use "off" to disable or a Linux video path to override.
thermal_reader = TC001Reader(os.environ.get("THERMAL_DEVICE", "auto").strip())

app = Flask(__name__, static_folder=None)


# ---------------- DS18B20 reading (from your script, hardened) ----------------
def find_sensors():
    return glob.glob(BASE_DIR + "28-*")


def read_temp_raw(device_file):
    try:
        with open(device_file, "r") as f:
            return f.readlines()
    except Exception:
        return []


def read_temp_c(device_file):
    lines = read_temp_raw(device_file)
    # need two lines, and first must end in "YES" (CRC ok)
    if len(lines) < 2 or lines[0].strip()[-3:] != "YES":
        return None
    pos = lines[1].find("t=")
    if pos != -1:
        try:
            return float(lines[1][pos + 2:]) / 1000.0
        except ValueError:
            return None
    return None


def read_all_sensors():
    out = []
    for path in find_sensors():
        sensor_id = path.split("/")[-1]
        temp = read_temp_c(path + "/w1_slave")
        out.append({
            "id": sensor_id,
            "temp_c": temp,                       # None if read failed
            "status": "ok" if temp is not None else "error",
            "type": "temp",
            "source": "pi_1wire",
        })
    # Pico temperature probes, updated by the background USB serial reader.
    out.extend(get_pico_sensors())
    # Pico door switches, including door-open duration warnings.
    out.extend(read_all_doors())
    thermal = thermal_reader.sensor()
    if thermal is not None:
        out.append(thermal)
    return out


# Default door-open duration threshold in seconds (five minutes).
DEFAULT_DOOR_ALERT_SECONDS = 300


def load_probeconfig():
    if os.path.exists(PROBECONFIG_PATH):
        try:
            with open(PROBECONFIG_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def read_all_doors():
    """
    Apply door-open duration warnings to the serial reader's raw door states.
    Configure thresholds by door ID in probeconfig.json, for example:
    {"door-gp5": {"name": "Fume hood sash", "door_alert_seconds": 180}}.
    Unconfigured doors use DEFAULT_DOOR_ALERT_SECONDS.
    """
    config = load_probeconfig()
    out = []
    for door in get_pico_doors():
        door_conf = config.get(door["id"], {})
        threshold = door_conf.get("door_alert_seconds", DEFAULT_DOOR_ALERT_SECONDS)

        status = door["status"]  # Serial availability: "ok" or "offline".
        if status == "ok" and door["state"] == "open" and door["seconds_in_state"] >= threshold:
            status = "warning"

        out.append({
            "id": door["id"],
            "state": door["state"],           # "open" / "closed", rather than 0/1.
            "seconds_in_state": door["seconds_in_state"],
            "alert_threshold_seconds": threshold,
            "status": status,
            "type": "door",
            "source": "pico",
        })
    return out


# ---------------- Persistent event log (CSV) ----------------
def classify_temp_level(sensor, config):
    """Classify a temperature probe as ok / warn / alert / offline."""
    if sensor.get("status") != "ok" or sensor.get("temp_c") is None:
        return "offline"
    temp = sensor["temp_c"]
    cfg = config.get(sensor["id"], {})
    caution = cfg.get("caution", DEFAULT_WARN_LEVEL)
    alert = cfg.get("alert", DEFAULT_ALERT_LEVEL)
    if temp >= alert:
        return "alert"
    if temp >= caution:
        return "warn"
    return "ok"


def classify_door_level(sensor):
    """Classify a door as ok / warn / offline, using one alarm threshold."""
    if sensor.get("status") == "offline":
        return "offline"
    return "warn" if sensor.get("status") == "warning" else "ok"


def append_event(sensor_id, sensor_name, sensor_type, level, detail):
    """Append a CSV row, adding the header when the file is created."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    row = [ts, sensor_id, sensor_name, sensor_type, level, detail]
    with _eventlog_lock:
        is_new = not os.path.exists(EVENTLOG_PATH)
        with open(EVENTLOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(EVENTLOG_COLUMNS)
            w.writerow(row)


def read_event_log(limit=200):
    """Read the last limit CSV rows, newest first."""
    if not os.path.exists(EVENTLOG_PATH):
        return []
    with _eventlog_lock:
        with open(EVENTLOG_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return list(reversed(rows[-limit:]))


def _event_logger_loop():
    global _history_error
    while not _history_stop.is_set():
        try:
            config = load_probeconfig()
            readings = []
            for s in read_all_sensors():
                if s["type"] == "door":
                    level = classify_door_level(s)
                    if level == "offline":
                        detail = "Offline"
                    else:
                        detail = ("Door open ({}s)".format(int(s["seconds_in_state"]))
                                   if s["state"] == "open" else "Door closed")
                else:
                    level = classify_temp_level(s, config)
                    detail = "Offline" if level == "offline" else "{:.2f} \u00b0C".format(s["temp_c"])

                default_name = "TOPDON TC001" if s["type"] == "thermal" else s["id"]
                name = (config.get(s["id"], {}).get("name") or default_name)
                def finite(value):
                    return value if isinstance(value, (int, float)) and math.isfinite(value) else None
                available = level != "offline"
                readings.append({
                    "sensor_id": s["id"], "sensor_name": name, "type": s["type"], "source": s.get("source", "pi"),
                    "level": level, "status": s.get("status", "offline"), "detail": detail,
                    "value": finite(s.get("temp_c")) if available else None,
                    "unit": "°C" if s["type"] != "door" else "", "state": s.get("state"),
                    **{key: finite(s.get(key)) if available else None for key in ("min_c", "max_c", "mean_c", "center_c")},
                })
            for event in history_store.capture(readings):
                append_event(event["sensor_id"], event["sensor_name"], event["type"], event["level"], event["detail"])
            history_store.prune()
            _history_error = ""
            interval = history_store.settings()["record_interval_seconds"]
        except Exception as e:
            _history_error = str(e)
            print(f"[history] Recording failed: {e}")
            interval = EVENT_LOG_POLL_SECONDS
        _history_stop.wait(interval)


def start_event_logger():
    global _history_thread
    history_store.add_system_event("Service started")
    _history_stop.clear()
    _history_thread = threading.Thread(target=_event_logger_loop, daemon=True)
    _history_thread.start()
    return _history_thread


def stop_history():
    _history_stop.set()
    if _history_thread:
        _history_thread.join(timeout=10)
        try:
            history_store.add_system_event("Service stopped")
        except Exception as error:
            print(f"[history] Could not record shutdown: {error}")


def terminate_service(*_):
    raise SystemExit(0)


# ---------------- routes ----------------
@app.route("/")
def index():
    return send_from_directory(HERE, "fumehood_dashboard.html")


@app.route("/api/sensors")
def api_sensors():
    return jsonify(read_all_sensors())


@app.route("/api/userinfo", methods=["GET", "POST"])
def api_userinfo():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        # keep only known fields
        allowed = ("hood", "student", "pi", "description", "hazard")
        clean = {k: str(data.get(k, "")) for k in allowed}
        with open(USERINFO_PATH, "w") as f:
            json.dump(clean, f, indent=2)
        return jsonify({"ok": True})

    # GET
    if os.path.exists(USERINFO_PATH):
        with open(USERINFO_PATH) as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route("/api/probeconfig", methods=["GET", "POST"])
def api_probeconfig():
    """Per-probe settings, keyed by sensor id: {id: {name, caution, alert}}."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        with open(PROBECONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"ok": True})

    if os.path.exists(PROBECONFIG_PATH):
        with open(PROBECONFIG_PATH) as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route("/api/eventlog")
def api_eventlog():
    """Return persistent CSV events, newest first; default to the last 200."""
    try:
        limit = int(request.args.get("limit", 200))
    except ValueError:
        limit = 200
    limit = max(1, min(limit, 2000))
    return jsonify(read_event_log(limit))


@app.route("/api/eventlog.csv")
def api_eventlog_csv():
    """Download the complete raw CSV file."""
    if not os.path.exists(EVENTLOG_PATH):
        return "no events logged yet", 404
    return send_from_directory(HERE, "event_log.csv", as_attachment=True,
                                download_name="fumehood_event_log.csv")


# ---------------- History, export and PC replication ----------------
def history_filters():
    before = request.args.get("before_seq")
    return {"kind": request.args.get("kind", "events"), "sensor_id": request.args.get("sensor_id", ""),
            "start": request.args.get("start", ""), "end": request.args.get("end", ""),
            "before_seq": int(before) if before else None}


@app.route("/api/history/status")
def api_history_status():
    try:
        data = history_store.status()
        data.update(recording=bool(_history_thread and _history_thread.is_alive() and not _history_error),
                    error=_history_error or None)
        response = jsonify(data)
        response.headers["Cache-Control"] = "no-store"
        return response
    except Exception as error:
        return jsonify(error="History unavailable: " + str(error)), 503


@app.route("/api/history/settings", methods=["GET", "POST"])
def api_history_settings():
    try:
        return jsonify(history_store.save_settings(request.get_json(silent=True))
                       if request.method == "POST" else history_store.settings())
    except ValueError as error:
        return jsonify(error=str(error)), 400


@app.route("/api/history")
def api_history():
    try:
        return jsonify(history_store.query(limit=int(request.args.get("limit", 200)), **history_filters()))
    except ValueError as error:
        return jsonify(error=str(error)), 400


@app.route("/api/history/export.csv")
def api_history_export():
    try:
        filters = history_filters()
        filters.pop("before_seq")  # Export the full selection, not only the visible page.
        rows = history_store.export_csv(**filters)
        return Response(rows, mimetype="text/csv", headers={
            "Content-Disposition": 'attachment; filename="sentinel_' + filters["kind"] + '.csv"',
            "Cache-Control": "no-store",
        })
    except ValueError as error:
        return jsonify(error=str(error)), 400


@app.route("/api/history/sync")
def api_history_sync():
    try:
        response = jsonify(history_store.sync_batch(int(request.args.get("after", 0)), int(request.args.get("limit", 500))))
        response.headers["Cache-Control"] = "no-store"
        return response
    except ValueError as error:
        return jsonify(error=str(error)), 400


# ---------------- thermal camera (TOPDON TC001) ----------------
@app.route("/api/thermal")
def api_thermal():
    response = jsonify(thermal_reader.snapshot())
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/thermal_feed")
def thermal_feed():
    if not thermal_reader.snapshot()["available"]:
        return "thermal camera unavailable", 503
    return Response(thermal_reader.frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


# ---------------- visible camera (GC0307 via V4L2 / OpenCV) ----------------
def visible_camera_candidates():
    devices = video_devices()
    thermal_paths = {os.path.realpath(item["path"]) for item in devices if item["tc001"]}
    if thermal_reader.enabled and thermal_reader.device != "auto":
        thermal_paths.add(os.path.realpath(thermal_reader.device))
    if CAMERA_DEVICE:
        candidates = [CAMERA_DEVICE]
    elif "CAMERA_INDEX" in os.environ:
        candidates = [f"/dev/video{CAMERA_INDEX}"] if os.name == "posix" else [CAMERA_INDEX]
    elif devices:
        candidates = [item["path"] for item in devices if item.get("vendor") and not item["tc001"]]
    else:
        candidates = [f"/dev/video{CAMERA_INDEX}"] if os.name == "posix" else [CAMERA_INDEX]
    return [path for path in candidates if not isinstance(path, str) or os.path.realpath(path) not in thermal_paths]


def get_camera():
    """Lazily open the camera. Returns a cv2.VideoCapture or None if unavailable.
    Requires opencv:  pip install opencv-python-headless"""
    global _camera, _camera_device
    try:
        import cv2
    except ImportError:
        return None
    with _camera_lock:
        if _camera is None or not _camera.isOpened():
            for device in visible_camera_candidates():
                cap = cv2.VideoCapture(device)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame.ndim == 3 and frame.shape[2] == 3:
                        _camera, _camera_device = cap, device
                        break
                cap.release()
            else:
                _camera = None
                return None
        return _camera


@app.route("/api/camera")
def api_camera():
    cap = get_camera()
    ok = False
    if cap is not None:
        with _camera_lock:
            ok, _ = cap.read()
    return jsonify({"available": bool(ok), "index": CAMERA_INDEX, "device": _camera_device})


def mjpeg_generator():
    import cv2
    cap = get_camera()
    if cap is None:
        return
    while True:
        with _camera_lock:
            ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.05)   # cap ~20 fps


@app.route("/video_feed")
def video_feed():
    if get_camera() is None:
        return "camera unavailable", 503
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    thermal_reader.start()
    atexit.register(thermal_reader.stop)
    # Start the Pico USB reader for temperature probes and door switches.
    start_pico_reader()
    # Record sensor transitions even when the dashboard is closed.
    start_event_logger()
    atexit.register(stop_history)
    # Let systemd termination run the normal cleanup and save the stop event.
    signal.signal(signal.SIGTERM, terminate_service)

    # host=0.0.0.0 so you can reach it from another machine on the LAN.
    # threaded=True so the MJPEG stream doesn't block the JSON endpoints.
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
