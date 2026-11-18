"""
Pi-side Pico USB serial reader.
  - Parse "DATA,<rom_id>,<temp>" temperature readings.
  - Parse "DOOR,<door_id>,<open|closed>" door states.

Normalize dashed ROM IDs, dropping the CRC byte, to match the Pi's native
/sys/bus/w1/devices/ IDs so both connections share probeconfig.json entries.

Dependency: conda install -c conda-forge pyserial
"""

import serial
import threading
import time
import glob

# Serial silence timeout, separate from the door-open duration alarm.
STALE_AFTER_SECONDS = 15

# Temperature probes: { normalized_id: {"temp_c": float, "last_seen": float} }
pico_sensor_data = {}
# Doors: { door_id: {"state": "open"/"closed", "last_seen": float, "changed_at": float} }
pico_door_data = {}
_lock = threading.Lock()


def normalize_temp_id(dashed_id: str) -> str:
    """
    Input: "28-67-74-23-00-00-00-d2" (family, six serial bytes, CRC).
    Output: "28677423000000" (family and serial bytes, without dashes or CRC).
    """
    parts = dashed_id.split("-")
    if len(parts) != 8:
        return dashed_id.replace("-", "")
    family = parts[0]
    serial_bytes = parts[1:7]
    return family + "".join(serial_bytes)


def find_pico_port():
    candidates = glob.glob("/dev/ttyACM*")
    if candidates:
        return candidates[0]
    return None


def _reader_loop(port, baudrate=115200):
    ser = None
    while True:
        try:
            if ser is None:
                port_path = port or find_pico_port()
                if port_path is None:
                    print("[pico_serial] Pico not connected; retrying in 3 seconds")
                    time.sleep(3)
                    continue
                ser = serial.Serial(port_path, baudrate, timeout=2)
                print(f"[pico_serial] Connected to {port_path}")

            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            if line.startswith("#"):
                print(f"[pico_serial] {line}")
                continue

            if line.startswith("DATA,"):
                parts = line.split(",")
                if len(parts) == 3:
                    _, raw_id, temp_str = parts
                    try:
                        temp = float(temp_str)
                        norm_id = normalize_temp_id(raw_id)
                        with _lock:
                            pico_sensor_data[norm_id] = {
                                "temp_c": temp,
                                "last_seen": time.time(),
                            }
                    except ValueError:
                        print(f"[pico_serial] Invalid temperature data: {line}")
                continue

            if line.startswith("DOOR,"):
                parts = line.split(",")
                if len(parts) == 3:
                    _, door_id, state = parts
                    if state in ("open", "closed"):
                        now = time.time()
                        with _lock:
                            prev = pico_door_data.get(door_id)
                            # Preserve the transition time while the state stays unchanged.
                            if prev is None or prev["state"] != state:
                                changed_at = now
                            else:
                                changed_at = prev["changed_at"]
                            pico_door_data[door_id] = {
                                "state": state,
                                "last_seen": now,
                                "changed_at": changed_at,
                            }
                    else:
                        print(f"[pico_serial] Unrecognized door state: {line}")
                continue

        except (serial.SerialException, OSError) as e:
            print(f"[pico_serial] Serial connection lost: {e}; reconnecting in 3 seconds")
            ser = None
            time.sleep(3)


def start_pico_reader(port=None, baudrate=115200):
    t = threading.Thread(target=_reader_loop, args=(port, baudrate), daemon=True)
    t.start()
    return t


def get_pico_sensors():
    """Return temperatures using the app.py sensor schema."""
    out = []
    now = time.time()
    with _lock:
        items = list(pico_sensor_data.items())
    for sensor_id, info in items:
        age = now - info["last_seen"]
        status = "ok" if age < STALE_AFTER_SECONDS else "offline"
        out.append({
            "id": sensor_id,
            "temp_c": info["temp_c"] if status == "ok" else None,
            "status": status,
            "type": "temp",
            "source": "pico",
        })
    return out


def get_pico_doors():
    """
    Return raw door states and elapsed seconds in the current state.
    Status reflects serial availability only; the caller applies each door's
    configurable duration threshold from probeconfig.json.
    """
    out = []
    now = time.time()
    with _lock:
        items = list(pico_door_data.items())
    for door_id, info in items:
        age = now - info["last_seen"]
        comm_status = "ok" if age < STALE_AFTER_SECONDS else "offline"
        seconds_in_state = now - info["changed_at"]
        out.append({
            "id": door_id,
            "state": info["state"],
            "seconds_in_state": round(seconds_in_state, 1),
            "status": comm_status,
            "type": "door",
            "source": "pico",
        })
    return out
