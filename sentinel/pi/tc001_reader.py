"""TOPDON TC001 raw V4L2 capture (Linux). See TC001.md for protocol references."""

import argparse
import importlib.util
import json
from pathlib import Path
import threading
import time

WIDTH, HEIGHT = 256, 192
STALE_SECONDS = 3.0
TEMPERATURE_FIELDS = ("min_c", "max_c", "mean_c", "center_c")


def video_devices(sysfs_root=Path("/sys/class/video4linux")):
    """Inspect sysfs without opening or reserving any cameras."""
    devices = []
    for node in sorted(sysfs_root.glob("video*")):
        try:
            info = {"path": "/dev/" + node.name, "name": (node / "name").read_text().strip()}
            for parent in (node.resolve(), *node.resolve().parents):
                if (parent / "idVendor").exists():
                    info["vendor"] = (parent / "idVendor").read_text().strip().lower()
                    info["product"] = (parent / "idProduct").read_text().strip().lower()
                    break
            info["tc001"] = (
                (info.get("vendor"), info.get("product")) == ("0bda", "5830")
                or "tc001" in info["name"].lower()
            )
            devices.append(info)
        except OSError:
            continue
    return devices


def decode_tc001_frame(frame):
    """Decode the lower half as little-endian uint16 Kelvin/64, never RGB pixels."""
    import numpy as np

    frame = np.asarray(frame)
    valid_layout = (
        frame.dtype == np.uint8 and frame.shape in (
            (HEIGHT * 2, WIDTH, 2), (HEIGHT * 2, WIDTH * 2),
            (1, HEIGHT * WIDTH * 4), (HEIGHT * WIDTH * 4,),
        )
    ) or (frame.dtype == np.dtype("<u2") and frame.shape == (HEIGHT * 2, WIDTH))
    if not valid_layout:
        raise ValueError(f"Expected raw 256x384 YUYV, got {frame.shape} / {frame.dtype}; RGB conversion must be disabled")
    raw = np.ascontiguousarray(frame).view(np.uint8).reshape(HEIGHT * 2, WIDTH, 2)
    words = raw[HEIGHT:, :, 0].astype(np.uint16) | (raw[HEIGHT:, :, 1].astype(np.uint16) << 8)
    temperatures = words.astype(np.float32) / 64.0 - 273.15
    valid = (words != 0) & (temperatures >= -40) & (temperatures <= 650)
    if valid.mean() < 0.95:
        raise ValueError("Invalid TC001 temperature plane; check raw format and device selection")
    values = temperatures[valid]
    center = temperatures[HEIGHT // 2, WIDTH // 2] if valid[HEIGHT // 2, WIDTH // 2] else None
    stats = {
        "min_c": round(float(values.min()), 2),
        "max_c": round(float(values.max()), 2),
        "mean_c": round(float(values.mean()), 2),
        "center_c": round(float(center), 2) if center is not None else None,
        "invalid_pixels": int((~valid).sum()),
    }
    return temperatures, valid, stats


def render_thermal_jpeg(temperatures, valid, stats):
    import cv2
    import numpy as np

    low, high = float(temperatures[valid].min()), float(temperatures[valid].max())
    scale = 255.0 / max(high - low, 0.1)
    pixels = np.clip((temperatures - low) * scale, 0, 255).astype(np.uint8)
    image = cv2.applyColorMap(pixels, cv2.COLORMAP_INFERNO)
    image[~valid] = 0
    image = cv2.resize(image, (WIDTH * 2, HEIGHT * 2), interpolation=cv2.INTER_NEAREST)
    cv2.drawMarker(image, (WIDTH, HEIGHT), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
    cv2.rectangle(image, (0, HEIGHT * 2 - 28), (WIDTH * 2, HEIGHT * 2), (0, 0, 0), -1)
    label = f"Min {stats['min_c']:.1f} C   Max {stats['max_c']:.1f} C   Mean {stats['mean_c']:.1f} C"
    cv2.putText(image, label, (8, HEIGHT * 2 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("Could not encode the thermal image")
    return encoded.tobytes()


class TC001Reader:
    """One capture thread and cached frames shared by every HTTP client."""

    def __init__(self, device="", *, stale_seconds=STALE_SECONDS):
        self.device = device
        self.enabled = bool(device) and device.lower() not in ("off", "none", "disabled")
        self.stale_seconds = stale_seconds
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._jpeg = None
        self._stats = {}
        self._last_seen = None
        self._measured_at = None
        self._sequence = 0
        self._active_device = None
        self._error = "Starting" if self.enabled else "Disabled"

    def start(self):
        if not self.enabled:
            return
        with self._condition:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="tc001-capture", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=2)
        self._unavailable("Stopped")

    def _publish(self, jpeg, stats, device):
        with self._condition:
            self._jpeg = jpeg
            self._stats = stats.copy()
            self._last_seen = time.monotonic()
            self._measured_at = time.time()
            self._active_device = device
            self._error = None
            self._sequence += 1
            self._condition.notify_all()

    def _unavailable(self, error):
        with self._condition:
            self._error = str(error)
            self._jpeg = None
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            age = time.monotonic() - self._last_seen if self._last_seen is not None else None
            available = bool(self.enabled and self._jpeg and not self._error and age is not None and age < self.stale_seconds)
            result = {
                "enabled": self.enabled, "available": available,
                "status": "ok" if available else "offline",
                "model": "TOPDON TC001", "device": self._active_device,
                "width": WIDTH, "height": HEIGHT,
                "measured_at": self._measured_at, "age_s": round(age, 2) if age is not None else None,
                "error": None if available else (self._error or "No fresh thermal frames"),
                "invalid_pixels": self._stats.get("invalid_pixels") if available else None,
            }
            result.update({key: self._stats.get(key) if available else None for key in TEMPERATURE_FIELDS})
            return result

    def sensor(self):
        if not self.enabled:
            return None
        data = self.snapshot()
        return {**data, "id": "thermal:tc001", "type": "thermal", "source": "topdon_tc001", "temp_c": data["max_c"], "unit": "°C"}

    def frames(self):
        sequence = -1
        while not self._stop.is_set():
            with self._condition:
                self._condition.wait_for(
                    lambda: self._sequence != sequence or self._error or self._stop.is_set(), timeout=1
                )
                if not self.snapshot()["available"]:
                    return
                if self._sequence == sequence:
                    continue
                sequence, jpeg = self._sequence, self._jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    def _candidates(self):
        if self.device != "auto":
            return [self.device]
        return [item["path"] for item in video_devices() if item["tc001"]]

    def _run(self):
        try:
            import cv2
        except ImportError:
            self._unavailable("Install requirements-thermal.txt in the Pi service environment")
            return
        while not self._stop.is_set():
            paths = self._candidates()
            if not paths:
                self._unavailable("TC001 not detected; check USB or set THERMAL_DEVICE=/dev/videoX")
            for path in paths:
                if self._stop.is_set():
                    break
                cap = None
                try:
                    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
                    if not cap.isOpened():
                        raise RuntimeError(f"Cannot open {path}; check video permissions or other camera apps")
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT * 2)
                    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    failures, last_publish = 0, 0.0
                    while not self._stop.is_set():
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError(f"No frames from {path}")
                        if time.monotonic() - last_publish < 0.1:
                            continue
                        try:
                            temperatures, valid, stats = decode_tc001_frame(frame)
                        except ValueError as error:
                            failures += 1
                            if failures >= 5:
                                raise error
                            continue
                        failures = 0
                        jpeg = render_thermal_jpeg(temperatures, valid, stats)
                        self._publish(jpeg, stats, path)
                        last_publish = time.monotonic()
                except Exception as error:
                    self._unavailable(error)
                finally:
                    if cap is not None:
                        cap.release()
            self._stop.wait(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect TC001 video devices without opening them")
    parser.add_argument("--probe", action="store_true", help="Print video devices and dependency availability")
    parser.parse_args()
    print(json.dumps({
        "devices": video_devices(),
        "opencv": importlib.util.find_spec("cv2") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
    }, indent=2))
