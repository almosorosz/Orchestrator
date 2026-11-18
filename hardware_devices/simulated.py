"""Deterministic device doubles. No serial, OPC or NI driver is opened here."""
import math
import random
import time
from threading import RLock
from hardware_devices.limits import DEFAULT_LIMITS, target_allowed


class SimulatedDevice:
    simulated = True
    TYPES = {"masterflex", "ismatec", "huber", "jkem", "mettler", "nidaq_valve", "opcua"}

    def __init__(self, name, device_type, *, seed=42, clock=time.monotonic,
                 noise=0.01, fail_after=None, response_time=2.0):
        if device_type not in self.TYPES:
            raise ValueError(f"Unsupported simulation type: {device_type}")
        if noise < 0 or response_time <= 0:
            raise ValueError("Noise must be nonnegative and response time positive")
        self.name, self.device_type = name, device_type
        self.port = "SIMULATED"
        self.is_connected = self.is_running = False
        self.target = 25.0 if device_type == "huber" else 0.0
        self.value = self.target
        self.weight = 0.0
        self.stage = "2"
        self.time_scale = 1.0
        self.target_limits = dict(DEFAULT_LIMITS.get(device_type, {}))
        self._clock, self._rng = clock, random.Random(seed)
        self._noise, self._fail_after, self._response_time = noise, fail_after, response_time
        self._commands = 0
        self._last = clock()
        self._origin = self._last
        self._lock = RLock()

    def connect(self):
        with self._lock:
            self.is_connected = True
            self._last = self._clock()
            self._origin = self._last
            return True

    def set_simulation_clock(self, clock):
        """Temporarily use the experiment's simulated time instead of wall time."""
        with self._lock:
            previous = (self._clock, self._origin)
            self._clock = clock
            self._last = self._origin = clock()
            self.time_scale = 1.0
            return previous

    def restore_simulation_clock(self, previous):
        with self._lock:
            self._clock, self._origin = previous
            self._last = self._clock()
            self.time_scale = 1.0

    def disconnect(self):
        with self._lock:
            self.is_running = self.is_connected = False
            return True

    def _require_connection(self):
        if not self.is_connected:
            raise RuntimeError(f"{self.name}: simulated device disconnected")

    def _advance(self):
        now = self._clock()
        dt = max(0.0, now - self._last) * self.time_scale
        self._last = now
        target = self.target if self.is_running else 0.0
        self.value += (target - self.value) * (1 - math.exp(-dt / self._response_time))
        return now

    def set_target(self, value, **kwargs):
        with self._lock:
            self._require_connection()
            self._commands += 1
            if self._fail_after is not None and self._commands > self._fail_after:
                return False
            value = float(value)
            if not math.isfinite(value):
                return False
            if self.device_type in {"mettler", "opcua"}:
                return False
            if self.device_type == "nidaq_valve" and value not in (0, 1):
                return False
            if not target_allowed(value, self.target_limits):
                return False
            self._advance()
            if self.device_type == "masterflex" and value > 0 and kwargs.get("direction") == "-":
                value = -value
            self.target = value
            self.stage = str(kwargs.get("stage") or self.stage)
            self.is_running = value != 0 if self.device_type != "huber" else True
            return True

    def start(self, **kwargs):
        with self._lock:
            self._require_connection()
            self._advance()
            self.is_running = True
            return True

    def stop(self, **kwargs):
        with self._lock:
            self._require_connection()
            self._advance()
            self.is_running = False
            return True

    def set_jacket_control(self):
        self._require_connection()
        return "OK"

    set_process_control = set_jacket_control

    def release_remote_mode(self):
        return self.stop()

    release_to_manual = release_remote_mode

    def test_connection(self):
        return self.read_status()

    def read_value(self):
        return self.read_status()["value"]

    def read_status(self):
        with self._lock:
            self._require_connection()
            now = self._advance()
            noise = self._rng.uniform(-self._noise, self._noise)
            value = self.value + noise
            status = {"name": self.name, "connected": True, "simulated": True}
            if self.device_type in {"masterflex", "jkem"}:
                status.update(speed_rpm=value, current_a=0.1, voltage_v=24)
                if self.device_type == "jkem":
                    status["torque_ozin"] = abs(value) * 0.02
            elif self.device_type == "ismatec":
                status.update(flow=value, stage=self.stage)
            elif self.device_type == "huber":
                status.update(setpoint=self.target, temp_internal=value, temp_external=value - 0.2)
            elif self.device_type == "nidaq_valve":
                state = int(self.target) if self.is_running else 0
                status.update(target_state=state, measured_voltage=5.0 * state, circuit_state=str(state))
            elif self.device_type == "mettler":
                status.update(weight_g=self.weight + noise)
            else:
                status.update(value=25 + math.sin((now - self._origin) * self.time_scale / 10) + noise)
            return status


def update_simulated_process(devices, configs, elapsed_s):
    """Simple demo coupling: integrate simulated pump delivery into balances.

    Masterflex RPM uses an illustrative 0.1 mL/min/RPM conversion and density
    1 g/mL. This is a demo model, never a physical calibration.
    """
    flow = 0.0
    for device_id, dev in devices.items():
        if not getattr(dev, "simulated", False) or not dev.is_running:
            continue
        kind = configs[device_id]["type"]
        if kind == "masterflex":
            flow += dev.target * 0.1
        elif kind == "ismatec":
            flow += dev.target
    for dev in devices.values():
        if getattr(dev, "simulated", False) and dev.device_type == "mettler":
            with dev._lock:
                dev.weight += flow * max(0, elapsed_s) / 60.0
