from enum import Enum
from threading import RLock
from uuid import uuid4


class HardwareState(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    EXPERIMENT = "experiment"
    SHUTDOWN = "shutdown"


class HardwareBusyError(RuntimeError):
    pass


class HardwareManager:
    """
    Centralized ownership coordinator for shared hardware.

    Does not communicate with hardware directly.
    Does not create or destroy drivers.

    A reservation token identifies the operation that
    owns the hardware.
    """

    def __init__(self, devices):
        self._devices = devices
        self._lock = RLock()
        self._state = HardwareState.IDLE
        self._token = None

    @property
    def devices(self):
        # Existing shared dictionary; not a copy.
        return self._devices

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def is_busy(self):
        with self._lock:
            return self._state != HardwareState.IDLE

    def can_manual_control(self):
        with self._lock:
            return self._state == HardwareState.IDLE

    def _reserve(self, state):
        with self._lock:
            if self._state != HardwareState.IDLE:
                raise HardwareBusyError(
                    f"Hardware is currently {self._state.value}"
                )

            token = uuid4().hex
            self._state = state
            self._token = token

            return token

    def reserve_connection(self):
        return self._reserve(HardwareState.CONNECTING)

    def reserve_experiment(self):
        return self._reserve(HardwareState.EXPERIMENT)

    def owns_experiment(self, token):
        with self._lock:
            return (
                self._state == HardwareState.EXPERIMENT
                and self._token == token
            )

    def release_connection(self, token):
        return self._release(token, HardwareState.CONNECTING)

    def release_experiment(self, token):
        return self._release(token, HardwareState.EXPERIMENT)

    def _release(self, token, expected_state):
        with self._lock:
            if (
                self._state != expected_state
                or self._token != token
            ):
                return False

            self._state = HardwareState.IDLE
            self._token = None
            return True

    def require_idle(self):
        with self._lock:
            if self._state != HardwareState.IDLE:
                raise HardwareBusyError(
                    f"Operation blocked: {self._state.value}"
                )

    def begin_shutdown(self):
        with self._lock:
            if self._state == HardwareState.SHUTDOWN:
                return False

            if self._state != HardwareState.IDLE:
                raise HardwareBusyError(
                    "Cannot shut down while hardware is reserved."
                )

            self._state = HardwareState.SHUTDOWN
            self._token = uuid4().hex
            return True

    def cancel_shutdown(self):
        with self._lock:
            if self._state == HardwareState.SHUTDOWN:
                self._state = HardwareState.IDLE
                self._token = None

    def assert_registry_identity(self, devices):
        if devices is not self._devices:
            raise ValueError(
                "HardwareManager and ControlPanel must share "
                "the exact same hardware dictionary."
            )