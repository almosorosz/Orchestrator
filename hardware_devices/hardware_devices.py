#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Hardware Driver Objects for Lab Automation
Standardized OOP design for QtPy6 integration.
"""

import time
import re
from abc import ABC, abstractmethod
import serial
import threading
from asyncua.sync import Client


class BaseDevice(ABC):
    """Abstract Base Class for all hardware devices."""

    def __init__(self, name: str, port: str):
        self.name = name
        self.port = port
        self.is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Establishes connection to the physical unit."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Safely disconnects from the physical unit."""
        pass

    @abstractmethod
    def start(self) -> bool:
        """Starts device action/actuation (e.g., pump running, temp regulation)."""
        pass

    @abstractmethod
    def stop(self) -> bool:
        """Stops device action/actuation."""
        pass

    @abstractmethod
    def set_target(self, value: float | int | bool, **kwargs) -> bool:
        """Sets target setpoint (speed, temperature, state, flow, etc.)."""
        pass

    @abstractmethod
    def read_status(self) -> dict:
        """Reads status / sensor values and returns standard dictionary payload."""
        pass

# =============================================================================
# 0.OPC
# ============================================================================

class OPCUAReader:

    def __init__(self, name, endpoint, node_id, timeout=3.0):
        self.name = name
        self.endpoint = endpoint
        self.node_id = node_id
        self.timeout = float(timeout)

        self.client = None
        self.is_connected = False

    def connect(self):

        if self.is_connected:
            return True

        self.client = Client(
            self.endpoint,
            timeout=self.timeout
        )

        try:
            self.client.connect()
            self.is_connected = True
            return True

        except Exception:
            self.client = None
            self.is_connected = False
            raise

    def disconnect(self):

        if self.client is not None:
            try:
                self.client.disconnect()
            finally:
                self.client = None

        self.is_connected = False
        return True

    def read_value(self):

        if not self.is_connected or self.client is None:
            raise RuntimeError("OPC server is not connected")

        node = self.client.get_node(self.node_id)

        return node.read_value()

    def read_status(self):

        value = self.read_value()

        return {
            "name": self.name,
            "server": self.endpoint,
            "node": self.node_id,
            "value": value
        }

    def test_connection(self):

        was_connected = self.is_connected

        try:
            if not was_connected:
                self.connect()

            value = self.read_value()

            return value

        finally:
            if not was_connected and self.is_connected:
                self.disconnect()



# =============================================================================
# 1. MASTERFLEX PUMP
# =============================================================================
class MasterflexPump(BaseDevice):
    """Masterflex Peristaltic Pump Driver (RS232 ASCII Protocol)."""

    ENQ = b'\x05'
    STX = b'\x02'
    CR = b'\r'

    def __init__(self, name: str, port: str, node_address: str = "P04", baudrate: int = 4800):
        super().__init__(name, port)
        self.node = node_address.encode('ascii')
        self.baudrate = baudrate
        self.ser = None
        self._lock = threading.Lock()
        self.target_rpm = 0.0
        self.is_running = False

    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.SEVENBITS,
                parity=serial.PARITY_ODD,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.5,
                write_timeout=2.0
            )

            time.sleep(2.0)  # Line settling time

            # Clear anything left in the buffers before synchronization
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

            # Sync sequence
            self.ser.write(self.ENQ)
            self.ser.flush()

            time.sleep(0.4)

            self.ser.read_all()

            # Address locking
            self.ser.write(
                self.STX + self.node + self.CR
            )

            self.ser.flush()

            time.sleep(0.4)

            ack = self.ser.read_all()

            self.is_connected = True

            print(
                f"[{self.name}] Connected on {self.port} "
                f"(Node: {self.node.decode()})"
            )

            return True

        except Exception as e:

            print(
                f"[{self.name}] Connection Error: {e}"
            )

            self.is_connected = False

            if self.ser:
                try:
                    self.ser.close()
                except Exception:
                    pass

                self.ser = None

            return False

    def _write_command(self, command: bytes) -> bytes:

        with self._lock:

            if not self.ser or not self.ser.is_open:
                raise RuntimeError(
                    f"[{self.name}] Serial port not open."
                )

            packet = (
                self.STX
                + self.node
                + command
                + self.CR
            )

            try:

                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()

                self.ser.write(packet)

                self.ser.flush()

                time.sleep(0.2)

                return self.ser.read_all()

            except Exception as e:

                print(
                    f"[{self.name}] Serial write error: "
                    f"{type(e).__name__}: {e}"
                )

                raise

    def set_target(self, rpm: float, direction: str = "+", **kwargs) -> bool:
        """Set pump speed. Negative RPM always means counterclockwise."""
    
        rpm = float(rpm)
    
        if rpm == 0:
    
            success = self.stop()
    
            if success:
                self.target_rpm = 0.0
    
            return success
    
        # Negative profile/setpoint always means counterclockwise.
        if rpm < 0:
            actual_direction = "-"
        else:
            actual_direction = direction
    
        speed = abs(rpm)
    
        val_str = f"{int(round(speed)):04d}"
        cmd = f"S{actual_direction}{val_str}".encode("ascii")
    
        reply = self._write_command(cmd)
        success = len(reply) > 0
    
        if not success:
            return False
    
        self.target_rpm = (
            speed
            if actual_direction == "+"
            else -speed
        )
    
        if not self.is_running:
            return self.start()
    
        return True
    
    def start(self) -> bool:
        reply = self._write_command(b'G0')
        success = len(reply) > 0
    
        if success:
            self.is_running = True
    
        return success
    
    def stop(self) -> bool:
    
        reply = self._write_command(b'H')
        success = len(reply) > 0
    
        if success:
            self.is_running = False
    
        return success
    
    def read_status(self) -> dict:
        return {
            "name": self.name,
            "connected": self.is_connected,
            "speed_rpm": self.target_rpm if self.is_running else 0.0
        }
            

    def release_remote_mode(self) -> bool:
        """Releases remote line control back to keypad."""
        reply = self._write_command(b'X')
        return len(reply) > 0

    def disconnect(self) -> None:
    
        if self.ser and self.ser.is_open:

            try:
                self.stop()
                self.release_remote_mode()

            except Exception:
                pass

            with self._lock:

                if self.ser and self.ser.is_open:
                    self.ser.close()

                self.is_connected = False

            print(
                f"[{self.name}] Disconnected safely."
            )



# =============================================================================
# 2. HUBER MINISTAT CONTROLLER
# =============================================================================

class HuberController(BaseDevice):
    """
    Huber Ministat Controller.

    BaseDevice-compatible implementation with:
      - Thread-safe serial access
      - RLock-protected command transactions
      - Deterministic serial-port teardown
      - Safe stop before disconnect
    """

    def __init__(self,name: str,port: str,baudrate: int = 9600,timeout: float = 0.2):
        super().__init__(name, port)

        self.baudrate = baudrate
        self.timeout = timeout

        self.ser = None
        self._lock = threading.RLock()

    def _parse_temperature(self, response):
        """Convert Huber temperature response to °C."""
        if response is None:
            return None
    
        text = str(response).strip()
    
        try:
            return float(text)
        except ValueError:
            pass
    
        match = re.search(r"([-+]?\d+)\s*$", text)
    
        if not match:
            return None
    
        try:
            return int(match.group(1)) / 100.0
        except Exception as e:
            print(f"[{self.name}] Could not parse temperature response '{response}': {e}")
            return None


    def connect(self) -> bool:

        with self._lock:

            try:

                # Clean up any stale serial object first.
                if self.ser is not None:

                    try:
                        if self.ser.is_open:
                            self.ser.close()
                    except Exception:
                        pass

                    self.ser = None

                self.ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=self.timeout,
                )

                self.is_connected = self.ser.is_open

                if self.is_connected:
                    print(f"[{self.name}] Connected on {self.port}")

                return self.is_connected

            except Exception as e:

                print(f"[{self.name}] Connection Error on{self.port}: {e}")

                self.ser = None
                self.is_connected = False

                return False


    def send_command(self, cmd: str) -> str:
        """
        Sends one Huber command and waits for its response.

        The entire write/read transaction is protected by the same lock
        used by disconnect(), preventing teardown from occurring while
        another thread is communicating with the Huber.
        """

        with self._lock:

            if (self.ser is None or not self.ser.is_open or not self.is_connected):
                return ""

            try:

                full_cmd = f"{cmd}\r\n".encode("ascii")

                self.ser.write(full_cmd)
                self.ser.flush()

                response = (self.ser.readline().decode("ascii", errors="ignore").strip())

                return response

            except Exception as ex:

                print(
                    f"[{self.name}] IO Error during "
                    f"command '{cmd}': {ex}"
                )

                # We are already holding the RLock.
                # Close the port immediately on communication failure.
                self._safe_close_unlocked()

                return ""


    def start(self) -> bool:
        """
        Starts Huber circulation / temperature control.
        """
        #response = self.send_command("CA@ +00001")
        
        response = True
        return bool(response)


    def stop(self) -> bool:
        """
        Stops Huber circulation / temperature control.
        """
        response = self.send_command("CA@ +00000")
        return bool(response)


    def set_target(self,temp_celsius: float,**kwargs) -> bool:
        """
        Sets Huber temperature setpoint.

        Uses the original Huber SP@ protocol from your driver.
        """

        try:

            value = int(round(temp_celsius * 100))
            command = f"SP@ {value:+06d}"
            response = self.send_command(command)
            return bool(response)

        except (TypeError, ValueError) as e:

            print(f"[{self.name}] Invalid temperature {temp_celsius}: {e}")

            return False


    def set_jacket_control(self) -> str:
        return self.send_command("TM 0")

    def set_process_control(self) -> str:
        return self.send_command("TM 1")


    def read_status(self) -> dict:
        """Reads Huber telemetry and returns numerical temperatures."""
    
        if not self.is_connected:
            return {
                "name": self.name,
                "connected": False
            }
    
        try:
            setpoint_raw = self.send_command("SP?")
            internal_raw = self.send_command("TI?")
            external_raw = self.send_command("TE?")
    
            return {
                "name": self.name,
                "connected": self.is_connected,
                "setpoint": self._parse_temperature(setpoint_raw),
                "temp_internal": self._parse_temperature(internal_raw),
                "temp_external": self._parse_temperature(external_raw)
            }
    
        except Exception as e:
            return {
                "name": self.name,
                "connected": False,
                "error": str(e)
            }



    def _safe_close_unlocked(self):
        """
        Actually closes the PySerial handle.

        IMPORTANT:
        Caller must already hold self._lock.
        """

        self.is_connected = False

        ser = self.ser

        # Remove the reference immediately so the rest of the application
        # cannot treat the old handle as active.
        self.ser = None

        if ser is not None:

            try:

                if ser.is_open:

                    print( f"[{self.name}] Closing serial port {self.port}..."  )

                    # Best-effort buffer cleanup.
                    try:
                        ser.reset_input_buffer()
                    except Exception:
                        pass

                    try:
                        ser.reset_output_buffer()
                    except Exception:
                        pass

                    # THE IMPORTANT OPERATION:
                    ser.close()

                    print( f"[{self.name}] Serial close() completed.")

            except Exception as e:

                print(f"[{self.name}] Error closing{self.port}: {e}")

            finally:

                # Second close attempt is harmless and gives us
                # additional protection against unusual PySerial states.
                try:
                    if ser.is_open:
                        ser.close()
                except Exception:
                    pass

        print(
            f"[{self.name}] Hardware handle released "
            f"({self.port})."
        )


    def disconnect(self) -> None:
        """
        Safely stop Huber and release COM port.

        This method is deliberately synchronous.
        When it returns, PySerial has been told to close the port.
        """

        print(  f"[{self.name}] DISCONNECT REQUESTED(port={self.port})")

        with self._lock:

            # Send stop command while the port is still valid.
            if (  self.ser is not None  and self.ser.is_open  and self.is_connected):

                try:

                    self.ser.write(b"CA@ +00000\r\n")
                    self.ser.flush()
                    # Give the Huber a short opportunity to process
                    # the stop command.
                    time.sleep(0.05)

                except Exception as e:

                    print(f"[{self.name}] Stop command failed  during disconnect: {e}")

            # Definitive COM-port release.
            self._safe_close_unlocked()

        print( f"[{self.name}] DISCONNECT COMPLETE(port={self.port}, is_connected={self.is_connected})")


# =============================================================================
# 3. J-KEM STIRRER CONTROLLER
# =============================================================================
class JKemStirrer(BaseDevice):
    """J-KEM Scientific Overhead Stirrer Driver."""

    def __init__(self, name: str, port: str, baudrate: int = 9600, timeout: float = 1.0):
        super().__init__(name, port)
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            time.sleep(0.3)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.is_connected = True
            print(f"[{self.name}] Connected on {self.port}")
            return True
        except Exception as e:
            print(f"[{self.name}] Connection Error: {e}")
            self.is_connected = False
            return False

    def _send_command(self, cmd: str) -> str:
        if not self.ser or not self.ser.is_open:
            raise RuntimeError(f"[{self.name}] Serial port not connected.")
        self.ser.reset_input_buffer()
        full_cmd = f"{cmd.strip().upper()}\r"
        self.ser.write(full_cmd.encode('ascii'))
        time.sleep(0.1)
        return self.ser.read_until(b'\r').decode('ascii', errors='ignore').strip()

    def set_target(self, rpm: int, **kwargs) -> bool:
        """Sets motor target speed (0 or 20-800 RPM)."""
        if rpm != 0 and not (20 <= rpm <= 800):
            print(f"[{self.name}] Invalid RPM: {rpm}. Range: 0, 20-800.")
            return False
        reply = self._send_command(f"SS{rpm}")
        return reply == f"SS{rpm}"

    def start(self) -> bool:
        """No explicit start command required; speed setting drives motor."""
        return True

    def stop(self) -> bool:
        return self.set_target(0)

    def release_to_manual(self) -> bool:
        return bool(self._send_command("RM"))

    def read_status(self) -> dict:
        raw_status = self._send_command("MS")
        status_map = {
            "MS0": "Normal Operation",
            "MS1": "Overloaded",
            "MS2": "Stalled / Excessive Current",
            "MS3": "Terminal Open",
            "MS4": "Stopped by Hardware"
        }
        return {
            "name": self.name,
            "speed_rpm": self._send_command("SS").replace("SS", ""),
            "torque_ozin": self._send_command("TQ").replace("TQ", ""),
            "current_a": self._send_command("CU").replace("CU", ""),
            "voltage_v": self._send_command("VL").replace("VL", ""),
            "status": status_map.get(raw_status, raw_status)
        }

    def disconnect(self) -> None:
        if self.ser and self.ser.is_open:
            self.stop()
            self.ser.close()
            self.is_connected = False
            print(f"[{self.name}] Disconnected safely.")


# =============================================================================
# 4. METTLER SCALE READER
# =============================================================================
class MettlerScale(BaseDevice):
    """Mettler Balance Scale Reader Driver."""

    def __init__(self, name: str, port: str, baudrate: int = 9600, timeout: float = 1.0):
        super().__init__(name, port)
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            self.is_connected = True
            print(f"[{self.name}] Connected on {self.port}")
            return True
        except Exception as e:
            print(f"[{self.name}] Connection Error: {e}")
            self.is_connected = False
            return False

    def start(self) -> bool:
        return True  # Passive monitor

    def stop(self) -> bool:
        return True

    def set_target(self, value=None, **kwargs) -> bool:
        return True  # Scale setpoints are read-only

    def read_weight(self) -> float | None:
        """Polls scale once for current mass in grams."""
        if not self.ser or not self.ser.is_open:
            return None
        self.ser.write(b"SI\r\n")
        raw = self.ser.read_until(b"\r").decode("utf-8", errors="ignore").strip()
        if raw:
            match = re.search(r"S\s+[SD]\s+([-+]?\d*\.?\d+)", raw)
            if match:
                return float(match.group(1))
        return None

    def read_status(self) -> dict:
        return {
            "name": self.name,
            "weight_g": self.read_weight()
        }

    def disconnect(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.is_connected = False
            print(f"[{self.name}] Disconnected safely.")


class IsmatecPump(BaseDevice):

    _shared_ports = {}
    _registry_lock = threading.RLock()

    def __init__(
        self,
        name: str,
        port: str,
        stage: str = "2",
        baudrate: int = 9600,
        timeout: float = 1.0
    ):
        super().__init__(name, port)

        self.baudrate = baudrate
        self.timeout = timeout

        self.stage = str(stage)

        if self.stage not in {"1", "2", "3"}:
            raise ValueError(
                f"Invalid Ismatec stage '{stage}'. "
                f"Expected 1, 2, or 3."
            )

        self.ser = None
        self._serial_lock = None

        self.target_flow = 0.0
        self.is_running = False

    def connect(self) -> bool:

        try:

            with self._registry_lock:

                shared = self._shared_ports.get(
                    self.port
                )

                # --------------------------------------------------
                # Reuse an already-open COM port
                # --------------------------------------------------

                if (
                    shared is not None
                    and shared["ser"] is not None
                    and shared["ser"].is_open
                ):

                    self.ser = shared["ser"]
                    self._serial_lock = shared["lock"]

                    shared["users"].add(self)

                    self.is_connected = True

                    print(
                        f"[{self.name}] Reusing {self.port} "
                        f"for Ismatec stage {self.stage}"
                    )

                    return True

                # --------------------------------------------------
                # First Ismatec logical device on this COM port
                # --------------------------------------------------

                ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=self.timeout,
                    xonxoff=True
                )

                lock = threading.RLock()

                self._shared_ports[self.port] = {
                    "ser": ser,
                    "lock": lock,
                    "users": {self}
                }

                self.ser = ser
                self._serial_lock = lock
                self.is_connected = True

            # Switch controller to individual-channel mode.
            self._send_cmd("1~1")

            print(
                f"[{self.name}] Connected on {self.port} "
                f"| stage {self.stage}"
            )

            return True

        except Exception as e:

            print(
                f"[{self.name}] Connection Error: {e}"
            )

            self.is_connected = False
            self.ser = None
            self._serial_lock = None

            return False

    def _resolve_stage(self, stage=None):

        stage = (
            self.stage
            if stage is None
            else str(stage)
        )

        if stage not in {"1", "2", "3"}:
            raise ValueError(
                f"Invalid stage '{stage}'. "
                f"Expected 1, 2, or 3."
            )

        return stage

    def _send_cmd(self, cmd: str) -> str:

        if (
            self.ser is None
            or not self.ser.is_open
        ):
            raise RuntimeError(
                f"[{self.name}] Serial port not connected."
            )

        if not cmd.endswith("\r"):
            cmd += "\r"

        lock = self._serial_lock

        if lock is None:
            raise RuntimeError(
                f"[{self.name}] Missing Ismatec serial lock."
            )

        with lock:

            self.ser.write(
                cmd.encode("ascii")
            )

            self.ser.flush()

            time.sleep(0.5)

            return self.ser.read_all().decode(
                "ascii",
                errors="ignore"
            )

    def set_target(
        self,
        flow_value: float,
        stage=None,
        **kwargs
    ) -> bool:
    
        stage = self._resolve_stage(
            stage
        )
    
        try:
    
            flow_value = float(
                flow_value
            )
    
            # --------------------------------------------------
            # ZERO FLOW -> STOP THIS STAGE
            # --------------------------------------------------
    
            if flow_value == 0:
    
                self._send_cmd(
                    f"{stage}S000000"
                )
    
                self._send_cmd(
                    f"{stage}I"
                )
    
                if stage == self.stage:
    
                    self.target_flow = 0.0
                    self.is_running = False
    
                return True
    
            # --------------------------------------------------
            # DIRECTION
            # --------------------------------------------------
    
            direction_cmd = (
                f"{stage}J"
                if flow_value < 0
                else f"{stage}K"
            )
    
            self._send_cmd(
                direction_cmd
            )
    
            # --------------------------------------------------
            # FLOW SETPOINT
            # --------------------------------------------------
    
            val_str = (
                f"{int(abs(flow_value) * 100):06d}"
            )
    
            self._send_cmd(
                f"{stage}S{val_str}"
            )
    
            # --------------------------------------------------
            # ENSURE STAGE IS RUNNING
            # --------------------------------------------------
    
            self._send_cmd(
                f"{stage}H"
            )
    
            if stage == self.stage:
    
                self.target_flow = flow_value
                self.is_running = True
    
            return True
    
        except Exception as e:
    
            print(
                f"[{self.name}] Failed to set "
                f"stage {stage} flow: {e}"
            )
    
            return False

    def start(
        self,
        stage=None
    ) -> bool:

        stage = self._resolve_stage(stage)

        try:

            self._send_cmd(
                f"{stage}H"
            )

            if stage == self.stage:
                self.is_running = True

            return True

        except Exception as e:

            print(
                f"[{self.name}] Failed to start "
                f"stage {stage}: {e}"
            )

            return False

    def stop(
        self,
        stage=None
    ) -> bool:

        stage = self._resolve_stage(stage)

        try:

            self._send_cmd(
                f"{stage}I"
            )

            if stage == self.stage:
                self.is_running = False

            return True

        except Exception as e:

            print(
                f"[{self.name}] Failed to stop "
                f"stage {stage}: {e}"
            )

            return False

    def read_status(self) -> dict:

        return {
            "name": self.name,
            "connected": self.is_connected,
            "stage": self.stage,
            "flow": (
                self.target_flow
                if self.is_running
                else 0.0
            )
        }

    def disconnect(self) -> None:

        if not self.is_connected:
            return

        try:

            self.stop()

        except Exception as e:

            print(
                f"[{self.name}] Could not stop "
                f"stage {self.stage}: {e}"
            )

        with self._registry_lock:

            shared = self._shared_ports.get(
                self.port
            )

            if shared is not None:

                shared["users"].discard(self)

                # Close the physical COM port only when
                # no Ismatec logical stages still use it.
                if not shared["users"]:

                    ser = shared["ser"]

                    try:

                        if (
                            ser is not None
                            and ser.is_open
                        ):
                            ser.close()

                    except Exception as e:

                        print(
                            f"[{self.name}] Error closing "
                            f"{self.port}: {e}"
                        )

                    self._shared_ports.pop(
                        self.port,
                        None
                    )

                    print(
                        f"[{self.name}] Physical COM port "
                        f"{self.port} closed."
                    )

                else:

                    print(
                        f"[{self.name}] Stage {self.stage} "
                        f"disconnected; {self.port} remains open "
                        f"for another Ismatec stage."
                    )

        self.ser = None
        self._serial_lock = None
        self.is_connected = False
        self.is_running = False

# =============================================================================
# 6. NI-DAQ DIGITAL VALVE CONTROLLER
# =============================================================================
class NIDAQValveController(BaseDevice):
    """National Instruments DAQ Valve & Voltage Monitor Driver."""

    def __init__(self, name: str, ai_channel: str = "Dev1/ai0", do_channel: str = "Dev1/port0/line0", threshold: float = 2.5):
        super().__init__(name, port=f"{ai_channel},{do_channel}")
        self.ai_channel = ai_channel
        self.do_channel = do_channel
        self.threshold = threshold
        self.state = False  # False = Closed, True = Open

    def connect(self) -> bool:
        import nidaqmx  # Dynamic import for DAQ setup check
        try:
            with nidaqmx.Task() as write_task:
                write_task.do_channels.add_do_chan(self.do_channel)
                write_task.write(False)  # Safety initialization to closed
            self.is_connected = True
            print(f"[{self.name}] NI-DAQ Initialized successfully.")
            return True
        except Exception as e:
            print(f"[{self.name}] DAQ Initialization Error: {e}")
            self.is_connected = False
            return False

    def set_target(self, state: bool, **kwargs) -> bool:
        """Set Valve Position: True = Open (5V), False = Closed (0V)."""
        import nidaqmx
        try:
            with nidaqmx.Task() as write_task:
                write_task.do_channels.add_do_chan(self.do_channel)
                write_task.write(bool(state))
            self.state = bool(state)
            return True
        except Exception as e:
            print(f"[{self.name}] Failed to actuate valve: {e}")
            return False

    def start(self) -> bool:
        return True

    def stop(self) -> bool:
        return self.set_target(False)  # Close valve safely

    def read_status(self) -> dict:
        """Reads digital feedback line voltage."""
        import nidaqmx
        from nidaqmx.constants import TerminalConfiguration
        try:
            with nidaqmx.Task() as read_task:
                read_task.ai_channels.add_ai_voltage_chan(
                    physical_channel=self.ai_channel,
                    min_val=0.0,
                    max_val=5.0,
                    terminal_config=TerminalConfiguration.DIFF
                )
                voltage = read_task.read()

            circuit_state = "OPEN (0V)" if voltage >= self.threshold else "CLOSED / SHORTED (5V)"
            return {
                "name": self.name,
                "target_state": self.state,
                "measured_voltage": round(voltage, 3),
                "circuit_state": circuit_state
            }
        except Exception as e:
            return {"name": self.name, "error": str(e)}

    def disconnect(self) -> None:
        self.stop()
        self.is_connected = False
        print(f"[{self.name}] Safe teardown complete.")

