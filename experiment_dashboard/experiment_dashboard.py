import time
import threading
import pandas as pd
import numpy as np
import logging
from uuid import uuid4

from experiment.models import Campaign, ExperimentDefinition, ExperimentRun, RunStatus
from hardware_devices.simulated import update_simulated_process, SimulatedDevice
from utilities.plot_sampling import simulation_ticks
from experiment.recording import RunRecorder, resume_sql

logger = logging.getLogger("orchestrator.execution")


def _debug(*values, **kwargs):
    logger.debug(" ".join(str(value) for value in values))

from hardware_devices.hardware_manager import HardwareBusyError
from hardware_devices.capabilities import is_monitor_only

from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
)



PROFILE_TYPES = {
    "masterflex",
    "huber",
    "jkem",
    "ismatec",
    "nidaq_valve"
}

# =============================================================================
# WORKER THREAD FOR ASYNCHRONOUS DEVICE EXECUTION & TELEMETRY
# =============================================================================
class ExperimentWorker(QObject):

    status_updated = Signal(dict)
    step_changed = Signal(int)
    finished = Signal()
    experiment_changed = Signal(int, int)
    run_finished = Signal(dict)
    recording_started = Signal(str)

    def __init__(self, devices: dict, device_configs: dict, experiments: list[pd.DataFrame], poll_interval: float = 1.0,
                 *, clock=time.monotonic, speed=1.0, end_hold_s=5.0, campaign=None, recording=None):
        super().__init__()

        self.devices = devices
        self.device_configs = device_configs

        if not experiments:
            raise ValueError("At least one experiment is required")
        if not all(np.isfinite(v) for v in (speed, poll_interval, end_hold_s)) or speed <= 0 or poll_interval <= 0 or end_hold_s < 0:
            raise ValueError("Invalid execution timing settings")
        if speed != 1 and any(not getattr(dev, "simulated", False) for dev in devices.values()):
            raise ValueError("Accelerated execution requires every device to be simulated")
        self.definitions = tuple(
            item if isinstance(item, ExperimentDefinition) else
            ExperimentDefinition.from_frame(item, device_configs, name=f"Experiment {i + 1}")
            for i, item in enumerate(experiments)
        )
        self.experiments = [definition.to_frame() for definition in self.definitions]
        # Validate typed definitions too, using the currently assigned devices.
        for frame in self.experiments:
            ExperimentDefinition.from_frame(frame, device_configs)
        for frame in self.experiments:
            for device_id in frame.columns.drop("time_s"):
                if device_id not in devices:
                    raise ValueError(f"Assigned device is missing: {device_id}")
        self.runs = [ExperimentRun(definition.experiment_id, run_id=f"{number}_{uuid4().hex}")
                     for number, definition in enumerate(self.definitions, start=1)]
        if campaign is not None and campaign.experiments != self.definitions:
            raise ValueError("Campaign definitions do not match the requested experiments")
        self.campaign = campaign or Campaign("Profile campaign", self.definitions)
        self.clock, self.speed, self.end_hold_s = clock, speed, end_hold_s
        self.outcome = RunStatus.PENDING
        self.error_message = ""
        self.recording_settings = recording or {}
        self.recorder = None
        self.sql_sync = None
        self._process_time = 0.0
        self._simulation_time = 0.0
        self._event_simulation = speed > 1 and all(isinstance(dev, SimulatedDevice) for dev in devices.values())
        self._simulation_clocks = []
        self._plot_samples = []

        self.poll_interval = poll_interval

        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._skip_lock = threading.Lock()
        self._skippable_run = None
        self._skip_requested = False

        self._teardown_lock = threading.Lock()
        self._teardown_done = False

        self.executor = None

    # ==============================================================
    # START
    # ==============================================================

    @Slot()
    def start_experiment(self):

        # Preserve a stop requested while QThread startup was pending.
        if self._stop_event.is_set():
            self.outcome = RunStatus.STOPPED
            self.run_finished.emit({"status": self.outcome.value, "error": ""})
            self.finished.emit()
            return
        self.outcome = RunStatus.RUNNING
        for dev in self.devices.values():
            if getattr(dev, "simulated", False):
                dev.time_scale = self.speed
        logger.info("Campaign started (%d experiments, speed %g×)", len(self.experiments), self.speed)

        with self._teardown_lock:
            self._teardown_done = False

        self.executor = ThreadPoolExecutor(
            max_workers=len(self.devices) + 2
        )

        try:
            if self._event_simulation:
                for dev in self.devices.values():
                    self._simulation_clocks.append((dev, dev.set_simulation_clock(lambda: self._simulation_time)))

            if self.recording_settings.get("enabled"):
                safe_settings = {key: value for key, value in self.recording_settings.items() if key != "remote_sql_url"}
                self.recorder = RunRecorder(self.recording_settings["output_folder"], self.campaign,
                                            self.device_configs, safe_settings)
                self.recording_started.emit(str(self.recorder.folder))
                if self.recording_settings.get("remote_sql_url"):
                    self.sql_sync = resume_sql(self.recording_settings)
                logger.info("Recording campaign to %s", self.recorder.folder)

            _debug(
                "[WORKER] Verifying hardware connections..."
            )

            disconnected = [
                dev_id
                for dev_id, dev in self.devices.items()
                if not getattr(
                    dev,
                    "is_connected",
                    False
                )
            ]

            if disconnected:

                raise RuntimeError(
                    "Cannot start experiment. "
                    "Disconnected devices: "
                    + ", ".join(disconnected)
                )

            _debug(
                "[WORKER] All devices are connected."
            )

            if self._stop_event.is_set():
                return

            # ======================================================
            # SET HUBER CONTROL MODE
            # ======================================================

            for dev_id, dev in self.devices.items():

                config = self.device_configs.get(
                    dev_id,
                    {}
                )

                if config.get("type") != "huber" or is_monitor_only(config):
                    continue

                control_mode = config.get(
                    "defaults",
                    {}
                ).get(
                    "control_mode",
                    "Jacket"
                )

                try:

                    if control_mode == "Jacket":

                        if not dev.set_jacket_control():
                            raise RuntimeError("Jacket control command rejected")

                        _debug(
                            f"[HUBER] {dev_id}: "
                            f"Jacket control selected."
                        )

                    elif control_mode == "Process":

                        if not dev.set_process_control():
                            raise RuntimeError("Process control command rejected")

                        _debug(
                            f"[HUBER] {dev_id}: "
                            f"Process control selected."
                        )

                    else:

                        raise ValueError(
                            f"Invalid Huber control mode: "
                            f"{control_mode}"
                        )

                except Exception as e:

                    raise RuntimeError(
                        f"Could not set Huber control mode "
                        f"for {dev_id}: {e}"
                    )

            # ======================================================
            # START ALL DEVICES
            # ======================================================

            start_futures = []

            for dev_id, dev in self.devices.items():

                if (
                    not getattr(
                        dev,
                        "is_connected",
                        False
                    )
                    or not callable(
                        getattr(
                            dev,
                            "start",
                            None
                        )
                    )
                ):
                    continue

                config = self.device_configs.get(
                    dev_id,
                    {}
                )

                device_type = config.get(
                    "type"
                )

                if is_monitor_only(config):
                    continue
                if device_type in {
                    "jkem",
                    "ismatec",
                    "masterflex",
                }:
                    continue

                try:

                    if device_type == "ismatec":

                        stage = config.get(
                            "defaults",
                            {}
                        ).get(
                            "stage"
                        )

                        future = self.executor.submit(
                            dev.start,
                            stage=stage
                        )

                    else:

                        future = self.executor.submit(
                            dev.start
                        )

                    start_futures.append(
                        (dev_id, future)
                    )

                except Exception as e:

                    _debug(
                        f"[START SUBMIT ERROR] "
                        f"{dev_id}: "
                        f"{type(e).__name__}: {e}"
                    )
                    raise RuntimeError(f"Could not start {dev_id}: {e}") from e

            start_failed = False

            for dev_id, future in start_futures:

                try:

                    success = future.result(
                        timeout=5
                    )

                    _debug(
                        f"Device '{dev_id}' "
                        f"start status: {success}"
                    )

                    if not success:
                        start_failed = True

                except TimeoutError:

                    _debug(
                        f"[START TIMEOUT] {dev_id}"
                    )

                    start_failed = True

                except Exception as e:

                    _debug(
                        f"[START ERROR] {dev_id}: "
                        f"{type(e).__name__}: {e}"
                    )

                    start_failed = True

            if start_failed:

                raise RuntimeError(
                    "One or more devices failed to start."
                )

            # ======================================================
            # RUN COMPLETE DOE
            # ======================================================

            total_experiments = len(
                self.experiments
            )

            for exp_idx, profile in enumerate(
                self.experiments
            ):

                if self._stop_event.is_set():
                    break

                with self._skip_lock:
                    self._skippable_run = exp_idx + 1 if exp_idx + 1 < total_experiments else None
                    self._skip_requested = False
                    self._wake_event.clear()

                self.experiment_changed.emit(
                    exp_idx + 1,
                    total_experiments
                )

                _debug(
                    f"\n========================================\n"
                    f"STARTING EXPERIMENT "
                    f"{exp_idx + 1}/{total_experiments}\n"
                    f"========================================"
                )

                profile = (
                    profile
                    .sort_values("time_s")
                    .reset_index(drop=True)
                )

                run = self.runs[exp_idx]
                self._active_profiles = {p.device_id: p for p in self.definitions[exp_idx].profiles}
                run.status = RunStatus.RUNNING
                if self.recorder:
                    self.recorder.submit("run_start", run.run_id, {
                        "experiment_id": run.experiment_id, "experiment_number": exp_idx + 1,
                        "batch_name": run.run_id, "planned": profile.to_dict(orient="list")})
                logger.info("Experiment %d/%d started (run %s)", exp_idx + 1, total_experiments, run.run_id)
                start_time = self.clock()
                self._process_time = 0.0
                last_gui_publish = float("-inf")
                self._plot_samples = []
                simulation_base = self._simulation_time
                wall_start = time.monotonic()

                total_steps = len(profile)
                current_step_idx = -1

                times = profile[
                    "time_s"
                ].to_numpy(
                    dtype=float
                )
                ticks = iter(simulation_ticks(profile, self.poll_interval, self.end_hold_s)) if self._event_simulation else None
                next_tick = next(ticks) if ticks is not None else None

                while not self._stop_event.is_set() and not self._wake_event.is_set():

                    elapsed = next_tick[0] if next_tick is not None else (
                        self.clock()
                        - start_time
                    ) * self.speed
                    if self._event_simulation:
                        self._simulation_time = simulation_base + elapsed

                    # ------------------------------------------
                    # Determine current profile interval
                    # ------------------------------------------

                    idx = (
                        np.searchsorted(
                            times,
                            elapsed,
                            side="right"
                        ) - 1
                    )

                    idx = max(
                        0,
                        min(
                            idx,
                            total_steps - 1
                        )
                    )

                    if idx != current_step_idx:

                        current_step_idx = idx

                        self.step_changed.emit(
                            current_step_idx
                        )

                    # ------------------------------------------
                    # Interpolated setpoints
                    # ------------------------------------------

                    interpolated_row = (
                        self._interpolate_profile_row(
                            profile,
                            elapsed
                        )
                    )

                    if self.recorder:
                        self.recorder.submit("command_attempt", run.run_id, interpolated_row.to_dict())
                    if self._event_simulation:
                        # Integrate delivery under the previous target before
                        # switching at this exact simulated-time boundary.
                        update_simulated_process(self.devices, self.device_configs, elapsed - self._process_time)
                        self._process_time = elapsed
                    self._apply_profile_step(interpolated_row)
                    if self.recorder:
                        self.recorder.submit("command_accepted", run.run_id, interpolated_row.to_dict())

                    # ------------------------------------------
                    # Telemetry
                    # ------------------------------------------

                    if not self._stop_event.is_set():

                        if not self._event_simulation:
                            update_simulated_process(self.devices, self.device_configs, elapsed - self._process_time)
                        self._process_time = elapsed

                        telemetry = (
                            self._poll_all_telemetry(
                                elapsed
                            )
                        )

                        telemetry[
                            "experiment_index"
                        ] = exp_idx
                        telemetry["profile_boundary"] = bool(next_tick[1]) if next_tick is not None else False
                        if self.recorder:
                            sample = {"elapsed_s": elapsed}
                            for key in self.devices:
                                sample[f"{key}.target"] = float(interpolated_row[key]) if key in interpolated_row else None
                                status = telemetry.get(key, {})
                                for field in ("connected", "speed_rpm", "flow", "setpoint", "temp_internal", "temp_external",
                                              "weight_g", "measured_voltage", "target_state", "value", "torque_ozin", "error"):
                                    value = status.get(field)
                                    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                                        value = None
                                    if isinstance(value, np.generic):
                                        value = value.item()
                                    sample[f"{key}.{field}"] = value
                            self.recorder.submit("sample", run.run_id, sample)

                        # Plot delivery uses wall time, independent of the
                        # accelerated experiment clock. Control/polling continues.
                        publish_time = time.monotonic()
                        self._plot_samples.append(telemetry)
                        if (publish_time - last_gui_publish >= 0.25 or len(self._plot_samples) >= 256
                                or elapsed >= times[-1] + self.end_hold_s):
                            self._publish_plot_samples()
                            last_gui_publish = publish_time

                    # ------------------------------------------
                    # Experiment finished?
                    # ------------------------------------------

                    final_time = float(
                        times[-1]
                    )

                    if elapsed >= final_time + self.end_hold_s:
                        break

                    if ticks is not None:
                        next_tick = next(ticks, None)
                        if next_tick is None:
                            break
                        delay = max(0., wall_start + next_tick[0] / self.speed - time.monotonic())
                    else:
                        delay = self.poll_interval / self.speed
                    self._wake_event.wait(delay)

                with self._skip_lock:
                    self._skippable_run = None
                    skipped = self._skip_requested and not self._stop_event.is_set()
                self._publish_plot_samples()
                if skipped:
                    # Profile commands have finished before this boundary.
                    self._idle_actuators()
                    if self.outcome == RunStatus.FAILED:
                        break
                if self._stop_event.is_set():
                    run.status = RunStatus.STOPPED
                else:
                    run.status = RunStatus.SKIPPED if skipped else RunStatus.COMPLETED
                if self.recorder:
                    self.recorder.submit("run_end", run.run_id, {"run_id": run.run_id,
                        "experiment_id": run.experiment_id, "experiment_number": exp_idx + 1,
                        "batch_name": run.run_id, "status": run.status.value})
                logger.info("Experiment %d %s", exp_idx + 1, run.status.value)

                _debug(
                    f"FINISHED EXPERIMENT "
                    f"{exp_idx + 1}/"
                    f"{total_experiments}"
                )

            _debug(
                "\nCOMPLETE DOE FINISHED."
            )

        except Exception as e:

            self.outcome = RunStatus.FAILED
            self.error_message = f"{type(e).__name__}: {e}"
            logger.exception("Experiment execution failed")
            for run in self.runs:
                if run.status == RunStatus.RUNNING:
                    run.status, run.error = RunStatus.FAILED, self.error_message

            _debug(
                f"[WORKER ERROR] "
                f"{type(e).__name__}: {e}"
            )

            logger.debug("Execution traceback", exc_info=True)

        finally:
            with self._skip_lock:
                self._skippable_run = None
            self._publish_plot_samples()
            if self.outcome != RunStatus.FAILED:
                self.outcome = RunStatus.STOPPED if self._stop_event.is_set() else RunStatus.COMPLETED
            for run in self.runs:
                if run.status == RunStatus.RUNNING:
                    run.status = self.outcome
            self._return_hardware_to_idle()
            for device, previous in self._simulation_clocks:
                device.restore_simulation_clock(previous)
            self._simulation_clocks.clear()
            if self.outcome == RunStatus.FAILED:
                for run in reversed(self.runs):
                    if run.status != RunStatus.PENDING:
                        run.status, run.error = RunStatus.FAILED, self.error_message
                        break
            logger.info("Campaign %s%s", self.outcome.value, f": {self.error_message}" if self.error_message else "")
            if self.recorder:
                try:
                    self.recorder.submit("summary", "", {"status": self.outcome.value, "error": self.error_message,
                        "runs": [{"run_id": run.run_id, "experiment_id": run.experiment_id,
                                  "status": run.status.value, "error": run.error} for run in self.runs]})
                    self.recorder.close()
                except Exception as error:
                    self.outcome = RunStatus.FAILED
                    self.error_message += f"; Recording failed: {error}"
                    logger.error("%s", self.error_message)
                    try:
                        if self.recorder.thread.is_alive():
                            self.recorder.close()
                        (self.recorder.folder / "RECORDING_FAILED.txt").write_text(self.error_message, encoding="utf-8")
                    except Exception:
                        pass
            self.run_finished.emit({
                "campaign_id": self.campaign.campaign_id,
                "recording_folder": str(self.recorder.folder) if self.recorder else None,
                "recording_pending": self.recorder.pending_csv if self.recorder else False,
                "status": self.outcome.value, "error": self.error_message,
                "runs": [{"run_id": run.run_id, "experiment_id": run.experiment_id,
                          "status": run.status.value, "error": run.error} for run in self.runs],
            })
            self.finished.emit()

    # ==============================================================
    # REQUEST STOP
    # ==============================================================

    def _publish_plot_samples(self):
        if self._plot_samples:
            packet = {**self._plot_samples[-1], "samples": tuple(self._plot_samples)}
            self._plot_samples = []
            self.status_updated.emit(packet)

    def request_stop(self):
        """Thread-safe emergency stop request.

        This method does NOT perform hardware I/O.
        It is safe to call directly from the GUI thread.
        """

        _debug(  "[STOP] Emergency stop requested.")

        self._stop_event.set()
        self._wake_event.set()

    def request_skip(self, experiment_number):
        """Wake the worker only if the displayed run still has a successor."""
        with self._skip_lock:
            if (self._stop_event.is_set() or self._skip_requested or self._skippable_run is None
                    or self._skippable_run != experiment_number):
                return False
            self._skip_requested = True
            self._wake_event.set()
            return True



    # ==============================================================
    # PROFILE
    # =======


    def _interpolate_profile_row(
        self,
        profile,
        elapsed
    ):

        times = profile[
            "time_s"
        ].to_numpy(dtype=float)

        row = {
            "time_s": elapsed
        }

        for dev_id in profile.columns:

            if dev_id == "time_s":
                continue

            values = profile[
                dev_id
            ].to_numpy(dtype=float)

            config = self.device_configs.get(
                dev_id,
                {}
            )

            device_type = config.get(
                "type"
            )

            # --------------------------------------------------
            # DISCRETE DEVICE
            # --------------------------------------------------

            assigned_profile = getattr(self, "_active_profiles", {}).get(dev_id)
            if device_type == "nidaq_valve" or (assigned_profile and assigned_profile.interpolation == "step"):

                idx = (
                    np.searchsorted(
                        times,
                        elapsed,
                        side="right"
                    ) - 1
                )

                idx = max(
                    0,
                    min(
                        idx,
                        len(values) - 1
                    )
                )

                value = values[idx]

            # --------------------------------------------------
            # CONTINUOUS DEVICES
            # --------------------------------------------------

            else:

                value = np.interp(
                    elapsed,
                    times,
                    values
                )

            row[
                dev_id
            ] = value

        return pd.Series(row)

    def _apply_profile_step(self, step_row):

        if self._stop_event.is_set():
            return

        _debug(
            f"\n--- [T={step_row['time_s']} s] "
            f"Executing Profile Step ---"
        )
        _debug(
            "[PROFILE STEP]",
            step_row.to_dict()
        )

        futures = []

        def submit_command(
            dev_id,
            description,
            func,
            *args,
            **kwargs
        ):

            if self._stop_event.is_set():
                return

            dev = self.devices.get(dev_id)

            if dev is None:
                raise RuntimeError(f"Assigned device is missing: {dev_id}")

            if not getattr(dev, "is_connected", False):

                _debug(
                    f"[PROFILE] {dev_id}: "
                    f"not connected, skipping {description}"
                )

                raise RuntimeError(f"{dev_id} disconnected during {description}")

            future = self.executor.submit(
                func,
                *args,
                **kwargs
            )

            futures.append(
                (future, dev_id, description)
            )

        # ==========================================================
        # APPLY PROFILE VALUES
        # ==========================================================

        for dev_id in step_row.index:

            if dev_id == "time_s":
                continue

            if dev_id not in self.devices:
                continue

            value = step_row[dev_id]

            if pd.isna(value):
                continue

            config = self.device_configs.get(dev_id)

            if config is None:
                continue

            device_type = config["type"]

            if device_type not in PROFILE_TYPES:
                continue

            dev = self.devices[dev_id]

            options = config.get(
                "defaults",
                {}
            )

            # ------------------------------------------------------
            # HUBER
            # ------------------------------------------------------

            if device_type == "huber":
                # Mode was applied before startup. Keep commands to one port
                # ordered rather than submitting mode and target concurrently.
                submit_command(
                    dev_id,
                    f"set_target({value})",
                    dev.set_target,
                    float(value)
                )

            # ------------------------------------------------------
            # MASTERFLEX
            # ------------------------------------------------------

            elif device_type == "masterflex":

                value = float(value)

                direction = (
                    "-"
                    if value < 0
                    else options.get("direction", "+")
                )

                submit_command(
                    dev_id,
                    f"set_target({value}, direction={direction})",
                    dev.set_target,
                    value,
                    direction=direction
                )

            # ------------------------------------------------------
            # ISMATEC
            # ------------------------------------------------------

            elif device_type == "ismatec":

                stage = options.get(
                    "stage"
                )

                submit_command(
                    dev_id,
                    f"set_target({value}, stage={stage})",
                    dev.set_target,
                    float(value),
                    stage=stage
                )

            # ------------------------------------------------------
            # JKEM
            # ------------------------------------------------------

            elif device_type == "jkem":

                submit_command(
                    dev_id,
                    f"set_target({value})",
                    dev.set_target,
                    int(round(value))
                )

            # ------------------------------------------------------
            # NI-DAQ VALVE
            # ------------------------------------------------------

            elif device_type == "nidaq_valve":

                valve_state = int(value)

                if valve_state not in (0, 1):
                    _debug(
                        f"[PROFILE] {dev_id}: "
                        f"invalid valve state {value}; expected 0 or 1"
                    )
                    continue

                submit_command(
                    dev_id,
                    f"set_target({valve_state})",
                    dev.set_target,
                    valve_state
                )

        # ==========================================================
        # WAIT FOR COMMANDS
        # ==========================================================

        for future, dev_id, description in futures:

            try:

                result = future.result(
                    timeout=5
                )

                if not result:
                    raise RuntimeError(f"{dev_id}: {description} was rejected")

                _debug(
                    f"[PROFILE] {dev_id}: "
                    f"{description} -> {result}"
                )

            except TimeoutError:

                _debug(
                    f"[PROFILE COMMAND TIMEOUT] "
                    f"{dev_id}: {description}"
                )
                raise RuntimeError(f"{dev_id}: {description} timed out")

            except Exception as e:

                _debug(
                    f"[PROFILE COMMAND ERROR] "
                    f"{dev_id}: {description} -> "
                    f"{type(e).__name__}: {e}"
                )
                raise RuntimeError(f"{dev_id}: {description} failed: {e}") from e

    # ==============================================================
    # TELEMETRY
    # ==============================================================

    def _poll_all_telemetry(self, timestamp):

        telemetry = { "timestamp": timestamp}

        if self._stop_event.is_set():
            return telemetry

        def safe_read(dev_key):
            try:
                dev = self.devices[dev_key]
                if getattr(dev, "is_connected", False):
                    return (dev_key,dev.read_status())

                return (dev_key,{"connected": False})

            except Exception as ex:
                return (dev_key,{"error": str(ex)})

        futures = []

        for key in self.devices:
            if self._stop_event.is_set():
                break
            futures.append(self.executor.submit(safe_read, key) )

        for future in futures:

            try:

                key, status = future.result(timeout=5 )
                telemetry[key] = status
                if key == "opc_temperature":
                    _debug(f"[OPC] Temperature = {status.get('value')}")

            except Exception as ex:

                _debug( f"[TELEMETRY ERROR] {ex}")

        return telemetry

    # ==============================================================
    # HARDWARE TEARDOWN
    # ==============================================================

    def _return_hardware_to_idle(self):

        with self._teardown_lock:

            if self._teardown_done:
                return

            self._teardown_done = True

        _debug(
            "\n[TEARDOWN] Returning hardware to safe idle state..."
        )

        self._stop_event.set()

        # ----------------------------------------------------------
        # STEP 1
        # Finish/cancel worker executor activity.
        # ----------------------------------------------------------

        if self.executor is not None:

            try:

                self.executor.shutdown(
                    wait=True,
                    cancel_futures=True
                )

            except Exception as e:

                _debug(
                    f"[TEARDOWN] Executor shutdown error: {e}"
                )

            finally:

                self.executor = None

        _debug(
            "[TEARDOWN] Executor stopped."
        )
        self._idle_actuators()
        for dev in self.devices.values():
            if getattr(dev, "simulated", False):
                dev.time_scale = 1.0

    def _idle_actuators(self):

        # ----------------------------------------------------------
        # STEP 2
        # Put actuating devices into safe idle state.
        #
        # IMPORTANT:
        # Devices remain CONNECTED.
        # ----------------------------------------------------------

        for dev_id, dev in self.devices.items():

            if not getattr(dev, "is_connected", False):
                continue

            config = self.device_configs.get(
                dev_id,
                {}
            )

            device_type = config.get(
                "type"
            )

            if is_monitor_only(config):
                continue
            try:

                # ----------------------------------------------
                # MASTERFLEX
                # ----------------------------------------------

                if device_type == "masterflex":

                    _debug(
                        f"[TEARDOWN] Stopping {dev_id}..."
                    )

                    if dev.stop() is False:
                        raise RuntimeError("Stop command rejected")

                # ----------------------------------------------
                # ISMATEC
                # ----------------------------------------------

                elif device_type == "ismatec":

                    stage = config.get(
                        "defaults",
                        {}
                    ).get(
                        "stage"
                    )


                    _debug(
                        f"[TEARDOWN] Stopping "
                        f"{dev_id}, stage {stage}..."
                    )

                    if dev.stop(stage=stage) is False:
                        raise RuntimeError("Stop command rejected")

                # ----------------------------------------------
                # JKEM STIRRER
                # ----------------------------------------------

                elif device_type == "jkem":

                    _debug(
                        f"[TEARDOWN] Stopping {dev_id}..."
                    )

                    if dev.stop() is False:
                        raise RuntimeError("Stop command rejected")

                # ----------------------------------------------
                # VALVE
                # 0 = closed
                # ----------------------------------------------

                elif device_type == "nidaq_valve":

                    _debug(
                        f"[TEARDOWN] Closing {dev_id}..."
                    )

                    if not dev.set_target(0):
                        raise RuntimeError("Valve close command rejected")

                # ----------------------------------------------
                # HUBER
                #
                # Leave connected and keep regulating at
                # its final setpoint.
                # ----------------------------------------------

                elif device_type == "huber":

                    _debug(
                        f"[TEARDOWN] {dev_id} remains "
                        f"connected and regulating."
                    )

                # ----------------------------------------------
                # READ-ONLY DEVICES
                # ----------------------------------------------

                elif device_type in {
                    "mettler",
                    "opcua"
                }:

                    _debug(
                        f"[TEARDOWN] {dev_id} remains connected."
                    )

            except Exception as e:

                _debug(
                    f"[TEARDOWN] Safe-idle error "
                    f"{dev_id}: {e}"
                )
                self.outcome = RunStatus.FAILED
                self.error_message += f"; Safe-idle failed for {dev_id}: {e}"
                logger.error("Safe-idle failed for %s: %s", dev_id, e)

        _debug(
            "[TEARDOWN] Hardware is idle and remains connected."
        )


# =============================================================================
# LIVE TELEMETRY PLOTS WITH ORIGINAL MATPLOTLIB SETTINGS
# =============================================================================
from gui.live_plot_widget import LivePlotWidget as RealTimePlotCanvas


class ExperimentDashboard(QMainWindow):
    execution_active_changed = Signal(bool)
    recording_started = Signal(str)
    campaign_finished = Signal(dict)

    def __init__(self, devices: dict,device_configs: dict, experiments: list[pd.DataFrame], campaign=None):
        super().__init__()
        self.setWindowTitle(
            "Automated Lab Reactor - Parallel Orchestrator"
        )

        self.resize(1100, 850)

        self.devices = devices
        self.device_configs = device_configs
        self.experiments = experiments
        self.campaign = campaign

        # ----------------------------------------------------------
        # UI
        # ----------------------------------------------------------

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        if campaign is not None:
            layout.addWidget(QLabel(f"{campaign.name}: {len(campaign.experiments)} experiments"))
        self.run_status_label = QLabel("Ready")
        simulated = sum(bool(getattr(dev, "simulated", False)) for dev in devices.values())
        self.mode_label = QLabel(f"Devices: {simulated} simulated / {len(devices) - simulated} real")
        layout.addWidget(self.mode_label)
        layout.addWidget(self.run_status_label)
        self.recording_status_label = QLabel("")
        layout.addWidget(self.recording_status_label)
        self._recording_timer = QTimer(self)
        self._recording_timer.setInterval(500)
        self._recording_timer.timeout.connect(self.update_recording_status)
        self._recording_timer.start()
        self._last_recorder = None
        self.plot_canvas = RealTimePlotCanvas(self,device_configs=self.device_configs)
        self.plot_canvas.setMinimumHeight(1000)
        self.plot_canvas.setObjectName("plotCanvas")
        layout.addWidget(self.plot_canvas)

        self.btn_start = QPushButton("Start Profile Experiment")
        self.btn_start.setObjectName("start_button")
        self.btn_start.clicked.connect(self.start_orchestrator)
        layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Emergency Stop / Abort Hardware")
        self.btn_stop.setObjectName("stop_button")
        self.btn_stop.clicked.connect(self.stop_orchestrator)
        self.btn_stop.setEnabled(False)
        layout.addWidget(self.btn_stop)
        self.btn_skip = QPushButton("Skip to next experiment")
        self.btn_skip.setEnabled(False)
        self.btn_skip.clicked.connect(self.skip_experiment)
        layout.addWidget(self.btn_skip)
        self._displayed_experiment = None

        self.setCentralWidget(central_widget)

        self.set_theme(dark=True)

        # ----------------------------------------------------------
        # Thread state
        # ----------------------------------------------------------

        self.thread = None
        self.worker = None

        self._closing_application = False

    def set_theme(self, dark=True):

        self.dark_mode = bool(dark)

        # Existing Qt light/dark stylesheet
        self.create_style(dark=self.dark_mode)

        # Explicitly style light-mode text fields and controls
        if not self.dark_mode:

            self.setStyleSheet(
                self.styleSheet() + """
                    QLineEdit,
                    QSpinBox,
                    QDoubleSpinBox,
                    QComboBox,
                    QTextEdit,
                    QPlainTextEdit,
                    QTableWidget {
                        background-color: #ffffff;
                        color: #000000;
                    }

                    QPushButton#start_button {
                        background-color: #16a34a;
                        color: #ffffff;
                    }

                    QPushButton#stop_button {
                        background-color: #dc2626;
                        color: #ffffff;
                    }

                    QPushButton#start_button:disabled,
                    QPushButton#stop_button:disabled {
                        background-color: #cbd5e1;
                        color: #64748b;
                    }
                """
            )

        self.plot_canvas.set_theme(self.dark_mode)

    def _release_hardware_ownership(self, token=None):
        """
        Release this dashboard's experiment reservation.

        If a token is supplied, release only that
        specific reservation.
        """

        manager = getattr(self, "hardware_manager", None)
        current_token = getattr(self, "hardware_token", None)

        if manager is None or current_token is None:
            return

        if token is not None and token != current_token:
            return

        released = manager.release_experiment(current_token)

        if released or not manager.owns_experiment(current_token):
            self.hardware_token = None


    # ====================================================== #
    # STYLE                                                  #
    # ====================================================== #

    def create_style(self, dark=True):
        if dark:
            self.setStyleSheet("""
            QGroupBox QLabel {
                background: transparent;
                border: none;
            }

            QLabel#dashboard_current {
                background: transparent;
            }

            QWidget#metricCard {
                background: #111d31;
                border: 1px solid #1e293b;
                border-radius: 12px;
            }

            QWidget {
                background: #07111f;
                color: #e5e7eb;
                font-family: "Segoe UI";
                font-size: 13px;
            }

            QGroupBox {
                font-size: 13px;
                font-weight: 800;
                background: #111d31;
                border: 1px solid #16263d;
                border-radius: 12px;
                margin-top: 16px;
                padding: 16px;
            }

            QGroupBox#exp_status_box {
                background: #111d31;
                border: 1px solid #16263d;
                border-radius: 12px;
                margin-top: 16px;
                padding: 16px;
            }

            QGroupBox::title {
                subcontrol-rigin: margin;
                subcontrol-position: top left;
                left: 26px;
                top: -2px;
                padding: 0px 6px;
                color: white;
                font-size: 16px;
                font-weight: 700;
            }

            QPushButton {
                background: transparent;
                color: #d6dde8;
                border: 1px solid #16263d;
                border-radius: 8px;
                padding: 11px 15px;
                text-align: left;
                font-weight: 600;
            }

                QPushButton:hover {
                    background: #162846;
                    border: 1px solid #2f5fa7;
                    color: white;
                }

                QPushButton:pressed {
                    background: #10203a;
                }

                QPushButton:checked {
                    background: #2563eb;
                    border: 1px solid #3b82f6;
                    color: white;
                }

                QPushButton#plot_button {
                    background: transparent;
                    color: #d6dde8;
                    border: 1px solid #16263d;
                    border-radius: 8px;
                    padding: 11px 15px;
                    text-align: center;
                    font-weight: 600;
                }

                QPushButton#plot_button:hover {
                    background: #162846;
                    border: 1px solid #2f5fa7;
                    color: white;
                }

                QPushButton#plot_button:pressed {
                    background: #10203a;
                }

                QPushButton#plot_button:checked {
                    background: #2563eb;
                    border: 1px solid #3b82f6;
                    color: white;
                }
                QWidget#plotCanvas {
                    background: #111827;
                }
                QPushButton#reset_button {
                    background: transparent;
                    color: #d6dde8;
                    border: 1px solid #16263d;
                    border-radius: 8px;
                    padding: 11px 15px;
                    text-align: center;
                    font-weight: 600;
                }

                QPushButton#reset_button:hover {
                    background: #162846;
                    border: 1px solid #2f5fa7;
                    color: white;
                }

                QPushButton#reset_button:pressed {
                    background: #10203a;
                }

                QPushButton#reset_button:checked {
                    background: #2563eb;
                    border: 1px solid #3b82f6;
                    color: white;
                }

                QPushButton#generate_button {
                    background: #3b82f6;
                    border: 1px solid #60a5fa;
                    color: white;
                    text-align: center;
                    font-weight: 700;
                    padding: 11px;
                }

                QPushButton#generate_button:hover {
                    background: #2563eb;
                }

                QPushButton#generate_button:pressed {
                    background: #10203a;
                }

                QPushButton#generate_button:checked {
                    background: #2563eb;
                    border: 1px solid #3b82f6;
                    color: white;
                }

                QPushButton#start_button {
                    background: #22c55e;
                    border: 1px solid #37d774;
                    color: white;
                    text-align: center;
                    font-size: 12px;
                    font-weight: 700;
                    padding: 12px;
                }

                QPushButton#start_button:hover {
                    background: #16a34a;
                }

                QPushButton#stop_button {
                    background: #ef4444;
                    border: 1px solid #f87171;
                    color: white;
                    text-align: center;
                    font-size: 12px;
                    font-weight: 700;
                    padding: 12px;
                }

                QPushButton#stop_button:hover {
                    background: #dc2626;
                }

                QPushButton#define_button {
                    background: #3b82f6;
                    border: 1px solid #60a5fa;
                    color: white;
                    text-align: center;
                    font-weight: 700;
                    padding: 11px;
                }

                QPushButton#define_button:hover {
                    background: #2563eb;
                }

                QPushButton#define_button:pressed {
                    background: #10203a;
                }

                QLineEdit {
                    background: #0b1728;
                    border: 1px solid #24364d;
                    border-radius: 8px;
                    padding: 8px 10px;
                    color: white;
                }

                QLineEdit:focus {
                    border: 1px solid #3b82f6;
                }

                QTextEdit {
                    background: #081321;
                    border: 1px solid #16263d;
                    border-radius: 10px;
                    padding: 8px;
                    color: #dbe5ef;
                    selection-background-color: #2563eb;
                }

                QCheckBox {
                    spacing: 8px;
                    color: white;
                    font-weight: 500;
                }

                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border-radius: 4px;
                    border: 2px solid #5b6c84;
                    background: #111827;
                }

                QCheckBox::indicator:hover {
                    border: 2px solid #60a5fa;
                }

                QCheckBox::indicator:checked {
                    background: #2563eb;
                    border: 2px solid #3b82f6;
                }

                QScrollBar:vertical {
                    background: transparent;
                    width: 10px;
                    margin: 0px;
                }

                QScrollBar::handle:vertical {
                    background: #2d415d;
                    border-radius: 5px;
                    min-height: 30px;
                }

                QScrollBar::handle:vertical:hover {
                    background: #456182;
                }

                QScrollBar::add-line:vertical,
                QScrollBar::sub-line:vertical {
                    height: 0px;
                }

                QScrollBar::add-page:vertical,
                QScrollBar::sub-page:vertical {
                    background: transparent;
                }

                QTableWidget {
                    background: #081321;
                    border: 1px solid #16263d;
                    gridline-color: #16263d;
                    border-radius: 10px;
                    alternate-background-color: #0d1b2d;
                }

                QHeaderView::section {
                    background: #111d31;
                    color: white;
                    padding: 6px;
                    border: none;
                    border-bottom: 1px solid #16263d;
                    font-weight: 700;
                }

                QComboBox {
                    background: #0b1728;
                    border: 1px solid #24364d;
                    border-radius: 8px;
                    padding: 6px 10px;
                }

                QComboBox:hover {
                    border: 1px solid #3b82f6;
                }

                QComboBox::drop-down {
                    border: none;
                    width: 24px;
                }

                QToolTip {
                    background: #162846;
                    color: white;
                    border: 1px solid #3b82f6;
                    padding: 6px;
                    border-radius: 6px;
                }



                QLabel#statusCard {
                    background: #111d31;
                    border: 1px solid #16263d;
                    border-radius: 12px;
                    color: #e5e7eb;
                    padding: 16px;
                    font-size: 14px;
                    font-weight: 700;
                }

                QWidget#centralContainer {
                    background: #07111f;
                    border: 1px solid #ffffff;
                    border-radius: 12px;
                }

                QWidget#bodyContainer {
                    background: transparent;
                }

                QStackedWidget {
                    background: transparent;
                    border-bottom-right-radius: 12px;
                }

                QWidget#sidebar {
                    background: #081426;
                    border-right: 1px solid #17283f;
                    border-bottom-left-radius: 12px;
                }

                QWidget#topHeader {
                    background: #081426;
                    border-top-left-radius: 12px;
                    border-top-right-radius: 12px;
                    border-bottom: 1px solid #17283f;
                }

                QLabel#statusPill {
                    background: #064e3b;
                    color: #34d399;
                    border-radius: 10px;
                    padding: 4px 12px;
                    font-weight: 600;
                }

                QPushButton#windowControlBtn {
                    background: transparent;
                    color: #94a3b8;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                    text-align: center;
                    padding: 0px;
                }

                QPushButton#windowControlBtn:hover {
                    background: #1e293b;
                    color: white;
                }

                QPushButton#windowCloseBtn {
                    background: transparent;
                    color: #94a3b8;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                    text-align: center;
                    padding: 0px;
                }

                QPushButton#windowCloseBtn:hover {
                    background: #ef4444;
                    color: white;
                }
                QScrollBar:horizontal {
                background: #111827;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
                }

                QScrollBar::handle:horizontal {
                    background: #374151;
                    min-width: 30px;
                    border-radius: 6px;
                }

                QScrollBar::handle:horizontal:hover {
                    background: #4B5563;
                }

                QScrollBar::add-line:horizontal,
                QScrollBar::sub-line:horizontal {
                    width: 0px;
                }

                QScrollBar::add-page:horizontal,
                QScrollBar::sub-page:horizontal {
                    background: transparent;
                }
                QPushButton#theme_toggle {
                    background-color: #2b2f36;
                    color: #f7f7f5;
                    border: 1px solid #4a505a;
                    border-radius: 8px;
                    font-size: 20px;
                    font-weight: bold;
                    padding: 0px;
                }

                QPushButton#theme_toggle:hover {
                    background-color: #363b44;
                    border: 1px solid #6b7280;
                }

                QPushButton#theme_toggle:pressed {
                    background-color: #22262c;
                }
                """)
            #self.logo_label.setStyleSheet("font-weight: bold; font-size: 15px; color: white; background: transparent;")

        else:
            self.setStyleSheet("""
                /* ==================================================
                   LIGHT THEME
                   ================================================== */

                QPushButton#plot_button {
                    text-align: center;
                }

                QPushButton#reset_button {
                    text-align: center;
                }

                QPushButton#stop_button {
                    text-align: center;
                }

                QPushButton#start_button {
                    text-align: center;
                }

                QGroupBox QLabel {
                    background: transparent;
                    border: none;
                }
                QPushButton#theme_toggle {
                    background-color: #f8fafc;
                    color: #1e293b;
                    border: 1px solid #6b7280;
                    border-radius: 8px;
                    font-size: 20px;
                    font-weight: bold;
                    padding: 0px;
                    text-align: right;
                }

                QPushButton#theme_toggle:hover {
                    background-color:#e8f0fe;
                    border: 1px solid #6b7280;
                }

                QPushButton#theme_toggle:pressed {
                    background-color: #dbeafe;
                }
                QWidget {
                    background: #f8fafc;
                    color: #1e293b;
                    font-family: "Segoe UI";
                    font-size: 13px;
                }

                QLabel#statusCard {
                    background: #111d31;
                    border: 1px solid #16263d;
                    border-radius: 12px;
                    color: #000000;
                    padding: 16px;
                    font-size: 14px;
                    font-weight: 700;
                }

                QWidget#centralContainer {
                    background: #f8fafc;
                    border: 1px solid #000000;
                    border-radius: 12px;
                }
                QWidget#plotCanvas {
                    background: #ffffff;
                }
                QWidget#bodyContainer {
                    background: transparent;

                }

                QWidget#sidebar {
                    background: #eef2f7;
                    border-right: 1px solid #d5dce5;
                    border-bottom-left-radius: 12px;
                }

                QWidget#topHeader {
                    background: #ffffff;
                    border-top-left-radius: 12px;
                    border-top-right-radius: 12px;
                    border-bottom: 1px solid #d5dce5;
                }

                QGroupBox {
                    font-size: 13px;
                    font-weight: 800;
                    background: #ffffff;
                    border: 1px solid #d8dee8;
                    border-radius: 12px;
                    margin-top: 16px;
                    padding: 16px;
                }

                QGroupBox::title {
                    subcontrol-rigin: margin;
                    subcontrol-position: top left;
                    left: 26px;
                    top: -2px;
                    padding: 0px 6px;
                    color: #1e293b;
                    font-size: 16px;
                    font-weight: 700;
                }

                QPushButton {
                    background: transparent;
                    color: #334155;
                    border: 1px solid #d8dee8;
                    border-radius: 8px;
                    padding: 11px 15px;
                    text-align: left;
                    font-weight: 600;
                }

                QPushButton:hover {
                    background: #e8f0fe;
                    border: 1px solid #93b4e8;
                    color: #1e40af;
                }

                QPushButton:pressed {
                    background: #dbeafe;
                }

                QPushButton:checked {
                    background: #2563eb;
                    border: 1px solid #3b82f6;
                    color: white;
                }

                QLabel#statusPill {
                    background: #dcfce7;
                    color: #15803d;
                    border-radius: 10px;
                    padding: 4px 12px;
                    font-weight: 600;
                }

                QPushButton#themeToggle {
                    background: #f1f5f9;
                    color: #475569;
                    border: 1px solid #cbd5e1;
                    border-radius: 7px;
                    padding: 0px;
                    font-size: 16px;
                    font-weight: 600;
                }

                QPushButton#themeToggle:hover {
                    background: #e2e8f0;
                    border: 1px solid #94a3b8;
                }

                QPushButton#windowControlBtn {
                    background: transparent;
                    color: #64748b;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                    text-align: center;
                    padding: 0px;
                }

                QPushButton#windowControlBtn:hover {
                    background: #e2e8f0;
                    color: #0f172a;
                }

                QPushButton#windowCloseBtn {
                    background: transparent;
                    color: #64748b;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                    text-align: center;
                    padding: 0px;
                }

                QPushButton#windowCloseBtn:hover {
                    background: #ef4444;
                    color: white;
                }

                QLineEdit {
                    background: #ffffff;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    padding: 8px 10px;
                    color: #1e293b;
                }

                QLineEdit:focus {
                    border: 1px solid #3b82f6;
                }

                QTextEdit {
                    background: #ffffff;
                    border: 1px solid #d8dee8;
                    border-radius: 10px;
                    padding: 8px;
                    color: #334155;
                    selection-background-color: #2563eb;
                }

                QComboBox {
                    background: #ffffff;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    padding: 6px 10px;
                    color: #334155;
                }

                QComboBox:hover {
                    border: 1px solid #3b82f6;
                }

                QScrollBar:vertical {
                    background: transparent;
                    width: 10px;
                }

                QScrollBar::handle:vertical {
                    background: #cbd5e1;
                    border-radius: 5px;
                    min-height: 30px;
                }

                QScrollBar::handle:vertical:hover {
                    background: #94a3b8;
                }

                QScrollBar::add-line:vertical,
                QScrollBar::sub-line:vertical {
                    height: 0px;
                }

                QScrollBar::add-page:vertical,
                QScrollBar::sub-page:vertical {
                    background: transparent;
                }
            """)
            #self.logo_label.setStyleSheet("font-weight: bold; font-size: 15px; color: black; background: transparent;")


    # ==============================================================
    # START
    # ==============================================================

    def start_orchestrator(self):

        _debug("[DASHBOARD] start_orchestrator() ENTERED")

        # --------------------------------------------------
        # 1. Do not start after dashboard closure
        # --------------------------------------------------

        if self._closing_application:
            self._release_hardware_ownership()
            return

        # --------------------------------------------------
        # 2. Prevent duplicate execution
        # --------------------------------------------------

        if self.thread is not None:
            try:
                if self.thread.isRunning():
                    _debug("[GUI] Experiment already running.")
                    return
            except RuntimeError:
                self.thread = None
                self.worker = None

        # --------------------------------------------------
        # 3. Obtain or validate hardware ownership
        # --------------------------------------------------

        manager = getattr(self, "hardware_manager", None)

        if manager is not None:

            token = getattr(self, "hardware_token", None)

            if token is None or not manager.owns_experiment(token):

                try:
                    token = manager.reserve_experiment()

                except HardwareBusyError as error:
                    QMessageBox.warning(
                        self,
                        "Hardware Busy",
                        str(error),
                    )
                    return

                self.hardware_token = token

        # --------------------------------------------------
        # 4. Create and start the worker
        # --------------------------------------------------

        thread_started = False
        self.execution_active_changed.emit(True)

        try:
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)

            self.thread = QThread(self)

            self.worker = ExperimentWorker(
                devices=self.devices,
                device_configs=self.device_configs,
                experiments=self.experiments,
                poll_interval=2.0,
                campaign=self.campaign,
                recording=getattr(self, "recording_settings", None),
                speed=(float(getattr(self, "simulation_speed", 1.0))
                       if self.devices and all(getattr(d, "simulated", False) for d in self.devices.values()) else 1.0),
            )

            self.worker.moveToThread(self.thread)

            self.thread.started.connect(
                self.worker.start_experiment
            )

            self.worker.status_updated.connect(
                self.plot_canvas.append_and_update
            )

            self.worker.experiment_changed.connect(
                self.plot_canvas.set_experiment_title
            )
            self.worker.experiment_changed.connect(self._show_skip_control)

            self.worker.finished.connect(
                self.on_experiment_finished
            )
            self.worker.run_finished.connect(self.on_run_result)
            self.worker.recording_started.connect(self.recording_started.emit)
            self.worker.finished.connect(self.worker.deleteLater)

            self.worker.finished.connect(
                self.thread.quit
            )

            # Release only once the QThread has terminated.
            self.thread.finished.connect(self.on_thread_finished)

            self.thread.start()
            thread_started = True
            self.run_status_label.setText("Running")

            _debug("[DASHBOARD] Experiment thread started.")

        except Exception as error:

            _debug(
                "[DASHBOARD] Startup error: "
                f"{type(error).__name__}: {error}"
            )

            logger.exception("Experiment startup failed")

            # Do not release ownership if the execution
            # thread might still be running.
            running = thread_started

            if self.thread is not None:
                try:
                    running = self.thread.isRunning()
                except RuntimeError:
                    pass

            if not running:
                self._release_hardware_ownership()
                self.thread = None
                self.worker = None
                self.btn_start.setEnabled(True)
                self.btn_stop.setEnabled(False)
                self.execution_active_changed.emit(False)
            else:
                # Retain ownership until worker/thread cleanup.
                if self.worker is not None:
                    self.worker.request_stop()

            QMessageBox.critical(
                self,
                "Experiment Startup Error",
                str(error),
            )

    # ==============================================================
    # EMERGENCY STOP
    # ==============================================================

    def stop_orchestrator(self):

        if self.worker is None:
            return

        _debug("[GUI] Emergency stop requested.")

        self.worker.request_stop()

        self.btn_stop.setEnabled(False)
        self.btn_skip.setEnabled(False)

    @Slot(int, int)
    def _show_skip_control(self, number, total):
        self._displayed_experiment = number
        self.btn_skip.setEnabled(number < total and self.btn_stop.isEnabled())
        self.run_status_label.setText(f"Running experiment {number} of {total}")

    def skip_experiment(self):
        if self.worker is not None and self.worker.request_skip(self._displayed_experiment):
            self.btn_skip.setEnabled(False)
            self.run_status_label.setText("Skipping experiment and idling actuators…")

    # ==============================================================
    # WORKER FINISHED
    # ==============================================================

    @Slot()
    def on_experiment_finished(self):

        _debug(
            "[GUI] Experiment finished. "
            "Hardware returned to safe idle state."
        )

        # QThread still owns the hardware until its finished signal arrives.
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_skip.setEnabled(False)

    @Slot(dict)
    def on_run_result(self, result):
        if self.worker is not None:
            self._last_recorder = self.worker.recorder
        if self.worker is not None and self.worker.sql_sync is not None:
            if not hasattr(self, "_sql_synchronizers"):
                self._sql_synchronizers = []
            self._sql_synchronizers.append(self.worker.sql_sync)
        self.plot_canvas.flush_pending()
        self.last_result = result
        self.campaign_finished.emit(result)
        self.run_status_label.setText(result["status"].capitalize() + (f": {result['error']}" if result["error"] else ""))

    @Slot()
    def update_recording_status(self):
        recorder = getattr(self.worker, "recorder", None) if self.worker is not None else self._last_recorder
        if recorder is not None and recorder.waiting_for_csv:
            self.recording_status_label.setText("CSV saving is waiting. Close execution.csv in Excel; pending data is retained and will be saved automatically.")
        elif recorder is not None and recorder.pending_csv:
            self.recording_status_label.setText("Flushing pending CSV data…")
        elif recorder is not None:
            self.recording_status_label.setText(f"CSV recording: {recorder.folder}")

    @Slot()
    def on_thread_finished(self):
        self._release_hardware_ownership()
        thread = self.thread
        self.thread = self.worker = None
        if thread is not None:
            thread.deleteLater()
        self.btn_start.setEnabled(not self._closing_application)
        self.execution_active_changed.emit(False)

    def is_execution_active(self):
        # Include reserved/pending startup and the entire cleanup interval.
        # QThread.finished must be handled before the session becomes closable.
        if self.thread is not None:
            return True
        manager = getattr(self, "hardware_manager", None)
        token = getattr(self, "hardware_token", None)
        return manager is not None and token is not None and manager.owns_experiment(token)

    # ==============================================================
    # CLOSE APPLICATION
    # ==============================================================

    def closeEvent(self, event):

        if self.is_execution_active():
            event.ignore()
            return

        _debug("\n==================================================")
        _debug("[GUI] APPLICATION CLOSE REQUESTED")
        _debug(
            f"[GUI] spontaneous={event.spontaneous()} | "
            f"visible={self.isVisible()} | "
            f"thread={self.thread} | "
            f"worker={self.worker}"
        )
        _debug("==================================================")

        self._closing_application = True

        # --------------------------------------------------------------
        # 1. Request the experiment to stop
        # --------------------------------------------------------------

        if self.worker is not None:

            try:
                _debug("[GUI] Requesting worker shutdown...")
                self.worker.request_stop()

            except RuntimeError:
                _debug(
                    "[GUI] Worker object already deleted."
                )

        # --------------------------------------------------------------
        # 2. Request thread shutdown
        # --------------------------------------------------------------

        thread = self.thread

        if thread is not None:

            try:
                thread.quit()

                # ------------------------------------------------------
                # 3. Wait for experiment and hardware teardown
                # ------------------------------------------------------

                if thread.isRunning():

                    _debug(
                        "[GUI] Waiting for worker thread "
                        "and hardware teardown..."
                    )

                    finished = thread.wait(10000)

                    if not finished:

                        _debug(
                            "[GUI] WARNING: Worker did not stop. "
                            "Dashboard will remain open."
                        )

                        self._closing_application = False
                        event.ignore()
                        return

                    _debug(
                        "[GUI] Worker thread completely stopped."
                    )

            except RuntimeError:

                _debug(
                    "[GUI] Cannot verify QThread termination. "
                    "Dashboard closure refused."
                )

                self._closing_application = False
                event.ignore()
                return

        # --------------------------------------------------------------
        # 4. Release ownership after confirmed thread termination
        # --------------------------------------------------------------

        if not self.plot_canvas.shutdown_renderer():
            self._closing_application = False
            event.ignore()
            return


        self._release_hardware_ownership()

        _debug(
            "[GUI] Experiment ownership released. "
            "Hardware remains connected for Control Panel."
        )

        # --------------------------------------------------------------
        # 5. Clear stale references
        # --------------------------------------------------------------

        self.thread = None
        self.worker = None

        _debug(
            "[GUI] Dashboard closed successfully."
        )
        self._recording_timer.stop()

        event.accept()

