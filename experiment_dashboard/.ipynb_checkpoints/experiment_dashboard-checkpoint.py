import time
import threading
import pandas as pd
import numpy as np
import traceback

from hardware_devices.hardware_manager import HardwareBusyError

from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QObject, QThread, Signal, Slot

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QMessageBox,
)

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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

    def __init__(self, devices: dict, device_configs: dict, experiments: list[pd.DataFrame], poll_interval: float = 1.0):
        super().__init__()
    
        self.devices = devices
        self.device_configs = device_configs
        
        self.experiments = [
            df.sort_values("time_s").reset_index(drop=True)
            for df in experiments
        ]
    
        self.poll_interval = poll_interval
    
        self._stop_event = threading.Event()
    
        self._teardown_lock = threading.Lock()
        self._teardown_done = False
    
        self.executor = None

    # ==============================================================
    # START
    # ==============================================================

    @Slot()
    def start_experiment(self):
    
        self._stop_event.clear()
    
        with self._teardown_lock:
            self._teardown_done = False
    
        self.executor = ThreadPoolExecutor(
            max_workers=len(self.devices) + 2
        )
    
        try:
    
            print(
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
    
            print(
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
    
                if config.get("type") != "huber":
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
                
                        dev.set_jacket_control()
                
                        print(
                            f"[HUBER] {dev_id}: "
                            f"Jacket control selected."
                        )
                
                    elif control_mode == "Process":
                
                        dev.set_process_control()
                
                        print(
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

                if device_type in {
                    "jkem",
                    "ismatec"
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
    
                    print(
                        f"[START SUBMIT ERROR] "
                        f"{dev_id}: "
                        f"{type(e).__name__}: {e}"
                    )
    
            start_failed = False
    
            for dev_id, future in start_futures:
    
                try:
    
                    success = future.result(
                        timeout=5
                    )
    
                    print(
                        f"Device '{dev_id}' "
                        f"start status: {success}"
                    )
    
                    if not success:
                        start_failed = True
    
                except TimeoutError:
    
                    print(
                        f"[START TIMEOUT] {dev_id}"
                    )
    
                    start_failed = True
    
                except Exception as e:
    
                    print(
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
    
                self.experiment_changed.emit(
                    exp_idx + 1,
                    total_experiments
                )
    
                print(
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
    
                start_time = time.monotonic()
    
                total_steps = len(profile)
                current_step_idx = -1
    
                times = profile[
                    "time_s"
                ].to_numpy(
                    dtype=float
                )
    
                while not self._stop_event.is_set():
    
                    elapsed = (
                        time.monotonic()
                        - start_time
                    )
    
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
    
                    self._apply_profile_step(
                        interpolated_row
                    )
    
                    # ------------------------------------------
                    # Telemetry
                    # ------------------------------------------
    
                    if not self._stop_event.is_set():
    
                        telemetry = (
                            self._poll_all_telemetry(
                                elapsed
                            )
                        )
    
                        telemetry[
                            "experiment_index"
                        ] = exp_idx
    
                        self.status_updated.emit(
                            telemetry
                        )
    
                    # ------------------------------------------
                    # Experiment finished?
                    # ------------------------------------------
    
                    final_time = float(
                        times[-1]
                    )
    
                    if elapsed >= final_time + 5:
                        break
    
                    self._stop_event.wait(
                        self.poll_interval
                    )
    
                print(
                    f"FINISHED EXPERIMENT "
                    f"{exp_idx + 1}/"
                    f"{total_experiments}"
                )
    
            print(
                "\nCOMPLETE DOE FINISHED."
            )
    
        except Exception as e:
    
            print(
                f"[WORKER ERROR] "
                f"{type(e).__name__}: {e}"
            )
    
            traceback.print_exc()
    
        finally:
        
            self._return_hardware_to_idle()
        
            self.finished.emit()

    # ==============================================================
    # REQUEST STOP
    # ==============================================================

    def request_stop(self):
        """Thread-safe emergency stop request.

        This method does NOT perform hardware I/O.
        It is safe to call directly from the GUI thread.
        """

        print(  "[STOP] Emergency stop requested.")

        self._stop_event.set()



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
    
            if device_type == "nidaq_valve":
    
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
    
        print(
            f"\n--- [T={step_row['time_s']} s] "
            f"Executing Profile Step ---"
        )
        print(
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
                return
    
            if not getattr(dev, "is_connected", False):
    
                print(
                    f"[PROFILE] {dev_id}: "
                    f"not connected, skipping {description}"
                )
    
                return
    
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
    
                control_mode = options.get(
                    "control_mode",
                    "Jacket"
                )
    
                if control_mode == "Jacket":
    
                    submit_command(
                        dev_id,
                        "set_jacket_control()",
                        dev.set_jacket_control
                    )
    
                elif control_mode == "Process":
    
                    submit_command(
                        dev_id,
                        "set_process_control()",
                        dev.set_process_control
                    )
    
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
                    print(
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
    
                print(
                    f"[PROFILE] {dev_id}: "
                    f"{description} -> {result}"
                )
    
            except TimeoutError:
    
                print(
                    f"[PROFILE COMMAND TIMEOUT] "
                    f"{dev_id}: {description}"
                )
    
            except Exception as e:
    
                print(
                    f"[PROFILE COMMAND ERROR] "
                    f"{dev_id}: {description} -> "
                    f"{type(e).__name__}: {e}"
                )

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
                    print(f"[OPC] Temperature = {status.get('value')}")

            except Exception as ex:

                print( f"[TELEMETRY ERROR] {ex}")

        return telemetry

    # ==============================================================
    # HARDWARE TEARDOWN
    # ==============================================================

    def _return_hardware_to_idle(self):

        with self._teardown_lock:

            if self._teardown_done:
                return

            self._teardown_done = True

        print(
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

                print(
                    f"[TEARDOWN] Executor shutdown error: {e}"
                )

            finally:

                self.executor = None

        print(
            "[TEARDOWN] Executor stopped."
        )

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

            try:

                # ----------------------------------------------
                # MASTERFLEX
                # ----------------------------------------------

                if device_type == "masterflex":

                    print(
                        f"[TEARDOWN] Stopping {dev_id}..."
                    )

                    dev.stop()

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


                    print(
                        f"[TEARDOWN] Stopping "
                        f"{dev_id}, stage {stage}..."
                    )

                    dev.stop(
                        stage=stage
                    )

                # ----------------------------------------------
                # JKEM STIRRER
                # ----------------------------------------------

                elif device_type == "jkem":

                    print(
                        f"[TEARDOWN] Stopping {dev_id}..."
                    )

                    dev.stop()

                # ----------------------------------------------
                # VALVE
                # 0 = closed
                # ----------------------------------------------

                elif device_type == "nidaq_valve":

                    print(
                        f"[TEARDOWN] Closing {dev_id}..."
                    )

                    dev.set_target(0)

                # ----------------------------------------------
                # HUBER
                #
                # Leave connected and keep regulating at
                # its final setpoint.
                # ----------------------------------------------

                elif device_type == "huber":

                    print(
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

                    print(
                        f"[TEARDOWN] {dev_id} remains connected."
                    )

            except Exception as e:

                print(
                    f"[TEARDOWN] Safe-idle error "
                    f"{dev_id}: {e}"
                )

        print(
            "[TEARDOWN] Hardware is idle and remains connected."
        )


# =============================================================================
# REAL-TIME MATPLOTLIB CANVAS FOR PYQT
# =============================================================================
class RealTimePlotCanvas(FigureCanvas):

    TELEMETRY_FIELDS = {
        "masterflex": [("speed_rpm", "RPM")],
        "jkem": [("speed_rpm", "RPM")],
        "ismatec": [("flow", "Flow")],
        "huber": [("setpoint", "Setpoint"),("temp_internal", "Jacket"),("temp_external", "Process")],
        "mettler": [("weight_g", "Mass")],
        "nidaq_valve": [("measured_voltage", "Voltage")],
        "opcua": [ ("value", "Value")]
    }

    SECONDARY_FIELDS = {
        "jkem": [
            ("torque_ozin", "Torque")
        ],
        "nidaq_valve": [
            ("target_state", "Command")
        ]
    }

    def __init__(self, parent=None, device_configs=None, width=12, height=12, dpi=100):

        self.device_configs = device_configs or {}
        self.dark_mode = True
        self.fig = Figure(
            figsize=(width, height),
            dpi=dpi,
            tight_layout=False
        )
        
        self.fig.subplots_adjust(
            left=0.07,
            right=0.96,
            bottom=0.055,
            top=0.94,
            wspace=0.35,
            hspace=0.42
        )

        self.axes = [
            self.fig.add_subplot(421),
            self.fig.add_subplot(422),
            self.fig.add_subplot(423),
            self.fig.add_subplot(424),
            self.fig.add_subplot(425),
            self.fig.add_subplot(426),
            self.fig.add_subplot(427),
            self.fig.add_subplot(428)
        ]

        super().__init__(self.fig)
        self.setParent(parent)

        self.fig.patch.set_facecolor("#111827")

        self.plot_counter = 0
        self.reset_canvas()

    def set_theme(self, dark=True):
    
        self.dark_mode = bool(dark)
    
        if self.dark_mode:
            bg = "#111827"
            fg = "#e5e7eb"
            border = "#475569"
            grid = "#64748b"
        else:
            bg = "#ffffff"
            fg = "#000000"
            border = "#334155"
            grid = "#94a3b8"
    
        self.fig.patch.set_facecolor(bg)
    
        for ax in self.fig.axes:
    
            ax.set_facecolor(bg)
    
            ax.tick_params(
                axis="both",
                colors=fg,
                labelsize=8
            )
    
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
    
            for spine in ax.spines.values():
                spine.set_color(border)
    
            ax.grid(
                True,
                color=grid,
                alpha=0.3,
                linestyle="--"
            )
    
            legend = ax.get_legend()
    
            if legend is not None:
                legend.get_frame().set_facecolor(bg)
                legend.get_frame().set_edgecolor(border)
    
                for label in legend.get_texts():
                    label.set_color(fg)
    
        if self.fig._suptitle is not None:
            self.fig._suptitle.set_color(fg)
    
        self.draw_idle()
        
    def set_experiment_title(self, exp_num: int, total_exp: int):

        self.reset_canvas()

        self.fig.suptitle(
            f"Experiment {exp_num} of {total_exp}",
            fontsize=11,
            fontweight="bold",
            color="#e5e7eb" if self.dark_mode else "#000000"
        )

        self.draw_idle()

    def reset_canvas(self):
    
        # Remove twinx axes created during the previous reset
        for ax in list(self.fig.axes):
            if ax not in self.axes:
                ax.remove()
    
        self.t_data = []
        self.history = {}
        self.lines = {}
        self.secondary_axes = {}
        self.secondary_lines = {}

        for i, ax in enumerate(self.axes, start=1):

            ax.cla()
            ax.set_facecolor("#111827")
            ax.grid(True, alpha=0.25)

            ax.tick_params(
                axis="both",
                colors="#e5e7eb",
                labelsize=8
            )

            ax.xaxis.label.set_color("#e5e7eb")
            ax.yaxis.label.set_color("#e5e7eb")
            ax.title.set_color("#e5e7eb")

            for spine in ax.spines.values():
                spine.set_color("#475569")

            ax.set_title(
                f"Graph {i}",
                fontsize=7,
                pad=32
            )

            ax.set_xlabel(
                "Elapsed Time (s)",
                fontsize=7,
                labelpad=2
            )

        for device_id, config in self.device_configs.items():

            device_type = config.get("type")

            fields = self.TELEMETRY_FIELDS.get(
                device_type,
                []
            )

            if not fields:
                continue

            plot_index = int(
                config.get("plot_index", 1)
            )

            plot_index = min(
                max(plot_index, 1),
                8
            )

            ax = self.axes[
                plot_index - 1
            ]

            device_name = config.get(
                "name",
                device_id
            )

            for field, field_label in fields:

                key = (
                    device_id,
                    field
                )

                self.history[key] = []

                if len(fields) == 1:
                    label = device_name
                else:
                    label = (
                        f"{device_name} - "
                        f"{field_label}"
                    )

                line, = ax.plot(
                    [],
                    [],
                    label=label
                )

                self.lines[key] = line

        for device_id, config in self.device_configs.items():

            device_type = config.get("type")
        
            fields = self.SECONDARY_FIELDS.get(
                device_type,
                []
            )
        
            if not fields:
                continue
        
            plot_index = int(
                config.get("plot_index", 1)
            )
        
            plot_index = min(
                max(plot_index, 1),
                8
            )
        
            ax = self.axes[
                plot_index - 1
            ]
        
            if plot_index not in self.secondary_axes:
        
                ax2 = ax.twinx()
        
                ax2.tick_params(
                    axis="y",
                    colors="#e5e7eb",
                    labelsize=8
                )
        
                ax2.yaxis.label.set_color(
                    "#e5e7eb"
                )
        
                ax2.spines["right"].set_color(
                    "#475569"
                )
        
                self.secondary_axes[
                    plot_index
                ] = ax2
        
            else:
        
                ax2 = self.secondary_axes[
                    plot_index
                ]
        
            device_name = config.get(
                "name",
                device_id
            )
        
            for field, field_label in fields:
        
                key = (
                    device_id,
                    field
                )
        
                self.history[key] = []
        
                line, = ax2.plot(
                    [],
                    [],
                    "-.",
                    label=(
                        f"{device_name} - "
                        f"{field_label}"
                    )
                )
        
                self.secondary_lines[
                    key
                ] = line

        for plot_index, ax2 in self.secondary_axes.items():
        
            secondary_types = {
                config.get("type")
                for config in self.device_configs.values()
                if int(config.get("plot_index", 1)) == plot_index
            }
        
            if "jkem" in secondary_types:
                ax2.set_ylabel(
                    "Torque (oz-in)",
                    fontsize=8
                )
        
            elif "nidaq_valve" in secondary_types:
            
                ax2.set_ylabel(
                    "Valve command",
                    fontsize=8
                )
            
                ax2.set_ylim(
                    -0.05,
                    1.05
                )
            
                ax2.set_yticks(
                    [0, 1]
                )

        for i, ax in enumerate(self.axes, start=1):
        
            handles, labels = ax.get_legend_handles_labels()
        
            ax2 = self.secondary_axes.get(i)
        
            if ax2 is not None:
        
                handles2, labels2 = (
                    ax2.get_legend_handles_labels()
                )
        
                handles += handles2
                labels += labels2
        
            if handles:
        
                legend = ax.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.01),
                    ncol=min(len(handles), 3),
                    fontsize=6,
                    frameon=True,
                    borderpad=0.25,
                    labelspacing=0.25,
                    handlelength=1.5
                )
        
                for text in legend.get_texts():
                    text.set_color("#e5e7eb")
        
                legend.get_frame().set_facecolor(
                    "#111827"
                )
        
                legend.get_frame().set_edgecolor(
                    "#475569"
                )

        self.set_theme(self.dark_mode)

    def _safe_float(self, value):

        if value is None:
            return np.nan

        try:
            value = float(value)

            return (
                value
                if np.isfinite(value)
                else np.nan
            )

        except (ValueError, TypeError):
            return np.nan

    def append_and_update(self, data: dict):

        t = self._safe_float(
            data.get("timestamp")
        )

        if np.isnan(t):
            return

        self.plot_counter += 1

        if t < 600:
            plot_every = 1

        elif t < 1800:
            plot_every = 2

        elif t < 3600:
            plot_every = 5

        elif t < 10800:
            plot_every = 10

        else:
            plot_every = 30

        if self.plot_counter % plot_every != 0:
            return

        self.t_data.append(t)

        for device_id, config in self.device_configs.items():

            device_type = config.get("type")

            fields = self.TELEMETRY_FIELDS.get(
                device_type,
                []
            )

            status = data.get(
                device_id,
                {}
            )

            for field, _ in fields:

                key = (
                    device_id,
                    field
                )

                if key not in self.history:
                    continue

                value = self._safe_float(
                    status.get(field)
                )

                self.history[key].append(
                    value
                )

                self.lines[key].set_data(
                    self.t_data,
                    self.history[key]
                )

        for device_id, config in self.device_configs.items():

            device_type = config.get("type")
        
            fields = self.SECONDARY_FIELDS.get(
                device_type,
                []
            )
        
            status = data.get(
                device_id,
                {}
            )
        
            for field, _ in fields:
        
                key = (
                    device_id,
                    field
                )
        
                if key not in self.history:
                    continue
        
                value = self._safe_float(
                    status.get(field)
                )
        
                self.history[key].append(
                    value
                )
        
                self.secondary_lines[key].set_data(
                    self.t_data,
                    self.history[key]
                )
        
        for ax in self.axes:
        
            ax.relim()
            ax.autoscale_view()
        
        for plot_index, ax2 in self.secondary_axes.items():
        
            secondary_types = {
                config.get("type")
                for config in self.device_configs.values()
                if int(config.get("plot_index", 1)) == plot_index
            }
        
            ax2.relim()
        
            if "nidaq_valve" in secondary_types:
                ax2.set_ylim(-0.05, 1.05)
                ax2.set_yticks([0, 1])
            else:
                ax2.autoscale_view()
        
        self.draw_idle()


class ExperimentDashboard(QMainWindow):

    def __init__(self, devices: dict,device_configs: dict, experiments: list[pd.DataFrame]):
        super().__init__()
        self.setWindowTitle(
            "Automated Lab Reactor - Parallel Orchestrator"
        )

        self.resize(1100, 850)

        self.devices = devices
        self.device_configs = device_configs
        self.experiments = experiments

        # ----------------------------------------------------------
        # UI
        # ----------------------------------------------------------

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
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
    
        print("[DASHBOARD] start_orchestrator() ENTERED")
    
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
                    print("[GUI] Experiment already running.")
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
    
        try:
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
    
            self.thread = QThread()
    
            self.worker = ExperimentWorker(
                devices=self.devices,
                device_configs=self.device_configs,
                experiments=self.experiments,
                poll_interval=2.0,
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
    
            self.worker.finished.connect(
                self.on_experiment_finished
            )
    
            self.worker.finished.connect(
                self.thread.quit
            )
    
            # Release only once the QThread has terminated.
            release_token = getattr(
                self,
                "hardware_token",
                None,
            )
            
            self.thread.finished.connect(
                lambda t=release_token:
                self._release_hardware_ownership(t)
            )
    
            self.thread.start()
            thread_started = True
    
            print("[DASHBOARD] Experiment thread started.")
    
        except Exception as error:
    
            print(
                "[DASHBOARD] Startup error: "
                f"{type(error).__name__}: {error}"
            )
    
            traceback.print_exc()
    
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

        print("[GUI] Emergency stop requested.")

        self.worker.request_stop()

        self.btn_stop.setEnabled(False)

    # ==============================================================
    # WORKER FINISHED
    # ==============================================================

    @Slot()
    def on_experiment_finished(self):

        print(
            "[GUI] Experiment finished. "
            "Hardware returned to safe idle state."
        )

        self.btn_start.setEnabled(not self._closing_application)
        self.btn_stop.setEnabled(False)

    # ==============================================================
    # CLOSE APPLICATION
    # ==============================================================

    def closeEvent(self, event):

        print("\n==================================================")
        print("[GUI] APPLICATION CLOSE REQUESTED")
        print(
            f"[GUI] spontaneous={event.spontaneous()} | "
            f"visible={self.isVisible()} | "
            f"thread={self.thread} | "
            f"worker={self.worker}"
        )
        print("==================================================")

        self._closing_application = True

        # --------------------------------------------------------------
        # 1. Request the experiment to stop
        # --------------------------------------------------------------

        if self.worker is not None:

            try:
                print("[GUI] Requesting worker shutdown...")
                self.worker.request_stop()

            except RuntimeError:
                print(
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

                    print(
                        "[GUI] Waiting for worker thread "
                        "and hardware teardown..."
                    )

                    finished = thread.wait(10000)

                    if not finished:

                        print(
                            "[GUI] WARNING: Worker did not stop. "
                            "Dashboard will remain open."
                        )

                        self._closing_application = False
                        event.ignore()
                        return

                    print(
                        "[GUI] Worker thread completely stopped."
                    )

            except RuntimeError:

                print(
                    "[GUI] Cannot verify QThread termination. "
                    "Dashboard closure refused."
                )

                self._closing_application = False
                event.ignore()
                return

        # --------------------------------------------------------------
        # 4. Release ownership after confirmed thread termination
        # --------------------------------------------------------------

        self._release_hardware_ownership()

        print(
            "[GUI] Experiment ownership released. "
            "Hardware remains connected for Control Panel."
        )

        # --------------------------------------------------------------
        # 5. Clear stale references
        # --------------------------------------------------------------

        self.thread = None
        self.worker = None

        print(
            "[GUI] Dashboard closed successfully."
        )

        event.accept()

