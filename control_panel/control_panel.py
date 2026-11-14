
import pandas as pd
import numpy as np
import traceback
import json
import sys

from copy import deepcopy
from app.paths import DATA_DIR
from hardware_devices.capabilities import is_monitor_only
from hardware_devices.simulated import SimulatedDevice
from hardware_devices.limits import DEFAULT_LIMITS, target_limits
from experiment.models import Campaign, ExperimentDefinition
from asyncua.sync import Client

from PySide6.QtCore import Qt, QTimer, QObject, QThread, Signal, Slot, QEvent
from gui.flow_layout import FlowLayout

from PySide6.QtWidgets import (QApplication,QFileDialog, 
    QWidget, QGroupBox, QFormLayout, QLineEdit, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QHBoxLayout, QVBoxLayout,
    QGridLayout, QScrollArea, QDialog, QDialogButtonBox, QTableWidget,
    QTableWidgetItem,
    QMessageBox, QSizePolicy
)


from hardware_devices.hardware_devices import (
    MasterflexPump,
    HuberController,
    JKemStirrer,
    MettlerScale,
    IsmatecPump,
    NIDAQValveController,
    OPCUAReader
)

from hardware_devices.hardware_manager import (
    HardwareManager,
    HardwareBusyError,
    HardwareState,
)

from experiment_dashboard.experiment_dashboard import ExperimentDashboard

dict_dev = {
        'STR_MAIN': 'COM30',
        'STR_SEED': 'COM31',
        'STR_SOLU': 'COM21',
        'TEM_MAIN': 'COM33',
        'PMP_CLEN': 'COM34',
        'PMP_SOLU': 'COM6',
        'PMP_SOLV': 'COM32',
        'PMP_SEED': 'COM26',
        'PMP_AS':   'COM36',
        'SCL_AS':   'COM35'
    }


# In[47]:


HARDWARE_TYPES = {

    "masterflex": {
        "driver": MasterflexPump,
        "connection": {
            "port": {
                "label": "Port",
                "default": ""
            },
            "node_address": {
                "label": "Node Address",
                "default": "P01"
            }
        },
        "target": {
            "label": "Speed",
            "type": "float",
            "unit": "RPM",
            "min": -300,
            "max": 300,
            "decimals": 0
        },
        "options": {
            "direction": {
                "label": "Direction",
                "type": "choice",
                "choices": ["+", "-"],
                "default": "+"
            }
        }
    },

    "huber": {
        "driver": HuberController,
        "connection": {
            "port": {
                "label": "Port",
                "default": ""
            }
        },
        "target": {
            "label": "Temperature",
            "type": "float",
            "unit": "°C",
            "min": -50,
            "max": 200,
            "decimals": 1
        },
        "options": {
            "control_mode": {
                "label": "Control",
                "type": "choice",
                "choices": ["Jacket", "Process"],
                "default": "Jacket"
            }
        }
    },

    "jkem": {
        "driver": JKemStirrer,
        "connection": {
            "port": {
                "label": "Port",
                "default": ""
            }
        },
        "target": {
            "label": "Speed",
            "type": "int",
            "unit": "RPM",
            "min": 0,
            "max": 800
        },
        "options": {}
    },

    "mettler": {
        "driver": MettlerScale,
        "connection": {
            "port": {
                "label": "Port",
                "default": ""
            }
        },
        "target": None,
        "options": {}
    },

    "ismatec": {
        "driver": IsmatecPump,
        "connection": {
            "port": {
                "label": "Port",
                "default": ""
            }
        },
        "target": {
            "label": "Flow",
            "type": "float",
            "unit": "mL/min",
            "min": -1000,
            "max": 1000,
            "decimals": 2
        },
        "options": {
            "stage": {
                "label": "Stage",
                "type": "choice",
                "choices": ["1", "2", "3"],
                "default": "2"
            }
        }
    },

    "nidaq_valve": {
        "driver": NIDAQValveController,
        "connection": {
            "ai_channel": {
                "label": "AI Channel",
                "default": "Dev1/ai0"
            },
            "do_channel": {
                "label": "DO Channel",
                "default": "Dev1/port0/line0"
            }
        },
        "target": {
            "label": "State",
            "type": "int",
            "min": 0,
            "max": 1,
            "default": 0
        },
        "options": {}
    },

    "opcua": {
        "driver": OPCUAReader,

        "connection": {
            "endpoint": {
                "label": "OPC Server",
                "default": "opc.tcp://localhost:62552/iCOpcUaServer",
                "modifiable": True
            },
            "node_id": {
                "label": "Node ID",
                "default": "ns=2;s=iCache.Temperature1",
                "modifiable": True
            }
        },

        "target": None,

        "options": {}
    }
}


DEFAULT_DEVICES = {

    "PMP_SOLU": {
        "name": "Solution Pump",
        "type": "masterflex",
        "connection": {
            "port": dict_dev["PMP_SOLU"],
            "node_address": "P01"
        },
        "default_target": 0,
        "plot_index": 2
    },

    "PMP_CLEN": {
        "name": "Cleaning Pump",
        "type": "masterflex",
        "connection": {
            "port": dict_dev["PMP_CLEN"],
            "node_address": "P02"
        },
        "default_target": 0,
        "plot_index": 2
    },

    "PMP_SOLV": {
        "name": "Solvent Pump",
        "type": "masterflex",
        "connection": {
            "port": dict_dev["PMP_SOLV"],
            "node_address": "P03"
        },
        "default_target": 0,
        "plot_index": 2
    },

    "PMP_SEED": {
        "name": "Seed Pump",
        "type": "masterflex",
        "connection": {
            "port": dict_dev["PMP_SEED"],
            "node_address": "P04"
        },
        "default_target": 0,
        "plot_index": 2
    },

    "TEM_MAIN": {
        "name": "Main Reactor Huber",
        "type": "huber",
        "connection": {
            "port": dict_dev["TEM_MAIN"]
        },
        "default_target": 25.0,
        "defaults": {
            "control_mode": "Jacket"
        },
        "plot_index": 4
    },

    "STR_MAIN": {
        "name": "Main Reactor Stirrer",
        "type": "jkem",
        "connection": {
            "port": dict_dev["STR_MAIN"]
        },
        "default_target": 300,
        "plot_index": 1
    },

    "STR_SEED": {
        "name": "Seed Reactor Stirrer",
        "type": "jkem",
        "connection": {
            "port": dict_dev["STR_SEED"]
        },
        "default_target": 300,
        "plot_index": 1
    },

    "STR_SOLU": {
        "name": "Solution Vessel Stirrer",
        "type": "jkem",
        "connection": {
            "port": dict_dev["STR_SOLU"]
        },
        "default_target": 300,
        "plot_index": 1
    },

    "SCL_AS": {
        "name": "Antisolvent Balance",
        "type": "mettler",
        "connection": {
            "port": dict_dev["SCL_AS"]
        },
        "plot_index": 6
    },

    "PMP_AS": {
        "name": "Antisolvent Pump",
        "type": "ismatec",
        "connection": {
            "port": dict_dev["PMP_AS"]
        },
        "default_target": 0.0,
        "defaults": {
            "stage": "2"
        },
        "plot_index": 3
    },

    "VALVE": {
        "name": "Outlet Valve",
        "type": "nidaq_valve",
        "connection": {
            "ai_channel": "Dev1/ai0",
            "do_channel": "Dev1/port0/line0"
        },
        "default_target": 0,
        "plot_index": 5
    }
}

DEVICES = deepcopy(DEFAULT_DEVICES)

PROFILE_TYPES = {
    "masterflex",
    "huber",
    "jkem",
    "ismatec",
    "nidaq_valve"
}

def create_device(config, hardware_types):

    if config.get("simulated", False):
        device = SimulatedDevice(config["name"], config["type"], **config.get("simulation", {}))
        if config["type"] in DEFAULT_LIMITS:
            device.target_limits = target_limits(config)
        return device

    schema = hardware_types[
        config["type"]
    ]

    kwargs = dict(
        config["connection"]
    )

    if config["type"] == "ismatec":

        kwargs["stage"] = (
            config.get(
                "defaults",
                {}
            ).get(
                "stage",
                "2"
            )
        )

    device = schema["driver"](
        name=config["name"],
        **kwargs
    )
    if config["type"] in DEFAULT_LIMITS:
        device.target_limits = target_limits(config)
    return device


def create_hardware(devices, hardware_types):

    return {device_id: create_device(config, hardware_types) for device_id, config in devices.items()}


class ConnectionWorker(QObject):

    finished = Signal()
    device_finished = Signal(str, bool, str)

    def __init__(self, devices, mode):
        super().__init__()

        self.devices = devices
        self.mode = mode

    @Slot()
    def run(self):

        try:

            for device_id, device in self.devices.items():

                try:

                    if device is None:

                        self.device_finished.emit(
                            device_id,
                            False,
                            "No device"
                        )

                        continue

                    if self.mode == "connect":

                        if getattr(
                            device,
                            "is_connected",
                            False
                        ):

                            success = True

                        else:

                            success = bool(
                                device.connect()
                            )

                        message = (
                            "Connected"
                            if success
                            else "Connection failed"
                        )

                    else:

                        if not getattr(
                            device,
                            "is_connected",
                            False
                        ):

                            success = True

                        else:

                            device.disconnect()

                            success = not bool(
                                getattr(
                                    device,
                                    "is_connected",
                                    False
                                )
                            )

                        message = (
                            "Disconnected"
                            if success
                            else "Disconnect failed"
                        )

                    self.device_finished.emit(
                        device_id,
                        success,
                        message
                    )

                except Exception as e:

                    self.device_finished.emit(
                        device_id,
                        False,
                        f"{type(e).__name__}: {e}"
                    )

        finally:

            self.finished.emit()


class ProfileDialog(QDialog):

    def __init__(self, device_id, config, schema, existing_profile=None, parent=None):
        super().__init__(parent)

        self.device_id = device_id
        self.config = config
        self.schema = schema
        self.profile = None

        target = schema.get("target")

        self.setWindowTitle(f"Profile - {config['name']}")
        self.resize(500, 450)

        main_layout = QVBoxLayout(self)

        label = target["label"]
        unit = target.get("unit", "")

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels([
            "Time (s)",
            f"{label} ({unit})" if unit else label
        ])

        main_layout.addWidget(self.table)

        controls = QHBoxLayout()

        self.add_button = QPushButton("Add Point")
        self.remove_button = QPushButton("Remove Point")

        controls.addWidget(self.add_button)
        controls.addWidget(self.remove_button)
        controls.addStretch()

        main_layout.addLayout(controls)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Ok
        )

        buttons.button(
            QDialogButtonBox.StandardButton.Ok
        ).setText("Save Profile")

        main_layout.addWidget(buttons)

        self.add_button.clicked.connect(self._add_row)
        self.remove_button.clicked.connect(self._remove_row)

        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

        if existing_profile is not None:
            for _, row in existing_profile.iterrows():
                self._add_row(
                    row["time_s"],
                    row["value"]
                )

        else:
            default = config.get(
                "default_target",
                target.get("default", 0)
            )

            self._add_row(0, default)

    def _add_row(self, time_value="", target_value=""):

        row = self.table.rowCount()
        self.table.insertRow(row)

        self.table.setItem(
            row,
            0,
            QTableWidgetItem(str(time_value))
        )

        self.table.setItem(
            row,
            1,
            QTableWidgetItem(str(target_value))
        )

    def _remove_row(self):

        row = self.table.currentRow()

        if row >= 0:
            self.table.removeRow(row)

    def _validate_and_accept(self):

        rows = []

        try:
            for row in range(self.table.rowCount()):

                time_item = self.table.item(row, 0)
                value_item = self.table.item(row, 1)

                if time_item is None or value_item is None:
                    raise ValueError("Incomplete profile row")

                t = float(time_item.text())
                value = float(value_item.text())

                if t < 0:
                    raise ValueError("Time cannot be negative")

                rows.append((t, value))

            if not rows:
                raise ValueError("Profile must contain at least one point")

            rows.sort(key=lambda x: x[0])

            if rows[0][0] != 0:
                raise ValueError("Profile must start at time 0")

            times = [row[0] for row in rows]

            if len(times) != len(set(times)):
                raise ValueError("Duplicate time points are not allowed")

            target = self.schema["target"]

            minimum = target.get("min", -float("inf"))
            maximum = target.get("max", float("inf"))

            for _, value in rows:
                if not minimum <= value <= maximum:
                    raise ValueError(
                        f"Value {value} outside allowed range "
                        f"{minimum} to {maximum}"
                    )

            self.profile = pd.DataFrame(
                rows,
                columns=["time_s", "value"]
            )

            self.accept()

        except Exception as e:
            QMessageBox.warning(
                self,
                "Invalid Profile",
                str(e)
            )


# In[51]:


class DeviceWidget(QGroupBox):

    def __init__(self, device_id, config, hardware_types, hardware=None, device=None, parent=None):
        super().__init__(parent)

        self.device_id = device_id
        self.config = deepcopy(config)
        self.schema = deepcopy(hardware_types[config["type"]])
        if config["type"] in DEFAULT_LIMITS:
            limits = target_limits(config)
            integer = self.schema["target"]["type"] == "int"
            self.schema["target"].update(min=int(limits["min"]) if integer else limits["min"],
                                         max=int(limits["max"]) if integer else limits["max"])
        self.hardware = hardware
        self.device = device

        self.connection_widgets = {}
        self.option_widgets = {}
        self.target_widget = None
        self.target_widget = None
        self.profile = None

        self.setTitle(config["name"] + (" [SIMULATED]" if config.get("simulated") else ""))
        self._build_ui()
        self._update_connection_state()

    def _modify_name(self):
        
        panel = self.parent()
        
        while panel is not None:
            if isinstance(panel, ControlPanel):
                break
            panel = panel.parent()
        
        if isinstance(panel, ControlPanel) and panel.experiment_running():
            QMessageBox.warning(
                self,
                "Experiment Running",
                "Manual hardware control is disabled "
                "during an active experiment."
            )
            return

        new_name = self.name_edit.text().strip()

        if not new_name:
            self.status_label.setText("Invalid name")
            return

        try:
            if self.is_connected():
                if not self.disconnect_device():
                    self.status_label.setText("Disconnect before modifying name")
                    return

            self.config["name"] = new_name

            self.device = self.schema["driver"](
                name=self.config["name"],
                **self.config["connection"]
            )

            if self.hardware is not None:
                self.hardware[self.device_id] = self.device

            self.setTitle(new_name)
            self.status_label.setText(f"Name changed to {new_name}")
            self._update_connection_state()

        except Exception as e:
            self.status_label.setText(f"Modify error: {e}")

    def _test_connection(self):

        if self.device is None:
            return

        if not hasattr(self.device, "test_connection"):
            self.status_label.setText(
                "Test not supported"
            )
            return

        try:
            value = self.device.test_connection()

            self.status_label.setText(
                f"Test OK | Value: {value}"
            )

        except Exception as e:
            self.status_label.setText(
                f"Test failed: {e}"
            )

        self._update_connection_state()

    def _edit_profile(self):

        if is_monitor_only(self.config):
            self.status_label.setText("Read-only monitor: select Controlled device before defining a profile.")
            return

        if self.schema.get("target") is None:
            return

        dialog = ProfileDialog(
            device_id=self.device_id,
            config=self.config,
            schema=self.schema,
            existing_profile=self.profile,
            parent=self
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.profile = dialog.profile

        duration = self.profile["time_s"].max()

        self.profile = dialog.profile
        
        print(
            f"\nProfile defined for {self.device_id}:"
        )
        
        print(self.profile)
        
        duration = self.profile["time_s"].max()

        self.status_label.setText(
            f"Profile ready | {len(self.profile)} points | {duration:g} s"
        )

        parent = self.parent()

        while parent is not None:

            if isinstance(parent, ControlPanel):
                parent.update_profile_status()
                break

            parent = parent.parent()

    def _build_ui(self):

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 14, 10, 10)
        main_layout.setSpacing(5)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(4)
        self.setObjectName("deviceCard")
        self.setMinimumWidth(0)
        self.setMaximumWidth(460)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        self.id_label = QLabel(self.device_id)
        form.addRow("ID", self.id_label)

        self.type_label = QLabel(self.config["type"])
        form.addRow("Type", self.type_label)

        self.plot_widget = QSpinBox()
        self.plot_widget.setRange(1, 8)
        self.plot_widget.setValue(int(self.config.get("plot_index", 1)))
        form.addRow("Graph", self.plot_widget)
        
        self.name_edit = QLineEdit(self.config["name"])

        name_row_widget = QWidget()
        name_row = QHBoxLayout(name_row_widget)
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.setSpacing(6)

        self.name_modify_button = QPushButton("Modify")
        self.name_modify_button.setFixedWidth(70)
        self.name_modify_button.clicked.connect(self._modify_name)

        name_row.addWidget(self.name_edit)
        name_row.addWidget(self.name_modify_button)

        form.addRow("Name", name_row_widget)

        for key, value in self.config.get("connection", {}).items():

            widget = QLineEdit(str(value))
            self.connection_widgets[key] = widget

            connection_spec = self.schema.get(
                "connection", {}
            ).get(key, {})

            label = connection_spec.get(
                "label",
                key.replace("_", " ").title()
            )

            modifiable = (
                key == "port"
                or connection_spec.get("modifiable", False)
            )

            if modifiable:

                row_widget = QWidget()
                row = QHBoxLayout(row_widget)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(6)

                modify_button = QPushButton("Modify")
                modify_button.setFixedWidth(70)

                modify_button.clicked.connect(
                    lambda checked=False, k=key: self._modify_connection(k)
                )

                row.addWidget(widget)
                row.addWidget(modify_button)

                form.addRow(label, row_widget)

            else:
                form.addRow(label, widget)

        target = self.schema.get("target")

        if target is not None:
            self.target_widget = self._make_input_widget(target)

            default = self.config.get(
                "default_target",
                target.get("default")
            )

            self._set_widget_value(
                self.target_widget,
                default
            )

            label = target["label"]

            if target.get("unit"):
                label += f" ({target['unit']})"

            form.addRow(label, self.target_widget)

        defaults = self.config.get("defaults", {})

        for key, option in self.schema.get("options", {}).items():
            widget = self._make_input_widget(option)

            default = defaults.get(
                key,
                option.get("default")
            )

            self._set_widget_value(
                widget,
                default
            )

            self.option_widgets[key] = widget
            form.addRow(option["label"], widget)

        main_layout.addLayout(form)

        # ---------------------------------------------------------
        # Action buttons
        # ---------------------------------------------------------

        from gui.device_connectivity import DeviceConnectivity
        self.connectivity_fields = DeviceConnectivity(self.config, self)
        main_layout.addWidget(self.connectivity_fields)
        button_layout = QGridLayout()
        button_layout.setSpacing(4)

        self.connect_button = QPushButton("Connect")
        self.set_button = QPushButton("Set")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.read_button = QPushButton("Read")
        self.profile_button = QPushButton("Profile")
        self.delete_button = QPushButton("Delete")

        if self.config["type"] not in PROFILE_TYPES:
            self.profile_button.setEnabled(False)

        if self.schema.get("target") is None:
            self.set_button.setEnabled(False)

        if self.config["type"] == "opcua":
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)

        self.test_button = None

        if hasattr(self.device, "test_connection"):
            self.test_button = QPushButton("Test")

        self.connect_button.setObjectName("device_connect_button")
        self.set_button.setObjectName("device_set_button")
        self.start_button.setObjectName("start_button")
        self.stop_button.setObjectName("stop_button")
        self.read_button.setObjectName("device_read_button")

        actions = [self.connect_button, self.set_button, self.start_button, self.stop_button,
                   self.read_button, self.delete_button]
        if self.schema.get("target") is not None:
            actions.append(self.profile_button)
        if self.test_button is not None:
            actions.append(self.test_button)
        for index, button in enumerate(actions):
            button_layout.addWidget(button, index // 4, index % 4)

        main_layout.addLayout(button_layout)

        # ---------------------------------------------------------
        # Device status
        # ---------------------------------------------------------

        status_layout = QHBoxLayout()

        self.connection_led = QLabel()
        self.connection_led.setFixedSize(12, 12)

        self.status_label = QLabel("Disconnected")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumWidth(0)

        status_layout.addWidget(self.connection_led)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        main_layout.addLayout(status_layout)
        for combo in self.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        for edit in self.findChildren(QLineEdit):
            edit.setMinimumWidth(0)

        # ---------------------------------------------------------
        # Signals
        # ---------------------------------------------------------

        self.connect_button.clicked.connect(self._toggle_connection)
        self.set_button.clicked.connect(self._set_device)
        self.start_button.clicked.connect(self._start_device)
        self.stop_button.clicked.connect(self._stop_device)
        self.read_button.clicked.connect(self._read_device)
        self.profile_button.clicked.connect(self._edit_profile) 

        if self.test_button is not None:
            self.test_button.clicked.connect(self._test_connection)

    def _modify_connection(self, key):

        panel = self.parent()
        
        while panel is not None:
            if isinstance(panel, ControlPanel):
                break
            panel = panel.parent()
        
        if isinstance(panel, ControlPanel) and panel.experiment_running():
            QMessageBox.warning(
                self,
                "Experiment Running",
                "Manual hardware control is disabled "
                "during an active experiment."
            )
            return

        new_value = self.connection_widgets[key].text().strip()

        if not new_value:
            self.status_label.setText("Invalid connection")
            return

        try:
            if self.is_connected():
                if not self.disconnect_device():
                    self.status_label.setText("Disconnect before modifying port")
                    return

            self.config["connection"][key] = new_value

            self.device = self.schema["driver"](
                name=self.config["name"],
                **self.config["connection"]
            )

            if self.hardware is not None:
                self.hardware[self.device_id] = self.device

            self.status_label.setText(
                f"{key.replace('_', ' ').title()} changed to {new_value}"
            )

            self._update_connection_state()

        except Exception as e:
            self.status_label.setText(f"Modify error: {e}")


    def _make_input_widget(self, spec):

        field_type = spec["type"]

        if field_type == "float":
            widget = QDoubleSpinBox()

            widget.setRange(
                spec.get("min", -1e9),
                spec.get("max", 1e9)
            )

            widget.setDecimals(
                spec.get("decimals", 2)
            )

            return widget

        if field_type == "int":
            widget = QSpinBox()

            widget.setRange(
                spec.get("min", -1_000_000),
                spec.get("max", 1_000_000)
            )

            return widget

        if field_type == "choice":
            widget = QComboBox()
            widget.addItems([str(x) for x in spec["choices"]])

            return widget

        return QLineEdit()


    def _set_widget_value(self, widget, value):

        if value is None:
            return

        if isinstance(widget, QComboBox):
            widget.setCurrentText(str(value))

        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))

        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))

        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))


    def _widget_value(self, widget):

        if isinstance(widget, QComboBox):
            return widget.currentText()

        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()

        if isinstance(widget, QLineEdit):
            return widget.text()

        return None


    def get_values(self):

        values = {
            "id": self.device_id,
            "name": self.name_edit.text(),
            "type": self.config["type"],
            "connection": {
                key: widget.text()
                for key, widget in self.connection_widgets.items()
            },
            "options": {
                key: self._widget_value(widget)
                for key, widget in self.option_widgets.items()
            }
        }

        if self.target_widget is not None:
            values["target"] = self._widget_value(
                self.target_widget
            )

        return values


    # ============================================================
    # CONNECTION
    # ============================================================

    def is_connected(self):

        if self.device is None:
            return False

        return bool(
            getattr(self.device, "is_connected", False)
        )


    def _update_connection_state(self):

        connected = self.is_connected()

        if connected:
            self.connection_led.setStyleSheet(
                """
                background: #22c55e;
                border: 1px solid #37d774;
                border-radius: 6px;
                """
            )

            self.connect_button.setText("Disconnect")

        else:
            self.connection_led.setStyleSheet(
                """
                background: #ef4444;
                border: 1px solid #f87171;
                border-radius: 6px;
                """
            )

            self.connect_button.setText("Connect")


    def _toggle_connection(self):
    
        # Find the owning ControlPanel.
        panel = self.parent()
    
        while panel is not None:
            if isinstance(panel, ControlPanel):
                break
            panel = panel.parent()
    
        if panel is not None and isinstance(panel, ControlPanel):
    
            if panel.experiment_running():
                QMessageBox.warning(
                    self,
                    "Experiment Running",
                    "Device connections cannot be changed "
                    "while an experiment is running.\n\n"
                    "Stop the experiment first."
                )
                return
    
            if panel.connection_operation_active():
                QMessageBox.warning(
                    self,
                    "Hardware Busy",
                    "Wait for the current connection operation "
                    "to finish."
                )
                return
    
        if self.is_connected():
            self.disconnect_device()
        else:
            self.connect_device()


    def connect_device(self):

        if self.device is None:
            self.status_label.setText("No device")
            return False

        try:
            success = self.device.connect()

            if success:
                self.status_label.setText("Connected")
            else:
                self.status_label.setText("Connection failed")

            self._update_connection_state()

            return bool(success)

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self._update_connection_state()

            return False


    def disconnect_device(self):

        panel = self.parent()
    
        while panel is not None:
            if isinstance(panel, ControlPanel):
                break
            panel = panel.parent()
    
        if isinstance(panel, ControlPanel) and panel.experiment_running():
            self.status_label.setText(
                "Cannot disconnect during active experiment"
            )
            return False


        if self.device is None:
            return False

        try:
            self.device.disconnect()

            still_connected = bool(
                getattr(
                    self.device,
                    "is_connected",
                    False
                )
            )

            ser = getattr(
                self.device,
                "ser",
                None
            )

            serial_open = bool(
                ser is not None
                and getattr(
                    ser,
                    "is_open",
                    False
                )
            )

            if still_connected or serial_open:
                self.status_label.setText(
                    "WARNING: port still open"
                )

                self._update_connection_state()

                return False

            self.status_label.setText(
                "Disconnected"
            )

            self._update_connection_state()

            return True

        except Exception as e:
            self.status_label.setText(
                f"Disconnect error: {e}"
            )

            self._update_connection_state()

            return False


    # ============================================================
    # DEVICE COMMANDS
    # ============================================================

    def _set_device(self):

        panel = self.parent()
        
        while panel is not None:
            if isinstance(panel, ControlPanel):
                break
            panel = panel.parent()
        
        if isinstance(panel, ControlPanel) and panel.experiment_running():
            QMessageBox.warning(
                self,
                "Experiment Running",
                "Manual hardware control is disabled "
                "during an active experiment."
            )
            return

        if self.device is None:
            return

        if self.target_widget is None:
            self.status_label.setText("No setpoint")
            return

        try:
            target = self._widget_value(self.target_widget)

            options = {
                key: self._widget_value(widget)
                for key, widget in self.option_widgets.items()
            }

            device_type = self.config["type"]

            if device_type == "huber":

                control_mode = options.get("control_mode")

                if control_mode == "Jacket" and not self.device.set_jacket_control():
                    raise RuntimeError("Huber did not confirm Jacket control.")

                if control_mode == "Process" and not self.device.set_process_control():
                    raise RuntimeError("Huber did not confirm Process control.")

                success = self.device.set_target(target)

            elif device_type == "ismatec":

                stage = options.get(
                    "stage"
                )
            
                success = self.device.set_target(
                    target,
                    stage=stage
                )

            elif device_type == "masterflex":

                success = self.device.set_target(
                    target,
                    direction=options.get("direction", "+")
                )

            elif device_type == "nidaq_valve":
        
                success = self.device.set_target(int(target))

            else:
                success = self.device.set_target(target)

            self.status_label.setText(
                "Setpoint applied"
                if success else
                "Set failed"
            )

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
        finally:
            self._update_connection_state()


    def _start_device(self):

        panel = self.parent()
        
        while panel is not None:
            if isinstance(panel, ControlPanel):
                break
            panel = panel.parent()
        
        if isinstance(panel, ControlPanel) and panel.experiment_running():
            QMessageBox.warning(
                self,
                "Experiment Running",
                "Manual hardware control is disabled "
                "during an active experiment."
            )
            return

        if self.device is None:
            return

        try:
            options = {
                key: self._widget_value(widget)
                for key, widget in self.option_widgets.items()
            }

            if self.config["type"] == "ismatec":
                success = self.device.start(
                    stage=options.get("stage", "2")
                )

            else:
                success = self.device.start()

            self.status_label.setText(
                "Running"
                if success else
                "Start failed"
            )

        except Exception as e:
            self.status_label.setText(f"Error: {e}")


    def _stop_device(self):

        panel = self.parent()
        
        while panel is not None:
            if isinstance(panel, ControlPanel):
                break
            panel = panel.parent()
        
        if isinstance(panel, ControlPanel) and panel.experiment_running():
            QMessageBox.warning(
                self,
                "Experiment Running",
                "Manual hardware control is disabled "
                "during an active experiment."
            )
            return

        if self.device is None:
            return

        try:
            options = {
                key: self._widget_value(widget)
                for key, widget in self.option_widgets.items()
            }

            if self.config["type"] == "ismatec":
            
                stage = options.get(
                    "stage"
                )
            
                success = self.device.stop(
                    stage=stage
                )

            else:
                success = self.device.stop()

            self.status_label.setText(
                "Stopped"
                if success else
                "Stop failed"
            )

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
        finally:
            self._update_connection_state()


    def _read_device(self):

        if self.device is None:
            return

        try:
            status = self.device.read_status()
            status = ''.join([f'| {i}: {j} ' for i,j in status.items() if i !='name'])
            self.status_label.setText(str(status))

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
        finally:
            self._update_connection_state()


# In[52]:


class ControlPanel(QWidget):
    connection_status_changed = Signal(bool, int, int)

    def __init__(self, devices, hardware_types, hardware, default_devices, parent=None, *, default_configuration=None):
        super().__init__(parent)

        self.setObjectName("controlPanel")

        self.devices = devices
        self.default_devices = deepcopy(default_devices)
        self.default_configuration = deepcopy(default_configuration)
        self.hardware_types = hardware_types
        self.hardware = hardware

        self.hardware_manager = HardwareManager(self.hardware)

        self._connection_token = None
        self._experiment_token = None
        
        self.device_widgets = {}
        self.experiments = []
        from experiment.system_topology import empty_topology
        self.system_topology = empty_topology()

        self.connection_thread = None
        self.connection_worker = None

        self._build_ui()
        self._apply_style()
        self.update_global_status()


    def connection_operation_active(self):
        """
        Return True while Connect All / Disconnect All
        is running.
        """
        thread = self.connection_thread
    
        if thread is None:
            return False
    
        try:
            return thread.isRunning()
        except RuntimeError:
            return False

    
    def experiment_running(self):
        return (self.hardware_manager.state == HardwareState.EXPERIMENT)

    def _build_ui(self):

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)

        # ========================================================
        # TOP CONTROL BAR
        # ========================================================

        top_section = QVBoxLayout()
        top_section.setSpacing(8)
        
        top_row = FlowLayout()
        bottom_row = FlowLayout()
        

        title = QLabel("Hardware Control Panel")
        title.setObjectName("control_title")

        self.global_led = QLabel()
        self.global_led.setFixedSize(14, 14)

        self.global_status = QLabel("Disconnected")

        self.connect_all_button = QPushButton("Connect All")
        self.disconnect_all_button = QPushButton("Disconnect All")
        self.new_connection_button = QPushButton("New Connection")
        self.new_connection_button.setObjectName("new_connection_button")
        
        self.reset_button = QPushButton("Reset Defaults")

        self.save_config_button = QPushButton("Save Configuration")
        self.load_config_button = QPushButton("Load Configuration")
        self.demo_button = QPushButton("Load Simulated Demo")
        self.demo_button.setToolTip("Load a hardware-free demonstration at real-time speed")
        self.limits_button = QPushButton("Device Limits")
        
        self.generate_experiment_button = QPushButton("Generate Experiment")
        self.simulation_speed = QDoubleSpinBox()
        self.simulation_speed.setRange(0.1, 100)
        self.simulation_speed.setValue(1.0)
        self.simulation_speed.setSuffix("×")
        self.simulation_speed.setToolTip("Simulation speed; 1× is real time. Real hardware always uses 1×.")
        self.start_experiment_button = QPushButton("Start Experiment")
        self.start_experiment_button.setEnabled(False)
        
        self.experiment_count_label = QLabel("Experiments: 0")
        
        self.delete_experiment_spin = QSpinBox()
        self.delete_experiment_spin.setRange(1, 1)
        self.delete_experiment_spin.setEnabled(False)
        
        self.delete_experiment_button = QPushButton("Delete Experiment")
        self.delete_experiment_button.setEnabled(False)

        self.connect_all_button.setObjectName("connect_all_button")
        self.disconnect_all_button.setObjectName("disconnect_all_button")



        # ========================================================
        # TOP ROW
        # ========================================================
        
        top_row.addWidget(title)
        
        top_row.addStretch()
        
        top_row.addWidget(self.global_led)
        top_row.addWidget(self.global_status)
        
        top_row.addSpacing(20)
        
        top_row.addWidget(self.new_connection_button)
        top_row.addWidget(self.reset_button)
        top_row.addWidget(self.save_config_button)
        top_row.addWidget(self.load_config_button)
        top_row.addWidget(self.demo_button)
        top_row.addWidget(self.limits_button)
        
        
        # ========================================================
        # BOTTOM ROW
        # ========================================================
        
        bottom_row.addWidget(self.experiment_count_label)
        bottom_row.addWidget(self.delete_experiment_spin)
        bottom_row.addWidget(self.delete_experiment_button)
        
        bottom_row.addSpacing(25)
        
        bottom_row.addWidget(self.generate_experiment_button)
        bottom_row.addWidget(self.start_experiment_button)
        bottom_row.addWidget(QLabel("Simulation speed"))
        bottom_row.addWidget(self.simulation_speed)
        
        bottom_row.addStretch()
        
        bottom_row.addWidget(self.connect_all_button)
        bottom_row.addWidget(self.disconnect_all_button)
            
        
        # ========================================================
        # COMBINE BOTH ROWS
        # ========================================================
        
        top_section.addLayout(top_row)
        top_section.addLayout(bottom_row)
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Control Panel experiment folder"))
        self.control_output_path = QLineEdit(str(DATA_DIR / "control_panel_runs"))
        path_row.addWidget(self.control_output_path, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.choose_control_output_folder)
        path_row.addWidget(browse)
        top_section.addLayout(path_row)
        self.calibration_button = QPushButton("Pump calibration files…")
        self.calibration_button.clicked.connect(self.edit_pump_calibrations)
        calibration_row = QHBoxLayout()
        calibration_row.addWidget(self.calibration_button)
        self.module_sensors_button = QPushButton("Module sensors…")
        self.module_sensors_button.clicked.connect(lambda: self.edit_system_topology(False, tab=1))
        calibration_row.addWidget(self.module_sensors_button)
        top_section.addLayout(calibration_row)
        system_row = QHBoxLayout()
        self.unit_operations_button = QPushButton("Unit operations / modules…")
        self.msi_button = QPushButton("Generate MSI / connectivity graph…")
        self.unit_operations_button.clicked.connect(lambda: self.edit_system_topology(False))
        self.msi_button.clicked.connect(lambda: self.edit_system_topology(True))
        system_row.addWidget(self.unit_operations_button)
        system_row.addWidget(self.msi_button)
        top_section.addLayout(system_row)
        
        main_layout.addLayout(top_section)

        # ========================================================
        # DEVICE GRID
        # ========================================================

        scroll = QScrollArea()
        self.device_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.viewport().installEventFilter(self)

        container = QWidget()

        self.device_grid = QGridLayout(container)
        self.device_grid.setSpacing(8)
        self.device_grid.setContentsMargins(0, 0, 0, 0)
        self.device_grid.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.n_columns = 2

        for i, (device_id, config) in enumerate(self.devices.items()):

            widget = DeviceWidget(
                device_id=device_id,
                config=config,
                hardware_types=self.hardware_types,
                hardware=self.hardware,
                device=self.hardware[device_id]
            )

            self.device_widgets[device_id] = widget

            widget.delete_button.clicked.connect(
                lambda checked=False, d=device_id:
                self.delete_connection(d)
            )

            row = i // self.n_columns
            col = i % self.n_columns

            self.device_grid.addWidget(
                widget,
                row,
                col
            )

        scroll.setWidget(container)
        QTimer.singleShot(0, self._adapt_device_columns)

        main_layout.addWidget(scroll)

        # ========================================================
        # SIGNALS
        # ========================================================

        self.connect_all_button.clicked.connect(
            self.connect_all
        )

        self.disconnect_all_button.clicked.connect(
            self.disconnect_all
        )

        self.new_connection_button.clicked.connect(
            self.new_connection
        )

        self.reset_button.clicked.connect(
            self.reset_defaults
        )
        
        self.delete_experiment_button.clicked.connect(
            self.delete_experiment
        )

        for widget in self.device_widgets.values():

            widget.connect_button.clicked.connect(
                self.update_global_status
            )

        self.generate_experiment_button.clicked.connect(
            self.generate_experiment
        )

        self.start_experiment_button.clicked.connect(
            self.start_experiment
        )

        self.save_config_button.clicked.connect(
            self.save_configuration
        )
        
        self.load_config_button.clicked.connect(
            self.load_configuration
        )
        self.demo_button.clicked.connect(self.load_simulated_demo)
        self.limits_button.clicked.connect(self.edit_device_limits)
        
        self.update_profile_status()

    def edit_pump_calibrations(self):
        if self.hardware_manager.is_busy or self.experiment_window_attached():
            QMessageBox.warning(self, "Session open", "End the experiment session before changing pump calibrations.")
            return None
        from gui.pump_calibration_dialog import PumpCalibrationDialog
        configs, _ = self.collect_current_configuration()
        dialog = PumpCalibrationDialog(configs, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        for key, config in dialog.configs.items():
            if key not in self.device_widgets:
                continue
            widget = self.device_widgets[key]
            if "pump_calibration" in config:
                widget.config["pump_calibration"] = deepcopy(config["pump_calibration"])
            else:
                widget.config.pop("pump_calibration", None)
        return self.collect_current_configuration()[0]

    def current_system_topology(self):
        from experiment.system_topology import encode_msi
        definition = deepcopy(self.system_topology)
        definition.pop("msi_code", None)
        configs, _ = self.collect_current_configuration()
        try:
            definition["msi_code"] = encode_msi(definition, configs)
        except ValueError:
            pass
        return definition

    def edit_system_topology(self, generate=False, tab=0):
        if self.hardware_manager.is_busy or self.experiment_window_attached():
            QMessageBox.warning(self, "Session open", "End the experiment session before editing the physical system definition.")
            return
        from gui.system_topology_dialog import SystemTopologyDialog
        configs, _ = self.collect_current_configuration()
        dialog = SystemTopologyDialog(self.current_system_topology(), configs, self, generate=generate, initial_tab=tab)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.system_topology = deepcopy(dialog.definition)
            self.refresh_connectivity_fields()
        dialog.deleteLater()

    def refresh_connectivity_fields(self):
        for widget in self.device_widgets.values():
            if hasattr(widget, "connectivity_fields"):
                widget.connectivity_fields.set_definition(self.system_topology)
                widget.connectivity_fields.setEnabled(not self.hardware_manager.is_busy and not self.experiment_window_attached())

    def load_simulated_demo(self):
        """Explicitly replace the current configuration with a short demo."""
        if self.experiment_window_attached():
            QMessageBox.warning(self, "Session Open", "End the current experiment session first.")
            return
        if self.hardware_manager.is_busy or any(w.is_connected() for w in self.device_widgets.values()):
            QMessageBox.warning(self, "Hardware Busy", "Disconnect devices and finish active operations before loading the demo.")
            return
        configs = deepcopy(DEFAULT_DEVICES)
        configs["OPC_DEMO"] = {
            "name": "Demo probe", "type": "opcua", "connection": {}, "plot_index": 7,
        }
        for index, config in enumerate(configs.values()):
            config["simulated"] = True
            config["simulation"] = {"seed": 42 + index}
            config["simulation_speed"] = 1.0
        frame = {"time_s": [0.0, 10.0, 30.0]}
        for device_id, config in configs.items():
            kind = config["type"]
            if kind in {"mettler", "opcua"}:
                continue
            frame[device_id] = (
                [25, 35, 30] if kind == "huber" else
                [100, 400, 200] if kind == "jkem" else
                [0, 1, 0] if kind == "nidaq_valve" else
                [0, 10, 0]
            )
        self.apply_loaded_configuration(configs, {}, [pd.DataFrame(frame).to_dict(orient="split")], self.control_output_path.text())

    def choose_control_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Control Panel experiment folder", self.control_output_path.text())
        if folder:
            self.control_output_path.setText(folder)

    def experiment_window_attached(self):
        return getattr(self, "experiment_window", None) is not None

    def edit_device_limits(self):
        if self.hardware_manager.is_busy or self.experiment_window_attached():
            QMessageBox.warning(self, "Session Active", "End the experiment session before changing device limits.")
            return
        dialog = DeviceLimitsDialog({key: widget.config for key, widget in self.device_widgets.items()}, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.apply_device_limits(dialog.overrides())

    def apply_device_limits(self, overrides):
        if self.hardware_manager.is_busy or self.experiment_window_attached():
            raise RuntimeError("Cannot change limits during an experiment or connection operation")
        # Validate every change before mutating any device.
        validated = {key: target_limits({**self.device_widgets[key].config, "target_limits": values})
                     for key, values in overrides.items()}
        for key, limits in validated.items():
            widget = self.device_widgets[key]
            values = {field: limits[field] for field in ("min", "max", "min_nonzero")}
            widget.config["target_limits"] = values
            self.devices[key]["target_limits"] = dict(values)
            self.hardware[key].target_limits = dict(limits)
            widget.schema["target"].update(min=limits["min"], max=limits["max"])
            if widget.target_widget is not None:
                if isinstance(widget.target_widget, QSpinBox):
                    widget.target_widget.setRange(int(limits["min"]), int(limits["max"]))
                else:
                    widget.target_widget.setRange(limits["min"], limits["max"])

    def dataframe_from_saved(self, data):
    
        return pd.DataFrame(
            data=data["data"],
            columns=data["columns"],
            index=data.get("index")
        )

    def apply_loaded_configuration(
        self,
        devices,
        profiles_data,
        experiments_data,
        recording_folder=None,
        system_topology=None,
    ):

        from experiment.system_topology import empty_topology, validate_topology
        definition = deepcopy(system_topology or empty_topology())
        validate_topology(definition, devices, complete=False)

        if self.experiment_window_attached():
            raise RuntimeError("End the existing experiment session before replacing configuration")
        if (
            self.experiment_running()
            or self.connection_operation_active()
        ):
            raise RuntimeError(
                "Cannot replace hardware configuration "
                "while a connection operation or experiment "
                "is active."
            )
    
        # ----------------------------------------------------------
        # Disconnect existing hardware
        # ----------------------------------------------------------
    
        if not self.shutdown_hardware():
            QMessageBox.warning(
                self,
                "Cannot Replace Hardware",
                "All devices must be disconnected "
                "successfully before changing the configuration.",
            )
            return
    
        # ----------------------------------------------------------
        # Remove existing widgets
        # ----------------------------------------------------------
    
        for widget in list(
            self.device_widgets.values()
        ):
    
            self.device_grid.removeWidget(
                widget
            )
    
            widget.deleteLater()
    
        self.device_widgets.clear()
        self.hardware.clear()
    
        # ----------------------------------------------------------
        # Replace working configuration
        # ----------------------------------------------------------
    
        self.control_output_path.setText(str(recording_folder or DATA_DIR / "control_panel_runs"))
        self.system_topology = definition
        self.devices.clear()
        self.devices.update(
            deepcopy(devices)
        )
    
        # ----------------------------------------------------------
        # Rebuild hardware + device widgets
        # ----------------------------------------------------------
    
        for i, (
            device_id,
            config
        ) in enumerate(
            self.devices.items()
        ):
    
            device = create_device(
                config,
                self.hardware_types
            )
    
            self.hardware[
                device_id
            ] = device
    
            widget = DeviceWidget(
                device_id=device_id,
                config=config,
                hardware_types=self.hardware_types,
                hardware=self.hardware,
                device=device
            )
    
            # ----------------------------------------------
            # Restore profile
            # ----------------------------------------------
    
            saved_profile = (
                profiles_data.get(
                    device_id
                )
            )
    
            if saved_profile is not None:
    
                widget.profile = (
                    self.dataframe_from_saved(
                        saved_profile
                    )
                )
    
            self.device_widgets[
                device_id
            ] = widget
    
            # Your existing signals
            widget.connect_button.clicked.connect(
                self.update_global_status
            )
    
            widget.delete_button.clicked.connect(
                lambda checked=False, d=device_id:
                self.delete_connection(d)
            )
    
            row = i // self.n_columns
            col = i % self.n_columns
    
            self.device_grid.addWidget(
                widget,
                row,
                col
            )
    
        # ----------------------------------------------------------
        # Restore generated experiments
        # ----------------------------------------------------------
    
        self.experiments = [
            self.dataframe_from_saved(
                experiment
            )
            for experiment in experiments_data
        ]
    
        # ----------------------------------------------------------
        # Refresh UI
        # ----------------------------------------------------------
    
        self.update_experiment_controls()
        self.update_profile_status()
        self.update_global_status()
        return True

    def load_configuration(self):
    
        if self.experiment_running():
            QMessageBox.warning(
                self,
                "Experiment Running",
                "Stop the experiment before changing "
                "the hardware configuration.",
            )
            return

        if (
            self.connection_thread is not None
            and self.connection_thread.isRunning()
        ):
        
            QMessageBox.warning(
                self,
                "Hardware Busy",
                "Wait for Connect All / Disconnect All "
                "to finish before resetting."
            )
        
            return
    
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "Lab Configuration (*.json)"
        )
    
        if not file_path:
            return
    
        try:
    
            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:
    
                data = json.load(f)
    
            devices = data.get(
                "devices",
                {}
            )
    
            profiles_data = data.get(
                "profiles",
                {}
            )
    
            experiments_data = data.get(
                "experiments",
                []
            )
    
            if not devices:
    
                raise ValueError(
                    "Configuration contains no devices."
                )
    
            self.apply_loaded_configuration(
                devices,
                profiles_data,
                experiments_data,
                data.get("recording_folder"),
                data.get("system_topology"),
            )
    
            print(
                f"[CONTROL PANEL] "
                f"Loaded configuration: {file_path}"
            )
    
            QMessageBox.information(
                self,
                "Configuration Loaded",
                f"Loaded {len(devices)} devices and "
                f"{len(self.experiments)} experiments."
            )
    
        except Exception as e:
    
            QMessageBox.critical(
                self,
                "Load Configuration Error",
                str(e)
            )
    
    def save_configuration(self):
    
        devices, profiles = (
            self.collect_current_configuration()
        )
    
        data = {
            "version": 3,
            "system_topology": self.current_system_topology(),
            "recording_folder": self.control_output_path.text().strip(),
            "devices": devices,
            "profiles": profiles,
            "experiments": [
                experiment.to_dict(
                    orient="split"
                )
                for experiment in self.experiments
            ]
        }
    
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",
            "Lab Configuration (*.json)"
        )
    
        if not file_path:
            return
    
        if not file_path.lower().endswith(
            ".json"
        ):
            file_path += ".json"
    
        try:
    
            with open(
                file_path,
                "w",
                encoding="utf-8"
            ) as f:
    
                json.dump(
                    data,
                    f,
                    indent=4
                )
    
            print(
                f"[CONTROL PANEL] "
                f"Configuration saved: {file_path}"
            )
    
            QMessageBox.information(
                self,
                "Configuration Saved",
                f"Saved configuration with "
                f"{len(devices)} devices and "
                f"{len(self.experiments)} experiments."
            )
    
        except Exception as e:
    
            QMessageBox.critical(
                self,
                "Save Configuration Error",
                str(e)
            )
    
    def collect_current_configuration(self):
    
        devices = {}
        profiles = {}
    
        for device_id, widget in self.device_widgets.items():
    
            config = deepcopy(
                widget.config
            )
    
            # Current graph assignment
            config["plot_index"] = (
                widget.plot_widget.value()
            )
    
            # Current default setpoint
            if widget.target_widget is not None:
    
                config["default_target"] = (
                    widget._widget_value(
                        widget.target_widget
                    )
                )
    
            # Current options:
            # Ismatec stage, Masterflex direction, Huber mode, etc.
            if widget.option_widgets:
    
                config["defaults"] = {
                    key: widget._widget_value(
                        option_widget
                    )
                    for key, option_widget
                    in widget.option_widgets.items()
                }
    
            devices[device_id] = config
    
            # Save editable profile too
            if widget.profile is not None:
    
                profiles[device_id] = (
                    widget.profile.to_dict(
                        orient="split"
                    )
                )
    
        return devices, profiles

    def update_experiment_controls(self):
    
        n = len(self.experiments)
    
        self.experiment_count_label.setText(
            f"Experiments: {n}"
        )
    
        has_experiments = n > 0
    
        self.start_experiment_button.setEnabled(
            has_experiments
        )
    
        self.delete_experiment_button.setEnabled(
            has_experiments
        )
    
        self.delete_experiment_spin.setEnabled(
            has_experiments
        )
    
        if has_experiments:
            self.delete_experiment_spin.setRange(1, n)
            self.delete_experiment_spin.setValue(n)
        else:
            self.delete_experiment_spin.setRange(1, 1)

    def apply_default_configuration(self):
        config = self.default_configuration
        return self.apply_loaded_configuration(
            config["devices"], config["profiles"], config["experiments"],
            DATA_DIR / "control_panel_runs" / "crystallization_demo", config["system_topology"],
        )

    def reset_defaults(self):

        if self.experiment_window_attached():
            QMessageBox.warning(self, "Session Open", "End the current experiment session first.")
            return

        if self.experiment_running():
            QMessageBox.warning(
                self,
                "Experiment Running",
                "Stop the experiment before changing "
                "the hardware configuration.",
            )
            return

        if (
            self.connection_thread is not None
            and self.connection_thread.isRunning()
        ):
        
            QMessageBox.warning(
                self,
                "Hardware Busy",
                "Wait for Connect All / Disconnect All "
                "to finish before resetting."
            )
        
            return
    
        if self.default_configuration is not None:
            self.apply_default_configuration()
            return

        if not self.shutdown_hardware():
            QMessageBox.warning(
                self,
                "Cannot Replace Hardware",
                "All devices must be disconnected "
                "successfully before changing the configuration.",
            )
            return
    
        self.experiments.clear()
    
        for device_id, widget in list(
            self.device_widgets.items()
        ):
            self.device_grid.removeWidget(widget)
            widget.deleteLater()
    
        self.device_widgets.clear()
        self.hardware.clear()
    
        self.devices.clear()
        
        self.devices.update(
            deepcopy(self.default_devices)
        )
    
        for i, (device_id, config) in enumerate(
            self.devices.items()
        ):
    
            device = create_device(
                config,
                self.hardware_types
            )
    
            self.hardware[device_id] = device
    
            widget = DeviceWidget(
                device_id=device_id,
                config=config,
                hardware_types=self.hardware_types,
                hardware=self.hardware,
                device=device
            )
            
            self.device_widgets[device_id] = widget
            
            widget.connect_button.clicked.connect(
                self.update_global_status
            )
            
            # Restore Delete button functionality after Reset Defaults.
            widget.delete_button.clicked.connect(
                lambda checked=False, d=device_id:
                self.delete_connection(d)
            )
    
            row = i // self.n_columns
            col = i % self.n_columns
    
            self.device_grid.addWidget(
                widget,
                row,
                col
            )
    
        self.update_experiment_controls()
        self.update_profile_status()
        self.update_global_status()
    
        print("[CONTROL PANEL] Reset to defaults.")

    def delete_experiment(self):
    
        if not self.experiments:
            return
    
        exp_num = self.delete_experiment_spin.value()
        index = exp_num - 1
    
        self.experiments.pop(index)
    
        print(
            f"Deleted Experiment {exp_num}"
        )
    
        self.update_experiment_controls()

    def start_experiment(self):
        return self._start_experiment()

    def start_campaign(self, campaign, recording_override=None):
        if not isinstance(campaign, Campaign) or not campaign.experiments:
            raise ValueError("A nonempty campaign is required")
        return self._start_experiment(campaign, recording_override)

    def _start_experiment(self, campaign=None, recording_override=None):

        if self.experiment_window_attached():
            QMessageBox.warning(self, "Session Open", "End the current experiment session before opening another.")
            return

        if self.connection_operation_active():
            QMessageBox.warning(
                self,
                "Hardware Busy",
                "Wait for the current connection operation "
                "to finish before starting an experiment."
            )
            return
        
        if self.experiment_running():
            QMessageBox.warning(
                self,
                "Experiment Running",
                "An experiment is already running."
            )
            return

        experiments = campaign.experiments if campaign is not None else self.experiments
        selected_ids = set(self.hardware)
        if campaign is not None:
            selected_ids = {profile.device_id for experiment in campaign.experiments for profile in experiment.profiles}
            selected_ids.update(campaign.monitor_device_ids)
            missing = selected_ids - self.hardware.keys()
            if missing:
                QMessageBox.warning(self, "Missing Devices", ", ".join(sorted(missing)))
                return

        print(
            f"[CONTROL PANEL] Start clicked | "
            f"{len(experiments)} experiment(s)"
        )
    
        if not experiments:
            QMessageBox.warning(
                self,
                "No Experiments",
                "Generate at least one experiment first."
            )
            return

        disconnected = [
            device_id
            for device_id, widget
            in self.device_widgets.items()
            if device_id in selected_ids and not widget.is_connected()
        ]

        if disconnected:

            QMessageBox.warning(
                self,
                "Devices Not Connected",
                "Connect all devices before starting "
                "the experiment.\n\n"
                "Disconnected devices:\n"
                + "\n".join(disconnected)
            )

            return
    
        try:

            # --------------------------------------------------
            # 1. Collect current device configurations
            # --------------------------------------------------

            for widget in self.device_widgets.values():

                widget.config["plot_index"] = (
                    widget.plot_widget.value()
                )

                if widget.target_widget is not None:
                    widget.config["default_target"] = (
                        widget._widget_value(
                            widget.target_widget
                        )
                    )

                if widget.option_widgets:
                    widget.config["defaults"] = {
                        key: widget._widget_value(option_widget)
                        for key, option_widget
                        in widget.option_widgets.items()
                    }

            device_configs = {
                device_id: deepcopy(widget.config)
                for device_id, widget
                in self.device_widgets.items()
                if device_id in selected_ids
            }
            # Validate all experiments before reserving or starting any hardware.
            for experiment in experiments:
                frame = experiment.to_frame() if isinstance(experiment, ExperimentDefinition) else experiment
                ExperimentDefinition.from_frame(frame, device_configs)
            if campaign is not None:
                if any(not is_monitor_only(device_configs[key]) for key in campaign.monitor_device_ids):
                    raise ValueError("Only read-only devices may be added as campaign monitors")

            # --------------------------------------------------
            # 2. Reserve hardware BEFORE dashboard creation
            # --------------------------------------------------

            try:
                token = self.hardware_manager.reserve_experiment()

            except HardwareBusyError as error:
                QMessageBox.warning(
                    self,
                    "Hardware Busy",
                    str(error),
                )
                return

            self._experiment_token = token

            # --------------------------------------------------
            # 3. Create and configure ExperimentDashboard
            # --------------------------------------------------

            dashboard = ExperimentDashboard(
                devices={key: device for key, device in self.hardware.items() if key in selected_ids},
                device_configs=device_configs,
                experiments=experiments,
                campaign=campaign,
            )

            dashboard.hardware_manager = self.hardware_manager
            from gui.recording_settings_dialog import load_recording_settings
            dashboard.recording_settings = load_recording_settings()
            dashboard.recording_settings["system_topology"] = self.current_system_topology()
            if campaign is None:
                output = self.control_output_path.text().strip()
                if not output:
                    raise ValueError("Choose a Control Panel experiment folder")
                dashboard.recording_settings["output_folder"] = output
            if recording_override:
                dashboard.recording_settings.update(recording_override)
            dashboard.recording_settings["enabled"] = True
            from app.logging_config import set_output_log_root
            set_output_log_root(dashboard.recording_settings["output_folder"])
            dashboard.simulation_speed = self.simulation_speed.value()
            dashboard.hardware_token = token

            self.experiment_window = dashboard
            self.refresh_connectivity_fields()

            print("[CONTROL PANEL] Dashboard created.")
            
            host = getattr(self, "dashboard_host", None)
            
            if host is None:
                raise RuntimeError(
                    "ExperimentDashboard host has not been configured."
                )
            
            host(dashboard)
            
            print("[CONTROL PANEL] Dashboard embedded in Experiment page.")

            # --------------------------------------------------
            # 4. Schedule experiment startup
            # --------------------------------------------------

            QTimer.singleShot(
                200,
                dashboard.start_orchestrator,
            )

            print(
                "[CONTROL PANEL] Experiment start scheduled."
            )
            return dashboard

        except Exception as error:

            print(
                "[CONTROL PANEL ERROR] "
                f"{type(error).__name__}: {error}"
            )

            traceback.print_exc()

            # --------------------------------------------------
            # 5. Check whether a worker may still be running
            # --------------------------------------------------

            dashboard = locals().get("dashboard", None)
            token = locals().get("token", None)

            thread_running = False

            if dashboard is not None:

                thread = getattr(dashboard, "thread", None)

                if thread is not None:
                    try:
                        thread_running = thread.isRunning()
                    except RuntimeError:
                        # Cannot prove that the thread stopped.
                        thread_running = True

            # --------------------------------------------------
            # 6. Release ownership only when safe

            if not thread_running and token is not None:

                self.hardware_manager.release_experiment(
                    token
                )

                if self._experiment_token == token:
                    self._experiment_token = None

                if dashboard is not None:
                    dashboard.hardware_token = None

            QMessageBox.critical(
                self,
                "Experiment Startup Error",
                str(error),
            )

    def generate_experiment(self):
    
        profiles = {}
    
        # ----------------------------------------------------------
        # Synchronize current GUI setpoints
        # ----------------------------------------------------------
    
        for widget in self.device_widgets.values():
    
            if widget.target_widget is not None:
                widget.config["default_target"] = widget._widget_value(
                    widget.target_widget
                )
    
        # ----------------------------------------------------------
        # Collect explicitly defined profiles
        # ----------------------------------------------------------
    
        for device_id, widget in self.device_widgets.items():
    
            if widget.config["type"] not in PROFILE_TYPES or is_monitor_only(widget.config):
                continue
    
            if widget.profile is not None:
                profiles[device_id] = widget.profile.copy()
    
        if not profiles:
    
            QMessageBox.warning(
                self,
                "No Profiles",
                "Define at least one device profile first."
            )
    
            return
    
        # ----------------------------------------------------------
        # Experiment duration
        # ----------------------------------------------------------
    
        final_time = max(
            profile["time_s"].max()
            for profile in profiles.values()
        )
    
        # ----------------------------------------------------------
        # Constant profiles for devices without explicit profiles
        # ----------------------------------------------------------
    
        for device_id, widget in self.device_widgets.items():
    
            if widget.config["type"] not in PROFILE_TYPES or is_monitor_only(widget.config):
                continue
    
            if device_id in profiles:
                continue
    
            if widget.target_widget is not None:
                target = widget._widget_value(
                    widget.target_widget
                )
            else:
                target = widget.config.get(
                    "default_target",
                    0
                )
    
            profiles[device_id] = pd.DataFrame(
                {
                    "time_s": [0, final_time],
                    "value": [target, target]
                }
            )
    
        # ----------------------------------------------------------
        # Build master timeline
        # ----------------------------------------------------------
    
        all_times = sorted(
            {
                time_value
                for profile in profiles.values()
                for time_value in profile["time_s"]
            }
        )
    
        experiment = pd.DataFrame(
            {
                "time_s": all_times
            }
        )
    
        # ----------------------------------------------------------
        # Merge profiles
        # ----------------------------------------------------------
    
        for device_id, profile in profiles.items():
    
            series = profile.rename(
                columns={
                    "value": device_id
                }
            )
    
            experiment = experiment.merge(
                series,
                on="time_s",
                how="left"
            )
    
        # ----------------------------------------------------------
        # Hold previous setpoint until next profile point
        # ----------------------------------------------------------
    
        # ----------------------------------------------------------
        # Fill master timeline using each device's own profile
        # ----------------------------------------------------------
        
        for device_id, profile in profiles.items():
        
            profile = (
                profile
                .sort_values("time_s")
                .reset_index(drop=True)
            )
        
            device_type = self.device_widgets[
                device_id
            ].config["type"]
        
            times = experiment[
                "time_s"
            ].to_numpy(dtype=float)
        
            profile_times = profile[
                "time_s"
            ].to_numpy(dtype=float)
        
            profile_values = profile[
                "value"
            ].to_numpy(dtype=float)
        
            if device_type == "nidaq_valve":
        
                # Discrete device: hold the previous state.
                indices = np.searchsorted(
                    profile_times,
                    times,
                    side="right"
                ) - 1
        
                indices = np.clip(
                    indices,
                    0,
                    len(profile_values) - 1
                )
        
                experiment[
                    device_id
                ] = profile_values[
                    indices
                ]
        
            else:
        
                # Continuous device: linear interpolation.
                experiment[
                    device_id
                ] = np.interp(
                    times,
                    profile_times,
                    profile_values
                )
    
        self.experiments.append(
            experiment
        )
        
        self.update_experiment_controls()
        
        print(
            f"\nGenerated Experiment "
            f"{len(self.experiments)}"
        )
        
        print(experiment)
        
    # ============================================================
    # GLOBAL CONNECTION
    # ============================================================

    def connect_all(self):
    
        self._start_connection_worker(
            "connect"
        )


    def disconnect_all(self):

        self._start_connection_worker(
            "disconnect"
        )

    def update_profile_status(self):

        has_any_profile = any(
            widget.profile is not None
            for widget in self.device_widgets.values()
            if widget.config["type"] in PROFILE_TYPES
        )
    
        self.generate_experiment_button.setEnabled(
            has_any_profile
        )


    @Slot()
    def _connection_thread_finished(self):
    
        # --------------------------------------------------
        # 1. Release hardware ownership
        # --------------------------------------------------
    
        if self._connection_token is not None:
    
            released = self.hardware_manager.release_connection(
                self._connection_token
            )
    
            if not released:
                print(
                    "[CONTROL PANEL WARNING] "
                    "Connection ownership release failed."
                )
    
            self._connection_token = None
    
        # --------------------------------------------------
        # 2. Clear completed worker references
        # --------------------------------------------------
    
        self.connection_worker = None
        self.connection_thread = None
    
        # --------------------------------------------------
        # 3. Restore UI controls
        # --------------------------------------------------
    
        self.connect_all_button.setEnabled(True)
        self.disconnect_all_button.setEnabled(True)
    
        self.reset_button.setEnabled(True)
        self.load_config_button.setEnabled(True)
        self.new_connection_button.setEnabled(True)
    
        # --------------------------------------------------
        # 4. Refresh hardware status
        # --------------------------------------------------
    
        self.update_global_status()
    
        print(
            "[CONTROL PANEL] "
            "Connection worker completely stopped."
        )
        self.load_config_button.setEnabled(True)
        
    @Slot(str, bool, str)
    def _connection_device_finished(
        self,
        device_id,
        success,
        message
    ):
    
        widget = self.device_widgets.get(
            device_id
        )
    
        if widget is None:
            return
    
        widget.status_label.setText(
            message
        )
    
        widget._update_connection_state()
    
    def _start_connection_worker(self, mode):
    
        # --------------------------------------------------
        # 1. Block operations during an experiment
        # --------------------------------------------------
        
        if self.experiment_running():
            QMessageBox.warning(
                self,
                "Experiment Running",
                "Cannot connect or disconnect hardware while "
                "an experiment is active.\n\n"
                "Stop the experiment before changing connections."
            )
            return
        
        # --------------------------------------------------
        # 2. Prevent overlapping connection operations
        # --------------------------------------------------
        
        if self.connection_operation_active():
            print(
                "[CONTROL PANEL] "
                "Connection operation already running."
            )
            return
        
        # --------------------------------------------------
        # 3. Reserve hardware
        # --------------------------------------------------
        
        try:
            token = self.hardware_manager.reserve_connection()
        
        except HardwareBusyError as error:
            QMessageBox.warning(
                self,
                "Hardware Busy",
                str(error),
            )
            return
        
        self._connection_token = token
        
        # --------------------------------------------------
        # 4. Create and start worker with error handling
        # --------------------------------------------------
        
        try:
        
            self.connect_all_button.setEnabled(False)
            self.disconnect_all_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.load_config_button.setEnabled(False)
            self.new_connection_button.setEnabled(False)
        
            self.global_status.setText(
                "Connecting..."
                if mode == "connect"
                else "Disconnecting..."
            )
        
            self.connection_thread = QThread(self)
        
            self.connection_worker = ConnectionWorker(
                dict(self.hardware),
                mode
            )
        
            self.connection_worker.moveToThread(
                self.connection_thread
            )
        
            self.connection_thread.started.connect(
                self.connection_worker.run
            )
        
            self.connection_worker.device_finished.connect(
                self._connection_device_finished
            )
        
            self.connection_worker.finished.connect(
                self.connection_thread.quit
            )
        
            self.connection_worker.finished.connect(
                self.connection_worker.deleteLater
            )
        
            self.connection_thread.finished.connect(
                self._connection_thread_finished
            )
        
            self.connection_thread.finished.connect(
                self.connection_thread.deleteLater
            )
        
            self.connection_thread.start()
        
            print(
                "[CONTROL PANEL] "
                f"Connection worker started: {mode}"
            )
        
        # --------------------------------------------------
        # 5. Handle startup failures
        # --------------------------------------------------
        
        except Exception as error:
        
            print(
                "[CONTROL PANEL ERROR] "
                f"Connection worker startup failed: "
                f"{type(error).__name__}: {error}"
            )
        
            traceback.print_exc()
        
            thread = self.connection_thread
        
            # Conservative default: retain ownership unless
            # we can confirm there is no running thread.
            thread_running = thread is not None
        
            if thread is not None:
                try:
                    thread_running = thread.isRunning()
                except RuntimeError:
                    thread_running = True
        
            # --------------------------------------------------
            # Thread did not start: release ownership
            # --------------------------------------------------
        
            if not thread_running:
        
                if self._connection_token is not None:
                    self.hardware_manager.release_connection(
                        self._connection_token
                    )
                    self._connection_token = None
        
                # Retain references to objects associated with
                # a failed, unstarted thread until controlled
                # cleanup. Do not delete a running QThread.
                if not hasattr(self, "_failed_connection_workers"):
                    self._failed_connection_workers = []
        
                self._failed_connection_workers.append(
                    (self.connection_thread, self.connection_worker)
                )
        
                self.connection_thread = None
                self.connection_worker = None
        
                self.connect_all_button.setEnabled(True)
                self.disconnect_all_button.setEnabled(True)
                self.reset_button.setEnabled(True)
                self.load_config_button.setEnabled(True)
                self.new_connection_button.setEnabled(True)
        
                self.update_global_status()
        
            else:
        
                # A thread might be executing hardware calls.
                # Preserve references and ownership.
                print(
                    "[CONTROL PANEL WARNING] "
                    "Thread may still be active. "
                    "Hardware reservation retained."
                )
        
            QMessageBox.critical(
                self,
                "Connection Worker Error",
                f"Could not start the connection worker.\n\n"
                f"{type(error).__name__}: {error}"
            )

        return
        
    def update_global_status(self):
        self.refresh_connectivity_fields()

        states = [
            widget.is_connected()
            for widget in self.device_widgets.values()
        ]

        all_connected = (
            len(states) > 0
            and all(states)
        )

        if all_connected:

            self.global_led.setStyleSheet(
                """
                background: #22c55e;
                border: 1px solid #37d774;
                border-radius: 7px;
                """
            )

            self.global_status.setText(
                "All connected"
            )

        else:

            self.global_led.setStyleSheet(
                """
                background: #ef4444;
                border: 1px solid #f87171;
                border-radius: 7px;
                """
            )

            connected = sum(states)

            self.global_status.setText(
                f"{connected}/{len(states)} connected"
            )

        # Notify MainWindow of the current connection state.
        self.connection_status_changed.emit(
            all_connected,
            sum(states),
            len(states)
        )


    def delete_connection(self, device_id):

        if (
            self.connection_thread is not None
            and self.connection_thread.isRunning()
        ):
        
            QMessageBox.warning(
                self,
                "Hardware Busy",
                "Wait for Connect All / Disconnect All "
                "to finish before resetting."
            )
        
            return
    
        widget = self.device_widgets.get(
            device_id
        )
    
        if widget is None:
            return
    
        try:
            if widget.is_connected():
                if not widget.disconnect_device():
                    QMessageBox.warning(
                        self,
                        "Delete Connection",
                        f"Could not safely disconnect {device_id}."
                    )
                    return
    
        except Exception as e:
            QMessageBox.warning(
                self,
                "Delete Connection",
                str(e)
            )
            return
    
        self.device_grid.removeWidget(
            widget
        )
    
        widget.deleteLater()
    
        self.device_widgets.pop(
            device_id,
            None
        )
    
        self.hardware.pop(
            device_id,
            None
        )
    
        self.devices.pop(
            device_id,
            None
        )
    
        self.rebuild_device_grid()
        self.update_profile_status()
        self.update_global_status()
    
        print( f"[CONTROL PANEL] Deleted {device_id}")


    def rebuild_device_grid(self):

        for i in reversed(
            range(self.device_grid.count())
        ):
            item = self.device_grid.takeAt(i)
    
            if item.widget() is not None:
                self.device_grid.removeWidget(
                    item.widget()
                )
    
        for i, widget in enumerate(
            self.device_widgets.values()
        ):
    
            row = i // self.n_columns
            col = i % self.n_columns
    
            self.device_grid.addWidget(
                widget,
                row,
                col
            )

    def eventFilter(self, watched, event):
        if hasattr(self, 'device_scroll') and watched is self.device_scroll.viewport() and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._adapt_device_columns)
        return super().eventFilter(watched, event)

    def _adapt_device_columns(self):
        if not hasattr(self, 'device_scroll'):
            return
        width = self.device_scroll.viewport().width()
        card_width = max(360, max((widget.minimumSizeHint().width() for widget in self.device_widgets.values()), default=360))
        columns = max(1, (width + 8) // (card_width + 8))
        for widget in self.device_widgets.values():
            self.device_grid.setAlignment(widget, Qt.AlignmentFlag.AlignTop)
        if columns != self.n_columns:
            old_columns = self.n_columns
            self.n_columns = columns
            for column in range(max(old_columns, columns)):
                self.device_grid.setColumnStretch(column, 1 if column < columns else 0)
            self.rebuild_device_grid()
        
    def new_connection(self):

        dialog = NewDeviceDialog(
            hardware_types=self.hardware_types,
            existing_ids=self.device_widgets.keys(),
            parent=self
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        device_id, config = dialog.get_device()

        try:
            device = create_device(
                config,
                self.hardware_types
            )

            self.hardware[device_id] = device

            widget = DeviceWidget(
                device_id=device_id,
                config=config,
                hardware_types=self.hardware_types,
                hardware=self.hardware,
                device=device
            )

            self.device_widgets[device_id] = widget
            self.devices[device_id] = config
            
            widget.connect_button.clicked.connect(
                self.update_global_status
            )
            
            widget.delete_button.clicked.connect(
                lambda checked=False, d=device_id:
                self.delete_connection(d)
            )

            i = len(self.device_widgets) - 1

            row = i // self.n_columns
            col = i % self.n_columns

            self.device_grid.addWidget(
                widget,
                row,
                col
            )

            self.update_global_status()

        except Exception as e:
            print(f"[NEW CONNECTION] {device_id}: {e}")


    def get_all_values(self):

        return {
            device_id: widget.get_values()
            for device_id, widget in self.device_widgets.items()
        }
    def shutdown_hardware(self):
        """
        Disconnect all hardware safely.
    
        Already-disconnected devices are treated as successful,
        provided no known serial connection remains open.
    
        Returns True only if every device is disconnected.
        """
    
        if self.connection_operation_active():
            print("[SHUTDOWN] Connection worker is active.")
            return False
    
        if self.experiment_running():
            print("[SHUTDOWN] Experiment owns hardware.")
            return False
    
        print("[SHUTDOWN] Checking hardware connections...")
    
        failed_devices = []
    
        for device_id, widget in self.device_widgets.items():
    
            device = widget.device
    
            if device is None:
                print(f"[SHUTDOWN] {device_id}: No device instance.")
                failed_devices.append(device_id)
                continue
    
            connected = bool(
                getattr(device, "is_connected", False)
            )
    
            ser = getattr(device, "ser", None)
    
            serial_open = bool(
                getattr(ser, "is_open", False)
            ) if ser is not None else False
    
            # --------------------------------------------------
            # 1. Already disconnected
            # --------------------------------------------------
    
            if not connected and not serial_open:
    
                print(
                    f"[SHUTDOWN] {device_id}: "
                    "Already disconnected."
                )
    
                widget._update_connection_state()
                continue
    
            # --------------------------------------------------
            # 2. Disconnect connected device
            # --------------------------------------------------
    
            print(
                f"[SHUTDOWN] {device_id}: "
                "Disconnecting..."
            )
    
            try:
    
                success = widget.disconnect_device()
    
                if not success:
                    failed_devices.append(device_id)
    
                    print(
                        f"[SHUTDOWN WARNING] "
                        f"{device_id}: Disconnect failed."
                    )
    
            except Exception as error:
    
                failed_devices.append(device_id)
    
                print(
                    f"[SHUTDOWN ERROR] "
                    f"{device_id}: "
                    f"{type(error).__name__}: {error}"
                )
    
        self.update_global_status()
    
        # --------------------------------------------------
        # 3. Report final status
        # --------------------------------------------------
    
        if failed_devices:
    
            print(
                "[SHUTDOWN] Failed devices: "
                + ", ".join(failed_devices)
            )
    
            return False
    
        print(
            "[SHUTDOWN] All devices disconnected successfully."
        )
    
        return True
            
    def prepare_shutdown(self):
        """
        Coordinate Control Panel shutdown.
    
        Return True if the application can proceed
        with closing, otherwise False.
        """
    
        # --------------------------------------------------
        # 1. Do not close during connection operations
        # --------------------------------------------------
    
        if self.connection_operation_active():
    
            QMessageBox.warning(
                self,
                "Hardware Busy",
                "A hardware connection operation is still "
                "running.\n\n"
                "Wait for it to finish before closing.",
            )
    
            return False
    
        # --------------------------------------------------
        # 2. Close the experiment dashboard first
        # --------------------------------------------------
    
        dashboard = getattr(
            self,
            "experiment_window",
            None,
        )
    
        if dashboard is not None:
    
            try:
                # Requests experiment shutdown and waits
                # for the execution thread to finish.
                closed = dashboard.close()
    
                if not closed:
                    return False
    
            except RuntimeError:
                self.experiment_window = None
    
        # --------------------------------------------------
        # 3. Ensure hardware ownership has been released
        # --------------------------------------------------
    
        try:
            self.hardware_manager.require_idle()
    
        except HardwareBusyError as error:
    
            QMessageBox.warning(
                self,
                "Hardware Busy",
                "Hardware ownership has not been released.\n\n"
                f"{error}",
            )
    
            return False
    
        # --------------------------------------------------
        # 4. Disconnect hardware
        # --------------------------------------------------
    
        if not self.shutdown_hardware():
    
            QMessageBox.warning(
                self,
                "Hardware Shutdown Incomplete",
                "One or more hardware devices could not "
                "be disconnected successfully.\n\n"
                "Check device status and try again.",
            )
    
            return False
    
        # --------------------------------------------------
        # 5. Mark hardware lifecycle as SHUTDOWN
        # --------------------------------------------------
    
        try:
            self.hardware_manager.begin_shutdown()
    
        except HardwareBusyError as error:
    
            QMessageBox.warning(
                self,
                "Hardware Busy",
                str(error),
            )
    
            return False
    
        print(
            "[CONTROL PANEL] "
            "Hardware shutdown completed successfully."
        )
    
        return True

    def closeEvent(self, event):
        """
        Standalone Control Panel closing behavior.
        """
    
        if self.prepare_shutdown():
            event.accept()
        else:
            event.ignore()

    # ============================================================
    # STYLE
    # ============================================================

    def _apply_style(self, dark=True):


        stylesheet = """

            QWidget#controlPanel {
                background: #07111f;
            }
            QWidget {
                background: #07111f;
                color: #e5e7eb;
                font-family: "Segoe UI";
                font-size: 13px;
            }

            QPushButton#new_connection_button {
                background: #2563eb;
                border: 1px solid #3b82f6;
                color: white;
                padding: 10px 18px;
                font-weight: 700;
            }

            QPushButton#new_connection_button:hover {
                background: #1d4ed8;
            }

            QLabel {
                background: transparent;
                border: none;
            }

            QLabel#control_title {
                color: white;
                font-size: 20px;
                font-weight: 700;
                padding: 8px;
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

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 20px;
                top: -2px;
                padding: 0px 6px;
                color: white;
                font-size: 15px;
                font-weight: 700;
            }

            QGroupBox#deviceCard {
                margin-top: 12px;
                padding: 8px;
                border-radius: 8px;
            }
            QGroupBox#deviceCard::title {
                left: 10px;
                font-size: 12px;
            }
            QGroupBox#deviceCard QLineEdit,
            QGroupBox#deviceCard QSpinBox,
            QGroupBox#deviceCard QDoubleSpinBox,
            QGroupBox#deviceCard QComboBox {
                padding: 3px 5px;
                border-radius: 5px;
                font-size: 12px;
            }
            QGroupBox#deviceCard QPushButton {
                padding: 4px 5px;
                font-size: 12px;
                min-width: 0px;
            }

            QLineEdit,
            QSpinBox,
            QDoubleSpinBox,
            QComboBox {
                background: #0b1728;
                border: 1px solid #24364d;
                border-radius: 8px;
                padding: 7px 9px;
                color: white;
            }

            QLineEdit:focus,
            QSpinBox:focus,
            QDoubleSpinBox:focus,
            QComboBox:focus {
                border: 1px solid #3b82f6;
            }

            QComboBox:hover,
            QSpinBox:hover,
            QDoubleSpinBox:hover {
                border: 1px solid #3b82f6;
            }

            QPushButton {
                background: transparent;
                color: #d6dde8;
                border: 1px solid #16263d;
                border-radius: 8px;
                padding: 8px 10px;
                font-weight: 600;
                text-align: center;
            }

            QPushButton:hover {
                background: #162846;
                border: 1px solid #2f5fa7;
                color: white;
            }

            QPushButton:pressed {
                background: #10203a;
            }

            QPushButton#start_button {
                background: #22c55e;
                border: 1px solid #37d774;
                color: white;
                font-weight: 700;
            }

            QPushButton#start_button:hover {
                background: #16a34a;
            }

            QPushButton#stop_button {
                background: #ef4444;
                border: 1px solid #f87171;
                color: white;
                font-weight: 700;
            }

            QPushButton#stop_button:hover {
                background: #dc2626;
            }

            QPushButton#connect_all_button {
                background: #2563eb;
                border: 1px solid #3b82f6;
                color: white;
                padding: 10px 18px;
                font-weight: 700;
            }

            QPushButton#connect_all_button:hover {
                background: #1d4ed8;
            }

            QPushButton#disconnect_all_button {
                background: #ef4444;
                border: 1px solid #f87171;
                color: white;
                padding: 10px 18px;
                font-weight: 700;
            }

            QPushButton#disconnect_all_button:hover {
                background: #dc2626;
            }

            QScrollArea {
                background: transparent;
                border: none;
            }

            QScrollBar:vertical {
                background: transparent;
                width: 10px;
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
        """
        
        if not dark:
            light_colors = {
                "#07111f": "#f8fafc",  # Main background
                "#111d31": "#ffffff",  # Device cards
                "#16263d": "#d8dee8",  # Borders
                "#0b1728": "#ededed",  # Inputs
                "#24364d": "#cbd5e1",  # Input borders
                "#10203a": "#dbeafe",  # Pressed buttons
                "#162846": "#e8f0fe",  # Hover background
                "#2d415d": "#cbd5e1",  # Scrollbar
                "#456182": "#94a3b8",  # Scrollbar hover
                "#e5e7eb": "#1e293b",  # Primary text
                "#d6dde8": "#334155",  # Secondary text
                "#2f5fa7": "#93b4e8",  # Hover borders
            }
    
            for original, replacement in light_colors.items():
                stylesheet = stylesheet.replace(
                    original,
                    replacement,
                )
    
            # Light-mode input controls
            stylesheet += """
                QLineEdit,
                QSpinBox,
                QDoubleSpinBox,
                QComboBox,
                QTextEdit,
                QPlainTextEdit {
                    background-color: #ffffff;
                    color: #000000;
                    selection-background-color: #2563eb;
                    selection-color: #ffffff;
                }
            
                QSpinBox,
                QDoubleSpinBox {
                    border: 1px solid #cbd5e1;
                    border-radius: 7px;
                    padding: 5px 8px;
                }
            """
    
        self.setStyleSheet(stylesheet)


    def set_theme(self, dark=True):
        self._apply_style(dark)

class DeviceLimitsDialog(QDialog):
    def __init__(self, configs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Device Limits — application overrides")
        self.resize(800, 500)
        layout = QVBoxLayout(self)
        description = QLabel("Edit the application limits per device. Minimum nonzero = 0 disables the low-speed gap. "
                             "Overrides do not change a physical instrument's capabilities. Valve states remain 0/1. "
                             "Save Configuration to retain changes.")
        description.setWordWrap(True)
        layout.addWidget(description)
        self.rows = {}
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Device", "Minimum", "Maximum", "Min. nonzero", "Unit"])
        for key, config in configs.items():
            if config["type"] not in DEFAULT_LIMITS:
                continue
            limits = target_limits(config)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(f"{key}: {config['name']}"))
            self.table.setItem(row, 4, QTableWidgetItem(limits["unit"]))
            editors = {}
            for col, field in enumerate(("min", "max", "min_nonzero"), 1):
                editor = QDoubleSpinBox()
                editor.setDecimals(2)
                editor.setRange(0 if field == "min_nonzero" else -1e6, 1e6)
                editor.setValue(limits[field])
                editor.setEnabled(config["type"] != "nidaq_valve")
                self.table.setCellWidget(row, col, editor)
                editors[field] = editor
            self.rows[key] = (config, editors)
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def overrides(self):
        return {key: {field: editor.value() for field, editor in editors.items()}
                for key, (_, editors) in self.rows.items()}

    def validate_and_accept(self):
        try:
            for key, values in self.overrides().items():
                target_limits({**self.rows[key][0], "target_limits": values})
        except ValueError as error:
            QMessageBox.warning(self, "Invalid Limits", str(error))
            return
        self.accept()


class NewDeviceDialog(QDialog):

    def __init__(self, hardware_types, existing_ids=None, parent=None):
        super().__init__(parent)

        self.hardware_types = hardware_types
        self.existing_ids = set(existing_ids or [])

        self.connection_widgets = {}
        self.option_widgets = {}
        self.target_widget = None

        self.setWindowTitle("New Connection")
        self.resize(420, 350)

        main_layout = QVBoxLayout(self)

        form = QFormLayout()

        self.id_edit = QLineEdit()
        self.name_edit = QLineEdit()

        self.type_combo = QComboBox()
        self.type_combo.addItems(self.hardware_types.keys())
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Real hardware", "Simulated"])

        form.addRow("Device ID", self.id_edit)
        form.addRow("Name", self.name_edit)
        form.addRow("Hardware Type", self.type_combo)
        form.addRow("Connection Mode", self.mode_combo)

        main_layout.addLayout(form)

        self.dynamic_widget = QWidget()
        self.dynamic_form = QFormLayout(self.dynamic_widget)

        main_layout.addWidget(self.dynamic_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Ok
        )

        buttons.button(
            QDialogButtonBox.StandardButton.Ok
        ).setText("Add")

        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

        main_layout.addWidget(buttons)

        self.type_combo.currentTextChanged.connect(
            self._build_dynamic_fields
        )

        self._build_dynamic_fields(
            self.type_combo.currentText()
        )

    def _clear_dynamic_form(self):

        while self.dynamic_form.rowCount():
            self.dynamic_form.removeRow(0)

        self.connection_widgets = {}
        self.option_widgets = {}
        self.target_widget = None

    def _build_dynamic_fields(self, device_type):

        self._clear_dynamic_form()

        schema = self.hardware_types[device_type]

        for key, spec in schema.get("connection", {}).items():

            widget = QLineEdit(str(spec.get("default", "")))

            self.connection_widgets[key] = widget

            self.dynamic_form.addRow(
                spec.get("label", key.replace("_", " ").title()),
                widget
            )

        target = schema.get("target")

        if target is not None:

            self.target_widget = self._make_input_widget(target)

            default = target.get("default", 0)

            self._set_widget_value(
                self.target_widget,
                default
            )

            label = f"Default {target['label']}"

            if target.get("unit"):
                label += f" ({target['unit']})"

            self.dynamic_form.addRow(
                label,
                self.target_widget
            )

        for key, spec in schema.get("options", {}).items():

            widget = self._make_input_widget(spec)

            self._set_widget_value(
                widget,
                spec.get("default")
            )

            self.option_widgets[key] = widget

            self.dynamic_form.addRow(
                spec["label"],
                widget
            )

    def _make_input_widget(self, spec):

        field_type = spec["type"]

        if field_type == "float":
            widget = QDoubleSpinBox()
            widget.setRange(
                spec.get("min", -1e9),
                spec.get("max", 1e9)
            )
            widget.setDecimals(
                spec.get("decimals", 2)
            )
            return widget

        if field_type == "int":
            widget = QSpinBox()
            widget.setRange(
                spec.get("min", -1_000_000),
                spec.get("max", 1_000_000)
            )
            return widget

        if field_type == "choice":
            widget = QComboBox()
            widget.addItems(
                [str(x) for x in spec["choices"]]
            )
            return widget

        return QLineEdit()

    def _set_widget_value(self, widget, value):

        if value is None:
            return

        if isinstance(widget, QComboBox):
            widget.setCurrentText(str(value))

        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))

        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))

        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))

    def _widget_value(self, widget):

        if isinstance(widget, QComboBox):
            return widget.currentText()

        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()

        if isinstance(widget, QLineEdit):
            return widget.text()

        return None

    def _validate_and_accept(self):

        device_id = self.id_edit.text().strip()
        name = self.name_edit.text().strip()

        if not device_id:
            self.id_edit.setFocus()
            return

        if device_id in self.existing_ids:
            self.id_edit.setFocus()
            return

        if not name:
            self.name_edit.setFocus()
            return

        self.accept()

    def get_device(self):

        device_type = self.type_combo.currentText()

        config = {
            "name": self.name_edit.text().strip(),
            "type": device_type,
            "simulated": self.mode_combo.currentText() == "Simulated",
            "connection": {
                key: widget.text().strip()
                for key, widget in self.connection_widgets.items()
            }
        }

        if self.target_widget is not None:
            config["default_target"] = self._widget_value(
                self.target_widget
            )

        if self.option_widgets:
            config["defaults"] = {
                key: self._widget_value(widget)
                for key, widget in self.option_widgets.items()
            }

        return self.id_edit.text().strip(), config



