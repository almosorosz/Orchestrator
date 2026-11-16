"""Assign generated process roles, inspect targets and launch a frozen campaign."""
import json
from pathlib import Path
from dataclasses import asdict

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
                               QTableWidgetItem, QComboBox, QPushButton,
                               QCheckBox, QFileDialog, QMessageBox, QAbstractItemView)

from experiment.doe_campaign import DoeCampaignService, ProfileAssignment, ROLES, DEFAULT_ASSIGNMENTS
from hardware_devices.capabilities import is_monitor_only


class DoeAssignmentDialog(QDialog):
    def __init__(self, creator, configs, hardware, existing_assignments=(), parent=None, calibration_editor=None):
        super().__init__(parent)
        self.setWindowTitle("Assign and preview generated DoE")
        self.resize(1100, 740)
        self.service = DoeCampaignService(creator)
        self.configs, self.hardware = configs, hardware
        self.campaign = None
        self.calibration_editor = calibration_editor
        existing = {item.role: item for item in existing_assignments}
        layout = QVBoxLayout(self)
        description = QLabel(
            "Assign active profiles (*) to distinct devices. Generated hours become seconds. "
            "Mass-flow profiles use pump calibrations. Ismatec defaults to g/h ÷ 60 → mL/min (1 g/mL). "
            "Existing return-pump segments are retained. Only assigned actuators will be controlled. "
            "Preview, validate and apply assignments. Start execution separately from Settings or the Dashboard."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self.table = QTableWidget(len(ROLES), 3)
        self.table.setHorizontalHeaderLabels(["Generated profile", "Device", "Conversion"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.rows = {}
        self.imed_rows = {}
        required = self.service.required_roles()
        for row, (role, spec) in enumerate(ROLES.items()):
            label = spec.label + (" *" if role in required else " (inactive)")
            self.table.setItem(row, 0, QTableWidgetItem(label))
            combo = QComboBox()
            combo.addItem("Not assigned", None)
            for key, config in configs.items():
                if config["type"] in spec.compatible_types and not is_monitor_only(config):
                    connected = getattr(hardware.get(key), "is_connected", False)
                    mode = "simulated" if config.get("simulated") else "real"
                    state = "connected" if connected else "disconnected"
                    combo.addItem(f"{key}: {config['name']} [{mode}, {state}]", key)
            self.rows[role] = (combo, row)
            self.table.setCellWidget(row, 1, combo)
            preferred = existing[role].device_id if role in existing else DEFAULT_ASSIGNMENTS.get(role)
            index = combo.findData(preferred)
            if index >= 0 and (role in required or role in existing):
                combo.setCurrentIndex(index)
            combo.currentIndexChanged.connect(lambda _, key=role: self.update_conversion(key))
        if self.service.imed_simulations:
            from imed_python.demo_campaign import MONITOR_TYPES
            defaults = next(iter(self.service.imed_simulations.values())).get("monitors", {})
            for role, kind in MONITOR_TYPES.items():
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(f"IMED: {role} *"))
                combo = QComboBox()
                combo.addItem("Not assigned", None)
                for key, config in configs.items():
                    if config.get("simulated") and config["type"] == kind and is_monitor_only(config):
                        combo.addItem(f"{key}: {config['name']}", key)
                combo.setCurrentIndex(max(0, combo.findData(defaults.get(role))))
                combo.currentIndexChanged.connect(self.invalidate)
                self.imed_rows[role] = combo
                self.table.setCellWidget(row, 1, combo)
                self.table.setItem(row, 2, QTableWidgetItem("Recorded channel"))
        layout.addWidget(self.table)
        self.selected_monitor_ids = None
        self.include_monitors = QCheckBox("Include connected read-only monitors")
        self.include_monitors.setChecked(True)
        self.include_monitors.toggled.connect(self.invalidate)
        self.include_monitors.clicked.connect(lambda: setattr(self, "selected_monitor_ids", None))
        layout.addWidget(self.include_monitors)
        self.summary = QLabel("Preview required before execution.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.preview_table = QTableWidget(0, 5)
        self.preview_table.setHorizontalHeaderLabels(["Experiment", "Duration (s)", "Device", "Target range", "Unit / interpolation"])
        self.preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.preview_table)
        actions = QHBoxLayout()
        self.preview_button = QPushButton("Preview / validate")
        self.calibration_button = QPushButton("Pump calibrations…")
        self.calibration_button.setEnabled(calibration_editor is not None)
        self.calibration_button.clicked.connect(self.edit_calibrations)
        self.save_button = QPushButton("Save assignments")
        self.load_button = QPushButton("Load assignments")
        self.export_button = QPushButton("Export campaign CSV")
        self.start_button = QPushButton("Apply assignments")
        self.cancel_button = QPushButton("Close")
        for button in (self.preview_button, self.calibration_button, self.save_button, self.load_button, self.export_button, self.start_button, self.cancel_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.preview_button.clicked.connect(self.preview)
        self.save_button.clicked.connect(self.save_assignments)
        self.load_button.clicked.connect(self.load_assignments)
        self.export_button.clicked.connect(self.export_campaign)
        self.start_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        for role in self.rows:
            self.update_conversion(role)
        self.table.resizeColumnsToContents()
        self.invalidate()

    def invalidate(self, *args):
        self.campaign = None
        if hasattr(self, "start_button"):
            self.start_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.summary.setText("Assignments changed. Preview and validate before execution.")
            self.preview_table.setRowCount(0)

    def update_conversion(self, role):
        combo, row = self.rows[role]
        device_id = combo.currentData()
        kind = self.configs[device_id]["type"] if device_id is not None else None
        text = "Not assigned"
        if device_id is not None:
            custom = self.configs[device_id].get("pump_calibration")
            text = (f"Loaded calibration: {custom.get('source_name', 'CSV')}" if custom and ROLES[role].unit == "g/h" else
                    "g/h ÷ 60 → mL/min (1 g/mL)" if kind == "ismatec" and ROLES[role].unit == "g/h" else
                    f"{role} calibration: g/h → RPM" if ROLES[role].unit == "g/h" else
                    f"Direct ({ROLES[role].unit})")
        self.table.setItem(row, 2, QTableWidgetItem(text))
        self.invalidate()

    def assignments(self):
        result = []
        for role, (combo, _) in self.rows.items():
            device_id = combo.currentData()
            if device_id is None:
                continue
            result.append(ProfileAssignment(role, device_id))
        return tuple(result)

    def imed_monitor_assignments(self):
        return {role: box.currentData() for role, box in self.imed_rows.items()
                if box.currentData() is not None}

    def edit_calibrations(self):
        if self.calibration_editor is not None:
            updated = self.calibration_editor()
            if updated is not None:
                self.configs = updated
                for role in self.rows:
                    self.update_conversion(role)

    def preview(self):
        try:
            monitors = tuple(key for key, config in self.configs.items()
                             if self.include_monitors.isChecked() and is_monitor_only(config)
                             and (self.selected_monitor_ids is None or key in self.selected_monitor_ids)
                             and getattr(self.hardware.get(key), "is_connected", False))
            campaign = self.service.compile(self.assignments(), self.configs, monitors,
                                            imed_monitors=self.imed_monitor_assignments())
            self.preview_table.setRowCount(0)
            total_duration = 0
            for experiment in campaign.experiments:
                duration = max(profile.points[-1].time_s for profile in experiment.profiles)
                total_duration += duration
                for profile in experiment.profiles:
                    row = self.preview_table.rowCount()
                    self.preview_table.insertRow(row)
                    values = [point.target for point in profile.points]
                    cells = (experiment.name, f"{duration:.2f}", profile.device_id,
                             f"{min(values):g} … {max(values):g}", f"{profile.unit} / {profile.interpolation}")
                    for col, text in enumerate(cells):
                        self.preview_table.setItem(row, col, QTableWidgetItem(text))
            self.preview_table.resizeColumnsToContents()
            self.campaign = campaign
            disconnected = [key for key in {a.device_id for a in self.assignments()}
                            if not getattr(self.hardware.get(key), "is_connected", False)]
            self.summary.setText(f"Validated {len(campaign.experiments)} experiments; {total_duration:g} s planned "
                                 f"({total_duration / 3600:.2f} h), plus 5 s final hold per experiment. "
                                 + ("Connect before starting: " + ", ".join(disconnected) if disconnected else "Assigned devices are connected."))
            self.start_button.setEnabled(True)
            self.export_button.setEnabled(True)
        except (ValueError, KeyError, TypeError) as error:
            self.invalidate()
            self.summary.setText(str(error))

    def accept(self):
        if self.campaign is not None and self.start_button.isEnabled():
            super().accept()

    def export_campaign(self):
        if self.campaign is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Export campaign CSV to a new subfolder")
        if folder:
            try:
                result = self.service.export_csv(folder, self.campaign, self.assignments(), self.configs)
                self.summary.setText(f"Exported planned campaign CSV files to {result}")
            except Exception as error:
                QMessageBox.warning(self, "Export Failed", str(error))

    def save_assignments(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save DoE assignments", "doe_assignments.json", "JSON (*.json)")
        if filename:
            try:
                payload = {"version": 2, "assignments": [asdict(item) for item in self.assignments()],
                           "include_monitors": self.include_monitors.isChecked(),
                           "monitor_device_ids": self.selected_monitor_ids}
                if self.imed_rows:
                    payload["imed_monitors"] = self.imed_monitor_assignments()
                from experiment.recording import atomic_json

                atomic_json(Path(filename), payload)
            except Exception as error:
                QMessageBox.warning(self, "Save Failed", str(error))

    def load_assignments(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load DoE assignments", "", "JSON (*.json)")
        if not filename:
            return
        try:
            payload = json.loads(Path(filename).read_text(encoding="utf-8"))
            if payload.get("version") not in (1, 2):
                raise ValueError("Unsupported assignment file version")
            loaded = tuple(ProfileAssignment(**item) for item in payload["assignments"])
            if len({item.role for item in loaded}) != len(loaded):
                raise ValueError("Duplicate profile roles in assignment file")
            for item in loaded:
                if item.role not in self.rows or self.rows[item.role][0].findData(item.device_id) < 0:
                    raise ValueError(f"Missing or incompatible device for {item.role}: {item.device_id}")
            monitored = payload.get("imed_monitors", {})
            for role, device in monitored.items():
                if role not in self.imed_rows or self.imed_rows[role].findData(device) < 0:
                    raise ValueError(f"Missing or incompatible IMED monitor for {role}: {device}")
            saved_ids = payload.get("monitor_device_ids")
            include = payload.get("include_monitors", True)
            if type(include) is not bool or (saved_ids is not None and (
                    not isinstance(saved_ids, list) or any(
                        not isinstance(key, str) or not is_monitor_only(self.configs.get(key, {}))
                        for key in saved_ids))):
                raise ValueError("The assignment file contains invalid monitor settings.")
            self.selected_monitor_ids = saved_ids
            self.include_monitors.setChecked(include)
            self.invalidate()
            for combo, _ in self.rows.values():
                combo.setCurrentIndex(0)
            for item in loaded:
                combo, _ = self.rows[item.role]
                combo.setCurrentIndex(combo.findData(item.device_id))
            for role, combo in self.imed_rows.items():
                combo.setCurrentIndex(max(0, combo.findData(monitored.get(role))))
        except Exception as error:
            QMessageBox.warning(self, "Load Failed", str(error))
