"""Load/preview CSV calibrations and save their points with a pump configuration."""
from copy import deepcopy
from pathlib import Path
import pandas as pd
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QLineEdit, QFileDialog, QMessageBox)
from experiment.pump_calibration_files import load_calibration, OUTPUT_UNITS


class PumpCalibrationDialog(QDialog):
    def __init__(self, configs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pump calibration files")
        self.resize(740, 580)
        self.configs = deepcopy(configs)
        self.candidate = None
        layout = QVBoxLayout(self)
        description = QLabel("Calibrations convert generated mass flow (g/h) to a pump target. "
            "Load a flow/target CSV or a time/target/mass log (mass in grams). "
            "Points are stored with the device and setup preset, so the original file is not needed later. "
            "Manual Control Panel profiles already use device units and are not converted again.")
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        self.device = QComboBox()
        for key, config in configs.items():
            if config["type"] in OUTPUT_UNITS:
                self.device.addItem(f"{key}: {config['name']} ({OUTPUT_UNITS[config['type']]})", key)
        form.addRow("Pump:", self.device)
        self.current = QLabel()
        self.current.setWordWrap(True)
        form.addRow("Current calibration:", self.current)
        path = QHBoxLayout()
        self.file = QLineEdit()
        self.file.setReadOnly(True)
        choose = QPushButton("Load CSV…")
        path.addWidget(self.file, 1)
        path.addWidget(choose)
        form.addRow("Calibration file:", path)
        self.header = QSpinBox()
        self.header.setRange(1, 20)
        form.addRow("CSV header row (first row = 1):", self.header)
        self.mode = QComboBox()
        self.mode.addItem("Flow / pump-target pairs", "pairs")
        self.mode.addItem("Time / pump-target / mass log", "mass_log")
        form.addRow("File format:", self.mode)
        self.flow = QComboBox()
        self.target = QComboBox()
        self.time = QComboBox()
        self.mass = QComboBox()
        self.time_unit = QComboBox()
        for label, value in (("Minutes", "min"), ("Seconds", "s"), ("Hours", "h")):
            self.time_unit.addItem(label, value)
        for label, widget in (("Measured flow (g/h):", self.flow), ("Pump target (device units):", self.target),
                              ("Time column:", self.time), ("Mass column (g):", self.mass), ("Log time unit:", self.time_unit)):
            form.addRow(label, widget)
        layout.addLayout(form)
        self.summary = QLabel("Select a pump and load a calibration CSV.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        actions = QHBoxLayout()
        preview = QPushButton("Preview / validate")
        self.apply_button = QPushButton("Apply calibration")
        self.apply_button.setEnabled(False)
        reset = QPushButton("Use built-in conversion")
        close = QPushButton("Cancel")
        for button in (preview, self.apply_button, reset, close):
            actions.addWidget(button)
        layout.addLayout(actions)
        choose.clicked.connect(self.choose_file)
        self.header.valueChanged.connect(self.read_columns)
        self.device.currentIndexChanged.connect(self.device_changed)
        for combo in (self.mode, self.flow, self.target, self.time, self.mass, self.time_unit):
            combo.currentIndexChanged.connect(self.invalidate)
        self.mode.currentIndexChanged.connect(self.update_mode)
        preview.clicked.connect(self.preview)
        self.apply_button.clicked.connect(self.apply)
        reset.clicked.connect(self.reset_calibration)
        close.clicked.connect(self.reject)
        self.device_changed()
        self.update_mode()

    def invalidate(self, *args):
        self.candidate = None
        self.apply_button.setEnabled(False)

    def device_changed(self, *args):
        config = self.configs.get(self.device.currentData(), {})
        calibration = config.get("pump_calibration")
        self.current.setText(f"{calibration['source_name']} — {calibration['input_unit']} → {calibration['output_unit']}" if calibration else
                             "Built-in role calibration (Masterflex), or g/h ÷ 60 at 1 g/mL (Ismatec).")
        self.invalidate()

    def update_mode(self, *args):
        log = self.mode.currentData() == "mass_log"
        self.flow.setEnabled(not log)
        for widget in (self.time, self.mass, self.time_unit):
            widget.setEnabled(log)

    def choose_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load pump calibration", "", "CSV (*.csv)")
        if not filename:
            return
        self.file.setText(filename)
        try:
            columns = pd.read_csv(filename, nrows=0).columns
            header = 2 if sum(str(c).startswith("Unnamed:") for c in columns) > len(columns) / 2 else 1
            self.header.blockSignals(True)
            self.header.setValue(header)
            self.header.blockSignals(False)
            self.read_columns()
        except Exception as error:
            self.summary.setText(str(error))

    def read_columns(self, *args):
        self.invalidate()
        if not self.file.text():
            return
        try:
            columns = list(map(str, pd.read_csv(self.file.text(), header=self.header.value() - 1, nrows=0).columns))
            for combo in (self.flow, self.target, self.time, self.mass):
                combo.clear()
                combo.addItems(columns)
            def select(combo, words):
                for index, column in enumerate(columns):
                    if any(word in column.lower() for word in words):
                        combo.setCurrentIndex(index)
                        return True
                return False
            select(self.flow, ("g/h", "g_h", "mass_rate", "flow"))
            select(self.target, ("rpm", "target", "setpt", "file  w", "command", "ml/min", "ml_min"))
            has_time = select(self.time, ("time",))
            has_mass = select(self.mass, ("mass", "weight"))
            self.mode.setCurrentIndex(1 if has_time and has_mass else 0)
            self.update_mode()
            self.summary.setText("Check the selected columns and units, then preview the calibration.")
        except Exception as error:
            self.summary.setText(f"Could not read calibration columns: {error}")

    def preview(self):
        try:
            config = self.configs[self.device.currentData()]
            self.candidate = load_calibration(self.file.text(), config["type"], self.flow.currentText(), self.target.currentText(),
                header_row=self.header.value() - 1, mode=self.mode.currentData(), time_column=self.time.currentText(),
                time_unit=self.time_unit.currentData(), mass_column=self.mass.currentText())
            points = self.candidate["points"]
            self.summary.setText(f"Validated {len(points)} calibration points: 0–{points[-1][0]:g} g/h → "
                                 f"0–{points[-1][1]:g} {self.candidate['output_unit']}. "
                                 "Linear interpolation between measured points; flow targets beyond the calibrated range are rejected.")
            self.apply_button.setEnabled(True)
        except Exception as error:
            self.invalidate()
            self.summary.setText(f"Calibration not applied: {error}")

    def apply(self):
        if self.candidate is not None:
            self.configs[self.device.currentData()]["pump_calibration"] = deepcopy(self.candidate)
            self.accept()

    def reset_calibration(self):
        if self.device.currentData() is not None:
            self.configs[self.device.currentData()].pop("pump_calibration", None)
            self.accept()
