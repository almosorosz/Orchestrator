"""Load/preview CSV calibrations and save their points with a pump configuration."""
from copy import deepcopy
from pathlib import Path
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QFileDialog, QTabWidget, QWidget)
from experiment.pump_calibration_files import delivered_flow, fit_liquid_flow, load_calibration, load_seed_calibration, OUTPUT_UNITS
from experiment.pump_profile_data import declared_time_unit
from experiment.pump_corrections import validate_corrections
from hardware_devices.limits import target_limits


def number_box(default=0.):
    box = QDoubleSpinBox()
    box.setDecimals(6)
    box.setRange(0., 1000000.)
    box.setValue(default)
    return box


class PumpCalibrationDialog(QDialog):
    def __init__(self, configs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pump calibrations and tubing corrections")
        self.resize(780, 630)
        self.configs = deepcopy(configs)
        self.candidate = None
        layout = QVBoxLayout(self)
        description = QLabel("Load a liquid mass-flow calibration or a dry seed dose calibration. "
            "Both curves and the tubing settings are saved with the pump. Use a filled line when measuring a seed dose "
            "if you will also enable the additional line-fill amount. Tubing corrections run once per active pump in each experiment.")
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
        self.mode.addItem("Seed dose at fixed RPM", "seed_dose")
        form.addRow("File format:", self.mode)
        self.flow = QComboBox()
        self.flow_unit = QComboBox()
        self.flow_unit.addItems(("g/h", "g/min", "g/s"))
        self.target = QComboBox()
        self.time = QComboBox()
        self.mass = QComboBox()
        self.time_unit = QComboBox()
        self.seed_rpm = number_box(50.)
        for label, value in (("Minutes", "min"), ("Seconds", "s"), ("Hours", "h")):
            self.time_unit.addItem(label, value)
        for label, widget in (("Measured flow column:", self.flow), ("Measured flow units:", self.flow_unit),
                              ("Pump target (device units):", self.target), ("Time column:", self.time),
                              ("Mass column (g):", self.mass), ("Time units:", self.time_unit), ("Fixed seed speed (RPM):", self.seed_rpm)):
            form.addRow(label, widget)
        self.calibration_form = form
        layout.addLayout(form)
        corrections = QFormLayout()
        self.prime = QCheckBox("Add a fixed line-fill amount (g)")
        self.prime_amount = number_box()
        corrections.addRow(self.prime, self.prime_amount)
        self.reverse = QCheckBox("Back rotation after delivery")
        corrections.addRow(self.reverse)
        self.reverse_speed = number_box()
        self.reverse_time = number_box(1.)
        self.reverse_label = QLabel()
        corrections.addRow(self.reverse_label, self.reverse_speed)
        corrections.addRow("Back-rotation time (min)", self.reverse_time)
        layout.addLayout(corrections)
        self.prime.toggled.connect(self.update_correction_fields)
        self.reverse.toggled.connect(self.update_correction_fields)
        self.summary = QLabel("Select a pump and load a calibration CSV.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        inspection = QHBoxLayout()
        preview = QPushButton("Preview / validate")
        self.plot_button = QPushButton("View calibration plot")
        self.plot_button.setToolTip("View a validated CSV preview or a saved calibration for this pump.")
        inspection.addWidget(preview)
        inspection.addWidget(self.plot_button)
        define_profile = QPushButton("Define pump profile…")
        define_profile.setEnabled(self.device.count() > 0)
        define_profile.clicked.connect(self.define_profile)
        inspection.addWidget(define_profile)
        inspection.addStretch()
        layout.addLayout(inspection)
        actions = QHBoxLayout()
        self.apply_button = QPushButton("Apply calibration")
        self.apply_button.setEnabled(False)
        save = QPushButton("Save tubing settings")
        reset = QPushButton("Remove selected calibration")
        close = QPushButton("Cancel")
        for button in (self.apply_button, save, reset, close):
            actions.addWidget(button)
        layout.addLayout(actions)
        choose.clicked.connect(self.choose_file)
        self.header.valueChanged.connect(self.read_columns)
        self.device.currentIndexChanged.connect(self.device_changed)
        for combo in (self.mode, self.flow, self.flow_unit, self.target, self.time, self.mass, self.time_unit):
            combo.currentIndexChanged.connect(self.invalidate)
        self.seed_rpm.valueChanged.connect(self.invalidate)
        self.mode.currentIndexChanged.connect(self.update_mode)
        preview.clicked.connect(self.preview)
        self.plot_button.clicked.connect(self.show_calibration_plot)
        self.apply_button.clicked.connect(self.apply)
        save.clicked.connect(self.save_corrections)
        reset.clicked.connect(self.reset_calibration)
        close.clicked.connect(self.reject)
        self.device_changed()
        self.update_mode()

    def invalidate(self, *args):
        self.candidate = None
        self.apply_button.setEnabled(False)
        config = self.configs.get(self.device.currentData(), {})
        self.plot_button.setEnabled(bool(config.get("pump_calibration") or config.get("seed_calibration")))
        self.update_reverse_unit(config)

    def update_reverse_unit(self, config, *, reset=True):
        unit = target_limits(config)["unit"] if config else ""
        if reset and getattr(self, "reverse_unit", unit) != unit:
            self.reverse_speed.setValue(0.)
        self.reverse_unit = unit
        self.reverse_label.setText(f"Reverse target ({unit})")

    def device_changed(self, *args):
        config = self.configs.get(self.device.currentData(), {})
        calibration = config.get("pump_calibration")
        seed = config.get("seed_calibration")
        liquid_text = (f"Liquid: {calibration['source_name']} · g/h → {calibration['output_unit']}" if calibration else
                       "Liquid: no calibration file loaded")
        seed_text = f"\nDry seed: {seed['source_name']} · {seed['rpm']:g} RPM" if seed else ""
        self.current.setText(liquid_text + seed_text)
        settings = config.get("pump_corrections", {})
        self.prime.setChecked("prime_g" in settings)
        self.prime_amount.setValue(settings.get("prime_g", 0.))
        self.reverse.setChecked("reverse_target" in settings)
        self.reverse_speed.setValue(settings.get("reverse_target", 0.))
        self.reverse_time.setValue(settings.get("reverse_s", 60.) / 60.)
        self.update_reverse_unit(config, reset=False)
        self.seed_rpm.setValue(seed["rpm"] if seed else 50.)
        self.update_correction_fields()
        self.invalidate()

    def update_mode(self, *args):
        mode = self.mode.currentData()
        pairs = mode == "pairs"
        for widget, visible in ((self.flow, pairs), (self.flow_unit, pairs), (self.target, mode != "seed_dose"),
                                (self.time, not pairs), (self.mass, not pairs), (self.time_unit, not pairs),
                                (self.seed_rpm, mode == "seed_dose")):
            self.calibration_form.setRowVisible(widget, visible)

    def update_correction_fields(self):
        self.prime_amount.setEnabled(self.prime.isChecked())
        self.reverse_speed.setEnabled(self.reverse.isChecked())
        self.reverse_time.setEnabled(self.reverse.isChecked())

    def choose_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load pump calibration", "", "CSV (*.csv)")
        if not filename:
            return
        self.invalidate()
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
            select(self.flow, ("g/h", "g_h", "g/min", "g/s", "mass_rate", "flow"))
            for unit in ("g/min", "g/s", "g/h"):
                if unit in self.flow.currentText().lower():
                    self.flow_unit.setCurrentText(unit)
                    break
            has_target = select(self.target, ("rpm", "target", "setpt", "file  w", "command", "ml/min", "ml_min"))
            self.seed_target_column = self.target.currentText() if has_target else None
            has_time = select(self.time, ("time", "duration"))
            has_mass = select(self.mass, ("mass", "weight", "dry", "dose"))
            if has_time:
                row = pd.read_csv(self.file.text(), header=self.header.value() - 1, nrows=1)
                column = self.time.currentText()
                unit = declared_time_unit(column, row[column].iloc[0] if len(row) else "")
                if unit is not None:
                    self.time_unit.setCurrentIndex(self.time_unit.findData(unit))
            config = self.configs.get(self.device.currentData(), {})
            if has_time and has_mass and config.get("type") == "masterflex" and "seed" in Path(self.file.text()).stem.lower():
                self.mode.setCurrentIndex(self.mode.findData("seed_dose"))
            elif self.mode.currentData() != "seed_dose":
                self.mode.setCurrentIndex(1 if has_time and has_mass else 0)
            self.update_mode()
            self.summary.setText("Check the selected columns and units, then preview the calibration.")
        except Exception as error:
            self.summary.setText(f"Could not read calibration columns: {error}")

    def preview(self):
        if self.file.text() and self.mode.currentData() == "mass_log":
            self.define_profile()
            return
        try:
            config = self.configs[self.device.currentData()]
            if self.mode.currentData() == "seed_dose":
                self.candidate = load_seed_calibration(self.file.text(), config, self.time.currentText(), self.mass.currentText(),
                    self.seed_rpm.value(), header_row=self.header.value() - 1, time_unit=self.time_unit.currentData(),
                    target_column=self.seed_target_column, fit_exponential=True)
            else:
                self.candidate = load_calibration(self.file.text(), config["type"], self.flow.currentText(), self.target.currentText(),
                    header_row=self.header.value() - 1, flow_unit=self.flow_unit.currentText())
                fit_liquid_flow(self.candidate, config["type"], config.get("reservoir_density_g_ml"))
            points = self.candidate["points"]
            if "rpm" in self.candidate:
                self.summary.setText(f"Exponential fit at {self.candidate['rpm']:g} RPM. "
                                     f"Measured through {points[-1][1]:g} g at {points[-1][0]:g} s.")
            else:
                maximum = delivered_flow({**config, "pump_calibration": self.candidate}, points[-1][1])
                self.summary.setText(f"Fitted 0–{maximum:g} g/h → 0–{points[-1][1]:g} {self.candidate['output_unit']}.")
            self.apply_button.setEnabled(True)
            self.plot_button.setEnabled(True)
            if "output_unit" in self.candidate:
                self.update_reverse_unit({**config, "pump_calibration": self.candidate})
        except Exception as error:
            self.invalidate()
            self.summary.setText(f"Calibration not applied: {error}")

    def show_calibration_plot(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from gui.pump_profile_dialog import delivery_figure, show_equations

        device = self.device.currentData()
        config = self.configs.get(device, {})
        curves = {name: config[key] for name, key in (("Liquid flow", "pump_calibration"), ("Dry seed", "seed_calibration"))
                  if config.get(key)}
        selected = "Dry seed" if self.mode.currentData() == "seed_dose" else "Liquid flow"
        if self.candidate is not None:
            curves[selected] = self.candidate
        if not curves:
            return

        dialog = QDialog(self)
        dialog.setObjectName("pumpCalibrationPlot")
        dialog.setWindowTitle(f"Pump calibration · {device}")
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.resize(640, 440)
        layout = QVBoxLayout(dialog)
        tabs = QTabWidget() if len(curves) > 1 or any(curve.get("dose_curves") for curve in curves.values()) else None
        for name, calibration in curves.items():
            page = QWidget() if tabs is not None else dialog
            plot_layout = QVBoxLayout(page) if tabs is not None else layout
            state = "Preview" if calibration is self.candidate else "Saved"
            caption = QLabel(f"{state} · {calibration['source_name']}")
            caption.setTextFormat(Qt.TextFormat.PlainText)
            caption.setWordWrap(True)
            plot_layout.addWidget(caption)
            plot_layout.addWidget(FigureCanvasQTAgg(calibration_figure(calibration)), 1)
            if tabs is not None:
                index = tabs.addTab(page, name)
                if name == selected:
                    tabs.setCurrentIndex(index)
                if calibration.get("dose_curves"):
                    dose_page = QWidget()
                    dose_layout = QVBoxLayout(dose_page)
                    dose_layout.addWidget(FigureCanvasQTAgg(delivery_figure(calibration["dose_curves"], calibration["output_unit"])))
                    index = tabs.addTab(dose_page, "Delivery over time")
                    if name == selected:
                        tabs.setCurrentIndex(index)
        if tabs is not None:
            layout.addWidget(tabs, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        equations = buttons.addButton("Show equations…", QDialogButtonBox.ButtonRole.ActionRole)
        equations.clicked.connect(lambda: show_equations(
            curves[tabs.tabText(tabs.currentIndex())] if tabs is not None and tabs.tabText(tabs.currentIndex()) in curves
            else curves.get("Liquid flow", next(iter(curves.values()))), dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.open()

    def define_profile(self):
        from gui.pump_profile_dialog import PumpProfileDialog

        device = self.device.currentData()
        if device is None:
            return
        dialog = PumpProfileDialog(device, self.configs[device], self)
        if self.file.text() and self.mode.currentData() == "mass_log":
            dialog.load_csv(self.file.text(), header_row=self.header.value() - 1,
                            columns=[self.time.currentText(), self.target.currentText(), self.mass.currentText()],
                            time_unit=self.time_unit.currentData())
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.calibration is not None:
            definition = dialog.calibration
            self.file.setText(definition["source_file"])
            self.header.setValue(definition["header_row"] + 1)
            self.read_columns()
            self.time_unit.setCurrentIndex(self.time_unit.findData(definition["time_unit"]))
            self.mode.setCurrentIndex(self.mode.findData("seed_dose" if "rpm" in definition else "mass_log"))
            for column, combo in (("time", self.time), ("target", self.target), ("mass", self.mass)):
                if definition["columns"].get(column) is not None:
                    combo.setCurrentText(definition["columns"][column])
            if "rpm" in definition:
                self.seed_rpm.setValue(definition["rpm"])
            self.candidate = dialog.calibration
            if "output_unit" in definition:
                self.update_reverse_unit({**self.configs[device], "pump_calibration": definition})
            self.summary.setText(f"Profile ready from {self.candidate['source_name']}. Apply calibration to save it for {device}.")
            self.apply_button.setEnabled(True)
            self.plot_button.setEnabled(True)
        dialog.deleteLater()

    def apply(self):
        if self.candidate is not None:
            field = "seed_calibration" if "rpm" in self.candidate else "pump_calibration"
            if field == "pump_calibration" and self.candidate.get("output_unit") != "RPM":
                self.summary.setText("Apply a calibration with measured RPM settings. Use the plot to review volumetric data.")
                return
            self.save_corrections(calibration=(field, deepcopy(self.candidate)))

    def save_corrections(self, *, calibration=None):
        key = self.device.currentData()
        if key is None:
            return
        config = dict(self.configs[key])
        if calibration:
            config[calibration[0]] = calibration[1]
        settings = {}
        if self.prime.isChecked():
            settings["prime_g"] = self.prime_amount.value()
        if self.reverse.isChecked():
            if self.reverse_unit != target_limits(config)["unit"]:
                self.summary.setText("Apply the calibration to save back rotation in the new command units.")
                return
            settings.update(reverse_target=self.reverse_speed.value(), reverse_s=self.reverse_time.value() * 60.)
        try:
            validate_corrections(settings, config)
        except ValueError as error:
            self.summary.setText(str(error))
            return
        if calibration is None or settings or "pump_corrections" in config:
            config["pump_corrections"] = settings
        self.configs[key] = config
        self.accept()

    def reset_calibration(self):
        if self.device.currentData() is not None:
            config = self.configs[self.device.currentData()]
            field = "seed_calibration" if self.mode.currentData() == "seed_dose" else "pump_calibration"
            original = config.pop(field, None)
            try:
                validate_corrections(config.get("pump_corrections", {}), config)
            except ValueError as error:
                if original:
                    config[field] = original
                self.summary.setText(str(error))
                return
            self.accept()


def calibration_figure(calibration):
    import numpy as np
    from matplotlib.figure import Figure
    from experiment.pump_curve_fits import seed_exponential_mass

    points = np.asarray(calibration["points"], dtype=float)
    added_origin = calibration.get("origin_added", False)
    if "rpm" in calibration:
        unit = calibration.get("time_unit") or "s"
        scale = {"s": 1., "min": 60., "h": 3600.}[unit]
        x, y = points[:, 0] / scale, points[:, 1]
        xlabel = f"Pumping time ({unit})"
        ylabel = f"Dry seed mass at {calibration['rpm']:g} RPM (g)"
    else:
        x, y = points[:, 1], points[:, 0]
        xlabel = f"Pump command ({calibration['output_unit']})"
        ylabel = "Delivered mass flow (g/h)"

    observed_x, observed_y = x[int(added_origin):], y[int(added_origin):]
    observation_label = "Measured samples" if "origin_added" in calibration else "Saved calibration points"
    if rates := calibration.get("plateau_rates"):
        observed_y, observed_x = np.asarray(rates, dtype=float).T
        observation_label = "Rates from mass log"

    figure = Figure(figsize=(6.2, 3.6), layout="constrained", facecolor="#f8fafc")
    axis = figure.add_subplot()
    axis.set_facecolor("white")
    curve_x, curve_y = x, y
    if model := calibration.get("seed_fit"):
        seconds = np.unique(np.r_[np.linspace(0., points[-1, 0], 128), model["delay_s"]])
        curve_x, curve_y = seconds / scale, seed_exponential_mass(model, seconds)
    elif "flow_fit" in calibration:
        curve_x = np.linspace(0., x[-1], 128)
        b, c = calibration["flow_fit"]
        curve_y = curve_x * (b + c*curve_x)
    if "rpm" not in calibration and (density := calibration.get("density_g_ml")):
        curve_y = curve_y / (60.*density)
        observed_y = observed_y / (60.*density)
        ylabel = "Delivered volume flow (mL/min)"
    axis.plot(curve_x, curve_y, color="#1d4ed8", linewidth=2, label="Conversion curve")
    axis.scatter(observed_x, observed_y, s=36, facecolors="white", edgecolors="#0f766e",
                 linewidths=1.6, label=observation_label, zorder=3)
    if added_origin:
        axis.scatter([0.], [0.], s=32, marker="D", facecolors="white", edgecolors="#64748b",
                     label="Added zero point", zorder=3)
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.tick_params(colors="#475569", labelsize=9)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(color="#e2e8f0", linewidth=.75)
    axis.legend(loc="best", frameon=True, framealpha=.95, edgecolor="#e2e8f0", fontsize=9)
    return figure
