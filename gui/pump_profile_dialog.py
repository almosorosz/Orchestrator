"""Review elapsed-time delivery curves before applying a pump calibration."""
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from app.paths import PUMP_CALIBRATION_DIR
from experiment.pump_calibration_files import load_seed_calibration, seed_dose_command
from experiment.pump_profile_data import (
    balance_plateaus, calibration_equations, dose_duration, liquid_profile_calibration, read_controller_target, read_profile_log,
    suggested_plateaus,
)


def delivery_figure(curves, unit):
    figure = Figure(figsize=(7., 3.4), layout="constrained", facecolor="#f8fafc")
    axis = figure.add_subplot()
    axis.set_facecolor("white")
    for index, run in enumerate(curves):
        minutes, mass = np.asarray(run["observed"]).T
        minutes = minutes / 60.
        included = run["included"]
        color = f"C{index % 10}" if included else "#64748b"
        label = f"{run['target']:g} {unit}" + (" · excluded" if not included else "")
        axis.scatter(minutes, mass, color=color, s=9, alpha=.42 if included else .3)
        end = run["duration_s"] / 60.
        time = np.linspace(0., end, 64)
        b, c = run.get("mass_time_fit", (run["rate_g_h"] / 60., 0.))
        axis.plot(time, time * (b + c*time), color=color, linewidth=1.8,
                  linestyle="-" if included else "--", label=label)
    axis.set(xlabel="Elapsed pumping time (min)", ylabel="Delivered mass (g)")
    axis.set_axisbelow(True)
    axis.grid(color="#e2e8f0")
    if curves:
        axis.legend(ncol=2 if len(curves) > 4 else 1, fontsize=8, loc="best")
    return figure


def number_entry():
    field = QDoubleSpinBox()
    field.setRange(0., 1e6)
    field.setDecimals(5)
    return field


def show_equations(calibration, parent):
    dialog = QDialog(parent)
    dialog.setWindowTitle("Pump calibration equations")
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.resize(635, 395)
    layout = QVBoxLayout(dialog)
    content = calibration_equations(calibration)
    editor = QPlainTextEdit(content)
    editor.setReadOnly(True)
    layout.addWidget(editor, 1)
    status = QLabel()
    status.setWordWrap(True)
    layout.addWidget(status)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    export = buttons.addButton("Save equations…", QDialogButtonBox.ButtonRole.ActionRole)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    def save():
        name = Path(calibration["source_name"]).stem + "_equations.txt"
        path, _ = QFileDialog.getSaveFileName(dialog, "Save calibration equations", name, "Text (*.txt)")
        if path:
            try:
                Path(path).write_text(content, encoding="utf-8")
            except OSError as error:
                status.setText(str(error))
            else:
                status.setText("Equations saved.")

    export.clicked.connect(save)
    dialog.open()


class PumpProfileDialog(QDialog):
    def __init__(self, device, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Define pump profile · {device}")
        available = self.screen().availableGeometry()
        self.resize(min(900, available.width() - 60), min(795, available.height() - 65))
        self.config = config
        self.log = None
        self.plateaus = ()
        self.calibration = None
        self.canvas = None
        layout = QVBoxLayout(self)
        files = QHBoxLayout()
        load = QPushButton("Load calibration CSV…")
        load.clicked.connect(self.choose_csv)
        controller = QPushButton("Load controller setpoint…")
        controller.setVisible(config["type"] == "ismatec")
        controller.clicked.connect(self.choose_setpoint)
        self.filename = QLabel("Select a timed mass measurement CSV.")
        self.filename.setTextFormat(Qt.TextFormat.PlainText)
        files.addWidget(load)
        files.addWidget(controller)
        files.addWidget(self.filename, 1)
        layout.addLayout(files)
        form = QFormLayout()
        self.kind = QComboBox()
        self.kind.addItem("Liquid delivery log", "liquid")
        if config["type"] == "masterflex":
            self.kind.addItem("Dry seed measurements", "seed")
        form.addRow("Measurements", self.kind)
        self.time_unit = QComboBox()
        for label, unit in (("Seconds", "s"), ("Minutes", "min"), ("Hours", "h")):
            self.time_unit.addItem(label, unit)
        form.addRow("CSV time units", self.time_unit)
        self.command_unit = QComboBox()
        if config["type"] == "ismatec":
            self.command_unit.addItem("Confirm the units used on the pump", "")
            self.command_unit.addItem("RPM · confirm the logged numbers are RPM", "RPM")
            self.command_unit.addItem("mL/min · plot and dose preview only", "mL/min")
        else:
            self.command_unit.addItem("RPM", "RPM")
        form.addRow("Recorded pump command", self.command_unit)
        self.density = number_entry()
        self.density.setSpecialValueText("Optional · show mL/min")
        self.density.setValue(config.get("pump_calibration", {}).get("density_g_ml") or config.get("reservoir_density_g_ml") or 0.)
        form.addRow("Liquid density (g/mL)", self.density)
        self.rpm = QComboBox()
        form.addRow("Fixed seed speed", self.rpm)
        self.reset_clock = QCheckBox("Use the first sample as elapsed time zero")
        form.addRow(self.reset_clock)
        self.fields = form
        layout.addLayout(form)
        self.runs = QTableWidget(0, 4)
        self.runs.setHorizontalHeaderLabels(["Use", "Pump setting", "Duration (min)", "Delivery (g/min)"])
        self.runs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.runs.verticalHeader().hide()
        self.runs.setFixedHeight(132)
        layout.addWidget(self.runs)
        self.plot = QVBoxLayout()
        layout.addLayout(self.plot, 1)
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        calculator = QHBoxLayout()
        self.target_label = QLabel("Pump setting")
        self.target = number_entry()
        self.mass = number_entry()
        self.duration = QLabel("—")
        calculator.addWidget(self.target_label)
        calculator.addWidget(self.target)
        calculator.addWidget(QLabel("Dose (g)"))
        calculator.addWidget(self.mass)
        calculator.addWidget(self.duration, 1)
        layout.addLayout(calculator)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.use_button = buttons.addButton("Use calibration", QDialogButtonBox.ButtonRole.AcceptRole)
        self.equations_button = buttons.addButton("Show equations…", QDialogButtonBox.ButtonRole.ActionRole)
        self.equations_button.clicked.connect(lambda: show_equations(self.calibration, self))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.kind.currentIndexChanged.connect(self.refresh_runs)
        self.time_unit.currentIndexChanged.connect(self.refresh_runs)
        self.command_unit.currentIndexChanged.connect(self.refresh)
        self.density.valueChanged.connect(self.refresh)
        self.rpm.currentIndexChanged.connect(self.refresh)
        self.reset_clock.toggled.connect(self.refresh)
        self.runs.itemChanged.connect(self.refresh)
        self.target.valueChanged.connect(self.preview_duration)
        self.mass.valueChanged.connect(self.preview_duration)
        self.refresh_runs()

    def choose_csv(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Timed pump calibration", str(PUMP_CALIBRATION_DIR), "CSV (*.csv)")
        if filename:
            self.load_csv(filename)

    def load_csv(self, filename, *, header_row=None, columns=None, time_unit=None):
        try:
            self.log = read_profile_log(filename, header_row=header_row, columns=columns)
        except (ValueError, OSError) as error:
            self.status.setText(str(error))
            return
        self.filename.setText(self.log["path"].name)
        with (QSignalBlocker(self.time_unit), QSignalBlocker(self.kind), QSignalBlocker(self.command_unit),
              QSignalBlocker(self.rpm), QSignalBlocker(self.reset_clock)):
            self.time_unit.setCurrentIndex(self.time_unit.findData(time_unit or self.log["time_unit"]))
            seed = columns is None and "seed" in self.log["path"].stem.lower() and self.kind.findData("seed") >= 0
            self.kind.setCurrentIndex(self.kind.findData("seed" if seed else "liquid"))
            self.reset_clock.setChecked(False)
            if self.config["type"] == "ismatec":
                self.command_unit.setCurrentIndex(self.command_unit.findData("RPM") if self.log["declared_unit"] == "RPM" else 0)
            self.rpm.clear()
            for speed in np.unique(self.log["data"][:, 1]):
                if speed > 0:
                    self.rpm.addItem(f"{speed:g} RPM", float(speed))
        self.refresh_runs()

    def choose_setpoint(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Controller setpoint", "", "Text (*.txt)")
        if not filename:
            return
        try:
            value = read_controller_target(filename)
            if value < 0:
                raise ValueError("A dispensing preview needs a nonnegative setpoint.")
            self.target.setValue(value)
            self.preview_duration()
            self.status.setText(f"Controller setpoint {value:g} loaded for the dose preview. Confirm its units against the pump log.")
        except (ValueError, OSError) as error:
            self.status.setText(str(error))

    def refresh_runs(self, *_):
        seed = self.kind.currentData() == "seed"
        self.fields.setRowVisible(self.rpm, seed)
        self.fields.setRowVisible(self.reset_clock, seed)
        self.fields.setRowVisible(self.command_unit, not seed and self.config["type"] == "ismatec")
        self.fields.setRowVisible(self.density, not seed)
        self.runs.setVisible(not seed and bool(self.log))
        self.target.setVisible(not seed)
        self.target_label.setVisible(not seed)
        self.plateaus = ()
        if self.log and not seed:
            try:
                self.plateaus = balance_plateaus(self.log, self.time_unit.currentData())
            except ValueError as error:
                self.refresh()
                self.status.setText(str(error))
                return
        with QSignalBlocker(self.runs):
            selected = set(suggested_plateaus(self.plateaus))
            self.runs.setRowCount(len(self.plateaus))
            for row, run in enumerate(self.plateaus):
                check = QTableWidgetItem()
                check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                check.setCheckState(Qt.CheckState.Checked if row in selected else Qt.CheckState.Unchecked)
                self.runs.setItem(row, 0, check)
                for column, value in enumerate((run["target"], run["duration_s"] / 60., run["rate_g_h"] / 60.), 1):
                    item = QTableWidgetItem(f"{value:.5g}")
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    self.runs.setItem(row, column, item)
            if selected and self.target.value() == 0:
                self.target.setValue(self.plateaus[min(selected)]["target"])
        self.refresh()

    def refresh(self, *_):
        from gui.pump_calibration_dialog import calibration_figure

        seed = self.kind.currentData() == "seed"
        unit = self.command_unit.currentData()
        self.calibration = None
        figure = None
        notice = ""
        if self.log:
            try:
                if seed:
                    time, target, mass = self.log["columns"]
                    if self.rpm.currentData() is None:
                        raise ValueError("Choose the fixed RPM used for the dry-seed measurements.")
                    self.calibration = load_seed_calibration(self.log["path"], self.config, time, mass, self.rpm.currentData(),
                        header_row=self.log["header"], time_unit=self.time_unit.currentData(), target_column=target,
                        reset_clock=self.reset_clock.isChecked(), fit_exponential=True)
                    figure = calibration_figure(self.calibration)
                    points = self.calibration["points"]
                    empty_until = max(time for time, mass in points if mass == 0.)
                    notice = (f"Includes {empty_until:g} s of empty-tube pumping. " if empty_until > 0 else "")
                    notice += "The exponential fit can estimate doses beyond the measured range."
                elif self.plateaus:
                    selected = [row for row in range(self.runs.rowCount()) if self.runs.item(row, 0).checkState() == Qt.CheckState.Checked]
                    curves = [{**run, "included": index in selected} for index, run in enumerate(self.plateaus)]
                    figure = delivery_figure(curves, unit or self.log["declared_unit"] or "setpoint")
                    if not unit:
                        declared = self.log["declared_unit"] or "no explicit units"
                        raise ValueError(f"The CSV records {declared}. Confirm the units the controller used for this run.")
                    if self.config["type"] == "masterflex" and self.log["declared_unit"] == "mL/min":
                        raise ValueError("Select a Masterflex log with measured RPM settings.")
                    if not selected:
                        raise ValueError("Select at least one delivery run to build the calibration.")
                    try:
                        self.calibration = liquid_profile_calibration(self.log, self.plateaus, selected, self.config["type"],
                                                                      unit, self.time_unit.currentData(), self.density.value() or None)
                    except ValueError as error:
                        raise ValueError("Selected runs must increase in delivery rate as the pump setting increases.") from error
                    notice = "Dots show the measured delivery. Lines show quadratic mass-versus-time fits for each run."
                    excluded = sorted({run["target"] for run in curves if not run["included"]})
                    if excluded:
                        settings = ", ".join(f"{target:g}" for target in excluded)
                        notice += f" Unchecked runs at {settings} {unit} will be excluded. Review them before continuing."
                    if self.log["declared_unit"] and self.log["declared_unit"] != unit:
                        notice += f" Interpreting the logged values as {unit} instead of the file's {self.log['declared_unit']} label."
                    if unit == "mL/min":
                        notice += " Applying a calibration requires measurements at known RPM settings."
            except (ValueError, OSError) as error:
                notice = str(error)
        self.status.setText(notice)
        self.use_button.setEnabled(self.calibration is not None and (seed or unit == "RPM"))
        self.equations_button.setEnabled(self.calibration is not None)
        if self.canvas is not None:
            self.plot.removeWidget(self.canvas)
            self.canvas.hide()
            self.canvas.deleteLater()
            self.canvas = None
        if figure is not None:
            self.canvas = FigureCanvasQTAgg(figure)
            self.plot.addWidget(self.canvas)
        self.preview_duration()

    def preview_duration(self, *_):
        self.target_label.setText(f"Pump setting ({self.command_unit.currentData() or 'confirm units'})")
        if not self.calibration or self.mass.value() == 0:
            self.duration.setText("—")
            return
        config = {**self.config, "seed_calibration" if "rpm" in self.calibration else "pump_calibration": self.calibration}
        try:
            if "rpm" in self.calibration:
                _, seconds = seed_dose_command(config, None, self.mass.value())
                last_time, last_dose = self.calibration["points"][-1]
                suffix = " · extrapolated" if seconds > last_time or self.mass.value() > last_dose else ""
                self.duration.setText(f"Pump for {seconds:.5g} s{suffix}")
            else:
                seconds, _ = dose_duration(config, self.target.value(), self.mass.value(), preview=True)
                self.duration.setText(f"Pump for {seconds / 60.:.5g} min")
            self.duration.setToolTip("")
        except ValueError as error:
            self.duration.setText("Check dose or setting")
            self.duration.setToolTip(str(error))
