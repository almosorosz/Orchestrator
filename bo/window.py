"""BO setup page shared by the standalone window and CrystalLoop."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtCharts import QChart, QChartView, QLineSeries
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from experiment.recording import atomic_json
from imed_python.campaign_dialog import CampaignImportDialog
from imed_python.campaign_import import campaign_runs
from imed_python.demo_campaign import import_simulated_campaign
from imed_python.solver import read_model

from .profiles import (
    CHANNELS, CONSTRAINTS, TARGETS, case_runs, components, constraint_values,
    encode, make_profile, profile_family, violations,
)
from .settings import empty_settings, validate_settings


def grid(headers):
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    return table


def item(text, *, editable=False):
    cell = QTableWidgetItem(str(text))
    if not editable:
        cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return cell


def show_value(value):
    return "—" if value is None else f"{value:.6g}"


class BOWindow(QWidget):
    imed_requested = Signal()

    def __init__(self, parent=None, *, imed_case=None):
        super().__init__(parent)
        self.setWindowTitle("CrystalLoop | Bayesian optimization")
        self.resize(1050, 710)
        self.imed_case = imed_case
        self.state = empty_settings()
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.source = QLabel()
        self.source.setWordWrap(True)
        self.source.setTextFormat(Qt.TextFormat.PlainText)
        self.tabs = QTabWidget()
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        title = QLabel("Bayesian optimization")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        toolbar.addWidget(title, 1)
        if imed_case is not None:
            toolbar.addWidget(self.button("IMED", self.imed_requested.emit))
        toolbar.addWidget(self.button("Load BO setup", self.load_file))
        toolbar.addWidget(self.button("Save BO setup", self.save_file))
        layout.addLayout(toolbar)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.status)
        self.build_data_page()
        self.build_settings_page()
        self.build_profile_page()
        self.restore(self.state)

    @staticmethod
    def button(label, handler):
        button = QPushButton(label)
        button.clicked.connect(handler)
        return button

    def page(self, name):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.tabs.addTab(page, name)
        return layout

    def build_data_page(self):
        layout = self.page("Experiments")
        actions = QHBoxLayout()
        actions.addWidget(self.button("Open MATLAB (.mat)", self.choose_matlab))
        actions.addWidget(self.button("Import CrystalLoop campaign", self.choose_campaign))
        if self.imed_case is not None:
            actions.addWidget(self.button("Use current IMED case", self.use_imed))
        actions.addStretch()
        layout.addLayout(actions)
        layout.addWidget(self.source)
        self.inventory = grid(("Use", "Experiment", "Duration (s)", "Profiles", "Yield (%)",
                               "D10 (µm)", "D50 (µm)", "D90 (µm)", "Constraints"))
        self.inventory.itemChanged.connect(self.select_run)
        layout.addWidget(self.inventory, 1)

    def build_settings_page(self):
        layout = self.page("Constraints and targets")
        self.bounds = grid(("Constraint", "Unit", "Minimum", "Maximum"))
        self.bounds.setRowCount(len(CONSTRAINTS))
        self.bound_fields = {}
        for row, (key, (label, unit)) in enumerate(CONSTRAINTS.items()):
            self.bounds.setItem(row, 0, item(label))
            self.bounds.setItem(row, 1, item(unit))
            fields = self.bound_fields[key] = [QLineEdit(), QLineEdit()]
            for col, field in enumerate(fields, 2):
                field.setPlaceholderText("Off")
                self.bounds.setCellWidget(row, col, field)
        self.bounds.setFixedHeight(self.bounds.verticalHeader().defaultSectionSize() * len(CONSTRAINTS) + 55)
        layout.addWidget(self.bounds)
        note = QLabel("Blank bounds are off. Cooling and dosing minimums apply to active segments. "
                      "Holds and zero feed are allowed.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.target_table = grid(("Use", "Target", "Unit", "Direction", "Weight", "Match value", "Scale"))
        self.target_table.setRowCount(len(TARGETS))
        self.target_fields = {}
        for row, (key, (label, unit, modes)) in enumerate(TARGETS.items()):
            enabled = QCheckBox()
            self.target_table.setCellWidget(row, 0, enabled)
            self.target_table.setItem(row, 1, item(label))
            self.target_table.setItem(row, 2, item(unit))
            mode = QComboBox()
            for value, name in (("min", "Minimize"), ("max", "Maximize"), ("match", "Match value")):
                if value in modes:
                    mode.addItem(name, value)
            self.target_table.setCellWidget(row, 3, mode)
            weight, target, scale = QLineEdit(), QLineEdit(), QLineEdit()
            for col, field in enumerate((weight, target, scale), 4):
                self.target_table.setCellWidget(row, col, field)
            mode.currentIndexChanged.connect(lambda _=0, m=mode, t=target: t.setEnabled(m.currentData() == "match"))
            target.setEnabled(False)
            self.target_fields[key] = enabled, mode, weight, target, scale
        self.target_table.resizeColumnsToContents()
        layout.addWidget(self.target_table, 1)
        self.yield_basis = QComboBox()
        self.yield_basis.addItem("Theoretical · solubility at final conditions", "theoretical")
        self.yield_basis.addItem("Estimated actual · measured concentration balance", "actual")
        form = QFormLayout()
        form.addRow("Yield basis", self.yield_basis)
        self.yield_basis.setToolTip("Yield uses a solvent and antisolvent mass balance. "
                                    "Estimated actual yield requires endpoint concentration and assumes no unrecorded withdrawals.")
        layout.addLayout(form)
        self.scores = QLabel()
        self.scores.setWordWrap(True)
        layout.addWidget(self.scores)
        layout.addWidget(self.button("Check bounds and preview objective", self.check_settings))

    def build_profile_page(self):
        layout = self.page("Profile encoding")
        row = QHBoxLayout()
        self.run_choice, self.channel_choice = QComboBox(), QComboBox()
        self.run_choice.currentIndexChanged.connect(self.refresh_channels)
        self.channel_choice.currentIndexChanged.connect(self.refresh_tolerance)
        self.tolerance = QLineEdit()
        self.tolerance.setMaximumWidth(95)
        row.addWidget(QLabel("Experiment"))
        row.addWidget(self.run_choice, 1)
        row.addWidget(self.channel_choice)
        row.addWidget(QLabel("Tolerance"))
        row.addWidget(self.tolerance)
        row.addWidget(self.button("Preview profile encoding", self.preview))
        layout.addLayout(row)
        self.chart = QChartView()
        self.chart.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart.setMinimumHeight(240)
        layout.addWidget(self.chart, 2)
        self.metrics = QLabel()
        self.metrics.setWordWrap(True)
        layout.addWidget(self.metrics)
        self.variables = grid(("Decision variable", "Component", "Encoded value", "Unit", "Across selected runs"))
        self.variables.itemChanged.connect(self.update_override)
        layout.addWidget(self.variables, 1)
        layout.addWidget(self.button("Plot edited components", self.plot_edit))
        self.encoded = None

    def restore(self, settings):
        self.state = validate_settings(settings)
        for key, bounds in self.state["constraints"].items():
            for field, value in zip(self.bound_fields[key], bounds):
                field.setText("" if value is None else str(value))
        for key, (enabled, mode, weight, target, scale) in self.target_fields.items():
            saved = self.state["targets"][key]
            enabled.setChecked(saved["enabled"])
            mode.setCurrentIndex(mode.findData(saved["mode"]))
            for field, value in ((weight, "weight"), (target, "target"), (scale, "scale")):
                field.setText(str(saved[value]))
        self.yield_basis.setCurrentIndex(self.yield_basis.findData(self.state["yield_basis"]))
        self.refresh_data()
        self.status.setText("Select experiments and check the settings." if self.state["runs"]
                            else "Open a MATLAB case or CrystalLoop campaign to get started.")

    def settings(self):
        data = dict(self.state)
        data["constraints"] = {key: [float(field.text()) if field.text().strip() else None
                                     for field in fields] for key, fields in self.bound_fields.items()}
        data["targets"] = {key: {"enabled": enabled.isChecked(), "mode": mode.currentData(),
                                 "weight": float(weight.text()), "target": float(target.text()) if target.isEnabled() else 0.,
                                 "scale": float(scale.text())}
                           for key, (enabled, mode, weight, target, scale) in self.target_fields.items()}
        data["yield_basis"] = self.yield_basis.currentData()
        if self.channel_choice.currentData() is not None:
            data["tolerances"] = {**self.state["tolerances"],
                                  self.channel_choice.currentData(): float(self.tolerance.text())}
        return validate_settings(data)

    def save_file(self):
        try:
            data = self.settings()
            path, _ = QFileDialog.getSaveFileName(self, "Save BO setup", "bo_setup.json", "JSON (*.json)")
            if path:
                self.state = data
                atomic_json(Path(path), data)
                self.status.setText(f"Saved BO setup to {Path(path).name}.")
        except (ValueError, OSError) as error:
            self.status.setText(str(error))

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load BO setup", "", "JSON (*.json)")
        if path:
            try:
                self.restore(json.loads(Path(path).read_text(encoding="utf-8")))
                self.status.setText(f"Loaded BO setup from {Path(path).name}.")
            except (OSError, ValueError) as error:
                self.status.setText(str(error))

    def choose_matlab(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open MATLAB case", "", "MAT files (*.mat)")
        if path:
            self.open_matlab(path)

    def open_matlab(self, path):
        try:
            self.set_case(read_model(path), str(path))
        except (ValueError, OSError, TypeError, KeyError) as error:
            self.status.setText(str(error))

    def use_imed(self):
        try:
            self.set_case(self.imed_case(), "Current IMED case")
        except (ValueError, TypeError, KeyError) as error:
            self.status.setText(str(error))

    def choose_campaign(self):
        path = QFileDialog.getExistingDirectory(self, "Select CrystalLoop campaign folder")
        if path:
            self.open_campaign(path)

    def open_campaign(self, folder):
        try:
            case = import_simulated_campaign(folder)
            if case is None:
                template = self.imed_case() if self.imed_case is not None else {}
                dialog = CampaignImportDialog(campaign_runs(folder), template, self)
                dialog.setWindowTitle("BO | Import CrystalLoop campaign")
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                case = {"ExpData": dialog.experiments, "ProcInf": dialog.processes,
                        "Solubility": template.get("Solubility")}
            self.set_case(case, str(folder), folder=folder)
        except (ValueError, OSError, TypeError, KeyError) as error:
            self.status.setText(str(error))

    def set_case(self, case, source, *, folder=None):
        runs = case_runs(case, folder)
        self.state.update(source=source, runs=runs, selected=list(range(len(runs))), overrides={})
        self.refresh_data()
        skipped = len(case.get("ExpData", [])) - len(runs)
        message = f"Loaded {len(runs)} experiment(s)."
        if skipped:
            message += f" Skipped {skipped} without a usable temperature timeline."
        self.status.setText(message)

    def refresh_data(self):
        self.source.setText(self.state["source"] or "No experiment data loaded")
        self.refresh_inventory()
        self.refresh_runs()

    def refresh_inventory(self):
        self.inventory.blockSignals(True)
        self.inventory.setRowCount(len(self.state["runs"]))
        for row, run in enumerate(self.state["runs"]):
            selected = item("")
            selected.setFlags(selected.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            selected.setCheckState(Qt.CheckState.Checked if row in self.state["selected"] else Qt.CheckState.Unchecked)
            self.inventory.setItem(row, 0, selected)
            response = run["metrics"]
            available = ", ".join(CHANNELS[key][0] for key in run["profiles"])
            values = [run["name"], show_value(response["duration"]), available,
                      show_value(response[f"yield_{self.state['yield_basis']}"]),
                      *[show_value(response[key]) for key in ("d10", "d50", "d90")],
                      self.check_constraints(run)]
            for col, value in enumerate(values, 1):
                self.inventory.setItem(row, col, item(value))
        self.inventory.blockSignals(False)

    def select_run(self, cell):
        if cell.column() != 0:
            return
        self.state["selected"] = [row for row in range(self.inventory.rowCount())
                                  if self.inventory.item(row, 0).checkState() == Qt.CheckState.Checked]
        self.refresh_runs()

    def refresh_runs(self):
        active = self.run_choice.currentData()
        self.run_choice.blockSignals(True)
        self.run_choice.clear()
        for index in self.state["selected"]:
            self.run_choice.addItem(self.state["runs"][index]["name"], index)
        self.run_choice.setCurrentIndex(max(0, self.run_choice.findData(active)))
        self.run_choice.blockSignals(False)
        self.refresh_channels()

    def refresh_channels(self):
        active = self.channel_choice.currentData()
        self.channel_choice.blockSignals(True)
        self.channel_choice.clear()
        index = self.run_choice.currentData()
        if index is not None:
            for key in CHANNELS:
                if key in self.state["runs"][index]["profiles"]:
                    self.channel_choice.addItem(CHANNELS[key][0], key)
        self.channel_choice.setCurrentIndex(max(0, self.channel_choice.findData(active)))
        self.channel_choice.blockSignals(False)
        self.refresh_tolerance()

    def refresh_tolerance(self):
        channel = self.channel_choice.currentData()
        self.tolerance.setText(str(self.state["tolerances"][channel]) if channel else "")
        self.tolerance.setEnabled(channel is not None)
        self.encoded = None
        self.variables.setRowCount(0)
        self.chart.setChart(QChart())
        self.metrics.setText("Click Preview profile encoding to inspect the selected trace.")

    @staticmethod
    def profile(run, channel):
        data = run["profiles"][channel]
        return make_profile(data["points"], data["source"], step=data["step"])

    def check_constraints(self, run):
        checks = {"duration": [run["metrics"]["duration"]]}
        for channel in run["profiles"]:
            key, values = constraint_values(channel, self.profile(run, channel))
            if key:
                checks[key] = values
        failed, missing = [], []
        for key, (lower, upper) in self.state["constraints"].items():
            if lower is None and upper is None:
                continue
            if key not in checks:
                missing.append(CONSTRAINTS[key][0])
            elif violations(checks[key], lower, upper):
                failed.append(CONSTRAINTS[key][0])
        return " · ".join((["Outside bounds: " + ", ".join(failed)] if failed else [])
                          + (["Missing: " + ", ".join(missing)] if missing else [])) or "—"

    def check_settings(self):
        try:
            self.state = self.settings()
            self.refresh_inventory()
        except ValueError as error:
            self.status.setText(str(error))
            return
        lines = []
        for index in self.state["selected"]:
            run = self.state["runs"][index]
            terms, missing = [], []
            for key, target in self.state["targets"].items():
                if not target["enabled"] or target["weight"] == 0:
                    continue
                response = run["metrics"].get(f"yield_{self.state['yield_basis']}" if key == "yield" else key)
                label = TARGETS[key][0]
                if response is None:
                    missing.append(label)
                    continue
                direction = target["mode"]
                loss = abs(response - target["target"]) if direction == "match" else response * (-1 if direction == "max" else 1)
                terms.append((label, target["weight"] * loss / target["scale"]))
            detail = ", ".join(f"{name} {value:.4g}" for name, value in terms)
            result = f"loss {sum(value for _, value in terms):.4g} ({detail})" if terms and not missing else "loss unavailable"
            if missing:
                result += f" · missing {', '.join(missing)}"
            lines.append(f"{run['name']}: {result}")
        self.scores.setText("\n".join(lines[:6]) + (f"\nAnd {len(lines) - 6} more selected runs." if len(lines) > 6 else "")
                            if lines else "Select one or more experiments to preview the objective.")
        self.status.setText("Checked bounds. Objective preview uses dimensionless weighted losses. Lower is better.")

    def preview(self):
        index, channel = self.run_choice.currentData(), self.channel_choice.currentData()
        if index is None or channel is None:
            self.status.setText("Select an experiment and a profile.")
            return
        try:
            tolerance = float(self.tolerance.text())
            if not np.isfinite(tolerance) or tolerance < 0:
                raise ValueError("Enter a finite, nonnegative encoding tolerance.")
        except ValueError as error:
            self.status.setText(str(error))
            return
        self.state["tolerances"][channel] = tolerance
        source = self.profile(self.state["runs"][index], channel)
        model = encode(source, tolerance, channel)
        model_values = components(channel, model)
        peers = []
        for other in self.state["selected"]:
            run = self.state["runs"][other]
            if channel in run["profiles"]:
                peer = encode(self.profile(run, channel), tolerance, channel)
                if profile_family(peer, channel) == profile_family(model, channel):
                    peers.append(components(channel, peer))
        family = ":".join(map(str, (channel, *profile_family(model, channel))))
        self.variables.blockSignals(True)
        self.variables.setRowCount(len(model_values))
        for row, (label, value) in enumerate(model_values.items()):
            sequence = [peer[label] for peer in peers]
            varied = bool(sequence and max(sequence) - min(sequence) > (1e-6 if label.endswith("duration") else tolerance))
            key = f"{family}:{label}"
            checked = item("")
            checked.setData(Qt.ItemDataRole.UserRole, key)
            checked.setFlags(checked.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked.setCheckState(Qt.CheckState.Checked if self.state["overrides"].get(key, varied) else Qt.CheckState.Unchecked)
            self.variables.setItem(row, 0, checked)
            self.variables.setItem(row, 1, item(label))
            self.variables.setItem(row, 2, item(f"{value:.12g}", editable=True))
            self.variables.setItem(row, 3, item("s" if label.endswith("duration") else CHANNELS[channel][1]))
            self.variables.setItem(row, 4, item("Varies" if varied else "Fixed" if len(peers) > 1 else "One run in family"))
        self.variables.blockSignals(False)
        self.variables.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.encoded = source, model, len(peers)
        self.draw_profiles(source, model, channel, len(peers))

    def update_override(self, cell):
        if cell.column() == 0:
            self.state["overrides"][cell.data(Qt.ItemDataRole.UserRole)] = cell.checkState() == Qt.CheckState.Checked
            if self.encoded is not None:
                self.draw_profiles(self.encoded[0], self.decoded, self.channel_choice.currentData(), self.encoded[2])

    def plot_edit(self):
        if self.encoded is None:
            self.preview()
            return
        source, model, peer_count = self.encoded
        channel = self.channel_choice.currentData()
        try:
            values = np.array([float(self.variables.item(row, 2).text()) for row in range(self.variables.rowCount())])
            if not np.isfinite(values).all() or np.any(values[1::2] <= 0):
                raise ValueError("Enter finite components and positive segment durations.")
            time, targets = np.r_[0., np.cumsum(values[1::2])], values[::2]
            limits_ok = np.all(targets > -273.15) if channel == "temperature" else np.all(targets >= 0)
            if not limits_ok:
                raise ValueError("The edited profile contains values outside the physical range.")
            changed = replace(model, time=time, values=targets)
            settings = self.settings()
            key, checks = constraint_values(channel, changed)
            if (key and violations(checks, *settings["constraints"][key])
                    or channel != "seed" and violations([time[-1]], *settings["constraints"]["duration"])):
                raise ValueError("The edited profile violates the enabled bounds.")
        except ValueError as error:
            self.status.setText(str(error))
            return
        self.draw_profiles(source, changed, channel, peer_count)
        self.status.setText("Previewed edited components. The imported experiment was preserved.")

    def draw_profiles(self, source, decoded, channel, peer_count=None):
        self.decoded = decoded
        chart = QChart()
        for label, profile, color, dashed in (("Source", source, "#b45309", True),
                                               ("Decoded", decoded, "#1d4ed8", False)):
            curve = QLineSeries()
            curve.setName(label)
            for i, (time, value) in enumerate(zip(profile.time, profile.values)):
                if profile.step and i:
                    curve.append(float(time), float(profile.values[i - 1]))
                curve.append(float(time), float(value))
            pen = QPen(QColor(color), 2)
            if dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            curve.setPen(pen)
            curve.setPointsVisible(label == "Decoded" or len(profile.time) == 1)
            chart.addSeries(curve)
        chart.createDefaultAxes()
        chart.axes(Qt.Orientation.Horizontal)[0].setTitleText("Time (s)")
        label, unit, _ = CHANNELS[channel]
        chart.axes(Qt.Orientation.Vertical)[0].setTitleText(f"{label} ({unit})")
        self.chart.setChart(chart)
        errors = source.values - decoded.sample(source.time)
        detail = (f"{source.source} · {len(source.time)} source points, {len(decoded.time)} decoded knots · "
                  f"max error {np.max(np.abs(errors)):.5g} {unit}, RMS {np.sqrt(np.mean(errors**2)):.5g} {unit}")
        if channel != "seed":
            detail += f" · duration {source.time[-1]:.6g} → {decoded.time[-1]:.6g} s"
        if channel == "feed":
            detail += f" · integrated dose {source.total():.6g} → {decoded.total():.6g} g"
        if channel == "temperature":
            _, rates = constraint_values(channel, decoded)
            detail += f" · peak cooling {max(rates, default=0):.5g} °C/min"
        if peer_count is not None:
            count = sum(self.variables.item(row, 0).checkState() == Qt.CheckState.Checked
                        for row in range(self.variables.rowCount()))
            detail += f" · {peer_count}/{len(self.state['selected'])} selected runs in this profile family"
            detail += f" · {count} decision variable(s) in this channel"
        self.metrics.setText(detail)
