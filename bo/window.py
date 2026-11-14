"""Bayesian experiment design shared by the standalone window and CrystalLoop."""
import json
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtCharts import QChart
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

from .profiles import CHANNELS, CONSTRAINTS, TARGETS, Profile, case_runs, constraint_values, make_profile, violations
from .proposal_view import BAND_SD, ProposalView, chart_view, profile_chart, replace_chart
from .settings import CHECKPOINT_CHANNELS, OUTPUTS, empty_settings, validate_settings


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


class BOJobSignals(QObject):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)


class BOJob(QRunnable):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.signals = BOJobSignals()

    def run(self):
        try:
            result = self.task(self.signals.progress.emit)
        except ImportError as error:
            hint = ("Use a build that includes the BO dependencies." if getattr(sys, "frozen", False)
                    else "Install requirements-bo.txt in this Python environment.")
            self.signals.failed.emit(f"Could not load the BO dependencies. {hint} {error}")
        except Exception as error:
            self.signals.failed.emit(str(error) or type(error).__name__)
        else:
            self.signals.completed.emit(result)


def bounds_fields(layout, definitions):
    table = grid(("Constraint", "Unit", "Minimum", "Maximum"))
    table.setRowCount(len(definitions))
    result = {}
    for row, (key, (label, unit)) in enumerate(definitions.items()):
        table.setItem(row, 0, item(label))
        table.setItem(row, 1, item(unit))
        fields = result[key] = [QLineEdit(), QLineEdit()]
        for column, field in enumerate(fields, 2):
            field.setPlaceholderText("Off")
            table.setCellWidget(row, column, field)
    table.setMinimumHeight(table.verticalHeader().defaultSectionSize() * len(definitions) + 38)
    layout.addWidget(table)
    return result


def read_bounds(fields):
    return {key: [float(control.text()) if control.text().strip() else None for control in pair]
            for key, pair in fields.items()}


class BOWindow(QWidget):
    imed_requested = Signal()

    def __init__(self, parent=None, *, imed_case=None):
        super().__init__(parent)
        self.setWindowTitle("CrystalLoop | Bayesian optimization")
        self.resize(1050, 710)
        self.imed_case = imed_case
        self.state = empty_settings()
        self.job = None
        self.revision = 0
        self.rendering = False
        self.proposal = None
        self.encoding = None
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
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
        self.build_output_page()
        self.build_profile_page()
        self.build_proposal_page()
        self.connect_settings()
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
        actions.addWidget(self.button("Open IMED data (.mat / .json)", self.choose_matlab))
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
        layout = self.page("Input bounds and targets")
        self.bound_fields = bounds_fields(layout, CONSTRAINTS)
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

    def build_output_page(self):
        layout = self.page("Output screening")
        self.output_fields = bounds_fields(layout, {key: TARGETS[key][:2] for key in OUTPUTS})
        form = QFormLayout()
        self.exploration = QLineEdit()
        self.output_margin = QLineEdit()
        form.addRow("Exploration weight", self.exploration)
        form.addRow("Output margin (standard deviations)", self.output_margin)
        layout.addLayout(form)
        note = QLabel("Blank bounds are off. The entire predicted interval must lie within the enabled bounds. "
                      "The interval is mean ± margin × SD. A margin of 0 screens the predicted mean. "
                      "The acquisition minimizes weighted loss minus the exploration weight times its SD.")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()

    def build_profile_page(self):
        layout = self.page("Profile encoding")
        row = QHBoxLayout()
        self.run_choice, self.channel_choice = QComboBox(), QComboBox()
        self.run_choice.currentIndexChanged.connect(self.refresh_channels)
        self.channel_choice.currentIndexChanged.connect(self.refresh_tolerance)
        self.tolerance = QLineEdit()
        self.tolerance.setMaximumWidth(90)
        self.tolerance.editingFinished.connect(self.store_tolerance)
        row.addWidget(QLabel("Experiment"))
        row.addWidget(self.run_choice, 1)
        row.addWidget(self.channel_choice)
        row.addWidget(QLabel("Variation tolerance"))
        row.addWidget(self.tolerance)
        self.preview_button = self.button("Compare source and decoded", self.preview)
        row.addWidget(self.preview_button)
        layout.addLayout(row)
        codec = QHBoxLayout()
        self.checkpoint_button = self.button("Load notebook checkpoint", self.choose_checkpoint)
        self.reset_checkpoint_button = self.button("Fit from selected runs", self.reset_checkpoint)
        self.codec_note = QLabel()
        self.codec_note.setWordWrap(True)
        codec.addWidget(self.checkpoint_button)
        codec.addWidget(self.reset_checkpoint_button)
        codec.addWidget(self.codec_note, 1)
        layout.addLayout(codec)
        self.chart = chart_view()
        self.chart.setMinimumHeight(215)
        layout.addWidget(self.chart, 2)
        self.metrics = QLabel()
        self.metrics.setWordWrap(True)
        layout.addWidget(self.metrics)
        self.variables = grid(("Vary in BO", "Decoder input", "Value", "Unit", "Selected runs"))
        self.variables.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                       | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.variables.itemChanged.connect(self.update_override)
        layout.addWidget(self.variables, 1)
        self.decode_button = self.button("Decode edited values", self.plot_edit)
        self.vary_duration = QCheckBox("Vary batch duration")
        self.vary_duration.toggled.connect(self.change_duration)
        row = QHBoxLayout()
        row.addWidget(self.decode_button)
        row.addWidget(self.vary_duration)
        row.addStretch()
        layout.addLayout(row)

    def build_proposal_page(self):
        layout = self.page("Propose experiment")
        row = QHBoxLayout()
        self.propose_button = self.button("Propose next experiment", self.suggest)
        self.export_button = self.button("Save proposed recipe", self.save_proposal)
        row.addWidget(self.propose_button)
        row.addWidget(self.export_button)
        row.addStretch()
        layout.addLayout(row)
        self.proposal_note = QLabel()
        self.proposal_note.setWordWrap(True)
        layout.addWidget(self.proposal_note)
        self.proposal_view = ProposalView()
        self.predictions = grid(("Output", "Unit", "Observations", "Mean", "SD", "Posterior lower", "Posterior upper"))
        self.proposal_view.insertTab(0, self.predictions, "Predictions")
        self.proposal_view.setCurrentIndex(0)
        layout.addWidget(self.proposal_view, 1)

    def connect_settings(self):
        for enabled, mode, weight, match, scale in self.target_fields.values():
            enabled.toggled.connect(self.invalidate_proposal)
            mode.currentIndexChanged.connect(self.invalidate_proposal)
            for control in (weight, match, scale):
                control.textEdited.connect(self.invalidate_proposal)
        for fields in (*self.bound_fields.values(), *self.output_fields.values()):
            for control in fields:
                control.textEdited.connect(self.invalidate_proposal)
        for control in (self.exploration, self.output_margin):
            control.textEdited.connect(self.invalidate_proposal)
        self.tolerance.textEdited.connect(self.tolerance_changed)
        self.yield_basis.currentIndexChanged.connect(self.invalidate_proposal)

    def restore(self, settings):
        self.state = validate_settings(settings)
        self.rendering = True
        for group, fields in (("constraints", self.bound_fields), ("output_bounds", self.output_fields)):
            for key, bounds in self.state[group].items():
                for field, value in zip(fields[key], bounds):
                    field.setText("" if value is None else str(value))
        self.exploration.setText(str(self.state["exploration"]))
        self.output_margin.setText(str(self.state["output_margin"]))
        self.vary_duration.setChecked(self.state["overrides"].get("duration", True))
        for key, (enabled, mode, weight, target, scale) in self.target_fields.items():
            saved = self.state["targets"][key]
            enabled.setChecked(saved["enabled"])
            mode.setCurrentIndex(mode.findData(saved["mode"]))
            for field, value in ((weight, "weight"), (target, "target"), (scale, "scale")):
                field.setText(str(saved[value]))
        self.yield_basis.setCurrentIndex(self.yield_basis.findData(self.state["yield_basis"]))
        self.rendering = False
        self.invalidate_proposal()
        self.refresh_data()
        self.status.setText("Select experiments and check the settings." if self.state["runs"]
                            else "Open a MATLAB case or CrystalLoop campaign to get started.")

    def settings(self):
        data = dict(self.state)
        data["constraints"] = read_bounds(self.bound_fields)
        data["output_bounds"] = read_bounds(self.output_fields)
        data["exploration"] = float(self.exploration.text())
        data["output_margin"] = float(self.output_margin.text())
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
        path, _ = QFileDialog.getOpenFileName(
            self, "Open experimental data", "", "IMED cases (*.mat *.imed.json);;MATLAB (*.mat);;JSON (*.json)")
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
        self.vary_duration.blockSignals(True)
        self.vary_duration.setChecked(True)
        self.vary_duration.blockSignals(False)
        self.invalidate_proposal()
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
        self.invalidate_proposal()
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
        checkpoint = self.state["vae_checkpoints"].get(channel)
        if channel is None:
            note = ""
        elif checkpoint is not None:
            note = checkpoint["name"]
        elif channel == "seed":
            note = "Seed mass is a scalar in grams."
        else:
            note = "Uses the notebook VAE fitted to the selected runs when the profile shape varies."
        self.codec_note.setText(note)
        self.clear_preview()
        self.update_buttons()

    def clear_preview(self):
        self.encoding = None
        self.variables.setRowCount(0)
        replace_chart(self.chart, QChart())
        self.metrics.setText("Select an experiment and compare its source profile with the decoder output.")
        self.update_buttons()

    def store_tolerance(self):
        channel = self.channel_choice.currentData()
        if channel is not None:
            try:
                value = float(self.tolerance.text())
                if not np.isfinite(value) or value < 0:
                    raise ValueError
            except ValueError:
                self.status.setText("Enter a finite, nonnegative variation tolerance.")
            else:
                self.state["tolerances"][channel] = value

    def tolerance_changed(self):
        self.invalidate_proposal()
        if self.encoding is not None:
            self.clear_preview()

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

    def invalidate_proposal(self, *_):
        if self.rendering:
            return
        self.revision += 1
        if self.proposal is not None or self.predictions.rowCount():
            self.set_proposal(None)
        else:
            self.proposal_note.setText("Select experiments, set the targets and bounds, then propose a recipe.")
            self.update_buttons()

    def change_duration(self, enabled):
        if not self.rendering:
            self.state["overrides"]["duration"] = enabled
            self.invalidate_proposal()

    def update_buttons(self):
        if not hasattr(self, "propose_button"):
            return
        available = self.job is None
        channel = self.channel_choice.currentData()
        self.propose_button.setEnabled(available and len(self.state["selected"]) >= 2)
        self.preview_button.setEnabled(available and channel is not None)
        self.checkpoint_button.setEnabled(available and channel in CHECKPOINT_CHANNELS)
        self.reset_checkpoint_button.setEnabled(available and channel in self.state["vae_checkpoints"])
        self.decode_button.setEnabled(available and self.encoding is not None and bool(self.encoding[0].variables))
        self.export_button.setEnabled(self.proposal is not None)

    def run_job(self, task, callback, *, check_profile=False):
        if self.job is not None:
            return
        self.status.setText("Preparing Bayesian optimization…")
        self.job = BOJob(task)
        self.completed_action = callback
        self.active_revision = self.revision
        self.active_profile = self.profile_selection() if check_profile else None
        self.job.signals.progress.connect(self.status.setText, Qt.ConnectionType.QueuedConnection)
        self.job.signals.completed.connect(self.work_completed, Qt.ConnectionType.QueuedConnection)
        self.job.signals.failed.connect(self.work_failed, Qt.ConnectionType.QueuedConnection)
        self.update_buttons()
        QThreadPool.globalInstance().start(self.job)

    def work_completed(self, result):
        self.job = None
        action = self.completed_action
        self.completed_action = None
        self.update_buttons()
        if self.active_revision != self.revision:
            self.status.setText("The BO settings changed during the calculation. Run the calculation again.")
            return
        if self.active_profile is not None and self.active_profile != self.profile_selection():
            self.status.setText("The selected profile changed. Compare the current profile to view its encoding.")
            return
        action(result)

    def work_failed(self, error):
        self.job = None
        self.completed_action = None
        self.update_buttons()
        self.status.setText(error if self.active_revision == self.revision
                            else "The BO settings changed during the calculation. Run the calculation again.")

    def profile_selection(self):
        return self.run_choice.currentData(), self.channel_choice.currentData()

    def choose_checkpoint(self):
        key = self.channel_choice.currentData()
        if key not in CHECKPOINT_CHANNELS:
            return
        path, _ = QFileDialog.getOpenFileName(self, f"Load {CHANNELS[key][0]} VAE checkpoint",
                                              "", "PyTorch checkpoints (*.pt *.pth)")
        if not path:
            return

        def load(report):
            from .vae import read_checkpoint

            report("Loading the notebook trajectory VAE weights…")
            return read_checkpoint(key, path)

        def loaded(checkpoint):
            self.state["vae_checkpoints"][key] = checkpoint
            self.invalidate_proposal()
            self.refresh_tolerance()
            self.status.setText(f"Loaded {checkpoint['name']}.")

        self.run_job(load, loaded, check_profile=True)

    def reset_checkpoint(self):
        key = self.channel_choice.currentData()
        if self.state["vae_checkpoints"].pop(key, None) is not None:
            self.invalidate_proposal()
            self.refresh_tolerance()
            self.status.setText(f"{CHANNELS[key][0]} will use a VAE fitted to the selected runs.")

    def preview(self):
        index, key = self.profile_selection()
        if index is None or key is None:
            self.status.setText("Select an experiment and a profile.")
            return
        try:
            state = self.settings()
        except ValueError as error:
            self.status.setText(str(error))
            return
        self.clear_preview()

        def prepare(report):
            from .optimizer import preview_channel

            channel, row = preview_channel(state, key, index, report)
            data = state["runs"][index]
            original = self.profile(data, key)
            features = channel.features[row:row + 1]
            decoded, = channel.decode(features, [data["metrics"]["duration"]])
            representation = None
            if channel.codec is not None:
                time, values = channel.codec.normalize(original)
                representation = Profile(time * original.time[-1], values * channel.codec.scale + channel.codec.offset,
                                         "VAE input knots", original.step)
            return channel, original, features[0], decoded, representation

        self.run_job(prepare, self.show_encoding, check_profile=True)

    def show_encoding(self, result):
        from .optimizer import variable_activity

        channel, original, features, decoded, representation = result
        self.encoding = channel, original, representation
        varied = variable_activity(channel.features)
        self.variables.blockSignals(True)
        self.variables.setRowCount(len(channel.variables))
        for row, (variable, value) in enumerate(zip(channel.variables, features)):
            cell = item("")
            cell.setData(Qt.ItemDataRole.UserRole, variable.key)
            if varied[row]:
                cell.setFlags(cell.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                cell.setCheckState(Qt.CheckState.Checked if self.state["overrides"].get(variable.key, True)
                                   else Qt.CheckState.Unchecked)
            self.variables.setItem(row, 0, cell)
            for column, content in enumerate((variable.label, f"{value:.12g}", variable.unit,
                                                "Varies" if varied[row] else "Fixed"), 1):
                self.variables.setItem(row, column, item(content, editable=column == 2))
        self.variables.blockSignals(False)
        self.show_profiles(original, decoded, representation, channel)
        self.update_buttons()
        self.status.setText("Compared the selected profile with its decoded values.")

    def update_override(self, cell):
        if cell.column() == 0 and cell.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            self.state["overrides"][cell.data(Qt.ItemDataRole.UserRole)] = cell.checkState() == Qt.CheckState.Checked
            self.invalidate_proposal()

    def plot_edit(self):
        channel, source, representation = self.encoding
        index = self.run_choice.currentData()
        try:
            settings = self.settings()
            values = np.array([float(self.variables.item(row, 2).text()) for row in range(len(channel.variables))])
            if not np.isfinite(values).all():
                raise ValueError("Enter finite decoder inputs.")
            duration = settings["runs"][index]["metrics"]["duration"]
            decoded, = channel.decode(values[None, :], [duration])
            key = channel.key
            if not np.isfinite(decoded.values).all() or np.any(decoded.values <= -273.15 if key == "temperature"
                                                               else decoded.values < 0):
                raise ValueError("The edited values produce a profile outside the physical range.")
        except (ValueError, RuntimeError, OverflowError) as error:
            self.status.setText(str(error))
            return
        self.show_profiles(source, decoded, representation, channel)
        key, checks = constraint_values(channel.key, decoded)
        outside = (key and violations(checks, *settings["constraints"][key]))
        self.status.setText("Previewed the edited values. This trace exceeds an input bound."
                            if outside else "Previewed the edited decoder inputs.")

    def show_profiles(self, original, decoded, representation, channel):
        key = channel.key
        traces = [("Source", original)]
        if representation is not None:
            traces.append(("VAE input knots", representation))
        traces.append(("Decoded", decoded))
        replace_chart(self.chart, profile_chart(key, traces))
        _, unit, _ = CHANNELS[key]
        difference = original.values - decoded.sample(original.time)
        detail = (f"{channel.description} · {len(original.time)} source points → {len(decoded.time)} decoded knots · "
                  f"RMSE {np.sqrt(np.mean(difference**2)):.5g} {unit} · maximum error {np.max(np.abs(difference)):.5g} {unit}")
        if key == "feed":
            detail += f" · total dose {original.total():.5g} → {decoded.total():.5g} g"
        if key == "temperature":
            _, rates = constraint_values(key, decoded)
            detail += f" · peak decoded cooling {max(rates, default=0):.5g} °C/min"
        self.metrics.setText(detail)

    def suggest(self):
        try:
            state = self.settings()
        except ValueError as error:
            self.status.setText(str(error))
            return
        self.set_proposal(None)

        def calculate(report):
            from .optimizer import propose

            return propose(state, report=report)

        self.run_job(calculate, self.set_proposal)

    def set_proposal(self, result):
        self.proposal = result
        self.predictions.setRowCount(0)
        self.proposal_view.set_result(result)
        if result is not None:
            for row, (key, (label, unit, _)) in enumerate(TARGETS.items()):
                self.predictions.insertRow(row)
                if key in result.predictions:
                    mean, sd = result.predictions[key]
                    values = (label, unit, "Exact" if key == "duration" else result.counts[key],
                              *(show_value(value) for value in (mean, sd, mean - BAND_SD * sd, mean + BAND_SD * sd)))
                else:
                    values = (label, unit, result.counts.get(key, 0), result.unavailable[key], "—", "—", "—")
                for column, value in enumerate(values):
                    self.predictions.setItem(row, column, item(value))
            beta = result.state["exploration"]
            active = int(result.vary.sum())
            self.proposal_note.setText(
                f"{active} varying inputs · {result.output_passed}/{result.input_passed} screened candidates passed. "
                f"Predicted loss {result.mean_loss:.5g} ± {result.loss_sd:.4g} SD · "
                f"exploration bonus {beta * result.loss_sd:.4g} · acquisition {result.acquisition:.5g}. "
                "The prediction table shows mean ± 1.96 SD.")
            self.status.setText("Proposal ready. Review the recipe and output predictions.")
        else:
            self.proposal_note.setText("Select experiments, set the targets and bounds, then propose a recipe.")
        self.update_buttons()

    def save_proposal(self):
        result = self.proposal
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save proposed recipe", "bo_proposed_recipe.json", "JSON (*.json)")
        if not path:
            return
        file = Path(path)
        payload = {
            "format": "crystalloop-bo-proposal", "version": 1,
            "yield_basis": result.state["yield_basis"], "duration_s": float(result.point[0]),
            "profiles": {key: {**profile.payload(), "unit": CHANNELS[key][1]} for key, profile in result.profiles.items()},
            "predictions": {key: {"mean": mean, "sd": sd, "unit": TARGETS[key][1]}
                            for key, (mean, sd) in result.predictions.items()},
            "decision_variables": {variable.key: {"decoder_input": float(control),
                                                  "evaluated_value": float(value), "unit": variable.unit}
                                   for variable, control, value in zip(result.variables, result.controls, result.point)},
            "input_bounds": result.state["constraints"], "output_bounds": result.state["output_bounds"],
            "output_margin": result.state["output_margin"], "exploration": result.state["exploration"],
            "weighted_loss": result.mean_loss, "weighted_loss_sd": result.loss_sd, "acquisition": result.acquisition,
        }
        try:
            atomic_json(file, payload)
        except (ValueError, OSError) as error:
            self.status.setText(str(error))
        else:
            self.status.setText(f"Saved the proposal to {file.name}.")
