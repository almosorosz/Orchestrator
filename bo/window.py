"""Bayesian experiment design shared by the standalone window and CrystalLoop."""
import json
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtCharts import QChart
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from experiment.recording import atomic_json
from imed_python.campaign_dialog import CampaignImportDialog
from imed_python.campaign_import import campaign_runs
from imed_python.demo_campaign import import_simulated_campaign
from imed_python.solver import read_model

from .decoder import cooling_landmarks, fixed_horizon
from .physical import physical_profile
from .physical_fit import initial_spec, read_spec, trace_spec
from .profiles import (
    CHANNELS, CONSTRAINTS, TARGETS, case_runs, constraint_values, make_profile, recipe_data,
    selected_horizon, violations,
)
from .proposal_view import BAND_SD, COLORS, ProposalView, chart_view, profile_chart, replace_chart
from .settings import OUTPUTS, STAGE_CHANNELS, empty_settings, validate_settings
from .stage_editor import StageEditor


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

    def __init__(self, parent=None, *, imed_case=None, execute_recipe=None):
        super().__init__(parent)
        self.setWindowTitle("CrystalLoop | Bayesian optimization")
        self.resize(1050, 710)
        self.imed_case = imed_case
        self.recipe_launcher = execute_recipe
        self.state = empty_settings()
        self.job = None
        self.loop_busy = False
        self.revision = 0
        self.rendering = False
        self.proposal = None
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        self.source = QLabel()
        self.source.setWordWrap(True)
        self.source.setTextFormat(Qt.TextFormat.PlainText)
        self.tabs = QTabWidget()
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        title = QLabel("📈  Bayesian optimization")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        toolbar.addWidget(title, 1)
        if imed_case is not None:
            toolbar.addWidget(self.button("IMED", self.imed_requested.emit))
        toolbar.addWidget(self.button("Load BO setup", self.load_file))
        toolbar.addWidget(self.button("Save BO setup", self.save_file))
        self.stages_button = self.button("BO Loop stages", self.show_loop_stages)
        toolbar.addWidget(self.stages_button)
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
        self.batch_note = QLabel()
        self.batch_note.setWordWrap(True)
        layout.addWidget(self.batch_note)

    def build_settings_page(self):
        layout = self.page("Input bounds and targets")
        self.bound_fields = bounds_fields(layout, CONSTRAINTS)
        note = QLabel("Set the batch horizon under Physical profiles. Blank bounds are off. "
                      "Cooling and dosing minimums apply to active segments.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.target_table = grid(("Use", "Target", "Unit", "Direction", "Weight", "Match value", "Scale"))
        self.target_table.setRowCount(len(OUTPUTS))
        self.target_fields = {}
        for row, key in enumerate(OUTPUTS):
            label, unit, modes = TARGETS[key]
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
        layout = self.page("Physical profiles")
        self.horizon_label = QLabel()
        self.horizon_label.setWordWrap(True)
        self.batch_horizon = QLineEdit()
        self.batch_horizon.setPlaceholderText("Longest selected run")
        self.batch_horizon.setMaximumWidth(190)
        self.batch_horizon.setToolTip("Leave blank to use the longest selected experiment. Enter a longer horizon to extend the recipe.")
        self.batch_horizon.textChanged.connect(self.horizon_changed)
        clock = QHBoxLayout()
        clock.addWidget(QLabel("Batch horizon (s)"))
        clock.addWidget(self.batch_horizon)
        clock.addWidget(self.horizon_label, 1)
        layout.addLayout(clock)
        self.profile_tabs = QTabWidget()
        layout.addWidget(self.profile_tabs, 1)
        edit = QWidget()
        edit_layout = QVBoxLayout(edit)
        self.stage_editor = StageEditor()
        self.stage_editor.changed.connect(self.stage_settings_changed)
        edit_layout.addWidget(self.stage_editor, 1)
        row = QHBoxLayout()
        self.export_template_button = self.button("Save template recipe", self.save_template)
        self.reset_stages_button = self.button("Reset templates from experiments", self.reset_stage_templates)
        row.addWidget(self.export_template_button)
        row.addWidget(self.reset_stages_button)
        row.addStretch()
        edit_layout.addLayout(row)
        self.profile_tabs.addTab(edit, "Edit stages")

        comparison = QWidget()
        compare_layout = QVBoxLayout(comparison)
        row = QHBoxLayout()
        self.run_choice, self.channel_choice = QComboBox(), QComboBox()
        self.run_choice.currentIndexChanged.connect(self.refresh_channels)
        self.channel_choice.currentIndexChanged.connect(self.refresh_preview)
        self.channel_choice.currentIndexChanged.connect(self.sync_stage_channel)
        self.stage_editor.tabs.currentChanged.connect(self.sync_comparison_channel)
        row.addWidget(QLabel("Reference experiment"))
        row.addWidget(self.run_choice, 1)
        row.addWidget(self.channel_choice)
        self.preview_button = self.button("Compare recorded and fitted", self.preview)
        row.addWidget(self.preview_button)
        self.use_trace_button = self.button("Use trace as template", self.use_trace_template)
        row.addWidget(self.use_trace_button)
        compare_layout.addLayout(row)
        self.compare_all = QCheckBox("Compare all selected experiments")
        self.compare_all.setChecked(True)
        self.compare_all.toggled.connect(self.refresh_preview)
        compare_layout.addWidget(self.compare_all)
        self.chart = chart_view()
        self.chart.setMinimumHeight(200)
        compare_layout.addWidget(self.chart, 2)
        self.metrics = QLabel()
        self.metrics.setWordWrap(True)
        compare_layout.addWidget(self.metrics)
        self.variables = grid(("Vary in BO", "Physical control", "Fitted value", "Unit"))
        self.variables.itemChanged.connect(self.update_override)
        self.fit_summary = grid(("Experiment", "RMSE"))
        self.fit_tabs = QTabWidget()
        self.fit_tabs.addTab(self.fit_summary, "Fit by experiment")
        self.fit_tabs.addTab(self.variables, "Decision controls")
        compare_layout.addWidget(self.fit_tabs, 1)
        hint = QLabel("Choose temperature, antisolvent and stirring controls under Edit stages. "
                      "Fixed means the fitted values are the same across the selected experiments.")
        hint.setWordWrap(True)
        compare_layout.addWidget(hint)
        self.profile_tabs.addTab(comparison, "Compare profiles")

    def build_proposal_page(self):
        layout = self.page("Propose experiment")
        row = QHBoxLayout()
        self.propose_button = self.button("Propose next experiment", self.suggest)
        self.execute_button = self.button("Execute recipe", self.execute_proposal)
        self.execute_button.setVisible(self.recipe_launcher is not None)
        self.export_button = self.button("Save proposed recipe", self.save_proposal)
        row.addWidget(self.propose_button)
        row.addWidget(self.execute_button)
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
        self.yield_basis.currentIndexChanged.connect(self.invalidate_proposal)

    def restore(self, settings):
        self.state = validate_settings(settings)
        self.state.pop("loop_proposal", None)
        self.rendering = True
        horizon = self.state["batch_horizon_s"]
        self.batch_horizon.setText(str(horizon) if horizon is not None else "")
        for group, fields in (("constraints", self.bound_fields), ("output_bounds", self.output_fields)):
            for key, bounds in self.state[group].items():
                for field, value in zip(fields[key], bounds):
                    field.setText("" if value is None else str(value))
        self.exploration.setText(str(self.state["exploration"]))
        self.output_margin.setText(str(self.state["output_margin"]))
        for key, (enabled, mode, weight, target, scale) in self.target_fields.items():
            saved = self.state["targets"][key]
            enabled.setChecked(saved["enabled"])
            mode.setCurrentIndex(mode.findData(saved["mode"]))
            for field, value in ((weight, "weight"), (target, "target"), (scale, "scale")):
                field.setText(str(saved[value]))
        self.yield_basis.setCurrentIndex(self.yield_basis.findData(self.state["yield_basis"]))
        self.load_stage_templates()
        self.rendering = False
        self.invalidate_proposal()
        self.refresh_data()
        self.status.setText("Select experiments and check the settings." if self.state["runs"]
                            else "Open a MATLAB case or CrystalLoop campaign to get started.")

    def settings(self):
        data = dict(self.state)
        data["batch_horizon_s"] = self.horizon_override()
        selected_horizon(data)
        data["constraints"] = read_bounds(self.bound_fields)
        data["output_bounds"] = read_bounds(self.output_fields)
        data["exploration"] = float(self.exploration.text())
        data["output_margin"] = float(self.output_margin.text())
        data["targets"] = dict(self.state["targets"])
        for key, (enabled, mode, weight, target, scale) in self.target_fields.items():
            data["targets"][key] = {"enabled": enabled.isChecked(), "mode": mode.currentData(),
                                    "weight": float(weight.text()), "scale": float(scale.text()),
                                    "target": float(target.text()) if target.isEnabled() else 0.}
        data["yield_basis"] = self.yield_basis.currentData()
        data["physical_profiles"] = self.stage_editor.specs()
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

    def show_loop_stages(self):
        stages = self.state["loop_stages"]
        if not stages:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Bayesian optimization | Loop stages")
        dialog.resize(875, 620)
        layout = QVBoxLayout(dialog)
        steps = QListWidget()
        steps.setMaximumHeight(138)
        layout.addWidget(steps)
        tabs = QTabWidget()
        observations = grid(("Experiment", "Source stage", "Duration (s)", "Yield (%)", "D10", "D50", "D90"))
        predictions = grid(("Output", "Unit", "Mean", "Standard deviation"))
        recipe_page = QWidget()
        recipe_layout = QVBoxLayout(recipe_page)
        channels = QComboBox()
        recipe_chart = chart_view()
        recipe_layout.addWidget(channels)
        recipe_layout.addWidget(recipe_chart, 1)
        tabs.addTab(observations, "Training observations")
        tabs.addTab(predictions, "Predictions")
        tabs.addTab(recipe_page, "Proposed recipe")
        layout.addWidget(tabs, 1)
        note = QLabel()
        note.setTextFormat(Qt.TextFormat.PlainText)
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(note)
        snapshot, profiles = {}, {}

        def draw_profile():
            key = channels.currentData()
            chart = profile_chart(key, (("Proposed recipe", profiles[key]),)) if key in profiles else QChart()
            replace_chart(recipe_chart, chart)

        channels.currentIndexChanged.connect(draw_profile)

        def load_settings():
            self.restore({**snapshot, "loop_stages": stages})
            self.tabs.setCurrentIndex(0)
            self.status.setText(f"Loaded {stages[steps.currentRow()]['name']} training data and settings.")
            dialog.accept()

        def execute_recipe():
            try:
                if self.recipe_launcher(profiles, source=stages[steps.currentRow()]["name"]):
                    dialog.accept()
            except ValueError as error:
                note.setText(str(error))

        row = QHBoxLayout()
        load = self.button("Load stage data and settings", load_settings)
        execute = self.button("Execute recipe", execute_recipe)
        execute.setVisible(self.recipe_launcher is not None)
        for button in (load, execute, self.button("Close", dialog.reject)):
            row.addWidget(button)
        layout.addLayout(row)

        def update(index):
            nonlocal snapshot, profiles
            snapshot, profiles = {}, {}
            observations.setRowCount(0)
            predictions.setRowCount(0)
            channels.clear()
            load.setEnabled(False)
            execute.setEnabled(False)
            if index < 0:
                return
            path = Path(stages[index]["path"])
            try:
                snapshot = validate_settings(json.loads(path.read_text(encoding="utf-8")))
                proposal = snapshot.get("loop_proposal", {})
                profiles = {key: make_profile(trace["points"], trace["source"], step=trace["step"])
                            for key, trace in proposal.get("profiles", {}).items()}
                if not all(profiles.values()):
                    raise ValueError("The saved BO recipe contains invalid profiles.")
                for key in profiles:
                    channels.addItem(CHANNELS[key][0], key)
                for number in snapshot["selected"]:
                    run = snapshot["runs"][number]
                    values = run["metrics"]
                    cells = (run["name"], run.get("recipe_stage", "—"),
                             *(show_value(values[key]) for key in
                               ("duration", f"yield_{snapshot['yield_basis']}", "d10", "d50", "d90")))
                    target_row = observations.rowCount()
                    observations.insertRow(target_row)
                    for column, value in enumerate(cells):
                        observations.setItem(target_row, column, item(value))
                for target, (mean, sd) in proposal.get("predictions", {}).items():
                    target_row = predictions.rowCount()
                    predictions.insertRow(target_row)
                    values = (*TARGETS[target][:2], show_value(mean), show_value(sd))
                    for column, value in enumerate(values):
                        predictions.setItem(target_row, column, item(value))
            except (OSError, ValueError, TypeError, KeyError) as error:
                note.setText(str(error))
                return
            detail = f"Acquisition {proposal['acquisition']:.5g}" if proposal else "Final observations"
            note.setText(f"{detail}\n{path}")
            load.setEnabled(not self.loop_busy and self.job is None)
            execute.setEnabled(bool(profiles) and self.recipe_launcher is not None and not self.loop_busy and self.job is None)

        for stage in stages:
            role = "Proposal" if stage["proposed"] else "Final observations"
            steps.addItem(f"{stage['name']} · {role} · {stage['observations']} observations")
        steps.currentRowChanged.connect(update)
        steps.setCurrentRow(len(stages) - 1)
        dialog.exec()
        dialog.deleteLater()

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
                dialog = CampaignImportDialog(campaign_runs(folder), template, self, require_process=False)
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
        self.state.pop("initial_input_groups", None)
        groups = case.get("PythonWorkflow", {}).get("initial_input_groups")
        if groups is not None:
            self.state["initial_input_groups"] = [str(value) for value in np.asarray(groups).reshape(-1)]
        self.state.update(source=source, runs=runs, selected=list(range(len(runs))), overrides={},
                          physical_profiles={})
        self.load_stage_templates()
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
        self.refresh_horizon()
        self.refresh_channels()

    def horizon_override(self):
        text = self.batch_horizon.text().strip()
        try:
            return float(text) if text else None
        except ValueError as error:
            raise ValueError("Enter the batch horizon in seconds or leave it blank for automatic.") from error

    def profile_horizon(self):
        return selected_horizon({**self.state, "batch_horizon_s": self.horizon_override()})

    def refresh_horizon(self):
        try:
            horizon = self.profile_horizon()
            seeds = [self.state["runs"][i]["profiles"].get("seed") for i in self.state["selected"]]
            seeded = any(seed and seed["points"][1][0] > 0 for seed in seeds)
            note = "Seeded runs start with seed at 0 s." if seeded else "Unseeded (0 g seed)." if any(seeds) else ""
            detail = f"Effective horizon {horizon:g} s · {note}" if horizon else ""
        except ValueError as error:
            horizon, detail = None, str(error)
        self.stage_editor.set_horizon(horizon)
        self.horizon_label.setText(detail)
        self.batch_note.setText(detail)

    def horizon_changed(self):
        if not self.rendering:
            self.invalidate_proposal()
            self.refresh_horizon()
            self.refresh_preview()

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
        self.sync_stage_channel()
        self.refresh_preview()

    def refresh_preview(self):
        self.variables.setRowCount(0)
        self.fit_summary.setRowCount(0)
        replace_chart(self.chart, QChart())
        self.metrics.setText("Compare the measured trace with the stage template fitted to that run.")
        self.update_buttons()

    def load_stage_templates(self):
        runs = [self.state["runs"][index] for index in self.state["selected"]]
        specs = self.state["physical_profiles"]
        for channel in STAGE_CHANNELS:
            if channel not in specs and (spec := initial_spec(channel, runs)) is not None:
                specs[channel] = spec
        try:
            horizon = self.profile_horizon()
        except ValueError:
            horizon = None  # Keep the stage editor available while the horizon is corrected.
        self.stage_editor.load(specs, horizon)

    def reset_stage_templates(self):
        self.state["physical_profiles"] = {}
        self.load_stage_templates()
        self.stage_settings_changed()
        self.status.setText("Reset the stage templates from the selected experiments.")

    def stage_settings_changed(self):
        self.invalidate_proposal()
        self.refresh_preview()

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

    def update_buttons(self):
        if not hasattr(self, "propose_button"):
            return
        available = self.job is None and not self.loop_busy
        channel = self.channel_choice.currentData()
        self.propose_button.setEnabled(available and len(self.state["selected"]) >= 2)
        self.preview_button.setEnabled(available and channel is not None)
        self.use_trace_button.setEnabled(available and channel in STAGE_CHANNELS)
        self.reset_stages_button.setEnabled(available and bool(self.state["selected"]))
        self.export_template_button.setEnabled(available and bool(self.state["selected"]))
        self.export_button.setEnabled(self.proposal is not None)
        self.execute_button.setEnabled(available and self.proposal is not None and self.recipe_launcher is not None)
        self.stages_button.setEnabled(bool(self.state["loop_stages"]))

    def run_job(self, task, callback, *, check_profile=False):
        if self.job is not None or self.loop_busy:
            return
        self.status.setText("Preparing Bayesian optimization…")
        self.job = BOJob(task)
        self.completed_action = callback
        self.active_revision = self.revision
        self.active_profile = self.comparison_selection() if check_profile else None
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
        if self.active_profile is not None and self.active_profile != self.comparison_selection():
            self.status.setText("The selected profile changed. Compare the current profile to view its fit.")
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

    def comparison_selection(self):
        return (*self.profile_selection(), self.compare_all.isChecked())

    def sync_stage_channel(self):
        editor = self.stage_editor.editors.get(self.channel_choice.currentData())
        if editor is not None:
            self.stage_editor.tabs.setCurrentWidget(editor)

    def sync_comparison_channel(self):
        channel = self.stage_editor.tabs.currentWidget().channel
        index = self.channel_choice.findData(channel)
        if index >= 0:
            self.channel_choice.setCurrentIndex(index)

    def use_trace_template(self):
        index, key = self.profile_selection()
        if index is None or key not in STAGE_CHANNELS:
            return
        try:
            replacement = trace_spec(key, self.profile(self.state["runs"][index], key), exact=True)
            horizon = self.profile_horizon()
            # Read the other editor independently so an invalid template can be replaced.
            specs = {other: spec for other, editor in self.stage_editor.editors.items() if other != key
                     and (spec := editor.spec()) is not None}
        except ValueError as error:
            self.status.setText(str(error))
            return
        specs[key] = replacement
        self.stage_editor.load(specs, horizon)
        self.stage_editor.tabs.setCurrentWidget(self.stage_editor.editors[key])
        self.profile_tabs.setCurrentIndex(0)
        self.stage_settings_changed()
        self.status.setText(f"Created {CHANNELS[key][0].lower()} stages from the trace’s profile changes. "
                            "Select the controls you want BO to vary.")

    def preview(self):
        index, key, compare_all = self.comparison_selection()
        if index is None or key is None:
            return
        try:
            state = self.settings()
        except ValueError as error:
            self.status.setText(str(error))
            return

        def prepare(report):
            from .optimizer import preview_channel, run_profile, variable_activity

            channel, runs, row = preview_channel(state, key, index, report)
            horizon = selected_horizon(state)
            rows = list(range(len(runs))) if compare_all else [row]
            decoded = channel.decode(channel.features[rows], [horizon] * len(rows))
            comparisons = [(runs[i]["name"], run_profile(runs[i], key), profile) for i, profile in zip(rows, decoded)]
            return channel, comparisons, row, variable_activity(channel.features)

        self.run_job(prepare, self.show_fit, check_profile=True)

    def show_fit(self, result):
        channel, comparisons, reference, varied = result
        key = channel.key
        features = channel.features[reference]
        self.render_comparison(key, comparisons, channel.stages)
        self.variables.blockSignals(True)
        self.variables.setRowCount(len(channel.variables))
        for row, (variable, value) in enumerate(zip(channel.variables, features)):
            cell = item("Varies" if varied[row] else "Fixed")
            if channel.stages is not None:
                tooltip = "Selected in Edit stages. "
                tooltip += ("Fitted values vary across the selected experiments." if varied[row]
                            else "The fitted value is the same across the selected experiments.")
                cell.setToolTip(tooltip)
            if channel.stages is None and varied[row]:
                cell.setText("")
                cell.setData(Qt.ItemDataRole.UserRole, variable.key)
                cell.setFlags(cell.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                cell.setCheckState(Qt.CheckState.Checked if self.state["overrides"].get(variable.key, True)
                                   else Qt.CheckState.Unchecked)
            self.variables.setItem(row, 0, cell)
            for column, content in enumerate((variable.label, show_value(float(value)), variable.unit), 1):
                self.variables.setItem(row, column, item(content))
        self.variables.blockSignals(False)
        self.status.setText("Review the fit before generating a recipe.")

    def render_comparison(self, key, comparisons, stages):
        _, unit, _ = CHANNELS[key]
        headers = ["Experiment", f"RMSE ({unit})"]
        if key == "feed":
            headers.extend(("Trace dose (g)", "Fitted dose (g)"))
        self.fit_summary.setColumnCount(len(headers))
        self.fit_summary.setHorizontalHeaderLabels(headers)
        self.fit_summary.setRowCount(len(comparisons))
        traces, styles, errors = [], [], []
        multiple = len(comparisons) > 1
        for row, (name, original, decoded) in enumerate(comparisons):
            observed = "Commanded" if original.source.startswith("Command ·") else "Recorded"
            prefix = f"{row + 1} · " if multiple else ""
            traces.extend(((prefix + observed, original), (prefix + "Fitted", decoded)))
            color = COLORS[row % len(COLORS)]
            styles.extend(((color, True), (color, False)))
            difference = original.values - decoded.sample(original.time)
            rmse = float(np.sqrt(np.mean(difference**2)))
            errors.append(rmse)
            values = [prefix + name, show_value(rmse)]
            if key == "feed":
                values += [show_value(profile.total()) for profile in (original, decoded)]
            for column, value in enumerate(values):
                self.fit_summary.setItem(row, column, item(value))
        if stages is not None:
            traces.insert(0, ("Template", physical_profile(key, stages, self.profile_horizon())))
            styles.insert(0, ("#626b89", False))
        replace_chart(self.chart, profile_chart(key, traces, styles=styles))
        if multiple:
            detail = f"{len(comparisons)} experiments · RMSE {min(errors):.5g}–{max(errors):.5g} {unit}"
            detail += " · Dashed traces and solid fits share a color per experiment."
        else:
            _, original, decoded = comparisons[0]
            detail = f"{observed} vs. fitted RMSE {errors[0]:.5g} {unit}"
            if key == "feed":
                detail += f" · total dose {original.total():.5g} → {decoded.total():.5g} g"
            if key == "temperature":
                _, rates = constraint_values(key, decoded)
                detail += f" · peak fitted cooling {max(rates, default=0.):.5g} °C/min"
                cooling = cooling_landmarks(decoded)
                if cooling is not None:
                    detail += f" · 10–90% cooling {cooling[-1] - cooling[0]:.5g} s"
        if self.compare_all.isChecked():
            missing = len(self.state["selected"]) - len(comparisons)
            if missing:
                detail += f" · {missing} selected experiments have no {CHANNELS[key][0].lower()} trace."
        self.metrics.setText(detail)
        self.fit_tabs.setCurrentWidget(self.fit_summary if multiple else self.variables)

    def update_override(self, cell):
        if cell.column() == 0 and cell.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            self.state["overrides"][cell.data(Qt.ItemDataRole.UserRole)] = cell.checkState() == Qt.CheckState.Checked
            self.invalidate_proposal()

    def save_template(self):
        try:
            from .optimizer import input_checks, run_profile

            state = self.settings()
            horizon = selected_horizon(state)
            stages, profiles = {}, {}
            for key, spec in state["physical_profiles"].items():
                rows, _ = read_spec(key, spec, horizon)
                stages[key] = rows
                profiles[key] = physical_profile(key, rows, horizon)
            if "temperature" not in profiles:
                raise ValueError("Add temperature stages before saving a recipe.")
            reference = state["runs"][self.run_choice.currentData()]
            for key in ("seed", "stirring", "volume"):
                if key in reference["profiles"] and key not in profiles:
                    profile = run_profile(reference, key)
                    profiles[key] = fixed_horizon(profile, horizon) if key == "stirring" else profile
            failed = input_checks(profiles, horizon, state["constraints"])
            if failed:
                raise ValueError("The template exceeds input bounds for " + ", ".join(failed) + ".")
            payload = recipe_data(profiles, stages, horizon)
            payload.update(format="crystalloop-bo-template", version=1)
            path, _ = QFileDialog.getSaveFileName(self, "Save template recipe", "bo_stage_recipe.json", "JSON (*.json)")
            if path:
                atomic_json(Path(path), payload)
                self.status.setText(f"Saved the stage recipe to {Path(path).name}.")
        except (ValueError, OSError) as error:
            self.status.setText(str(error))

    def suggest(self):
        try:
            state = self.settings()
            for channel in STAGE_CHANNELS:
                if (channel not in state["physical_profiles"]
                        and any(channel in state["runs"][index]["profiles"] for index in state["selected"])):
                    raise ValueError(f"Add {CHANNELS[channel][0].lower()} stages or reset the templates.")
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
                f"Fixed horizon {result.horizon:g} s · {active} varying inputs · "
                f"{result.output_passed}/{result.input_passed} screened candidates passed. "
                f"Predicted loss {result.mean_loss:.5g} ± {result.loss_sd:.4g} SD · "
                f"exploration bonus {beta * result.loss_sd:.4g} · acquisition {result.acquisition:.5g}. "
                "The prediction table shows mean ± 1.96 SD.")
            errors = [
                f"{CHANNELS[key][0]} {np.median(channel.fit_errors):.4g} {CHANNELS[key][1]}"
                for key, channel in result.channels.items() if channel.fit_errors is not None
            ]
            self.status.setText("Proposal ready. Median stage fit RMSE · " + " · ".join(errors) if errors
                                else "Proposal ready. Review the recipe and output predictions.")
        else:
            self.proposal_note.setText("Select experiments, set the targets and bounds, then propose a recipe.")
        self.update_buttons()

    def execute_proposal(self):
        if self.proposal is None or self.recipe_launcher is None or self.loop_busy or self.job is not None:
            return
        try:
            if self.recipe_launcher(self.proposal.profiles):
                self.status.setText("BO recipe sent to the experiment runner.")
        except ValueError as error:
            self.status.setText(str(error))

    def save_proposal(self):
        result = self.proposal
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save proposed recipe", "bo_proposed_recipe.json", "JSON (*.json)")
        if not path:
            return
        file = Path(path)
        payload = {
            **recipe_data(result.profiles, result.stages, result.horizon),
            "format": "crystalloop-bo-proposal", "version": 2,
            "yield_basis": result.state["yield_basis"],
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
