"""PySide6 port of the MATLAB ControlPanel workflow and loaded-case inspection."""
from pathlib import Path

import numpy as np
from PySide6.QtCharts import QChartView
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from .candidate_workers import max_cpu_workers
from .model_file import save_checkpoint
from .parameter_labels import PARAMETERS
from .solver import read_model
from .solver_choice import SolverChoice
from .session import history, record_step


def table(headers):
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    return widget


def fill_table(widget, rows):
    widget.setRowCount(len(rows))
    for row, values in enumerate(rows):
        for column, value in enumerate(values):
            display = f"{value:.7g}" if isinstance(value, (float, np.floating)) else str(value)
            widget.setItem(row, column, QTableWidgetItem(display))
    widget.resizeColumnsToContents()


def item_count(value):
    if isinstance(value, dict):
        return int(bool(value))
    if isinstance(value, np.ndarray):
        return value.size
    if isinstance(value, (tuple, list)):
        return len(value)
    return int(value is not None)


class ControlPanel(QMainWindow):
    WIDTH = 687
    HEIGHT = 538

    def __init__(self, simulation_window, parent=None):
        super().__init__(parent)
        if parent is not None:
            self.setWindowFlags(Qt.WindowType.Widget)
        self.simulation_window = simulation_window
        self.simulator = None
        self.case = {}
        self.source = None
        self.campaign_path = None
        self.parameter_edits = {}
        self.setting_edits = {}
        self.message_label = None
        self.setWindowTitle("IMED | Control Panel")
        self.body = QWidget()
        self.body.setObjectName("imedWorkflow")
        self.setCentralWidget(self.body)
        self.status = QPlainTextEdit(self.body)
        self.status.setReadOnly(True)
        self.checklist = QFrame(self.body)
        self.checks = {}
        for key, label in (
            ("Solubility", "Solubility"), ("ProcInf", "Process Inf."),
            ("ExpData", "Experimental Data"), ("ModelParameter", "Model parameters"),
            ("MDIteration", "Iterations"),
        ):
            check = QCheckBox(label, self.checklist)
            check.setEnabled(False)
            self.checks[key] = check

        self.import_button = self.button("Import MATLAB (.mat)", self.choose_file)
        self.save_button = self.button("Save Model", self.save_case)
        self.campaign_button = self.button("Import CrystalLoop", self.choose_campaign)
        self.input_buttons = [self.button("Initial DoE / Inputs", self.show_inputs)]
        self.configurations = self.button("Model Configurations", self.show_configurations)
        self.estimation_buttons = [self.button("Parameter Estimation", self.show_estimation)]
        self.uncertainty_button = self.button("Parameter Uncertainty", self.show_uncertainty)
        self.uncertainty_button.setToolTip("Calculate local confidence intervals at the current "
                                           "parameter values using measured data.")
        self.md_button = self.button("Model Discrimination", lambda: self.show_design(False))
        self.dopt_button = self.button("D-Optimal Design", lambda: self.show_design(True))
        self.design = self.button("Digital Design", self.show_simulation)
        self.reliability = self.button("Model reliability", lambda: self.show_probabilistic(False))
        self.sensitivity = self.button("Probabilistic sensitivity", lambda: self.show_probabilistic(True))
        self.history_button = self.button("Executed steps", self.show_history)
        self.design.setToolTip("Open forward simulation. Sensitivity analysis is pending.")
        self.numerics = SolverChoice(self.body)
        self.workers = QSpinBox(self.body)
        self.workers.setRange(1, max_cpu_workers())
        self.workers.setSpecialValueText("1 (serial)")
        self.workers.setToolTip("CPU workers for the global search in Parameter Estimation, "
                                "Model Discrimination and D-Optimal Design with a SciPy solver.")
        self.numerics.currentIndexChanged.connect(
            lambda: self.workers.setEnabled(self.numerics.currentData()[0] != "jax"))
        if parent is None:
            self._build_diagram()
        else:
            self._build_embedded()
        self.show_message("Import a MATLAB IMED case or a CrystalLoop recording.")
        self.refresh_status()

    def _build_embedded(self):
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        setup = QGroupBox("Case setup")
        setup_layout = QVBoxLayout(setup)
        self.import_button.setProperty("imedPrimary", True)
        for button in (self.import_button, self.campaign_button, self.save_button):
            button.setMinimumHeight(42)
            setup_layout.addWidget(button)
        solver = QFormLayout()
        self.numerics.setMinimumHeight(38)
        solver.addRow("PBM solver", self.numerics)
        solver.addRow("CPU workers", self.workers)
        setup_layout.addLayout(solver)
        self.message_label = QLabel()
        self.message_label.setTextFormat(Qt.TextFormat.PlainText)
        self.message_label.setWordWrap(True)
        setup_layout.addWidget(self.message_label)
        layout.addWidget(setup)

        workflow = QGroupBox("Modeling workflow")
        workflow_layout = QVBoxLayout(workflow)
        self.design.setProperty("imedPrimary", True)
        for button in (*self.input_buttons, self.configurations, *self.estimation_buttons,
                       self.uncertainty_button, self.md_button, self.dopt_button, self.design,
                       self.reliability, self.sensitivity, self.history_button):
            button.setMinimumHeight(42)
            workflow_layout.addWidget(button)
        layout.addWidget(workflow)

        status = QGroupBox("Case status")
        status_layout = QVBoxLayout(status)
        self.status.setFixedHeight(105)
        status_layout.addWidget(self.status)
        checklist = QGridLayout(self.checklist)
        checklist.setContentsMargins(0, 0, 0, 0)
        for index, check in enumerate(self.checks.values()):
            checklist.addWidget(check, index // 2, index % 2)
        status_layout.addWidget(self.checklist)
        layout.addWidget(status)
        layout.addStretch()

    def _build_diagram(self):
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        artwork = QLabel(self.body)
        artwork.setPixmap(QPixmap(str(Path(__file__).with_name("workflow.png"))).scaled(
            548, 430, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        artwork.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.place(artwork, -5, 43, 548, 430)
        artwork.lower()

        heading = QLabel("Current Status:", self.body)
        heading_font = QFont("Arial")
        heading_font.setBold(True)
        heading.setFont(heading_font)
        self.place(heading, 334, 451, 130, 22)
        status_font = QFont(heading_font)
        status_font.setPixelSize(11)
        self.status.setFont(status_font)
        self.place(self.status, 302, 345, 177, 101)
        self.checklist.setFrameShape(QFrame.Shape.StyledPanel)
        self.place(self.checklist, 495, 184, 162, 197)
        for index, check in enumerate(self.checks.values()):
            check.setGeometry(15, 22 + 32 * index, 140, 24)

        self.input_buttons.extend(self.button("Experimental Input", self.show_inputs) for _ in range(2))
        self.estimation_buttons.append(self.button("Parameter Estimation", self.show_estimation))
        for widget, position in (
            (self.input_buttons[0], (106, 425, 155, 30)),
            (self.input_buttons[1], (302, 257, 157, 20)),
            (self.input_buttons[2], (104, 71, 156, 20)),
            (self.configurations, (107, 367, 154, 22)),
            (self.estimation_buttons[0], (106, 295, 154, 22)),
            (self.estimation_buttons[1], (302, 70, 154, 22)),
            (self.uncertainty_button, (302, 222, 157, 22)),
            (self.md_button, (105, 222, 156, 22)),
            (self.dopt_button, (106, 148, 157, 20)),
            (self.design, (302, 148, 157, 20)),
            (self.reliability, (302, 120, 157, 20)),
            (self.sensitivity, (302, 92, 157, 20)),
        ):
            font = QFont("Arial")
            font.setPixelSize(11)
            font.setBold(True)
            widget.setFont(font)
            self.place(widget, *position)
        self.place(self.import_button, 495, 443, 162, 26)
        self.place(self.save_button, 495, 404, 162, 26)
        self.place(self.campaign_button, 495, 474, 162, 26)
        self.place(QLabel("PBM solver", self.body), 495, 145, 162, 20)
        self.place(self.numerics, 495, 118, 162, 26)
        self.place(QLabel("CPU workers", self.body), 495, 90, 162, 20)
        self.place(self.workers, 495, 62, 162, 26)
        self.place(self.history_button, 495, 24, 162, 26)

    def place(self, widget, x, bottom, width, height):
        # MATLAB figure coordinates are measured up from the bottom.
        widget.setGeometry(x, self.HEIGHT - bottom - height, width, height)

    def button(self, label, action):
        button = QPushButton(label, self.body)
        button.clicked.connect(action)
        return button

    def show_message(self, message):
        if self.message_label is None:
            self.statusBar().showMessage(message)
        else:
            self.message_label.setText(message)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import MATLAB IMED case", "", "MAT files (*.mat)")
        if path:
            self.open_case(path)

    def open_case(self, path):
        try:
            case = read_model(path)
        except Exception as error:
            QMessageBox.warning(self, "Import MATLAB", str(error))
            return
        self.case = case
        self.source = Path(path)
        self.campaign_path = None
        self.parameter_edits.clear()
        self.setting_edits.clear()
        self.numerics.load_settings(case.get("pbmsetting"))
        workflow = case.get("PythonWorkflow", {})
        self.workers.setValue(int(workflow.get("cpu_workers", 1)))
        self.show_message(f"Imported {self.source.name}")
        self.refresh_status()

    def choose_campaign(self):
        from .campaign_dialog import CampaignImportDialog
        from .campaign_import import campaign_runs, read_json
        from .demo_campaign import import_simulated_campaign

        folder = QFileDialog.getExistingDirectory(self, "Select CrystalLoop campaign folder")
        if not folder:
            return
        try:
            simulated = import_simulated_campaign(folder)
            definition = read_json(Path(folder) / "definition.json")
            doe_factors = definition.get("campaign", {}).get("doe_factors")
            if simulated is not None:
                self.case = simulated
                self.source = None
                self.parameter_edits.clear()
                self.setting_edits.clear()
                self.numerics.load_settings(simulated.get("pbmsetting"))
                self.use_campaign(folder, simulated["ExpData"], simulated["ProcInf"], doe_factors=doe_factors)
                return
            dialog = CampaignImportDialog(campaign_runs(folder), self.case, self)
        except (ValueError, OSError, KeyError) as error:
            QMessageBox.warning(self, "Import CrystalLoop", str(error))
            return
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.use_campaign(folder, dialog.experiments, dialog.processes, doe_factors=doe_factors)
        dialog.deleteLater()

    def use_campaign(self, folder, experiments, processes, *, doe_factors=None):
        self.campaign_path = Path(folder)
        self.case.update(ExpData=experiments, ProcInf=processes)
        history(self.case)
        workflow = self.case["PythonWorkflow"]
        workflow.pop("profile_doe_factors", None)
        if doe_factors is None:
            workflow.pop("doe_factors", None)
        else:
            workflow["doe_factors"] = np.asarray(doe_factors, dtype=object)
        for key in ("MDIteration", "MDsimulation", "PPsimulation", "summaryTable"):
            self.case.pop(key, None)
        old_weights = self.case.get("objweightfactor")
        weights = dict(old_weights) if isinstance(old_weights, dict) else {"W1": 0, "W2": 1, "W3": 1}
        for weight, field in (("W1", "CountsData"), ("W2", "ConcentrationData"),
                              ("W3", "FinalD10D50D90")):
            if not all(exp[field].size for exp in experiments):
                weights[weight] = 0
        weights["Terms"] = "D10 D50 D90"
        self.case["objweightfactor"] = weights
        record_step(self.case, "campaign_import", {"folder": str(folder), "runs": len(experiments)})
        self.refresh_status()
        message = f"Imported {len(experiments)} CrystalLoop run(s)."
        if not self.design.isEnabled():
            message += " Open Experimental Input to inspect recordings."
        self.show_message(message)

    def save_case(self):
        suggested = (self.source.with_name(f"{self.source.stem}_python.mat")
                     if self.source is not None else Path("imed_checkpoint.mat"))
        path, _ = QFileDialog.getSaveFileName(self, "Save IMED checkpoint", str(suggested),
                                              "MAT files (*.mat)")
        if not path:
            return
        destination = Path(path)
        if not destination.suffix:
            destination = destination.with_suffix(".mat")
        try:
            case = self.simulation_case()
            case["PythonWorkflow"]["cpu_workers"] = self.workers.value()
            save_checkpoint(self.source, destination, case)
        except Exception as error:
            QMessageBox.warning(self, "Save Model", str(error))
        else:
            self.show_message(f"Saved complete IMED checkpoint {destination.name}")

    def refresh_status(self):
        counts = {key: item_count(self.case.get(key)) for key in
                  (*self.checks, "MDsimulation", "PPsimulation", "summaryTable")}
        for key, check in self.checks.items():
            check.setChecked(counts[key] > 0)
        self.status.setPlainText("\n".join((
            f"Experiments: {counts['ExpData']}",
            f"Models: {counts['ModelParameter']}",
            f"Discrimination runs: {counts['MDsimulation']}",
            f"D-optimal runs: {counts['PPsimulation']}",
            f"Estimation runs: {counts['MDIteration'] + bool(counts['summaryTable'])}",
            f"Executed Python steps: {len(history(self.case))}",
        )))
        for button in self.input_buttons:
            button.setEnabled(any(counts[key] for key in ("Solubility", "ProcInf", "ExpData")))
        self.configurations.setEnabled(counts["ModelParameter"] > 0)
        self.save_button.setEnabled(counts["ModelParameter"] > 0 or counts["ExpData"] > 0)
        self.save_button.setToolTip("Save current IMED inputs, results, confidence intervals and executed steps.")
        self.history_button.setEnabled(bool(history(self.case)))
        self.design.setEnabled(
            counts["ExpData"] > 0 and counts["ProcInf"] >= counts["ExpData"]
            and counts["Solubility"] > 0 and counts["ModelParameter"] > 0)
        for button in self.estimation_buttons:
            button.setEnabled(self.design.isEnabled() and counts["ProcInf"] == counts["ExpData"])
        self.uncertainty_button.setEnabled(self.estimation_buttons[0].isEnabled())
        self.reliability.setEnabled(self.design.isEnabled())
        self.sensitivity.setEnabled(self.design.isEnabled())
        fitted = [model for model in self.case.get("ModelParameter", [])
                  if item_count(model.get("para"))]
        self.md_button.setEnabled(self.design.isEnabled() and len(fitted) >= 2)
        self.md_button.setToolTip("Requires a loaded case and at least two fitted models.")
        self.dopt_button.setEnabled(self.design.isEnabled()
                                    and any(item_count(model.get("covmat")) for model in fitted))
        self.dopt_button.setToolTip("Requires a fitted model with parameter covariance.")

    def dialog(self, title):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(760, 510)
        return dialog, QVBoxLayout(dialog)

    @staticmethod
    def show_dialog(dialog, layout):
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()
        dialog.deleteLater()

    def show_inputs(self):
        dialog, layout = self.dialog("IMED | Initial DoE / Experimental input")
        layout.addWidget(QLabel("Review saved experimental inputs"))
        tabs = QTabWidget()
        layout.addWidget(tabs)
        solubility = table(("Coefficient", "Value"))
        coefficients = self.case.get("Solubility")
        if isinstance(coefficients, dict):
            fill_table(solubility, [(key, coefficients[key]) for key in "abcdefgh" if key in coefficients])
        tabs.addTab(solubility, "Solubility Curve")

        experiments = self.case.get("ExpData", [])
        processes = self.case.get("ProcInf", [])
        for label, entries, is_process in (
            ("Process Information", processes, True), ("ExpData Content", experiments, False),
        ):
            page = QWidget()
            page_layout = QVBoxLayout(page)
            choices = QComboBox()
            choices.addItems(entry.get("Name", f"Experiment {index + 1}")
                             for index, entry in enumerate(entries))
            values = table(("Field", "Value" if is_process else "Stored data"))

            def update(index, entries=entries, values=values, is_process=is_process):
                if index < 0:
                    return
                fill_table(values, [
                    (key, value if is_process or not isinstance(value, np.ndarray)
                     else " × ".join(map(str, value.shape)))
                    for key, value in entries[index].items()
                ])

            choices.currentIndexChanged.connect(update)
            update(choices.currentIndex())
            page_layout.addWidget(choices)
            page_layout.addWidget(values)
            tabs.addTab(page, label)
        if self.campaign_path is not None:
            self.add_recording_tab(tabs, experiments)
            tabs.setCurrentIndex(tabs.count() - 1)
        self.show_dialog(dialog, layout)

    def add_recording_tab(self, tabs, experiments):
        page = QWidget()
        layout = QVBoxLayout(page)
        runs = QComboBox()
        runs.addItems(experiment["Name"] for experiment in experiments)
        signals = QComboBox()
        for field, label in (
            ("TemperatureData", "Temperature (°C)"), ("ConcentrationData", "Concentration (g/g solution)"),
            ("CountsData", "FBRM counts"), ("D10Data", "D10 (µm)"), ("D50Data", "D50 (µm)"),
            ("D90Data", "D90 (µm)"), ("AntisolventScaleData", "AS scale (g)"),
            ("AddedAntisolventData", "Estimated antisolvent added (g)"),
            ("FlowRateData", "Estimated antisolvent feed (g/s)"),
        ):
            signals.addItem(label, field)
        status = QLabel()
        view = QChartView()
        for widget in (runs, signals, status, view):
            layout.addWidget(widget)

        def refresh():
            points = experiments[runs.currentIndex()][signals.currentData()]
            count = points.shape[1]
            status.setText(f"{count:,} samples" if count else "No valid recorded samples")
            plot = points[:, np.linspace(0, count - 1, min(count, 2000), dtype=int)]
            lines = [(signals.currentText(), plot)] if count else []
            self.simulation_window.set_chart(view, lines, ("Time since import start (s)", signals.currentText()))

        runs.currentIndexChanged.connect(refresh)
        signals.currentIndexChanged.connect(refresh)
        refresh()
        tabs.addTab(page, "Recorded channels")

    def show_configurations(self):
        dialog, layout = self.dialog("IMED | Model Configurations")
        layout.addWidget(QLabel("Edit parameter values, bounds, and estimation flags. "
                                "Changes clear fitted results for the affected model."))
        models = self.case["ModelParameter"]
        tabs = QTabWidget()
        layout.addWidget(tabs)
        tables = {}
        for index, model in enumerate(models):
            parameters = table(("Parameter", "Meaning", "Nominal", "Lower Bound", "Upper Bound", "Est."))
            tabs.addTab(parameters, f"Model {index + 1}")
            entries = np.asarray(model.get("paraset", []), dtype=float)
            if entries.shape != (len(PARAMETERS), 4):
                continue
            fill_table(parameters, [
                (name, meaning, str(nominal), str(lower), str(upper), "")
                for (name, meaning), (estimate, nominal, lower, upper) in zip(PARAMETERS, entries)
            ])
            for row, estimate in enumerate(entries[:, 0]):
                for column in (0, 1, 5):
                    item = parameters.item(row, column)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                flag = parameters.item(row, 5)
                flag.setFlags(flag.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                flag.setCheckState(Qt.CheckState.Checked if estimate else Qt.CheckState.Unchecked)
            parameters.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                       | QAbstractItemView.EditTrigger.EditKeyPressed)
            tables[index] = (parameters, entries)

        def apply_changes():
            changes = {}
            for index, (parameters, original) in tables.items():
                updated = original.copy()
                for row, (name, _) in enumerate(PARAMETERS):
                    try:
                        active = parameters.item(row, 5).checkState() == Qt.CheckState.Checked
                        nominal, lower, upper = (float(parameters.item(row, c).text())
                                                 for c in (2, 3, 4))
                        values = np.array([active, nominal, lower, upper])
                        if np.array_equal(values, original[row], equal_nan=True):
                            continue
                        if active and (not np.isfinite(values).all() or not lower <= nominal <= upper):
                            raise ValueError("An estimated parameter needs finite bounds around its nominal value.")
                        if not active:
                            if not np.isfinite(nominal) and not (name.startswith("log_") and nominal == -np.inf):
                                raise ValueError("Use a finite nominal value or -inf for a disabled log rate.")
                            values[2:] = nominal
                        updated[row] = values
                    except ValueError as error:
                        tabs.setCurrentWidget(parameters)
                        parameters.selectRow(row)
                        QMessageBox.warning(dialog, f"Model {index + 1} · {name}", str(error))
                        return
                if not np.array_equal(updated, original, equal_nan=True):
                    changes[index] = updated
            for index, updated in changes.items():
                models[index] = {"paraset": updated, "para": np.empty(0)}
                self.parameter_edits[index] = models[index]
                record_step(self.case, "model_configuration", {"paraset": updated}, model_index=index)
            if changes:
                self.refresh_status()
                self.show_message("Updated model parameters. Use Save Model to write a MAT file.")
            dialog.accept()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                  | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(apply_changes)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()
        dialog.deleteLater()

    def simulation_case(self):
        history(self.case)
        settings = self.numerics.apply_settings(self.case.get("pbmsetting"))
        return {**self.case, "pbmsetting": settings}

    def prediction_case(self):
        case = self.simulation_case()
        creator = getattr(self.window(), "create_experiments", None)
        if creator is not None and "doe_factors" not in case["PythonWorkflow"]:
            from experiment.doe_campaign import sampled_doe_factors

            factors = sampled_doe_factors(creator)
            if factors:
                case["active_doe_factors"] = factors
        return case

    def cpu_workers(self):
        return 1 if self.numerics.currentData()[0] == "jax" else self.workers.value()

    def show_estimation(self):
        from .estimation_dialog import ParameterEstimation

        dialog = ParameterEstimation(self.simulation_case(), self, workers=self.cpu_workers())
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        if dialog.saved_settings:
            self.case.update(dialog.saved_settings)
            self.setting_edits.update(dialog.saved_settings)
            self.show_message("Saved PE settings for this case. Use Save Model to write a MAT file.")
        if accepted:
            for index, model in dialog.fitted_models.items():
                self.case["ModelParameter"][index] = model
                self.parameter_edits[index] = model
            self.refresh_status()
            self.show_message("Applied fitted parameters. Use Save Model to write a MAT file.")
        self.refresh_status()
        dialog.deleteLater()

    def show_design(self, optimal):
        from .design_dialog import DesignDialog

        dialog = DesignDialog(self.simulation_case(), optimal, self, workers=self.cpu_workers())
        dialog.exec()
        dialog.deleteLater()
        self.refresh_status()

    def show_uncertainty(self):
        from .uncertainty_dialog import ParameterUncertainty

        dialog = ParameterUncertainty(self.simulation_case(), self)
        dialog.exec()
        dialog.deleteLater()
        self.refresh_status()

    def show_history(self):
        dialog, layout = self.dialog("IMED | Executed steps")
        steps = table(("Completed (UTC)", "Step", "Model"))
        fill_table(steps, [(step["completed_at"], step["action"].replace("_", " "),
                            int(step["model_index"]) + 1 if int(step["model_index"]) >= 0 else "—")
                           for step in history(self.case)])
        layout.addWidget(steps)
        self.show_dialog(dialog, layout)

    def show_probabilistic(self, sensitivity):
        from .probabilistic_dialog import ProbabilisticDialog

        case = self.prediction_case() if sensitivity else self.simulation_case()
        dialog = ProbabilisticDialog(case, self, sensitivity=sensitivity,
                                     workers=self.cpu_workers())
        dialog.exec()
        dialog.deleteLater()
        self.refresh_status()

    def show_simulation(self):
        if self.simulator is not None and self.simulator.isVisible():
            self.simulator.raise_()
            self.simulator.activateWindow()
            return
        if self.simulator is None:
            self.simulator = self.simulation_window(self)
            if hasattr(self.simulator, "workflow_changed"):
                self.simulator.workflow_changed.connect(self.refresh_status)
        self.simulator.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.simulator.setWindowTitle("IMED | Digital Design | Forward simulation")
        self.simulator.set_case(self.prediction_case(),
                                "Choose a model and experiment, then run the forward simulation.")
        self.simulator.show()

    def close_simulation(self):
        return self.simulator is None or self.simulator.close()

    def closeEvent(self, event):
        if not self.close_simulation():
            event.ignore()
            return
        super().closeEvent(event)
