"""Parameter Estimation dialog backed by the standalone fitting engine."""
from datetime import datetime
from functools import partial
import json
from pathlib import Path
from math import isfinite
from threading import Event

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)

from experiment.recording import atomic_json

from .control_panel import fill_table, table
from .estimation import FitObjective, fit_model
from .mesh_grid import DEFAULT_LOG_OFFSET, is_log_grid
from .model_screening import matlab_screening_fields, pack_screening, screen_models, unpack_screening
from .native_case import NATIVE_FILTER, save_native_case
from .pe_settings import (PE_PBM_KEY, PE_WEIGHTS_KEY, PYFIT_DEFAULTS, PYFIT_SETTINGS_KEY, WEIGHT_DEFAULTS,
                          effective_pe_pbm, pbm_for_pe, pe_file, pe_mesh, read_pe_file)
from .parameter_labels import PARAMETERS
from .solver import pbm_backend_label
from .size_data import SIZE_SAMPLE_OPTIONS, TREND_SIZE_TERM
from .session import display_name, history, latest_pe_step, pack_fit, pe_signatures, record_step, unpack_fit


class AdvancedSettings(QDialog):
    def __init__(self, pbm, optimizer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("IMED | Advanced PE Settings")
        self.resize(810, 435)
        self.pbm, self.optimizer = dict(pbm), dict(optimizer)
        mode = pbm.get("Numerics", "conservative")
        jax = mode == "jax"
        defaults = pbm_for_pe({"Numerics": mode})

        self.mesh_fields = (
            ("Lmin", "Minimum size, Lmin (µm)", defaults["Lmin"], float),
            ("Lmax", "Maximum size, Lmax (µm)", defaults["Lmax"], float),
            ("N", "Number of bins", defaults["N"], int),
            ("RelTol", "Relative error tolerance", defaults["RelTol"], float),
            ("AbsTol", "Absolute error tolerance", defaults["AbsTol"], float),
            ("MaxSteps", "Maximum JAX steps", defaults["MaxSteps"], int) if jax else
            ("MaxIterationTime", "Max. time per simulation (s)", defaults["MaxIterationTime"], float),
        )
        self.optimizer_fields = tuple(
            (key, label, PYFIT_DEFAULTS[key], type(PYFIT_DEFAULTS[key]))
            for key, label in (
                ("population_size", "Population size"),
                ("global_iterations", "Generations (0 = local only)"),
                ("global_tolerance", "Relative convergence tolerance"),
                ("mutation_min", "Mutation factor, minimum"),
                ("mutation_max", "Mutation factor, maximum"),
                ("recombination", "Crossover probability"),
                ("local_evaluations", "Maximum local evaluations"),
            )
        )
        self.inputs = {}
        self.grid_spacing = QComboBox()
        self.grid_spacing.addItem("Uniform", "uniform")
        self.grid_spacing.addItem("Logarithmic", "log")
        self.grid_spacing.setCurrentIndex(int(is_log_grid(pbm) and mode != "matlab_pe"))
        self.grid_spacing.setEnabled(mode != "matlab_pe")
        self.grid_spacing.setToolTip("Archived MATLAB PE uses uniform spacing." if mode == "matlab_pe"
                                    else "Logarithmic spacing puts more bins near Lmin.")
        self.log_offset = QLineEdit(str(pbm.get("LogGridOffset", DEFAULT_LOG_OFFSET)))
        self.log_offset.setToolTip("Edges are uniform in log(size + offset). Smaller positive offsets "
                                  "concentrate more bins near Lmin.")
        self.comparison_button = QPushButton("Compare HR-FVM to QMOM")
        self.mesh_validation_button = QPushButton("Mesh validation")
        comparison_actions = QHBoxLayout()
        for button, callback in ((self.comparison_button, self.show_comparison),
                                 (self.mesh_validation_button, self.show_mesh_validation)):
            button.setEnabled(hasattr(parent, "case"))
            button.clicked.connect(callback)
            comparison_actions.addWidget(button)
        layout = QVBoxLayout(self)
        groups = QHBoxLayout()
        layout.addLayout(groups)
        for title, fields, values in (
            ("PBM grid and numerical error", self.mesh_fields, self.pbm),
            ("SciPy differential evolution", self.optimizer_fields, self.optimizer),
        ):
            group = QGroupBox(title)
            form = QFormLayout(group)
            for key, label, default, convert in fields:
                value = values.get(key, default)
                try:
                    number = float(value)
                    if not isfinite(number) or (convert is int and not number.is_integer()):
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    number = default
                field = QLineEdit(str(int(number)) if convert is int else str(number))
                field.setObjectName(f"pe_{key}")
                self.inputs[key] = field
                form.addRow(label, field)
            if fields is self.mesh_fields:
                form.addRow("Grid spacing", self.grid_spacing)
                form.addRow("Log offset (µm)", self.log_offset)
                form.addRow(comparison_actions)
            groups.addWidget(group, 1)

        note = QLabel("The global search uses differential evolution, followed by local "
                      "least-squares refinement.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        for field in self.inputs.values():
            field.textChanged.connect(self.error.clear)
        self.log_offset.textChanged.connect(self.error.clear)
        self.grid_spacing.currentIndexChanged.connect(self.update_grid_spacing)
        self.update_grid_spacing()
        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        actions.accepted.connect(self.accept)
        actions.rejected.connect(self.reject)
        layout.addWidget(actions)

    def read_inputs(self, fields):
        values = {}
        for key, label, _, convert in fields:
            minimum = {"N": 2, "population_size": 5, "local_evaluations": 1,
                       "MaxSteps": 1, "global_iterations": 0, "Lmin": 0,
                       "global_tolerance": 0, "mutation_min": 0, "mutation_max": 0,
                       "recombination": 0}.get(key)
            try:
                number = convert(self.inputs[key].text())
                valid = number > 0 if minimum is None else number >= minimum
                if not isfinite(number) or not valid:
                    raise ValueError
            except (ValueError, OverflowError):
                requirement = "a positive number" if minimum is None else f"a value of at least {minimum}"
                if convert is int:
                    requirement = f"a whole number of at least {minimum}"
                self.error.setText(f"{label} requires {requirement}.")
                self.inputs[key].setFocus()
                return
            values[key] = number
        return values

    def show_comparison(self):
        from .comparison_dialog import ComparisonDialog
        self.show_mesh_dialog(ComparisonDialog)

    def show_mesh_validation(self):
        from .mesh_validation_dialog import MeshValidationDialog
        self.show_mesh_dialog(MeshValidationDialog)

    def update_grid_spacing(self):
        self.log_offset.setEnabled(self.grid_spacing.currentData() == "log")
        self.error.clear()

    def read_mesh_inputs(self):
        values = self.read_inputs(self.mesh_fields)
        if values is None:
            return
        if values["Lmax"] <= values["Lmin"]:
            self.error.setText("Lmax must be greater than Lmin.")
            self.inputs["Lmax"].setFocus()
            return
        spacing = self.grid_spacing.currentData()
        if spacing == "log":
            try:
                offset = float(self.log_offset.text())
                if not isfinite(offset) or offset <= 0:
                    raise ValueError
            except ValueError:
                self.error.setText("Log offset requires a finite positive number.")
                self.log_offset.setFocus()
                return
            values.update(GridSpacing="log", LogGridOffset=offset)
        elif "GridSpacing" in self.pbm:
            values["GridSpacing"] = "uniform"
        return values

    def show_mesh_dialog(self, dialog_type):
        values = self.read_mesh_inputs()
        if values is None:
            return
        parent = self.parent()
        case = parent.analysis_case()
        settings = {**self.pbm, **values}
        case = {**case, "pbmsetting": settings}
        index = parent.model.currentData()
        dialog = dialog_type(case, self, model_index=index)
        dialog.exec()
        dialog.deleteLater()

    def accept(self):
        mesh = self.read_mesh_inputs()
        if mesh is None:
            return
        values = self.read_inputs(self.optimizer_fields)
        if values is None:
            return
        if not values["mutation_min"] <= values["mutation_max"] < 2:
            self.error.setText("Mutation factors must be ordered and smaller than 2.")
            self.inputs["mutation_max"].setFocus()
            return
        if values["recombination"] > 1:
            self.error.setText("Crossover probability must be between 0 and 1.")
            self.inputs["recombination"].setFocus()
            return

        self.pbm.pop("LogGridOffset", None)
        self.pbm.update(mesh)
        if self.pbm.get("Numerics") == "matlab_pe":
            self.pbm["UseArchivedTolerances"] = True
        self.optimizer.update({key: values[key] for key, *_ in self.optimizer_fields})
        super().accept()


class EstimationJob(QThread):
    message = Signal(str)
    generation = Signal(int, int, float, float)
    completed = Signal(object, object)
    failed = Signal(str)

    def __init__(self, case, models, weights, terms, limits, parent):
        super().__init__(parent)
        self.case, self.models = case, models
        self.weights, self.terms, self.limits = weights, terms, limits
        self.cancel_requested = Event()

    def run(self):
        case = self.case
        cancelled = self.cancel_requested.is_set
        try:
            settings = effective_pe_pbm(case)
            self.message.emit(f"Solver · {pbm_backend_label(settings)}")
            objective = FitObjective(case["ExpData"], case["ProcInf"], case["Solubility"],
                                     settings, self.weights, self.terms, cancelled=cancelled)
            results, errors = {}, {}
            for index in self.models:
                self.message.emit(f"Fitting model {index + 1}…")
                try:
                    results[index] = fit_model(case["ModelParameter"][index], objective,
                                               cancelled=cancelled, progress=self.message.emit,
                                               generation_progress=partial(self.generation.emit, index),
                                               **self.limits)
                except (ValueError, RuntimeError, TimeoutError, FloatingPointError) as error:
                    errors[index] = str(error)
                    self.message.emit(f"Model {index + 1} failed: {error}")
                else:
                    self.message.emit(f"Model {index + 1} finished · WMSE {results[index].ssr:.5g}")
                    fit = results[index]
                    self.message.emit(
                        f"Model {index + 1} · 95% parameter CIs ready" if fit.intervals is not None else
                        f"Model {index + 1} · CIs unavailable · {fit.uncertainty_error}")
            if cancelled():
                raise InterruptedError("Parameter estimation stopped.")
        except InterruptedError:
            return
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.completed.emit(results, errors)


class OptimizationProgress(QWidget):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.figure = Figure(layout="constrained")
        self.view = FigureCanvasQTAgg(self.figure)
        self.view.setMinimumHeight(180)
        self.axes = self.figure.add_subplot()
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.note)
        self.clear()

    def clear(self):
        self.series = {}
        self.axes.clear()
        self.axes.set(xlabel="Generation", ylabel="WMSE")
        self.axes.xaxis.get_major_locator().set_params(integer=True)
        self.axes.grid(alpha=.22)
        self.view.draw_idle()
        self.note.setText("Generation data appears during differential evolution.")

    def append(self, model, generation, mean, best):
        if model not in self.series:
            label = display_name(self.models[model], f"Model {model + 1}")
            mean_line, = self.axes.plot([], [], "o-", markersize=3, label=f"{label} · mean")
            best_line, = self.axes.plot([], [], "o--", markersize=3, color=mean_line.get_color(),
                                        label=f"{label} · best")
            self.series[model] = mean_line, best_line, [], [], []
            self.axes.legend(fontsize="small", ncols=2)
            self.note.setText("Mean and best weighted mean squared error (WMSE) per model. "
                              "The mean includes penalties for any rejected trials retained in the population.")
        mean_line, best_line, generations, means, bests = self.series[model]
        generations.append(generation)
        means.append(mean)
        bests.append(best)
        mean_line.set_data(generations, means)
        best_line.set_data(generations, bests)
        self.axes.relim()
        self.axes.autoscale_view()
        self.view.draw_idle()

    def restore(self, results):
        self.clear()
        for model, fit in results.items():
            for generation, mean, best in fit.generation_history:
                self.append(model, int(generation), mean, best)


class ParameterEstimation(QDialog):
    def __init__(self, case, parent=None, *, workers=1):
        super().__init__(parent)
        from .uncertainty_dialog import IntervalView

        self.setWindowTitle("IMED | Parameter Estimation")
        self.resize(1040, 760)
        history(case)
        self.case = dict(case)
        self.workers = workers
        self.job = None
        self.fitted_models = {}
        self.results = {}
        self.screening = {}
        self.saved_settings = {}
        saved_optimizer = case.get(PYFIT_SETTINGS_KEY)
        saved_optimizer = saved_optimizer if isinstance(saved_optimizer, dict) else {}
        self.optimizer = {key: saved_optimizer.get(key, default)
                          for key, default in PYFIT_DEFAULTS.items()}

        self.model = QComboBox()
        self.model.addItem("All models", None)
        for index, model in enumerate(case["ModelParameter"]):
            self.model.addItem(display_name(model, f"Model {index + 1}"), index)
        saved = case.get(PE_WEIGHTS_KEY, case.get("objweightfactor"))
        saved = saved if isinstance(saved, dict) else {}
        self.weights = []
        self.terms = QComboBox()
        self.terms.addItems(["CSD", "D10 D50 D90"])
        if saved.get("Terms") in {value for _, value in SIZE_SAMPLE_OPTIONS}:
            self.terms.setCurrentIndex(1)
        self.size_samples = QComboBox()
        for label, value in SIZE_SAMPLE_OPTIONS:
            self.size_samples.addItem(label, value)
        if saved.get("Terms") == TREND_SIZE_TERM:
            self.size_samples.setCurrentIndex(1)
        self.size_samples.setEnabled(self.terms.currentText() != "CSD")
        form = QFormLayout()
        form.setVerticalSpacing(4)
        for selector in (self.model, self.terms, self.size_samples):
            selector.setFixedWidth(220)
        form.addRow("Model", self.model)
        for index, label in enumerate(("Count weight", "Concentration weight", "Size weight"), 1):
            weight = QDoubleSpinBox()
            weight.setRange(0, 10000)
            weight.setDecimals(8)
            weight.setFixedWidth(140)
            value = saved.get(f"W{index}", WEIGHT_DEFAULTS[f"W{index}"])
            try:
                weight.setValue(float(value))
            except (ValueError, TypeError):
                weight.setValue(WEIGHT_DEFAULTS[f"W{index}"])
            self.weights.append(weight)
            form.addRow(label, weight)
        form.addRow("Size terms", self.terms)
        form.addRow("Size samples", self.size_samples)
        self.advanced_button = QPushButton("Advanced Settings")
        self.advanced_button.clicked.connect(self.edit_settings)
        self.save_case_button = QPushButton("Save PE case…")
        self.save_case_button.setToolTip("Save experiments, model definitions, current PE settings, and completed fits.")
        self.save_case_button.clicked.connect(self.save_pe_case)
        self.save_settings_button = QPushButton("Save PE settings…")
        self.load_settings_button = QPushButton("Load PE settings…")
        self.save_settings_button.clicked.connect(self.save_pe_settings)
        self.load_settings_button.clicked.connect(self.load_pe_settings)
        self.uncertainty_button = QPushButton("Calculate parameter CIs")
        self.uncertainty_button.clicked.connect(self.show_uncertainty)
        settings_buttons = QGridLayout()
        settings_buttons.addWidget(self.advanced_button, 0, 0)
        settings_buttons.addWidget(self.uncertainty_button, 0, 1)
        settings_buttons.addWidget(self.save_case_button, 1, 0)
        settings_buttons.addWidget(self.save_settings_button, 1, 1)
        settings_buttons.addWidget(self.load_settings_button, 2, 1)
        header = QHBoxLayout()
        header.addLayout(form)
        header.addStretch(1)
        header.addLayout(settings_buttons)
        header.setAlignment(settings_buttons, Qt.AlignmentFlag.AlignTop)

        self.summary = table(("Model", "Parameters", "WMSE", "AIC", "AIC rank", "F-test", "t-tests",
                              "Decision", "Fit status", "95% CIs"))
        self.summary.setMinimumHeight(155)
        self.summary.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.progress_log = QPlainTextEdit()
        self.progress_log.setReadOnly(True)
        self.progress_log.setMaximumBlockCount(250)
        self.progress_log.setMinimumHeight(110)
        self.interval_view = IntervalView()
        self.optimization_progress = OptimizationProgress(case["ModelParameter"])
        chart_group = QGroupBox("Generation progress")
        QVBoxLayout(chart_group).addWidget(self.optimization_progress)
        log_group = QGroupBox("Optimization log")
        QVBoxLayout(log_group).addWidget(self.progress_log)
        self.progress_panel = QSplitter(Qt.Orientation.Horizontal)
        self.progress_panel.addWidget(chart_group)
        self.progress_panel.addWidget(log_group)
        self.progress_panel.setChildrenCollapsible(False)
        self.progress_panel.setHandleWidth(8)
        self.progress_panel.setSizes([580, 400])
        self.output_tabs = QTabWidget()
        self.output_tabs.addTab(self.interval_view, "Parameters and covariance")
        self.output_tabs.addTab(self.progress_panel, "Optimization progress")
        self.output_tabs.setCurrentWidget(self.progress_panel)
        results = QSplitter(Qt.Orientation.Vertical)
        results.addWidget(self.summary)
        results.addWidget(self.output_tabs)
        results.setChildrenCollapsible(False)
        results.setHandleWidth(8)
        results.setStretchFactor(0, 0)
        results.setStretchFactor(1, 1)
        results.setSizes([175, 385])
        self.status = QLabel(f"{len(case['ExpData'])} experiments loaded.")
        self.status.setWordWrap(True)
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.start_fit)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_fit)
        self.save_button = QPushButton("Apply fitted parameters")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.apply_fit)
        self.details_button = QPushButton("Parameter Details")
        self.details_button.setEnabled(False)
        self.details_button.clicked.connect(self.show_parameters)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        actions = QHBoxLayout()
        for widget in (self.run_button, self.stop_button, self.details_button,
                       self.save_button, self.close_button):
            actions.addWidget(widget)
        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(results, 1)
        layout.addWidget(self.status)
        layout.addLayout(actions)
        for control in self.weights:
            control.valueChanged.connect(self.objective_changed)
        self.terms.currentIndexChanged.connect(self.objective_changed)
        self.size_samples.currentIndexChanged.connect(self.objective_changed)
        self.summary.currentCellChanged.connect(self.update_interval_view)
        restored, screening = {}, {}
        for index in range(len(case["ModelParameter"])):
            step = latest_pe_step(case, index)
            if step is not None:
                restored[index] = unpack_fit(step["result"])
                if "screening" in step["result"]:
                    screening[index] = unpack_screening(step["result"]["screening"])
        if restored:
            self.show_results(restored, {}, remember=False, screening={**screen_models(restored), **screening})
            self.save_button.setEnabled(True)
            self.details_button.setEnabled(True)

    def save_pe_case(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save PE case", "imed_pe.imed.json", NATIVE_FILTER)
        if not path:
            return
        try:
            case = {**self.case, **self.current_pe_settings()}
            # Results remain in the workflow history until applied to the model.
            destination = save_native_case(path, case)
        except (OSError, TypeError, ValueError) as error:
            self.status.setText(str(error))
        else:
            self.status.setText(f"Saved PE case {destination.name}. Reopen it from IMED.")

    def current_pe_settings(self):
        return {PYFIT_SETTINGS_KEY: dict(self.optimizer),
                PE_WEIGHTS_KEY: {**{f"W{i}": control.value()
                                    for i, control in enumerate(self.weights, 1)},
                                 "Terms": self.size_term()}}

    def save_pe_settings(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save PE settings", "imed_pe.json", "JSON (*.json)")
        if not path:
            return
        try:
            pbm = effective_pe_pbm(self.case)
            for key in ("N", "MaxSteps"):
                if key in pbm and float(pbm[key]).is_integer():
                    pbm[key] = int(pbm[key])
            if "UseArchivedTolerances" in pbm:
                pbm["UseArchivedTolerances"] = bool(pbm["UseArchivedTolerances"])
            settings = self.current_pe_settings()
            optimizer = settings[PYFIT_SETTINGS_KEY]
            for key, default in PYFIT_DEFAULTS.items():
                if type(default) is int and float(optimizer[key]).is_integer():
                    optimizer[key] = int(optimizer[key])
            data = pe_file(pbm, optimizer, settings[PE_WEIGHTS_KEY])
            read_pe_file(data)
            atomic_json(Path(path), data)
        except (OSError, TypeError, ValueError) as error:
            self.status.setText(str(error))
        else:
            self.status.setText(f"Saved PE settings to {Path(path).name}.")

    def load_pe_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load PE settings", "", "JSON (*.json)")
        if not path:
            return
        try:
            pbm, optimizer, weights = read_pe_file(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError) as error:
            self.status.setText(str(error))
            return
        backend = effective_pe_pbm(self.case).get("Numerics", "conservative")
        settings = pe_mesh(pbm_for_pe({"Numerics": backend, **pbm}))
        self.optimizer = optimizer
        loaded = {PE_PBM_KEY: settings, PYFIT_SETTINGS_KEY: optimizer, PE_WEIGHTS_KEY: weights}
        self.saved_settings.update(loaded)
        self.case.update(loaded)
        for index, control in enumerate(self.weights, 1):
            control.setValue(weights[f"W{index}"])
        term = weights["Terms"]
        self.terms.setCurrentIndex(int(term != "CSD"))
        self.size_samples.setCurrentIndex(self.size_samples.findData(term) if term != "CSD" else 0)
        self.objective_changed()
        self.status.setText(f"Loaded PE settings from {Path(path).name}.")

    def edit_settings(self):
        dialog = AdvancedSettings(effective_pe_pbm(self.case), self.optimizer, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.optimizer = dialog.optimizer
            self.saved_settings.update({PE_PBM_KEY: pe_mesh(dialog.pbm), PYFIT_SETTINGS_KEY: dialog.optimizer})
            self.case = {**self.case, **self.saved_settings}
            self.clear_results()
            self.status.setText("Advanced settings saved. Run parameter estimation to use them.")
        dialog.deleteLater()

    def clear_results(self):
        self.results = {}
        self.screening = {}
        self.summary.setRowCount(0)
        self.interval_view.clear_result()
        self.optimization_progress.clear()
        self.save_button.setEnabled(False)
        self.details_button.setEnabled(False)

    def objective_changed(self):
        self.size_samples.setEnabled(self.terms.currentText() != "CSD")
        self.clear_results()
        self.progress_log.clear()
        self.output_tabs.setCurrentWidget(self.progress_panel)
        self.status.setText("Objective settings changed. Click Run to calculate a new objective value.")

    def size_term(self):
        if self.terms.currentText() == "CSD":
            return "CSD"
        return self.size_samples.currentData()

    def start_fit(self):
        weights = [control.value() for control in self.weights]
        if not any(weights):
            self.status.setText("Set at least one positive objective weight before running.")
            return
        terms = self.size_term()
        choice = self.model.currentData()
        models = range(len(self.case["ModelParameter"])) if choice is None else [choice]
        self.clear_results()
        self.progress_log.clear()
        self.output_tabs.setCurrentWidget(self.progress_panel)
        self.run_button.setEnabled(False)
        self.advanced_button.setEnabled(False)
        self.save_case_button.setEnabled(False)
        self.save_settings_button.setEnabled(False)
        self.load_settings_button.setEnabled(False)
        self.uncertainty_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        for control in (self.model, self.terms, self.size_samples, *self.weights):
            control.setEnabled(False)
        self.log_progress("Preparing parameter estimation…")
        count, concentration, size = weights
        self.log_progress(f"Objective weights · count {count:g} · concentration {concentration:g}"
                          f" · size {size:g} ({terms})")
        limits = {**self.optimizer, "workers": self.workers}
        self.job = EstimationJob(self.case, models, weights, terms, limits, self)
        self.job.message.connect(self.log_progress)
        self.job.generation.connect(self.optimization_progress.append)
        self.job.completed.connect(self.show_results)
        self.job.failed.connect(self.log_progress)
        self.job.finished.connect(self.finish_fit)
        self.job.start()

    def log_progress(self, message):
        self.status.setText(message)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.progress_log.appendPlainText(f"{timestamp}  {message}")

    def stop_fit(self):
        if self.job is not None:
            self.job.cancel_requested.set()
            self.log_progress("Stopping parameter estimation…")
            self.stop_button.setEnabled(False)

    def reject(self):
        if self.job is None:
            super().reject()

    def finish_fit(self):
        stopped = self.job.cancel_requested.is_set()
        self.job.deleteLater()
        self.job = None
        self.run_button.setEnabled(True)
        self.advanced_button.setEnabled(True)
        self.save_case_button.setEnabled(True)
        self.save_settings_button.setEnabled(True)
        self.load_settings_button.setEnabled(True)
        self.uncertainty_button.setEnabled(True)
        self.close_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        for control in (self.model, self.terms, *self.weights):
            control.setEnabled(True)
        self.size_samples.setEnabled(self.terms.currentText() != "CSD")
        if stopped:
            self.clear_results()
            self.log_progress("Parameter estimation stopped.")
        self.save_button.setEnabled(bool(self.results))
        self.details_button.setEnabled(bool(self.results))

    def show_results(self, results, errors, *, remember=True, screening=None):
        self.results = results
        self.screening = screen_models(results) if screening is None else screening
        if not remember:
            self.optimization_progress.restore(results)
        weights = {**{f"W{i}": control.value() for i, control in enumerate(self.weights, 1)},
                   "Terms": self.size_term()}
        if remember:
            self.saved_settings[PE_WEIGHTS_KEY] = weights
            self.case[PE_WEIGHTS_KEY] = weights
        if remember and results:
            for index, fit in results.items():
                result = {**pack_fit(fit), "screening": pack_screening(self.screening[index])}
                record_step(self.case, "parameter_estimation", result, model_index=index,
                            context=pe_signatures(self.case, index)[0])
        rows = []
        for index, fit in results.items():
            diagnosis = self.screening[index]
            rows.append((index + 1, len(fit.parameters), fit.ssr, fit.aic, diagnosis["rank"] or "—",
                         "Pass" if diagnosis["f_pass"] else "Fail", "Pass" if diagnosis["t_pass"] else "Fail",
                         "Excluded" if diagnosis["reasons"] else "Retained",
                         "Converged" if fit.converged else "Evaluation limit reached",
                         "Ready" if fit.intervals is not None else fit.uncertainty_error))
        rows.extend((index + 1, *("—",) * 7, message, "—") for index, message in errors.items())
        rows.sort()
        fill_table(self.summary, rows)
        for row, (model, *_) in enumerate(rows):
            if diagnosis := self.screening.get(model - 1):
                self.summary.item(row, 5).setToolTip(f"F = {diagnosis['f_value']:.5g} · limit {diagnosis['f_limit']:.5g}")
                self.summary.item(row, 7).setToolTip(", ".join(diagnosis["reasons"]) or "Passed model screening")
            for column in (8, 9):
                item = self.summary.item(row, column)
                item.setToolTip(item.text())
        if self.results:
            first_model = min(self.results) + 1
            first_row = sorted(row[0] for row in rows).index(first_model)
            self.summary.setCurrentCell(first_row, 0)
        self.log_progress(f"Fitted {len(results)} model(s). {len(errors)} model(s) failed.")
        self.update_interval_view()

    def update_interval_view(self):
        self.interval_view.clear_result()
        item = self.summary.item(self.summary.currentRow(), 0)
        if item is not None:
            fit = self.results.get(int(item.text()) - 1)
            if fit is not None and fit.intervals is not None:
                self.interval_view.show_result(fit.intervals)

    def show_uncertainty(self):
        from .uncertainty_dialog import ParameterUncertainty

        index = self.model.currentData()
        item = self.summary.item(self.summary.currentRow(), 0)
        if index is None:
            index = int(item.text()) - 1 if item is not None else 0
        case = self.analysis_case()
        fit = self.results.get(index)
        dialog = ParameterUncertainty(case, self, model_index=index,
                                       initial_result=fit.intervals if fit is not None else None)
        dialog.exec()
        dialog.deleteLater()

    def analysis_case(self):
        models = list(self.case["ModelParameter"])
        for number, fit in self.results.items():
            models[number] = {**models[number], "para": fit.parameters.copy()}
        case = {**self.case, "ModelParameter": models, "pbmsetting": effective_pe_pbm(self.case),
                "objweightfactor": {**{f"W{i}": control.value()
                                      for i, control in enumerate(self.weights, 1)},
                                    "Terms": self.size_term()}}
        return case

    def show_parameters(self):
        item = self.summary.item(self.summary.currentRow(), 0)
        if item is None:
            return
        index = int(item.text()) - 1
        fit = self.results.get(index)
        if fit is None:
            self.status.setText("Select a model with a completed fit.")
            return
        active = np.asarray(self.case["ModelParameter"][index]["paraset"])[:, 0] == 1
        names = [name for (name, _), enabled in zip(PARAMETERS, active) if enabled]
        details = QDialog(self)
        details.setWindowTitle(f"Model {index + 1} | Parameter Details")
        details.resize(820, 620)
        layout = QVBoxLayout(details)
        diagnosis = self.screening[index]
        values = table(("Estimated parameter", "Value", "|t|", "95% cutoff", "t-test"))
        threshold = diagnosis["t_limit"]
        fill_table(values, [(name, value, statistic, threshold,
                             "Unavailable" if np.isnan(statistic) else "Pass" if statistic >= threshold else "Fail")
                            for name, value, statistic in zip(names, fit.parameters, diagnosis["t_statistics"])])
        if fit.intervals is not None:
            from .uncertainty_dialog import IntervalView

            view = IntervalView()
            view.show_result(fit.intervals)
            view.addTab(values, "All fitted values")
            layout.addWidget(view)
        else:
            layout.addWidget(values)
            note = QLabel(f"Confidence intervals unavailable. {fit.uncertainty_error}")
            note.setWordWrap(True)
            layout.addWidget(note)
        if not fit.converged:
            note = QLabel("The fit reached its evaluation limit. Treat its local intervals as provisional.")
            note.setWordWrap(True)
            layout.addWidget(note)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(details.reject)
        layout.addWidget(close)
        details.exec()
        details.deleteLater()

    def apply_fit(self):
        for index, fit in self.results.items():
            model = {key: value for key, value in self.case["ModelParameter"][index].items()
                     if key in ("Name", "paraset")}
            model.update(para=fit.parameters.reshape(1, -1), covmat=fit.covariance,
                         resnorm=float(fit.residuals @ fit.residuals),
                         resid=fit.residuals, jacobian=fit.jacobian, nvars=len(fit.parameters),
                         SSR=fit.ssr, AICVector=fit.aic)
            model.update(matlab_screening_fields(self.screening[index]))
            self.fitted_models[index] = model
        self.saved_settings.update(self.current_pe_settings())
        self.accept()
