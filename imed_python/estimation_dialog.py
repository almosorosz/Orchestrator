"""Parameter Estimation dialog backed by the standalone fitting engine."""
from datetime import datetime
from math import isfinite
from threading import Event

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QTabWidget, QVBoxLayout,
)

from .control_panel import fill_table, table
from .estimation import PYFIT_DEFAULTS, PYFIT_SETTINGS_KEY, FitObjective, fit_model
from .parameter_labels import PARAMETERS
from .solver import pbm_backend_label
from .size_data import SIZE_SAMPLE_OPTIONS, TREND_SIZE_TERM
from .session import display_name, latest_step, pack_fit, record_step, signature, unpack_fit


class AdvancedSettings(QDialog):
    def __init__(self, pbm, optimizer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("IMED | Advanced PE Settings")
        self.resize(810, 435)
        self.pbm, self.optimizer = dict(pbm), dict(optimizer)
        mode = pbm.get("Numerics", "conservative")
        jax = mode == "jax"
        rtol, atol = (1e-7, 1e-9) if jax else (1e-3, 1e-6)

        self.mesh_fields = (
            ("Lmin", "Minimum size, Lmin (µm)", 0., float),
            ("Lmax", "Maximum size, Lmax (µm)", 250., float),
            ("N", "Number of bins", 250, int),
            ("RelTol", "Relative error tolerance", rtol, float),
            ("AbsTol", "Absolute error tolerance", atol, float),
            ("MaxSteps", "Maximum JAX steps", 100_000, int) if jax else
            ("MaxIterationTime", "Max. time per simulation (s)", 30., float),
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
        self.comparison_button = QPushButton("Compare PBM with QMOM")
        self.comparison_button.setEnabled(hasattr(parent, "case"))
        self.comparison_button.clicked.connect(self.show_comparison)
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
                form.addRow(self.comparison_button)
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

        values = self.read_inputs(self.mesh_fields)
        if values is None:
            return
        if values["Lmax"] <= values["Lmin"]:
            self.error.setText("Lmax must be greater than Lmin.")
            return
        parent = self.parent()
        case = parent.analysis_case()
        settings = {**self.pbm, **values}
        case = {**case, "pbmsetting": settings}
        index = parent.model.currentData()
        dialog = ComparisonDialog(case, self, model_index=index)
        dialog.exec()
        dialog.deleteLater()

    def accept(self):
        values = self.read_inputs(self.mesh_fields + self.optimizer_fields)
        if values is None:
            return
        if values["Lmax"] <= values["Lmin"]:
            self.error.setText("Lmax must be greater than Lmin.")
            self.inputs["Lmax"].setFocus()
            return
        if not values["mutation_min"] <= values["mutation_max"] < 2:
            self.error.setText("Mutation factors must be ordered and smaller than 2.")
            self.inputs["mutation_max"].setFocus()
            return
        if values["recombination"] > 1:
            self.error.setText("Crossover probability must be between 0 and 1.")
            self.inputs["recombination"].setFocus()
            return

        self.pbm.update({key: values[key] for key, *_ in self.mesh_fields})
        if self.pbm.get("Numerics") == "matlab_pe":
            self.pbm["UseArchivedTolerances"] = True
        self.optimizer.update({key: values[key] for key, *_ in self.optimizer_fields})
        super().accept()


class EstimationJob(QThread):
    message = Signal(str)
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
            settings = case.get("pbmsetting")
            settings = settings if isinstance(settings, dict) else {}
            self.message.emit(f"Solver · {pbm_backend_label(settings)}")
            objective = FitObjective(case["ExpData"], case["ProcInf"], case["Solubility"],
                                     settings, self.weights, self.terms, cancelled=cancelled)
            results, errors = {}, {}
            for index in self.models:
                self.message.emit(f"Fitting model {index + 1}…")
                try:
                    results[index] = fit_model(case["ModelParameter"][index], objective,
                                               cancelled=cancelled, progress=self.message.emit,
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


class ParameterEstimation(QDialog):
    def __init__(self, case, parent=None, *, workers=1):
        super().__init__(parent)
        from .uncertainty_dialog import IntervalView

        self.setWindowTitle("IMED | Parameter Estimation")
        self.resize(900, 720)
        self.case = case
        self.workers = workers
        self.job = None
        self.fitted_models = {}
        self.results = {}
        self.saved_settings = {}
        saved_optimizer = case.get(PYFIT_SETTINGS_KEY)
        saved_optimizer = saved_optimizer if isinstance(saved_optimizer, dict) else {}
        self.optimizer = {key: saved_optimizer.get(key, default)
                          for key, default in PYFIT_DEFAULTS.items()}

        self.model = QComboBox()
        self.model.addItem("All models", None)
        for index, model in enumerate(case["ModelParameter"]):
            self.model.addItem(display_name(model, f"Model {index + 1}"), index)
        saved = case.get("objweightfactor")
        saved = saved if isinstance(saved, dict) else {}
        self.weights = []
        self.terms = QComboBox()
        self.terms.addItems(["CSD", "D10 D50 D90"])
        if saved.get("Terms", "CSD") != "CSD":
            self.terms.setCurrentIndex(1)
        self.size_samples = QComboBox()
        for label, value in SIZE_SAMPLE_OPTIONS:
            self.size_samples.addItem(label, value)
        if saved.get("Terms") == TREND_SIZE_TERM:
            self.size_samples.setCurrentIndex(1)
        self.size_samples.setEnabled(self.terms.currentText() != "CSD")
        form = QFormLayout()
        form.addRow("Model", self.model)
        for index, label in enumerate(("Count weight", "Concentration weight", "Size weight"), 1):
            weight = QDoubleSpinBox()
            weight.setRange(0, 10000)
            weight.setDecimals(8)
            weight.setValue(float(saved.get(f"W{index}", int(index == 2))))
            self.weights.append(weight)
            form.addRow(label, weight)
        form.addRow("Size terms", self.terms)
        form.addRow("Size samples", self.size_samples)
        self.advanced_button = QPushButton("Advanced Settings")
        self.advanced_button.clicked.connect(self.edit_settings)
        form.addRow(self.advanced_button)
        self.uncertainty_button = QPushButton("Calculate parameter CIs")
        self.uncertainty_button.clicked.connect(self.show_uncertainty)
        form.addRow(self.uncertainty_button)

        self.summary = table(("Model", "Parameters", "WMSE", "AIC", "Fit status", "95% CIs"))
        self.progress_log = QPlainTextEdit()
        self.progress_log.setReadOnly(True)
        self.progress_log.setMaximumBlockCount(250)
        self.progress_log.setMinimumHeight(110)
        self.interval_view = IntervalView()
        self.output_tabs = QTabWidget()
        self.output_tabs.addTab(self.interval_view, "Parameters and covariance")
        self.output_tabs.addTab(self.progress_log, "Optimization log")
        self.output_tabs.setCurrentIndex(1)
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
        layout.addLayout(form)
        layout.addWidget(self.summary, 1)
        layout.addWidget(self.output_tabs, 2)
        layout.addWidget(self.status)
        layout.addLayout(actions)
        for control in self.weights:
            control.valueChanged.connect(self.objective_changed)
        self.terms.currentIndexChanged.connect(self.objective_changed)
        self.size_samples.currentIndexChanged.connect(self.objective_changed)
        self.summary.currentCellChanged.connect(self.update_interval_view)
        restored = {}
        for index in range(len(case["ModelParameter"])):
            step = latest_step(case, "parameter_estimation", model_index=index,
                               context=signature(case, index, include_values=False))
            if step is not None:
                restored[index] = unpack_fit(step["result"])
        if restored:
            self.show_results(restored, {}, remember=False)
            self.save_button.setEnabled(True)
            self.details_button.setEnabled(True)

    def edit_settings(self):
        settings = self.case.get("pbmsetting")
        dialog = AdvancedSettings(settings if isinstance(settings, dict) else {}, self.optimizer, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.optimizer = dialog.optimizer
            self.saved_settings = {"pbmsetting": dialog.pbm, PYFIT_SETTINGS_KEY: dialog.optimizer}
            self.case = {**self.case, **self.saved_settings}
            self.clear_results()
            self.status.setText("Advanced settings saved. Run parameter estimation to use them.")
        dialog.deleteLater()

    def clear_results(self):
        self.results = {}
        self.summary.setRowCount(0)
        self.interval_view.clear_result()
        self.save_button.setEnabled(False)
        self.details_button.setEnabled(False)

    def objective_changed(self):
        self.size_samples.setEnabled(self.terms.currentText() != "CSD")
        self.clear_results()
        self.progress_log.clear()
        self.output_tabs.setCurrentIndex(1)
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
        self.output_tabs.setCurrentIndex(1)
        self.run_button.setEnabled(False)
        self.advanced_button.setEnabled(False)
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

    def show_results(self, results, errors, *, remember=True):
        self.results = results
        weights = {**{f"W{i}": control.value() for i, control in enumerate(self.weights, 1)},
                   "Terms": self.size_term()}
        self.saved_settings["objweightfactor"] = weights
        self.case["objweightfactor"] = weights
        if remember:
            for index, fit in results.items():
                record_step(self.case, "parameter_estimation", pack_fit(fit), model_index=index,
                            context=signature(self.case, index, include_values=False))
        rows = [(index + 1, len(fit.parameters), fit.ssr, fit.aic,
                 "Converged" if fit.converged else "Evaluation limit reached",
                 "Ready" if fit.intervals is not None else fit.uncertainty_error)
                for index, fit in results.items()]
        rows.extend((index + 1, "—", "—", "—", message, "—") for index, message in errors.items())
        fill_table(self.summary, sorted(rows))
        if self.results:
            first_model = min(self.results) + 1
            first_row = sorted(row[0] for row in rows).index(first_model)
            self.summary.setCurrentCell(first_row, 0)
        self.log_progress(f"Fitted {len(results)} model(s). {len(errors)} model(s) failed.")
        self.update_interval_view()
        if any(fit.intervals is not None for fit in results.values()):
            self.output_tabs.setCurrentIndex(0)

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
        case = {**self.case, "ModelParameter": models,
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
        values = table(("Estimated parameter", "Value"))
        fill_table(values, list(zip(names, fit.parameters)))
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
            self.fitted_models[index] = model
        self.saved_settings[PYFIT_SETTINGS_KEY] = dict(self.optimizer)
        self.saved_settings["objweightfactor"] = {
            **{f"W{index}": control.value() for index, control in enumerate(self.weights, 1)},
            "Terms": self.size_term(),
        }
        self.accept()
