"""Parameter Estimation dialog backed by the standalone fitting engine."""
from datetime import datetime
from math import isfinite
from threading import Event

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout,
)

from .control_panel import fill_table, table
from .estimation import PYFIT_DEFAULTS, PYFIT_SETTINGS_KEY, FitObjective, fit_model
from .parameter_labels import PARAMETERS
from .solver import pbm_backend_label


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

    def accept(self):
        values = {}
        for key, label, _, convert in self.mesh_fields + self.optimizer_fields:
            try:
                number = convert(self.inputs[key].text())
                minimum = {"N": 2, "population_size": 5, "local_evaluations": 1,
                           "MaxSteps": 1, "global_iterations": 0, "Lmin": 0,
                           "global_tolerance": 0, "mutation_min": 0, "mutation_max": 0,
                           "recombination": 0}.get(key)
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
        self.setWindowTitle("IMED | Parameter Estimation")
        self.resize(820, 650)
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
        for index, _ in enumerate(case["ModelParameter"]):
            self.model.addItem(f"Model {index + 1}", index)
        saved = case.get("objweightfactor")
        saved = saved if isinstance(saved, dict) else {}
        self.weights = []
        self.terms = QComboBox()
        self.terms.addItems(["CSD", "D10 D50 D90"])
        if saved.get("Terms", "CSD") != "CSD":
            self.terms.setCurrentIndex(1)
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
        self.advanced_button = QPushButton("Advanced Settings")
        self.advanced_button.clicked.connect(self.edit_settings)
        form.addRow(self.advanced_button)

        self.summary = table(("Model", "Parameters", "WMSE", "AIC", "Fit status"))
        self.progress_log = QPlainTextEdit()
        self.progress_log.setReadOnly(True)
        self.progress_log.setMaximumBlockCount(250)
        self.progress_log.setMinimumHeight(110)
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
        layout.addWidget(QLabel("Optimization log"))
        layout.addWidget(self.progress_log, 1)
        layout.addWidget(self.status)
        layout.addLayout(actions)
        for control in self.weights:
            control.valueChanged.connect(self.objective_changed)
        self.terms.currentIndexChanged.connect(self.objective_changed)

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
        self.save_button.setEnabled(False)
        self.details_button.setEnabled(False)

    def objective_changed(self):
        self.clear_results()
        self.progress_log.clear()
        self.status.setText("Objective settings changed. Click Run to calculate a new objective value.")

    def start_fit(self):
        weights = [control.value() for control in self.weights]
        if not any(weights):
            self.status.setText("Set at least one positive objective weight before running.")
            return
        terms = self.terms.currentText()
        choice = self.model.currentData()
        models = range(len(self.case["ModelParameter"])) if choice is None else [choice]
        self.clear_results()
        self.progress_log.clear()
        self.run_button.setEnabled(False)
        self.advanced_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        for control in (self.model, self.terms, *self.weights):
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
        self.close_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        for control in (self.model, self.terms, *self.weights):
            control.setEnabled(True)
        if stopped:
            self.clear_results()
            self.log_progress("Parameter estimation stopped.")
        self.save_button.setEnabled(bool(self.results))
        self.details_button.setEnabled(bool(self.results))

    def show_results(self, results, errors):
        self.results = results
        rows = [(index + 1, len(fit.parameters), fit.ssr, fit.aic,
                 "Converged" if fit.converged else "Evaluation limit reached")
                for index, fit in results.items()]
        rows.extend((index + 1, "—", "—", "—", message) for index, message in errors.items())
        fill_table(self.summary, sorted(rows))
        if self.results:
            first_model = min(self.results) + 1
            first_row = sorted(row[0] for row in rows).index(first_model)
            self.summary.setCurrentCell(first_row, 0)
        self.log_progress(f"Fitted {len(results)} model(s). {len(errors)} model(s) failed.")

    def show_parameters(self):
        item = self.summary.item(self.summary.currentRow(), 0)
        if item is None:
            return
        index = int(item.text()) - 1
        fit = self.results.get(index)
        if fit is None:
            self.status.setText("Select a model with a completed fit.")
            return
        active = self.case["ModelParameter"][index]["paraset"][:, 0] == 1
        names = [name for (name, _), enabled in zip(PARAMETERS, active) if enabled]
        details = QDialog(self)
        details.setWindowTitle(f"Model {index + 1} | Parameter Details")
        details.resize(470, 320)
        layout = QVBoxLayout(details)
        values = table(("Estimated parameter", "Value"))
        fill_table(values, list(zip(names, fit.parameters)))
        layout.addWidget(values)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(details.reject)
        layout.addWidget(close)
        details.exec()
        details.deleteLater()

    def apply_fit(self):
        for index, fit in self.results.items():
            model = {"paraset": self.case["ModelParameter"][index]["paraset"]}
            model.update(para=fit.parameters.reshape(1, -1), covmat=fit.covariance,
                         resnorm=float(fit.residuals @ fit.residuals),
                         resid=fit.residuals, jacobian=fit.jacobian, nvars=len(fit.parameters),
                         SSR=fit.ssr, AICVector=fit.aic)
            self.fitted_models[index] = model
        self.saved_settings[PYFIT_SETTINGS_KEY] = dict(self.optimizer)
        self.saved_settings["objweightfactor"] = {
            **{f"W{index}": control.value() for index, control in enumerate(self.weights, 1)},
            "Terms": self.terms.currentText(),
        }
        self.accept()
