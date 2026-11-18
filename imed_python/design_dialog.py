"""Dialogs for constrained model-discrimination and D-optimal recipe searches."""
import csv
from functools import partial

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from .control_panel import fill_table, table
from .design import discrimination_score, d_optimal_log_score, optimize_recipe, parameter_samples
from .solver import pbm_backend_label
from .session import latest_step, record_step, signature


def fitted_models(case):
    return [(index, model) for index, model in enumerate(case.get("ModelParameter", []))
            if np.asarray(model.get("para", [])).size]


class DesignJob(QThread):
    completed = Signal(object)
    failed = Signal(str)
    message = Signal(str)

    def __init__(self, experiment, process, conditions, score, iterations, parent, *,
                 settings=None, workers=1):
        super().__init__(parent)
        self.inputs = experiment, process, conditions, score
        self.iterations = iterations
        self.settings, self.workers = settings or {}, workers

    def run(self):
        try:
            backend = pbm_backend_label(self.settings)
            worker_note = f" using {self.workers} CPU workers" if self.workers > 1 else ""
            if not self.isInterruptionRequested():
                self.message.emit(f"Simulating candidate recipes with {backend}{worker_note}…")
            result = optimize_recipe(*self.inputs, iterations=self.iterations,
                                     cancelled=self.isInterruptionRequested, workers=self.workers)
            if self.isInterruptionRequested():
                raise InterruptedError("Experiment design cancelled")
        except InterruptedError:
            self.failed.emit("Experiment design stopped.")
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.completed.emit(result)


class DesignDialog(QDialog):
    def __init__(self, case, optimal, parent=None, *, workers=1):
        super().__init__(parent)
        name = "D-Optimal Design" if optimal else "Model Discrimination"
        self.setWindowTitle(f"IMED | {name}")
        self.resize(800, 650)
        self.case, self.optimal = case, optimal
        self.workers = workers
        self.models = fitted_models(case)
        self.job = None
        self.recipe = None
        self.template = QComboBox()
        self.template.addItems(f"Experiment {i + 1}" for i in range(len(case["ExpData"])))
        self.model = QComboBox()
        for index, _ in self.models:
            self.model.addItem(f"Model {index + 1}")
        self.batch = self.number(0.1, 1e8)
        self.temperatures = [self.number(-273, 1000) for _ in range(2)]
        self.temp_rates = [self.number(0, 10000) for _ in range(2)]
        self.masses = [self.number(0, 1e9) for _ in range(2)]
        self.concentration = self.number(0, 0.99999)
        self.final_anti = self.number(0, 0.99999)
        self.anti_rates = [self.number(0, 10000) for _ in range(2)]
        self.seed = [self.number(0, 1e9) for _ in range(2)]
        self.iterations = QSpinBox()
        self.iterations.setRange(0, 1000)
        self.iterations.setValue(5)
        self.samples = QSpinBox()
        self.samples.setRange(1, 10000)
        self.samples.setValue(8)
        self.terms = QComboBox()
        self.terms.addItems(["CSD", "D10 D50 D90"])
        weights = case.get("objweightfactor")
        weights = weights if isinstance(weights, dict) else {}
        self.weights = []
        form = QFormLayout()
        form.addRow("Experiment template (includes seed CSD)", self.template)
        if optimal:
            form.addRow("Fitted model", self.model)
        form.addRow("Batch time (s)", self.batch)
        form.addRow("Initial and final temperature (°C)", self.row(self.temperatures))
        form.addRow("Min and max ramp rate (°C/s)", self.row(self.temp_rates))
        form.addRow("Initial solvent and antisolvent masses (g)", self.row(self.masses))
        form.addRow("Initial solute concentration (g/g solution)", self.concentration)
        form.addRow("Final antisolvent mass fraction", self.final_anti)
        form.addRow("Min and max antisolvent rate (g/s)", self.row(self.anti_rates))
        form.addRow("Min and max seed mass (g)", self.row(self.seed))
        for i, label in enumerate(("Count", "Concentration", "Size"), 1):
            value = float(weights.get(f"W{i}", int(i > 1)))
            if optimal:
                control = QCheckBox()
                control.setChecked(value > 0)
            else:
                control = self.number(0, 10000)
                control.setValue(value)
            self.weights.append(control)
            form.addRow(label + (" response" if optimal else " weight"), control)
        if optimal:
            form.addRow("Latin hypercube samples", self.samples)
        else:
            self.terms.setCurrentIndex(int(weights.get("Terms", "CSD") != "CSD"))
            form.addRow("Size terms", self.terms)
        form.addRow("Search iterations", self.iterations)

        self.status = QLabel("Searches one temperature ramp and one antisolvent pulse.")
        self.status.setWordWrap(True)
        self.result_table = table(("Time (s)", "Temperature (°C)", "Antisolvent (g/s)"))
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.start_design)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_design)
        self.export_button = QPushButton("Export recipe")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_recipe)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.result_table)
        layout.addWidget(self.status)
        layout.addWidget(self.row((self.run_button, self.stop_button, self.export_button,
                                   self.close_button)))
        self.template.currentIndexChanged.connect(self.load_template)
        self.load_template(0)
        action = "d_optimal_design" if optimal else "model_discrimination"
        context = "|".join(signature(case, index) for index, _ in self.models)
        step = latest_step(case, action, context=context)
        if step is not None:
            saved = step["result"]
            request = saved.get("request", {})
            self.template.setCurrentIndex(int(request.get("template_index", 0)))
            self.model.setCurrentIndex(int(request.get("model_index", 0)))
            self.iterations.setValue(int(request.get("iterations", 5)))
            self.samples.setValue(int(request.get("samples", 8)))
            position = self.terms.findText(str(request.get("terms", "CSD")))
            if position >= 0:
                self.terms.setCurrentIndex(position)
            for control, weight in zip(self.weights, np.asarray(request.get("weights", [])).reshape(-1)):
                if optimal:
                    control.setChecked(bool(weight))
                else:
                    control.setValue(float(weight))
            conditions = request.get("conditions", {})
            for key, control in (("batch", self.batch), ("concentration", self.concentration),
                                 ("final_anti", self.final_anti)):
                if key in conditions:
                    control.setValue(float(conditions[key]))
            for key, controls in (("temperature", self.temperatures), ("temperature_rates", self.temp_rates),
                                  ("liquid_masses", self.masses), ("antisolvent_rates", self.anti_rates),
                                  ("seed_bounds", self.seed)):
                for control, value in zip(controls, np.asarray(conditions.get(key, [])).reshape(-1)):
                    control.setValue(float(value))
            self.show_recipe((float(saved["score"]), saved["experiment"], saved["process"]),
                             remember=False)
            self.export_button.setEnabled(True)

    @staticmethod
    def number(lower, upper):
        box = QDoubleSpinBox()
        box.setDecimals(6)
        box.setRange(lower, upper)
        return box

    @staticmethod
    def row(widgets):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        return row

    def load_template(self, index):
        experiment = self.case["ExpData"][index]
        process = self.case["ProcInf"][index]
        temperature = np.asarray(experiment.get("TemperatureData", []))
        if temperature.ndim == 2 and temperature.shape[0] == 2 and temperature.size:
            self.batch.setValue(float(np.max(temperature[0])))
            for box, value in zip(self.temperatures, temperature[1, [0, -1]]):
                box.setValue(float(value))
        concentration = np.asarray(experiment.get("ConcentrationData", []))
        if "InitialConcentration" in experiment:
            self.concentration.setValue(float(experiment["InitialConcentration"]))
        elif concentration.size:
            self.concentration.setValue(float(concentration.reshape(2, -1)[1, 0]))
        solvent = float(process["SolvMass"])
        anti = float(process["AntiSolvMass"])
        for box, value in zip(self.masses, (solvent, anti)):
            box.setValue(value)
        flow = np.asarray(experiment.get("FlowRateData", []))
        addition = np.trapezoid(np.maximum(flow[1], 0), flow[0]) if flow.ndim == 2 and flow.size else 0
        if solvent + anti + addition > 0:
            self.final_anti.setValue((anti + addition) / (solvent + anti + addition))
        self.temp_rates[0].setValue(0)
        self.temp_rates[1].setValue(1)
        self.anti_rates[0].setValue(0)
        self.anti_rates[1].setValue(max(0.1, 2 * addition / max(self.batch.value(), 1)))
        for box in self.seed:
            box.setValue(float(process["SeedMass"]))

    def start_design(self):
        index = self.template.currentIndex()
        experiment, process = self.case["ExpData"][index], self.case["ProcInf"][index]
        settings = self.case.get("pbmsetting")
        settings = settings if isinstance(settings, dict) else {}
        shared = {"solubility": self.case["Solubility"], "settings": settings}
        try:
            if self.optimal:
                model = self.models[self.model.currentIndex()][1]
                score = partial(d_optimal_log_score, model, **shared,
                                enabled_terms=[w.isChecked() for w in self.weights],
                                samples=parameter_samples(model, self.samples.value()))
            else:
                score = partial(discrimination_score, [model for _, model in self.models],
                                **shared, weights=[w.value() for w in self.weights],
                                size_term=self.terms.currentText(),
                                experiment_count=len(self.case["ExpData"]))
        except (ValueError, IndexError) as error:
            self.status.setText(str(error))
            return

        if self.optimal and sum(weight.isChecked() for weight in self.weights) < 2:
            self.status.setText("Select at least two response terms.")
            return
        if not self.optimal and not any(weight.value() for weight in self.weights):
            self.status.setText("Set at least one positive response weight.")
            return
        conditions = {"batch": self.batch.value(), "concentration": self.concentration.value(),
                      "final_anti": self.final_anti.value()}
        for key, fields in (("temperature", self.temperatures), ("temperature_rates", self.temp_rates),
                            ("liquid_masses", self.masses), ("antisolvent_rates", self.anti_rates),
                            ("seed_bounds", self.seed)):
            conditions[key] = tuple(field.value() for field in fields)
        self.recipe = None
        self.result_table.setRowCount(0)
        self.export_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText("Preparing the design search…")
        self.job = DesignJob(experiment, process, conditions, score, self.iterations.value(), self,
                             settings=settings, workers=self.workers)
        self.job.message.connect(self.status.setText)
        self.job.completed.connect(self.show_recipe)
        self.job.failed.connect(self.status.setText)
        self.job.finished.connect(self.finish_design)
        self.job.start()

    def show_recipe(self, result, *, remember=True):
        score, experiment, process = result
        t, flow = experiment["TemperatureData"], experiment["FlowRateData"]
        time = np.union1d(t[0], flow[0])
        self.recipe = list(zip(time, np.interp(time, *t), np.interp(time, *flow)))
        self.seed_mass = process["SeedMass"]
        fill_table(self.result_table, self.recipe)
        label = "Log determinant" if self.optimal else "Model separation"
        self.status.setText(f"{label}: {score:.6g}   Seed mass: {self.seed_mass:.6g} g")
        if remember:
            action = "d_optimal_design" if self.optimal else "model_discrimination"
            context = "|".join(signature(self.case, index) for index, _ in self.models)
            request = {"template_index": self.template.currentIndex(), "model_index": self.model.currentIndex(),
                       "iterations": self.iterations.value(), "samples": self.samples.value(),
                       "terms": self.terms.currentText(),
                       "weights": [int(control.isChecked()) if self.optimal else control.value()
                                   for control in self.weights],
                       "conditions": self.job.inputs[2] if self.job is not None else {}}
            record_step(self.case, action,
                        {"score": score, "experiment": experiment, "process": process, "request": request},
                        context=context)

    def finish_design(self):
        self.job.deleteLater()
        self.job = None
        self.run_button.setEnabled(True)
        self.close_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(self.recipe is not None)

    def stop_design(self):
        if self.job is not None:
            self.job.requestInterruption()
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping after active candidates finish…")

    def reject(self):
        if self.job is None:
            super().reject()

    def export_recipe(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export recipe", "imed_design.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("time_s", "temperature_c", "antisolvent_g_s", "seed_mass_g"))
                writer.writerows((*row, self.seed_mass) for row in self.recipe)
        except OSError as error:
            QMessageBox.warning(self, "Export recipe", str(error))
