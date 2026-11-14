"""IMED dialog for local confidence intervals, independent of optimization."""
import csv
from threading import Event

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QTabWidget, QVBoxLayout,
)

from .control_panel import fill_table, item_count, table
from .estimation import FitObjective, series
from .parameter_labels import PARAMETERS
from .solver import model_parameters, pbm_backend_label
from .size_data import TREND_SIZE_TERM
from .uncertainty import local_intervals
from .session import latest_pe_step, latest_step, pack_intervals, record_step, signature, unpack_fit, unpack_intervals


class UncertaintyJob(QThread):
    message = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, case, model, weights, terms, confidence, parent, *, convention="matlab"):
        super().__init__(parent)
        self.case, self.model = case, model
        self.weights, self.terms, self.confidence = weights, terms, confidence
        self.convention = convention
        self.cancel_requested = Event()

    def run(self):
        case = self.case
        try:
            settings = case.get("pbmsetting")
            settings = settings if isinstance(settings, dict) else {}
            self.message.emit(f"Evaluating current values with {pbm_backend_label(settings)}…")
            objective = FitObjective(case["ExpData"], case["ProcInf"], case["Solubility"],
                                     settings, self.weights, self.terms,
                                     at_measurements=self.convention == "student_t",
                                     cancelled=self.cancel_requested.is_set)
            result = local_intervals(self.model, objective, confidence=self.confidence,
                                     progress=self.message.emit, convention=self.convention)
        except InterruptedError:
            self.failed.emit("Uncertainty calculation stopped.")
        except Exception as error:
            self.failed.emit(str(error))
        else:
            if self.cancel_requested.is_set():
                self.failed.emit("Uncertainty calculation stopped.")
            else:
                self.completed.emit(result)


class IntervalView(QTabWidget):
    """Show intervals and the exact covariance used to calculate them."""

    HEADERS = ("Parameter", "Value", "Standard error", "Lower limit", "Upper limit")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.summary = table(self.HEADERS)
        self.result = None
        self.covariance_window = None
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.addTab(self.summary, "Confidence intervals")
        self.addTab(self.canvas, "Covariance")
        self.popout_button = QPushButton("Open covariance window")
        self.popout_button.clicked.connect(self.open_covariance)
        self.popout_button.setEnabled(False)
        self.setCornerWidget(self.popout_button)
        self.setTabEnabled(1, False)

    @staticmethod
    def rows(result):
        labels = (PARAMETERS[index][0] for index in result.indices)
        return list(zip(labels, result.values, result.standard_errors, result.lower, result.upper))

    def clear_result(self):
        self.result = None
        self.summary.setRowCount(0)
        self.setCurrentIndex(0)
        self.setTabEnabled(1, False)
        self.figure.clear()
        self.popout_button.setEnabled(False)
        if self.covariance_window is not None:
            self.covariance_window.hide()

    @staticmethod
    def draw_covariance(figure, result, *, aspect="auto"):
        figure.clear()
        axes = figure.add_subplot()
        matrix = result.covariance
        labels = [PARAMETERS[index][0] for index in result.indices]
        limit = float(np.max(np.abs(matrix))) or 1.
        heatmap = axes.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect=aspect)
        positions = np.arange(len(labels))
        axes.set_xticks(positions, labels, rotation=55, ha="right")
        axes.set_yticks(positions, labels)
        axes.tick_params(labelsize=8 if len(labels) > 12 else 10)
        axes.set_title("Parameter covariance")
        figure.colorbar(heatmap, ax=axes, label="Covariance")
        if len(labels) <= 8:
            for (row, column), value in np.ndenumerate(matrix):
                axes.text(column, row, f"{value:.3g}", ha="center", va="center",
                          color="white" if abs(value) > .6 * limit else "black", fontsize=8)

    def show_result(self, result):
        self.result = result
        fill_table(self.summary, self.rows(result))
        self.draw_covariance(self.figure, result)
        self.setTabEnabled(1, True)
        self.popout_button.setEnabled(True)
        self.canvas.draw_idle()

    def open_covariance(self):
        if self.result is None:
            return
        if self.covariance_window is None:
            self.covariance_window = QDialog(self)
            self.covariance_window.setWindowTitle("IMED | Parameter covariance")
            self.covariance_window.resize(900, 720)
            self.covariance_figure = Figure(figsize=(8, 6), layout="compressed")
            self.covariance_canvas = FigureCanvasQTAgg(self.covariance_figure)
            layout = QVBoxLayout(self.covariance_window)
            layout.addWidget(NavigationToolbar2QT(self.covariance_canvas, self.covariance_window))
            layout.addWidget(self.covariance_canvas, 1)
            close = QPushButton("Close")
            close.clicked.connect(self.covariance_window.close)
            layout.addWidget(close)
        self.draw_covariance(self.covariance_figure, self.result, aspect="equal")
        self.covariance_canvas.draw_idle()
        self.covariance_window.show()
        self.covariance_window.raise_()
        self.covariance_window.activateWindow()


class ParameterUncertainty(QDialog):
    @staticmethod
    def saved_result(case, model_index):
        step = latest_step(case, "parameter_uncertainty", model_index=model_index,
                           context=signature(case, model_index, include_weights=False))
        result = unpack_intervals(step["result"].get("intervals")) if step is not None else None
        if result is None:
            fit_step = latest_pe_step(case, model_index)
            if fit_step is not None:
                fit = unpack_fit(fit_step["result"])
                model = case["ModelParameter"][model_index]
                active = np.asarray(model["paraset"])[:, 0] == 1
                if np.array_equal(model_parameters(model)[active], fit.parameters):
                    result = fit.intervals
        return result, step

    def __init__(self, case, parent=None, *, model_index=0, initial_result=None):
        super().__init__(parent)
        self.setWindowTitle("IMED | Parameter Uncertainty")
        self.resize(760, 575)
        self.case = case
        self.job = None
        self.result = None
        restore_request = initial_result is None
        if initial_result is None:
            initial_result, saved_step = self.saved_result(case, model_index)
        else:
            saved_step = None

        self.model = QComboBox()
        for index, _ in enumerate(case["ModelParameter"], 1):
            self.model.addItem(f"Model {index}")
        self.model.setCurrentIndex(model_index)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.01, 99.99)
        self.confidence.setValue(95)
        self.confidence.setSuffix("%")
        self.convention = QComboBox()
        self.convention.addItem("MATLAB IMED (normal interval)", "matlab")
        self.convention.addItem("Student t (measured points)", "student_t")
        if initial_result is not None:
            self.confidence.setValue(100 * initial_result.confidence)
            self.convention.setCurrentIndex(self.convention.findData(initial_result.convention))

        def has_response(field):
            return any(item_count(experiment.get(field)) for experiment in case["ExpData"])

        def has_concentration_measurements():
            for experiment in case["ExpData"]:
                data = experiment.get("ConcentrationData")
                if item_count(data):
                    try:
                        points = series(data, "ConcentrationData")
                    except ValueError:
                        continue
                    if np.any(points[0] > 0):
                        return True
            return False

        saved = case.get("objweightfactor")
        saved = saved if isinstance(saved, dict) else {}
        self.terms = QComboBox()
        for label, field in (("CSD", "FinalCSDsData"), ("D10 D50 D90", "FinalD10D50D90")):
            if has_response(field):
                self.terms.addItem(label)
        if any(has_response(field) for field in ("D10Data", "D50Data", "D90Data")):
            self.terms.addItem(TREND_SIZE_TERM)
        preferred = self.terms.findText(str(saved.get("Terms", "CSD")))
        if preferred >= 0:
            self.terms.setCurrentIndex(preferred)
        available = (has_response("CountsData"), has_concentration_measurements(),
                     self.terms.count() > 0)

        form = QFormLayout()
        form.addRow("Model", self.model)
        form.addRow("Confidence level", self.confidence)
        form.addRow("CI convention", self.convention)
        self.weights = []
        for index, label in enumerate(("Count weight", "Concentration weight", "Size weight")):
            control = QDoubleSpinBox()
            control.setRange(0, 10000)
            control.setDecimals(8)
            try:
                weight = float(saved.get(f"W{index + 1}", int(index == 1)))
            except (TypeError, ValueError):
                weight = 0.
            if available[index] and np.isfinite(weight):
                control.setValue(weight)
            control.setEnabled(available[index])
            self.weights.append(control)
            form.addRow(label, control)
        if not any(control.value() for control in self.weights):
            for index in (1, 0, 2):
                if available[index]:
                    self.weights[index].setValue(1)
                    break
        if available[2]:
            form.addRow("Size terms", self.terms)

        note = QLabel("Intervals use the current fitted or nominal values and measured responses. "
                      "Select continuous parameters using Estimate in Model Configurations. "
                      "MATLAB IMED uses the 95% normal multiplier by default. "
                      "Intervals at nominal values describe local uncertainty and are provisional.")
        note.setWordWrap(True)
        self.view = IntervalView()
        self.summary = self.view.summary
        self.status = QLabel("Ready to calculate." if any(available)
                             else "Load measured responses to calculate uncertainty.")
        self.status.setWordWrap(True)
        self.run_button = QPushButton("Calculate intervals")
        self.run_button.setEnabled(any(available))
        self.run_button.clicked.connect(self.start)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.export_button = QPushButton("Export CSV")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_csv)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        actions = QHBoxLayout()
        for button in (self.run_button, self.stop_button, self.export_button, self.close_button):
            actions.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.status)
        layout.addLayout(actions)

        self.controls = (self.model, self.confidence, self.convention, self.terms,
                         *(control for control in self.weights if control.isEnabled()))
        self.model.currentIndexChanged.connect(self.restore_selected_result)
        self.terms.currentIndexChanged.connect(self.clear_result)
        self.confidence.valueChanged.connect(self.clear_result)
        self.convention.currentIndexChanged.connect(self.clear_result)
        for control in self.weights:
            control.valueChanged.connect(self.clear_result)
        if initial_result is not None:
            if saved_step is not None and restore_request:
                request = saved_step["result"].get("request", {})
                for control, weight in zip(self.weights, np.asarray(request.get("weights", [])).reshape(-1)):
                    if control.isEnabled():
                        control.setValue(float(weight))
                position = self.terms.findText(str(request.get("terms", "")))
                if position >= 0:
                    self.terms.setCurrentIndex(position)
            self.show_result(initial_result, remember=False)
            self.export_button.setEnabled(True)

    def restore_selected_result(self):
        self.clear_result()
        result, step = self.saved_result(self.case, self.model.currentIndex())
        if result is None:
            return
        self.confidence.setValue(100*result.confidence)
        self.convention.setCurrentIndex(self.convention.findData(result.convention))
        if step is not None:
            request = step["result"].get("request", {})
            for control, weight in zip(self.weights, np.asarray(request.get("weights", [])).reshape(-1)):
                if control.isEnabled():
                    control.setValue(float(weight))
            position = self.terms.findText(str(request.get("terms", "")))
            if position >= 0:
                self.terms.setCurrentIndex(position)
        self.show_result(result, remember=False)
        self.export_button.setEnabled(True)

    def clear_result(self):
        self.result = None
        self.view.clear_result()
        self.export_button.setEnabled(False)
        if self.job is None:
            self.status.setText("Ready to calculate.")

    def start(self):
        weights = [control.value() for control in self.weights]
        if not any(weights):
            self.status.setText("Set a positive weight for a measured response.")
            return
        self.clear_result()
        self.job = UncertaintyJob(
            self.case, self.case["ModelParameter"][self.model.currentIndex()],
            weights, self.terms.currentText(), self.confidence.value() / 100, self,
            convention=self.convention.currentData())
        for control in (*self.controls, self.run_button, self.close_button):
            control.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.job.message.connect(self.status.setText)
        self.job.completed.connect(self.show_result)
        self.job.failed.connect(self.status.setText)
        self.job.finished.connect(self.finish)
        self.job.start()

    def stop(self):
        if self.job is not None:
            self.job.cancel_requested.set()
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping uncertainty calculation…")

    def finish(self):
        self.job.deleteLater()
        self.job = None
        for control in (*self.controls, self.run_button, self.close_button):
            control.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(self.result is not None)

    def reject(self):
        if self.job is None:
            super().reject()

    def show_result(self, result, *, remember=True):
        self.result = result
        self.view.show_result(result)
        if remember:
            index = self.model.currentIndex()
            record_step(self.case, "parameter_uncertainty",
                        {"intervals": pack_intervals(result),
                         "request": {"weights": [control.value() for control in self.weights],
                                     "terms": self.terms.currentText()}},
                        model_index=index,
                        context=signature(self.case, index, include_weights=False))
        self.status.setText(
            f"{result.source} · approximate {100 * result.confidence:.4g}% intervals · "
            f"{result.degrees_of_freedom} residual degrees of freedom · "
            + ("MATLAB IMED" if result.convention == "matlab" else "Student t") + ".")

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export parameter uncertainty",
                                              "imed_parameter_uncertainty.csv", "CSV (*.csv)")
        if not path:
            return
        result = self.result
        metadata = (self.model.currentText(), result.source, result.convention, result.confidence,
                    result.degrees_of_freedom)
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("Model", "Values used", "CI convention", "Confidence level", "Residual degrees of freedom",
                                 *self.view.HEADERS))
                writer.writerows((*metadata, *row) for row in self.view.rows(result))
        except OSError as error:
            QMessageBox.warning(self, "Export parameter uncertainty", str(error))
