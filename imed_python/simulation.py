"""IMED forward simulation window and background worker."""

import numpy as np
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QTabWidget, QVBoxLayout, QWidget,
)

from .distribution import percent_density, volume_distribution
from .simulated_profile import editable_groups, prediction_inputs, remember_active_doe
from .simulated_profile_dialog import SimulatedProfileDialog
from .solver import load_case, model_parameters, pbm_backend_label, simulate
from .solver_choice import SolverChoice
from .session import history, record_step, signature


class Simulation(QThread):
    result = Signal(object)
    failed = Signal(str)
    message = Signal(str)

    def __init__(self, case, model, experiment, parent, numerics=None, *, simulated_profile=None):
        super().__init__(parent)
        self.case = case
        self.model = model
        self.experiment = experiment
        self.numerics = numerics
        self.simulated_profile = simulated_profile
        self.backend = ""

    def run(self):
        case, index = self.case, self.experiment
        settings = case.get("pbmsetting")
        settings = dict(settings) if isinstance(settings, dict) else {}
        if self.numerics is not None:
            settings["Numerics"] = self.numerics
        try:
            self.backend = pbm_backend_label(settings)
            self.message.emit(f"Simulating with {self.backend}…")
            parameters = model_parameters(case["ModelParameter"][self.model])
            experiment, process = prediction_inputs(case, index, self.simulated_profile)
            result = simulate(parameters, experiment, process, case["Solubility"], settings)
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.result.emit(result)


class Window(QMainWindow):
    workflow_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("IMED | PBM simulation")
        self.resize(950, 650)
        self.case = None
        self.worker = None
        self.last_profile = None
        self.open_button = QPushButton("Import MATLAB (.mat)")
        self.open_button.clicked.connect(self.choose_file)
        self.model = QComboBox()
        self.experiment = QComboBox()
        self.numerics = SolverChoice()
        self.run_button = QPushButton("Simulate")
        self.run_button.clicked.connect(self.run_simulation)
        self.run_button.setEnabled(False)
        self.profile_button = QPushButton("Simulate custom profile…")
        self.profile_button.clicked.connect(self.run_custom_simulation)
        self.profile_button.setEnabled(False)
        self.comparison_button = QPushButton("Experiment comparison")
        self.comparison_button.setEnabled(False)
        self.comparison_button.clicked.connect(lambda: self.show_evaluation(False))
        self.size_summary_button = QPushButton("Size summary")
        self.size_summary_button.setEnabled(False)
        self.size_summary_button.clicked.connect(lambda: self.show_evaluation(True))
        self.status = QLabel("Open a saved IMED case to get started.")
        self.status.setWordWrap(True)
        self.tabs = QTabWidget()
        self.plots = []
        for label in ("Particle count", "Concentration", "Volume-based distribution"):
            view = QChartView()
            self.plots.append(view)
            self.tabs.addTab(view, label)
        percentile_page = QWidget()
        percentile_layout = QVBoxLayout(percentile_page)
        self.percentile_summary = QLabel("Run a simulation to calculate D10, D50, and D90.")
        self.percentile_summary.setWordWrap(True)
        self.percentile_plot = QChartView()
        percentile_layout.addWidget(self.percentile_summary)
        percentile_layout.addWidget(self.percentile_plot)
        self.tabs.addTab(percentile_page, "D10 / D50 / D90")

        controls = QHBoxLayout()
        for widget in (self.open_button, QLabel("Model"), self.model,
                       QLabel("Experiment"), self.experiment, QLabel("PBM solver"),
                       self.numerics, self.run_button):
            controls.addWidget(widget)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.addLayout(controls)
        layout.addWidget(self.status)
        evaluation_buttons = QHBoxLayout()
        evaluation_buttons.addWidget(self.profile_button)
        evaluation_buttons.addWidget(self.comparison_button)
        evaluation_buttons.addWidget(self.size_summary_button)
        layout.addLayout(evaluation_buttons)
        layout.addWidget(self.tabs)
        self.setCentralWidget(body)
        self.model.currentIndexChanged.connect(self.restore_result)
        self.experiment.currentIndexChanged.connect(self.restore_result)
        self.numerics.currentIndexChanged.connect(self.restore_result)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import MATLAB IMED case", "", "MAT files (*.mat)")
        if path:
            self.open_case(path)

    def open_case(self, path):
        try:
            case = load_case(path)
        except Exception as error:
            self.status.setText(f"Unable to open case. {error}")
            return
        self.set_case(case, str(path))

    def set_case(self, case, message):
        self.case = case
        history(case)
        for combo, key in ((self.model, "ModelParameter"), (self.experiment, "ExpData")):
            combo.clear()
            combo.addItems(item.get("Name", str(index + 1)) for index, item in enumerate(case[key]))
        self.numerics.load_settings(case.get("pbmsetting"))
        selection = case.get("PythonWorkflow", {}).get("simulation_selection", {})
        if selection:
            self.model.setCurrentIndex(min(int(selection["model_index"]), self.model.count() - 1))
            self.experiment.setCurrentIndex(min(int(selection["experiment_index"]), self.experiment.count() - 1))
            self.numerics.load_settings(selection.get("pbmsetting"))
        self.run_button.setEnabled(self.model.count() > 0 and self.experiment.count() > 0)
        self.comparison_button.setEnabled(self.run_button.isEnabled())
        self.size_summary_button.setEnabled(self.run_button.isEnabled())
        self.status.setText(message)
        self.restore_result()

    def run_custom_simulation(self):
        dialog = SimulatedProfileDialog(self.case, self.model.currentIndex(), self.experiment.currentIndex(),
                                        self, previous=self.last_profile)
        if dialog.exec() == dialog.DialogCode.Accepted:
            self.run_simulation(simulated_profile=dialog.result_profile)
        dialog.deleteLater()

    def run_simulation(self, *, simulated_profile=None):
        settings = self.numerics.apply_settings(self.case.get("pbmsetting"))
        case = {**self.case, "pbmsetting": settings}
        self.worker = Simulation(case, self.model.currentIndex(), self.experiment.currentIndex(), self,
                                 simulated_profile=simulated_profile)
        self.worker.result.connect(self.show_result)
        self.worker.failed.connect(self.status.setText)
        self.worker.message.connect(self.status.setText)
        self.worker.finished.connect(self.finish_simulation)
        for widget in (self.open_button, self.run_button, self.experiment, self.model, self.numerics,
                       self.comparison_button, self.size_summary_button, self.profile_button):
            widget.setEnabled(False)
        self.status.setText("Preparing the simulation…")
        self.worker.start()

    def finish_simulation(self):
        self.worker.deleteLater()
        self.worker = None
        for widget in (self.open_button, self.run_button, self.experiment, self.model, self.numerics,
                       self.comparison_button, self.size_summary_button):
            widget.setEnabled(True)
        self.update_profile_button()

    def show_evaluation(self, summary):
        from .evaluation_dialog import EvaluationDialog

        case = {**self.case, "pbmsetting": self.numerics.apply_settings(self.case.get("pbmsetting"))}
        dialog = EvaluationDialog(case, self.model.currentIndex(), self, summary=summary)
        dialog.exec()
        dialog.deleteLater()
        self.workflow_changed.emit()

    def closeEvent(self, event):
        if self.worker is not None:
            event.ignore()
        else:
            super().closeEvent(event)

    def restore_result(self):
        if (self.case is None or self.worker is not None or self.model.currentIndex() < 0
                or self.experiment.currentIndex() < 0):
            self.update_profile_button()
            return
        case = {**self.case, "pbmsetting": self.numerics.apply_settings(self.case.get("pbmsetting"))}
        context = signature(case, self.model.currentIndex(), include_weights=False)
        self.last_profile = None
        for step in reversed(history(self.case)):
            if (step["action"] == "simulation" and step["signature"] == context
                    and int(step["model_index"]) == self.model.currentIndex()
                    and int(step["result"]["experiment_index"]) == self.experiment.currentIndex()):
                result = step["result"]
                self.show_result(tuple(np.asarray(result[key]) for key in
                                       ("count", "concentration", "csd")), remember=False,
                                 simulated_profile=result.get("simulated_profile"))
                label = self.last_profile["name"] if self.last_profile else "simulation"
                self.status.setText(f"Restored {label} · completed {step['completed_at']}.")
                self.update_profile_button()
                return
        for view in (*self.plots, self.percentile_plot):
            view.setChart(QChart())
        self.percentile_summary.setText("Run a simulation to calculate D10, D50, and D90.")
        self.status.setText("Choose a model and experiment, then run the forward simulation.")
        self.update_profile_button()

    def update_profile_button(self):
        valid = (self.case is not None and self.worker is None
                 and self.model.currentIndex() >= 0 and self.experiment.currentIndex() >= 0)
        groups = editable_groups(self.case, self.model.currentIndex(), self.last_profile) if valid else set()
        self.profile_button.setEnabled(bool(groups))
        self.profile_button.setToolTip("" if groups else "No editable DoE factors were found for this case and model.")

    def show_result(self, outputs, *, remember=True, simulated_profile=None):
        index = self.experiment.currentIndex()
        if self.worker is not None:
            simulated_profile = self.worker.simulated_profile
        self.last_profile = simulated_profile
        experiment, process = prediction_inputs(self.case, index, simulated_profile)
        settings = self.case.get("pbmsetting")
        count, concentration, csd = outputs
        volume, predicted = volume_distribution(
            csd, process, settings if isinstance(settings, dict) else {})
        axes = (("Time (s)", "Simulated particles per liquid volume"),
                ("Time (s)", "Solute mass fraction (g/g solution)"),
                ("Size (µm)", "Volume-based density (%/µm)"))
        fields = ("CountsData", "ConcentrationData", "FinalCSDsData")
        for view, values, axis_titles, field in zip(self.plots, (count, concentration, volume),
                                                     axes, fields):
            measured = self.measured_series(experiment.get(field, []))
            if measured is not None and field == "FinalCSDsData":
                measured = percent_density(measured[0], np.maximum(measured[1], 0))
            lines = [] if measured is None else [("Measured", measured)]
            lines.append(("Simulated", values))
            self.set_chart(view, lines, axis_titles,
                           measured_axis="Measured particle count" if field == "CountsData" else None)

        observed = np.asarray(experiment.get("FinalD10D50D90", [])).reshape(-1)
        if observed.size != 3 or not np.isfinite(observed).all() or not (observed > 0).all():
            observed = None
        lines = []
        summary = []
        for name, values, unavailable in (
            ("Simulated", predicted, "no predicted particle volume"),
            ("Measured", observed, "no measured percentiles available"),
        ):
            if values is None:
                summary.append(f"{name} — {unavailable}")
                continue
            lines.append((name, np.vstack((values, [10, 50, 90]))))
            sizes = "   ".join(f"D{p} {size:.4g} µm" for p, size in zip((10, 50, 90), values))
            summary.append(f"{name} — {sizes}")
        if predicted is not None and observed is not None and np.allclose(
                predicted, observed, rtol=1e-3, atol=0.001):
            summary.append("The simulated and measured points overlap at this scale.")
        # Draw the broad, dashed measured trace underneath the solid simulated trace
        # so both remain visible when their coordinates coincide.
        self.set_chart(self.percentile_plot, reversed(lines),
                       ("Size (µm)", "Cumulative volume (%)"), percentiles=True)
        self.percentile_summary.setText("\n".join(summary))
        backend = self.worker.backend if self.worker is not None else self.numerics.currentText()
        label = simulated_profile["name"] if simulated_profile else "Simulation"
        self.status.setText(f"{label} complete with {backend}.")
        if remember:
            case = self.worker.case if self.worker is not None else self.case
            self.case["PythonWorkflow"]["simulation_selection"] = {
                "model_index": self.model.currentIndex(), "experiment_index": index,
                "pbmsetting": case.get("pbmsetting", {})}
            result = {"experiment_index": index, "count": count, "concentration": concentration,
                      "csd": csd, "backend": backend, "pbmsetting": case.get("pbmsetting", {})}
            if simulated_profile:
                result["simulated_profile"] = simulated_profile
                remember_active_doe(self.case)
            record_step(self.case, "simulation", result,
                        model_index=self.model.currentIndex(),
                        context=signature(case, self.model.currentIndex(), include_weights=False))
            self.workflow_changed.emit()

    @staticmethod
    def measured_series(values):
        try:
            points = np.asarray(values, dtype=float)
        except (ValueError, TypeError):
            return None
        if points.shape == (2,):
            points = points[:, None]
        if points.ndim != 2 or points.shape[0] != 2:
            return None
        points = points[:, np.isfinite(points).all(axis=0)]
        return points[:, np.argsort(points[0], kind="stable")] if points.shape[1] else None

    @staticmethod
    def set_chart(view, lines, axis_titles, *, percentiles=False, measured_axis=None):
        chart = QChart()
        for name, values in lines:
            series = QLineSeries()
            series.setName(name)
            for x, y in values.T:
                series.append(float(x), float(y))
            chart.addSeries(series)
            measured = name == "Measured"
            pen = QPen(QColor("#D46B21" if measured else "#185ABD"))
            pen.setWidthF(5 if measured else 2)
            pen.setStyle(Qt.PenStyle.DashLine if measured else Qt.PenStyle.SolidLine)
            series.setPen(pen)
            series.setPointsVisible(percentiles or series.count() == 1)
        has_series = bool(chart.series())
        chart.legend().setVisible(has_series)
        if has_series:
            chart.createDefaultAxes()
            horizontal, vertical = chart.axes()
            horizontal.setTitleText(axis_titles[0])
            vertical.setTitleText(axis_titles[1])
            if percentiles:
                vertical.setRange(0, 100)
            if measured_axis and len(chart.series()) == 2:
                measured, simulated = chart.series()
                simulated.detachAxis(vertical)
                right = QValueAxis()
                chart.addAxis(right, Qt.AlignmentFlag.AlignRight)
                simulated.attachAxis(right)
                for series, axis, title in ((measured, vertical, measured_axis),
                                            (simulated, right, axis_titles[1])):
                    values = [point.y() for point in series.points()]
                    lower, upper = min(values), max(values)
                    if lower == upper:
                        padding = abs(lower) * 0.05 or 1
                        lower, upper = lower - padding, upper + padding
                    axis.setRange(lower, upper)
                    axis.setLabelFormat("%.3g")
                    axis.setTitleText(title)
                    axis.setTitleBrush(series.pen().brush())
        view.setChart(chart)
