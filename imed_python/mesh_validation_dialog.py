"""Plot size-mesh convergence and PBM solve times."""

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from .analysis_budget import AnalysisBudget, budget_field
from .control_panel import fill_table, table
from .mesh_grid import grid_edges, is_log_grid
from .mesh_validation import mesh_counts, validate_meshes
from .session import display_name, history, record_step, signature
from .solver_choice import SolverChoice


CONVERGENCE_TABS = ("D10 / D50 / D90", "Mass balance")


def add_plot_tab(tabs, title):
    page = QWidget()
    layout = QVBoxLayout(page)
    figure = Figure(layout="constrained")
    canvas = FigureCanvasQTAgg(figure)
    layout.addWidget(NavigationToolbar2QT(canvas, page))
    layout.addWidget(canvas, 1)
    tabs.addTab(page, title)
    return figure, canvas, layout


def plot_runtime(axis, result):
    bins = [int(row["bins"]) for row in result["runs"]]
    seconds = [float(row["pbm_seconds"]) for row in result["runs"]]
    axis.plot(bins, seconds, "o--", color="0.25", label="PBM solve time")
    label = "Second JAX solve (s)" if result["uses_jax"] else "PBM solve time (s)"
    axis.set_ylabel(label)
    axis.set_ylim(bottom=0)


def convergence_tabs(tabs):
    figures = tuple(add_plot_tab(tabs, title)[:2] for title in CONVERGENCE_TABS)
    tabs.setTabToolTip(tabs.count() - 1, "Peak absolute total API error relative to initial API. "
                                       "Includes overflow beyond Lmax.")
    return figures


def plot_convergence(figures, result):
    runs = result["runs"]
    bins = [int(row["bins"]) for row in runs]
    sizes = np.stack([np.asarray(row["percentiles"], dtype=float).reshape(3) for row in runs])
    mass_errors = np.asarray([row["mass_error_ppm"] for row in runs], dtype=float)
    datasets = (
        ("Final size (µm)", zip(("D10", "D50", "D90"), sizes.T)),
        ("Peak absolute API balance error (ppm)", (("Mass balance error", mass_errors),)),
    )
    for (figure, canvas), (ylabel, series) in zip(figures, datasets):
        axis = figure.subplots()
        for label, values in series:
            axis.plot(bins, values, "o-", label=label)
        axis.set(xlabel="Number of size bins", ylabel=ylabel)
        axis.grid(alpha=.2)
        timing = axis.twinx()
        plot_runtime(timing, result)
        handles, labels = axis.get_legend_handles_labels()
        time_handles, time_labels = timing.get_legend_handles_labels()
        axis.legend(handles + time_handles, labels + time_labels)
        canvas.draw_idle()


class MeshConvergenceDialog(QDialog):
    def __init__(self, result, parent):
        super().__init__(parent)
        self.setWindowTitle("IMED | Mesh convergence")
        self.resize(760, 480)
        self.setSizeGripEnabled(True)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        plot_convergence(convergence_tabs(tabs), result)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        layout.addWidget(close)


class MeshValidationJob(QThread):
    completed = Signal(object)
    failed = Signal(str)
    message = Signal(str)

    def __init__(self, case, model, experiment, bins, budget, parent):
        super().__init__(parent)
        self.case, self.model, self.experiment = case, model, experiment
        self.bins, self.budget = bins, budget

    def run(self):
        try:
            result = validate_meshes(self.case, self.model, self.experiment, self.bins,
                                     cancelled=self.budget.check, progress=self.message.emit,
                                     budget_expired=self.budget.expired)
        except InterruptedError:
            self.failed.emit("Mesh validation stopped.")
        except Exception as error:
            self.failed.emit(str(error))
        else:
            result["time_budget_s"] = self.budget.seconds
            self.completed.emit(result)


class MeshValidationDialog(QDialog):
    def __init__(self, case, parent=None, *, model_index=None):
        super().__init__(parent)
        self.setWindowTitle("IMED | Mesh validation")
        screen = self.screen().availableGeometry()
        self.resize(min(1000, screen.width() - 24), min(610, screen.height() - 48))
        self.setSizeGripEnabled(True)
        self.case, self.job, self.result = case, None, None
        self.budget = AnalysisBudget()
        self.model, self.experiment = QComboBox(), QComboBox()
        self.model.addItems(display_name(model, f"Model {i+1}") for i, model in enumerate(case["ModelParameter"]))
        self.model.setCurrentIndex(0 if model_index is None else model_index)
        self.experiment.addItems(display_name(exp, f"Experiment {i+1}") for i, exp in enumerate(case["ExpData"]))
        self.minimum, self.maximum, self.stages = QSpinBox(), QSpinBox(), QSpinBox()
        mesh_inputs = QHBoxLayout()
        for label, widget, default in (("Minimum bins", self.minimum, 100),
                                       ("Maximum bins", self.maximum, 1000),
                                       ("Stages (including endpoints)", self.stages, 3)):
            widget.setRange(2, 10000)
            widget.setValue(default)
            mesh_inputs.addWidget(QLabel(label))
            mesh_inputs.addWidget(widget)
        settings = case["pbmsetting"]
        self.numerics = SolverChoice(allow_archived=not is_log_grid(settings))
        self.numerics.load_settings(settings)
        self.time_budget = budget_field()
        self.plan = QLabel()
        form = QFormLayout()
        selection = QHBoxLayout()
        selection.addWidget(self.model, 1)
        selection.addWidget(QLabel("Experiment"))
        selection.addWidget(self.experiment, 1)
        form.addRow("Model", selection)
        solver_controls = QHBoxLayout()
        solver_controls.addWidget(self.numerics, 1)
        solver_controls.addWidget(QLabel("Total time budget"))
        solver_controls.addWidget(self.time_budget)
        form.addRow("PBM solver", solver_controls)
        spacing = "logarithmic" if is_log_grid(settings) else "uniform"
        form.addRow("Size domain", QLabel(f"{settings['Lmin']:g}–{settings['Lmax']:g} µm · {spacing}"))
        form.addRow("Mesh stages", mesh_inputs)
        form.addRow("Bin counts", self.plan)
        self.tabs = QTabWidget()
        distribution, distribution_canvas, _ = add_plot_tab(self.tabs, "Final volume distributions")
        runtime, runtime_canvas, timing_layout = add_plot_tab(self.tabs, "Simulation time")
        self.convergence = convergence_tabs(self.tabs)
        self.figures = ((distribution, distribution_canvas), (runtime, runtime_canvas), *self.convergence)
        self.timings = table(("Bins", "PBM solve time (s)", "Status"))
        self.timings.setMaximumHeight(155)
        timing_layout.addWidget(self.timings)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.run_button = QPushButton("Run validation")
        self.stop_button = QPushButton("Stop")
        self.details_button = QPushButton("Convergence plots…")
        self.close_button = QPushButton("Close")
        self.stop_button.setEnabled(False)
        self.run_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.details_button.clicked.connect(self.show_convergence)
        self.close_button.clicked.connect(self.reject)
        actions = QHBoxLayout()
        for button in (self.run_button, self.stop_button, self.details_button, self.close_button):
            actions.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(actions)
        layout.addWidget(self.status)
        layout.addWidget(self.tabs, 1)
        self.controls = (self.model, self.experiment, self.numerics, self.time_budget,
                         self.minimum, self.maximum, self.stages)
        self.update_plan()
        self.restore_result(model_index)
        for combo in (self.model, self.experiment, self.numerics):
            combo.currentIndexChanged.connect(self.update_plan)
        for spin in (self.minimum, self.maximum, self.stages, self.time_budget):
            spin.valueChanged.connect(self.update_plan)

    def update_plan(self):
        self.result = None
        self.details_button.setEnabled(False)
        self.timings.setRowCount(0)
        for figure, canvas in self.figures:
            figure.clear()
            canvas.draw_idle()
        try:
            self.counts = mesh_counts(self.minimum.value(), self.maximum.value(), self.stages.value())
        except ValueError as error:
            self.plan.setText("—")
            self.status.setText(str(error))
            self.run_button.setEnabled(False)
            return
        preview = ", ".join(f"{bins:,}" for bins in self.counts[:6])
        if len(self.counts) > 6:
            preview += f", …, {self.counts[-1]:,}"
        self.plan.setText(preview)
        self.run_button.setEnabled(self.model.count() > 0 and self.experiment.count() > 0)
        self.status.setText("JAX runs each mesh twice and reports the second solve time."
                             if self.numerics.currentData()[0] == "jax"
                            else "Times measure the PBM solve on each mesh.")

    def restore_result(self, model_index):
        for step in reversed(history(self.case)):
            if step["action"] != "mesh_validation":
                continue
            model = int(step["model_index"])
            result = step["result"]
            experiment = int(result["experiment_index"])
            if (0 <= model < self.model.count() and 0 <= experiment < self.experiment.count()
                    and model_index in (None, model)
                    and step["signature"] == signature(self.case, model, include_weights=False)):
                runs = [result["runs"]] if isinstance(result["runs"], dict) else result["runs"]
                planned = result.get("planned_bins", [row["bins"] for row in runs])
                planned = np.asarray(planned).reshape(-1)
                self.model.setCurrentIndex(model)
                self.experiment.setCurrentIndex(experiment)
                self.minimum.setValue(int(planned[0]))
                self.maximum.setValue(int(planned[-1]))
                self.stages.setValue(len(planned))
                self.numerics.load_settings(result.get("settings", self.case["pbmsetting"]))
                self.time_budget.setValue(float(result.get("time_budget_s", 0.)))
                self.update_plan()
                self.show_result(result, remember=False)
                break

    def start(self):
        if self.job is not None or not self.run_button.isEnabled():
            return
        self.update_plan()
        self.budget.start(self.time_budget.value())
        settings = self.budget.apply(self.numerics.apply_settings(self.case["pbmsetting"]))
        case = {**self.case, "pbmsetting": settings}
        self.job = MeshValidationJob(case, self.model.currentIndex(), self.experiment.currentIndex(),
                                     self.counts, self.budget, self)
        for control in (*self.controls, self.run_button, self.close_button):
            control.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.job.message.connect(self.status.setText)
        self.job.failed.connect(self.status.setText)
        self.job.completed.connect(self.show_result)
        self.job.finished.connect(self.finish)
        self.job.start()

    def stop(self):
        if self.job is not None:
            self.budget.stop()
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping mesh validation…")

    def finish(self):
        self.job.deleteLater()
        self.job = None
        for control in (*self.controls, self.run_button, self.close_button):
            control.setEnabled(True)
        self.stop_button.setEnabled(False)

    def reject(self):
        if self.job is None:
            super().reject()

    def show_result(self, result, *, remember=True):
        if isinstance(result["runs"], dict):
            result = {**result, "runs": [result["runs"]]}
        self.result = result
        runs = result["runs"]
        lower, upper = result["bounds"]
        for figure, _ in self.figures:
            figure.clear()
        distribution, _ = self.figures[0]
        axis = distribution.subplots()
        traces = [row for row in runs if np.size(row["density"])]
        color_map = colormaps["viridis"]
        norm = Normalize(int(runs[0]["bins"]), int(runs[-1]["bins"]))
        settings = {"Lmin": lower, "Lmax": upper, **result.get("settings", {})}
        for row in traces:
            bins = int(row["bins"])
            axis.stairs(np.asarray(row["density"]).reshape(-1), grid_edges(settings, bins),
                        color=color_map(norm(bins)), label=f"{bins:,} bins")
        if len(traces) > 10:
            distribution.colorbar(ScalarMappable(norm=norm, cmap=color_map), ax=axis, label="Number of size bins")
        elif traces:
            axis.legend()
        axis.set(xlabel="Size (µm)", ylabel="Volume density (%/µm)", xlim=(lower, upper))
        axis.grid(alpha=.2)
        timing = self.figures[1][0].subplots()
        plot_runtime(timing, result)
        timing.set_xlabel("Number of size bins")
        timing.grid(alpha=.2)
        plot_convergence(self.convergence, result)
        for _, canvas in self.figures:
            canvas.draw_idle()
        fill_table(self.timings, [(row["bins"], f"{row['pbm_seconds']:.4g}" if np.isfinite(row["pbm_seconds"]) else "—",
                                  row["status"]) for row in runs])
        self.details_button.setEnabled(bool(traces))
        planned_count = np.size(result.get("planned_bins", runs))
        prefix = "Time budget reached · " if result.get("budget_expired") else ""
        self.status.setText(f"{prefix}{result['backend']} · {len(traces)}/{planned_count} meshes completed")
        if remember:
            model = self.model.currentIndex()
            record_step(self.case, "mesh_validation", result, model_index=model,
                        context=signature(self.case, model, include_weights=False))

    def show_convergence(self):
        dialog = MeshConvergenceDialog(self.result, self)
        dialog.exec()
        dialog.deleteLater()
