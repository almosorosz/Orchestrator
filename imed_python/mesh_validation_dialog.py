"""Plot size-mesh convergence and PBM solve times."""

from threading import Event

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

from .control_panel import fill_table, table
from .mesh_validation import mesh_counts, validate_meshes
from .session import display_name, history, record_step, signature


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


class MeshConvergenceDialog(QDialog):
    def __init__(self, result, parent):
        super().__init__(parent)
        self.setWindowTitle("IMED | Mesh convergence")
        self.resize(1020, 590)
        self.setSizeGripEnabled(True)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        runs = result["runs"]
        bins = [int(row["bins"]) for row in runs]
        sizes = np.stack([np.asarray(row["percentiles"], dtype=float).reshape(3) for row in runs])
        mass_errors = np.asarray([row["mass_error_ppm"] for row in runs], dtype=float)
        for title, ylabel, series in (
            ("D10 / D50 / D90", "Final size (µm)", zip(("D10", "D50", "D90"), sizes.T)),
            ("Material balance", "Peak absolute API balance error (ppm)", (("Mass balance error", mass_errors),)),
        ):
            figure, canvas, _ = add_plot_tab(tabs, title)
            axis = figure.subplots()
            for label, values in series:
                axis.plot(bins, values, "o-", label=label)
            axis.set_xlabel("Number of size bins")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=.2)
            timing = axis.twinx()
            plot_runtime(timing, result)
            handles, labels = axis.get_legend_handles_labels()
            time_handles, time_labels = timing.get_legend_handles_labels()
            axis.legend(handles + time_handles, labels + time_labels)
            canvas.draw_idle()
        tabs.setTabToolTip(1, "Maximum absolute total API error over the saved time points, "
                              "relative to initial API. Includes overflow beyond Lmax.")
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        layout.addWidget(close)


class MeshValidationJob(QThread):
    completed = Signal(object)
    failed = Signal(str)
    message = Signal(str)

    def __init__(self, case, model, experiment, bins, cancelled, parent):
        super().__init__(parent)
        self.case, self.model, self.experiment = case, model, experiment
        self.bins, self.cancelled = bins, cancelled

    def run(self):
        try:
            result = validate_meshes(self.case, self.model, self.experiment, self.bins,
                                     cancelled=self.cancelled, progress=self.message.emit)
        except InterruptedError:
            self.failed.emit("Mesh validation stopped.")
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.completed.emit(result)


class MeshValidationDialog(QDialog):
    def __init__(self, case, parent=None, *, model_index=None):
        super().__init__(parent)
        self.setWindowTitle("IMED | Mesh validation")
        self.resize(1060, min(700, self.screen().availableGeometry().height() - 48))
        self.setSizeGripEnabled(True)
        self.case, self.job, self.result = case, None, None
        self.cancel_requested = Event()
        # Keep the Diffrax event callback stable across warm-up and timed solves.
        self.cancelled = self.cancel_requested.is_set
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
        self.plan = QLabel()
        form = QFormLayout()
        form.addRow("Model", self.model)
        form.addRow("Experiment", self.experiment)
        form.addRow("Size domain", QLabel(f"{settings['Lmin']:g}–{settings['Lmax']:g} µm"))
        form.addRow("Mesh stages", mesh_inputs)
        form.addRow("Bin counts", self.plan)
        self.tabs = QTabWidget()
        distribution, distribution_canvas, _ = add_plot_tab(self.tabs, "Final volume distributions")
        runtime, runtime_canvas, timing_layout = add_plot_tab(self.tabs, "Simulation time")
        self.figures = ((distribution, distribution_canvas), (runtime, runtime_canvas))
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
        self.controls = (self.model, self.experiment, self.minimum, self.maximum, self.stages)
        self.update_plan()
        self.restore_result(model_index)
        for combo in (self.model, self.experiment):
            combo.currentIndexChanged.connect(self.update_plan)
        for spin in (self.minimum, self.maximum, self.stages):
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
                            if self.case["pbmsetting"].get("Numerics") == "jax"
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
                runs = result["runs"]
                self.model.setCurrentIndex(model)
                self.experiment.setCurrentIndex(experiment)
                self.minimum.setValue(int(runs[0]["bins"]))
                self.maximum.setValue(int(runs[-1]["bins"]))
                self.stages.setValue(len(runs))
                self.update_plan()
                self.show_result(result, remember=False)
                break

    def start(self):
        if self.job is not None or not self.run_button.isEnabled():
            return
        self.cancel_requested.clear()
        self.update_plan()
        self.job = MeshValidationJob(self.case, self.model.currentIndex(), self.experiment.currentIndex(),
                                     self.counts, self.cancelled, self)
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
            self.cancel_requested.set()
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
        for row in traces:
            bins = int(row["bins"])
            axis.stairs(np.asarray(row["density"]).reshape(-1), np.linspace(lower, upper, bins + 1),
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
        for _, canvas in self.figures:
            canvas.draw_idle()
        fill_table(self.timings, [(row["bins"], f"{row['pbm_seconds']:.4g}" if np.isfinite(row["pbm_seconds"]) else "—",
                                  row["status"]) for row in runs])
        self.details_button.setEnabled(bool(traces))
        self.status.setText(f"{result['backend']} · {len(traces)}/{len(runs)} meshes completed")
        if remember:
            model = self.model.currentIndex()
            record_step(self.case, "mesh_validation", result, model_index=model,
                        context=signature(self.case, model, include_weights=False))

    def show_convergence(self):
        dialog = MeshConvergenceDialog(self.result, self)
        dialog.exec()
        dialog.deleteLater()
