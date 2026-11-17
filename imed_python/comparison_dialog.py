"""PBM mesh comparison plots against an independent QMOM solution."""
import csv

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from .analysis_budget import AnalysisBudget, budget_field
from .comparison import compare_model
from .mesh_grid import is_log_grid
from .session import history, record_step, signature
from .solver_choice import SolverChoice


class ComparisonJob(QThread):
    message = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, case, model_index, experiment_index, parent, *, budget):
        super().__init__(parent)
        self.case, self.model_index, self.experiment_index = case, model_index, experiment_index
        self.budget = budget

    def run(self):
        try:
            result = compare_model(
                self.case["ModelParameter"][self.model_index],
                self.case["ExpData"][self.experiment_index], self.case["ProcInf"][self.experiment_index],
                self.case["Solubility"], self.case["pbmsetting"],
                cancelled=self.budget.check, progress=self.message.emit, budget_expired=self.budget.expired)
        except InterruptedError:
            self.failed.emit("Time budget reached before the PBM comparison finished." if self.budget.expired()
                             else "PBM–QMOM comparison stopped.")
        except Exception as error:
            self.failed.emit(str(error))
        else:
            result["time_budget_s"] = self.budget.seconds
            self.completed.emit(result)


class ComparisonDialog(QDialog):
    def __init__(self, case, parent=None, *, model_index=None):
        super().__init__(parent)
        self.setWindowTitle("IMED | PBM–QMOM mesh comparison")
        screen = self.screen().availableGeometry()
        self.resize(min(1040, screen.width() - 24), min(670, screen.height() - 48))
        self.setSizeGripEnabled(True)
        self.case = case
        self.result = None
        self.job = None
        self.budget = AnalysisBudget()
        settings = case.get("pbmsetting")
        settings = settings if isinstance(settings, dict) else {}
        self.numerics = SolverChoice(allow_archived=not is_log_grid(settings))
        self.numerics.load_settings(settings)
        self.time_budget = budget_field()
        self.model = QComboBox()
        self.model.addItems(f"Model {i+1}" for i in range(len(case["ModelParameter"])))
        self.model.setCurrentIndex(0 if model_index is None else model_index)
        self.experiment = QComboBox()
        self.experiment.setObjectName("qmom_experiment")
        self.experiment.addItems(f"Experiment {i+1}" for i in range(len(case["ExpData"])))
        self.lmax = QDoubleSpinBox()
        self.lmax.setRange(float(settings.get("Lmin", 0)) + .001, 1e9)
        self.lmax.setDecimals(3)
        self.lmax.setValue(float(settings.get("Lmax", 250)))
        self.cells = QSpinBox()
        self.cells.setRange(2, 10000)
        self.cells.setValue(int(settings.get("N", 250)))
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
        mesh = QWidget()
        row = QHBoxLayout(mesh)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Lmax (µm)"))
        row.addWidget(self.lmax)
        row.addWidget(QLabel("Bins"))
        row.addWidget(self.cells)
        form.addRow("PBM mesh", mesh)
        self.tabs = QTabWidget()
        self.figures = {}
        for name in ("Material balance", "Moments", "D10–D50–D90", "Concentration and conditions",
                     "Final volume distribution"):
            figure = Figure(figsize=(10.5, 5.5), layout="constrained")
            canvas = FigureCanvasQTAgg(figure)
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.addWidget(NavigationToolbar2QT(canvas, page))
            layout.addWidget(canvas)
            self.tabs.addTab(page, name)
            self.figures[name] = figure, canvas
        note = QLabel("QMOM uses six moments and up to three quadrature nodes. "
                      "Its final size distribution is based on those nodes.")
        note.setWordWrap(True)
        self.status = QLabel("Choose a model, experiment and mesh, then run the comparison.")
        self.status.setWordWrap(True)
        self.run_button = QPushButton("Run comparison")
        self.run_button.clicked.connect(self.start)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.export_button = QPushButton("Export trajectories CSV")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_csv)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        for button in (self.run_button, self.stop_button, self.export_button, self.close_button):
            buttons.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        layout.addWidget(note)
        layout.addWidget(self.tabs, 1)
        self.controls = (self.model, self.experiment, self.lmax, self.cells, self.numerics, self.time_budget)
        for step in reversed(history(case)):
            if step["action"] != "pbm_qmom_comparison":
                continue
            record = step["result"]
            index = int(step["model_index"])
            if (not 0 <= index < self.model.count() or model_index not in (None, index)
                    or not 0 <= int(record["experiment_index"]) < self.experiment.count()):
                continue
            mesh = record["settings"]
            legacy_case = {**case, "pbmsetting": {"Lmin": 0., **settings, "Lmax": mesh["Lmax"], "N": mesh["N"]}}
            if step["signature"] in (signature(case, index, include_weights=False),
                                     signature(legacy_case, index, include_weights=False)):
                self.model.setCurrentIndex(index)
                self.experiment.setCurrentIndex(int(record["experiment_index"]))
                self.lmax.setValue(float(mesh["Lmax"]))
                self.cells.setValue(int(mesh["N"]))
                self.numerics.load_settings(mesh)
                self.time_budget.setValue(float(record.get("time_budget_s", 0.)))
                self.show_result(record, remember=False)
                break
        self.model.currentIndexChanged.connect(self.clear_result)
        self.experiment.currentIndexChanged.connect(self.clear_result)
        self.lmax.valueChanged.connect(self.clear_result)
        self.cells.valueChanged.connect(self.clear_result)
        self.numerics.currentIndexChanged.connect(self.clear_result)
        self.time_budget.valueChanged.connect(self.clear_result)

    def clear_result(self):
        self.result = None
        self.status.setText("Choose a model, experiment and mesh, then run the comparison.")
        self.export_button.setEnabled(False)
        for figure, canvas in self.figures.values():
            figure.clear()
            canvas.draw_idle()

    def start(self):
        if self.job is not None:
            return
        self.clear_result()
        self.budget.start(self.time_budget.value())
        settings = {"Lmin": 0., **self.case.get("pbmsetting", {}),
                    "Lmax": self.lmax.value(), "N": self.cells.value()}
        settings = self.budget.apply(self.numerics.apply_settings(settings))
        request_case = {**self.case, "pbmsetting": settings}
        self.job = ComparisonJob(request_case, self.model.currentIndex(),
                                 self.experiment.currentIndex(), self, budget=self.budget)
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
            self.status.setText("Stopping comparison…")

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
        if remember:
            result["experiment_index"] = self.experiment.currentIndex()
        self.result = result
        time = np.asarray(result["time"])
        methods = [("PBM", result["pbm"], "-")]
        if "qmom" in result:
            methods.append(("QMOM", result["qmom"], "--"))
        for figure, _ in self.figures.values():
            figure.clear()
        inventory, drift = self.figures["Material balance"][0].subplots(2, 1, sharex=True)
        for name, data, style in methods:
            dissolved = np.asarray(data["dissolved"])
            crystalline = np.asarray(data["crystal"]) + data["overflow"]
            inventory.plot(time, dissolved, style, color="tab:blue", label=f"{name} dissolved")
            inventory.plot(time, crystalline, style, color="tab:orange", label=f"{name} crystalline")
            total = dissolved + crystalline
            drift.plot(time, 1e6*(total-result["initial_api"])/max(result["initial_api"], 1e-30),
                       style, label=name)
        inventory.set_ylabel("API mass (g)")
        drift.set_ylabel("Balance error (ppm)")
        drift.set_xlabel("Time (s)")
        inventory.legend(ncols=2)
        drift.legend()
        axes = self.figures["Moments"][0].subplots(2, 3).ravel()
        for order, ax in enumerate(axes):
            for name, data, style in methods:
                ax.plot(time, np.asarray(data["moments"])[order], style, label=name)
            ax.set_title(f"M{order}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("particles" + (f" · µm^{order}" if order else ""))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axes[0].legend()
        axes = self.figures["D10–D50–D90"][0].subplots(3, 1, sharex=True)
        pbm_percentiles = np.asarray(result["pbm"]["percentiles"])
        for row, (ax, percentile) in enumerate(zip(axes, (10, 50, 90))):
            ax.plot(time, pbm_percentiles[row], label="PBM")
            ax.set_ylabel(f"D{percentile} (µm)")
        axes[0].legend()
        axes[-1].set_xlabel("Time (s)")
        axes = self.figures["Concentration and conditions"][0].subplots(3, 1, sharex=True)
        for name, data, style in methods:
            axes[0].plot(time, data["concentration"], style, label=f"{name} concentration")
        axes[0].plot(time, result["solubility"], ":", label="Solubility")
        axes[0].set_ylabel("Solute (g/g solution)")
        axes[0].legend()
        axes[1].plot(time, result["temperature"])
        axes[1].set_ylabel("Temperature (°C)")
        axes[2].plot(time, result["antisolvent"])
        axes[2].set_ylabel("AS in carrier (g/g)")
        axes[2].set_xlabel("Time (s)")
        ax = self.figures["Final volume distribution"][0].subplots()
        for name, data, _ in methods:
            ax.stairs(data["final_density"], result["bounds"], label=name, linewidth=1.5)
        ax.set_xlabel("Size (µm)")
        ax.set_ylabel("Volume density (%/µm)")
        ax.legend()
        for figure, canvas in self.figures.values():
            for ax in figure.axes:
                ax.grid(alpha=.2)
            canvas.draw_idle()
        pbm_label = "PBM (second solve)" if result.get("warm_timing") else "PBM"
        timing = (f" {pbm_label} {result['pbm_seconds']:.3f} s · QMOM {result['qmom_seconds']:.3f} s"
                  f" · total {result['total_seconds']:.3f} s") if "pbm_seconds" in result else ""
        bins_label = "log-spaced bins" if is_log_grid(result["settings"]) else "bins"
        self.status.setText(
            f"Model {self.model.currentIndex()+1} · Experiment {int(result['experiment_index'])+1} · "
            f"{result.get('backend', self.numerics.currentText())} · "
            f"Lmax {result['settings']['Lmax']:g} µm · {int(result['settings']['N'])} {bins_label} · "
            + (f"QMOM unavailable. {result['qmom_error']}" if "qmom_error" in result
               else "PBM and QMOM comparison complete.") + timing)
        self.export_button.setEnabled(True)
        if remember:
            record_step(self.case, "pbm_qmom_comparison", result, model_index=self.model.currentIndex(),
                         context=signature(self.case, self.model.currentIndex(), include_weights=False))

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export comparison", "pbm_qmom_comparison.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        data = self.result
        columns = {"time_s": np.asarray(data["time"]), "temperature_C": data["temperature"],
                   "antisolvent_fraction": data["antisolvent"], "solubility": data["solubility"]}
        for key in ("pbm_seconds", "qmom_seconds", "total_seconds"):
            if key in data:
                columns[key] = np.full(len(data["time"]), data[key])
        for method in ("pbm", "qmom"):
            if method not in data:
                continue
            traces = data[method]
            for key in ("concentration", "dissolved", "crystal", "overflow"):
                columns[f"{method}_{key}"] = traces[key]
            for order in range(6):
                columns[f"{method}_M{order}"] = np.asarray(traces["moments"])[order]
            for row, percentile in enumerate((10, 50, 90)):
                columns[f"{method}_D{percentile}"] = np.asarray(traces["percentiles"])[row]
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(columns)
                writer.writerows(zip(*columns.values()))
        except OSError as error:
            QMessageBox.warning(self, "Export comparison", str(error))
