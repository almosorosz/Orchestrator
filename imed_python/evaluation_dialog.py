"""Two Digital Design views for cross-experiment model evaluation."""
import csv
from threading import Event

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QVBoxLayout,
)

from .evaluation import evaluate_model
from .session import latest_step, record_step, signature


class EvaluationJob(QThread):
    message = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, case, model_index, parent):
        super().__init__(parent)
        self.case, self.model_index = case, model_index
        self.cancel_requested = Event()

    def run(self):
        try:
            result = evaluate_model(self.case, self.model_index,
                                    cancelled=self.cancel_requested.is_set, progress=self.message.emit)
        except InterruptedError:
            self.failed.emit("Model evaluation stopped.")
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.completed.emit(result)


class EvaluationDialog(QDialog):
    def __init__(self, case, model_index, parent=None, *, summary=False):
        super().__init__(parent)
        self.case, self.model_index, self.summary = case, model_index, summary
        self.job, self.result = None, None
        title = "Size summary" if summary else "Experiment comparison"
        self.setWindowTitle(f"IMED | {title} | Model {model_index+1}")
        self.resize(1150, 800)
        self.figure = Figure(figsize=(11, 6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)
        self.uncertainty = QComboBox()
        self.uncertainty.addItems(("No measurement error bars", "PAX reference error bars (6 experiments)",
                                   "RSV reference relative error bars"))
        self.uncertainty.setVisible(summary)
        self.uncertainty.currentIndexChanged.connect(self.redraw)
        self.run_button, self.stop_button = QPushButton("Simulate all experiments"), QPushButton("Stop")
        self.export_button, self.close_button = QPushButton("Export MAPE CSV"), QPushButton("Close")
        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.run_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.export_button.clicked.connect(self.export_csv)
        self.close_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        for button in (self.run_button, self.stop_button, self.export_button, self.close_button):
            buttons.addWidget(button)
        self.status = QLabel("Simulate all experiments to compare this model with the loaded measurements.")
        self.status.setWordWrap(True)
        note = QLabel("MAPE uses measured values as denominators and omits zero measurements. "
                      "Counts use separate measured and predicted axes. Count-shape MAPE uses peak-normalized traces.")
        note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.uncertainty)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(scroll, 1)
        layout.addWidget(note)
        layout.addWidget(self.status)
        layout.addLayout(buttons)
        step = latest_step(case, "model_evaluation", model_index=model_index,
                           context=signature(case, model_index, include_weights=False))
        if step is not None:
            self.show_result(step["result"], remember=False)

    def start(self):
        self.result = None
        self.figure.clear()
        self.canvas.draw_idle()
        self.job = EvaluationJob(self.case, self.model_index, self)
        for control in (self.run_button, self.close_button, self.export_button, self.uncertainty):
            control.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.job.message.connect(self.status.setText)
        self.job.failed.connect(self.status.setText)
        self.job.completed.connect(self.show_result)
        self.job.finished.connect(self.finish)
        self.job.start()

    def stop(self):
        if self.job is not None:
            self.job.cancel_requested.set()
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping model evaluation…")

    def finish(self):
        self.job.deleteLater()
        self.job = None
        for control in (self.run_button, self.close_button, self.uncertainty):
            control.setEnabled(True)
        self.export_button.setEnabled(self.result is not None)
        self.stop_button.setEnabled(False)

    def reject(self):
        if self.job is None:
            super().reject()

    def show_result(self, result, *, remember=True):
        if isinstance(result["records"], dict):
            result["records"] = [result["records"]]
        self.result = result
        self.redraw()
        records = result["records"]
        failed = sum(record["error"] != "Success" for record in records)
        self.status.setText(f"Model {self.model_index+1} · {len(records)-failed}/{len(records)} experiments successful · "
                            f"PBM compute {sum(record['seconds'] for record in records):.3f} s · "
                            f"{failed} failed")
        self.export_button.setEnabled(True)
        if remember:
            record_step(self.case, "model_evaluation", result, model_index=self.model_index,
                        context=signature(self.case, self.model_index, include_weights=False))

    def redraw(self):
        if self.result is None:
            return
        self.figure.clear()
        if self.summary:
            self.plot_summary()
        else:
            self.plot_comparison()
        for ax in self.figure.axes:
            ax.grid(alpha=.2)
        self.canvas.draw_idle()

    def plot_comparison(self):
        records = self.result["records"]
        self.canvas.setMinimumHeight(max(500, 280*len(records)))
        self.figure.set_size_inches(11, max(6, 2.8*len(records)))
        axes = self.figure.subplots(len(records), 3, squeeze=False)
        for row, record in enumerate(records):
            size, concentration, count = axes[row]
            if record["error"] != "Success":
                for ax in axes[row]:
                    ax.text(.5, .5, f"Experiment {int(record['experiment_index'])+1}\n{record['error']}",
                            transform=ax.transAxes, ha="center", va="center", wrap=True, fontsize=8)
                continue
            size.plot((10, 50, 90), record["observed_percentiles"], "r^-", mfc="none", label="Measured")
            size.plot((10, 50, 90), record["predicted_percentiles"], "bs--", mfc="none", label="Predicted")
            size.set_xticks((10, 50, 90), ("D10", "D50", "D90"))
            size.set_ylabel("Particle size (µm)")
            size.set_title("MAPE (%) · " + " / ".join(f"{value:.2f}" for value in record["size_errors"]), fontsize=10)
            size.legend(fontsize=8)
            for ax, measured_key, predicted_key in ((concentration, "observed_concentration", "concentration"),
                                                     (count, "observed_count", "count")):
                measured = np.asarray(record[measured_key])
                if measured.size:
                    measured = measured.reshape(2, -1)
                    ax.plot(measured[0]/3600, measured[1], "r-", label="Measured")
                predicted = np.asarray(record[predicted_key])
                target = ax.twinx() if predicted_key == "count" else ax
                target.plot(predicted[0]/3600, predicted[1], "b--", label="Predicted")
                ax.set_xlabel("Process time (h)")
                if predicted_key == "count":
                    target.set_ylabel("Predicted number density (#/mL)", color="blue")
                    target.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            concentration.set_ylabel("Concentration (g/g)")
            concentration.set_title(f"Experiment {int(record['experiment_index'])+1} · MAPE {record['concentration_mape']:.2f}%", fontsize=10)
            count.set_ylabel("Measured counts", color="red")
            count.set_title(f"Count-shape MAPE {record['count_shape_mape']:.2f}%", fontsize=10)

    def plot_summary(self):
        records = [record for record in self.result["records"] if record["error"] == "Success"]
        x = np.array([int(record["experiment_index"])+1 for record in records])
        measured = np.array([record["observed_percentiles"] for record in records])
        predicted = np.array([record["predicted_percentiles"] for record in records])
        errors = np.array([record["size_errors"] for record in records])
        ax = self.figure.subplots()
        for i, (percentile, style) in enumerate(zip((10, 50, 90), (":", "-", "--"))):
            ax.plot(x, measured[:, i], color="red", marker="s", ms=4, ls=style, label=f"Measured D{percentile}")
            ax.plot(x, predicted[:, i], color="blue", marker="x", ms=6, ls=style, label=f"Predicted D{percentile}")
        error_bars = None
        if self.uncertainty.currentIndex() == 1:
            if len(self.case["ExpData"]) == 6:
                preset = np.array([[.3449412,1.0939812,.07893697,.2569032,.01555587,3],
                                   [1.42499676,2.35920132,.13057254,.541832,.3440556,4],
                                   [2.50321686,4.0263467,3.49542015,1.51850412,4.81156804,9]]).T
                error_bars = preset[x-1]
            else:
                ax.text(.01, .02, "PAX reference error bars require six experiments.", transform=ax.transAxes)
        elif self.uncertainty.currentIndex() == 2:
            error_bars = np.abs(measured)*[.067425679, .074012011, .103615207]
        if error_bars is not None:
            for i in range(3):
                ax.errorbar(x, measured[:, i], yerr=error_bars[:, i], fmt="none", ecolor="red", capsize=3)
        summaries = []
        for i, percentile in enumerate((10, 50, 90)):
            values = errors[:, i][np.isfinite(errors[:, i])]
            summaries.append(f"D{percentile} {values.mean():.2f} ± {values.std():.2f}%" if len(values) else f"D{percentile} unavailable")
        ax.set_title("MAPE across experiments (mean ± SD)\n" + " · ".join(summaries))
        ax.set_xticks(x)
        ax.set_xlabel("Experiment number")
        ax.set_ylabel("Particle size (µm)")
        ax.legend(ncols=3, fontsize=9)

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export model evaluation", "model_evaluation.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("experiment", "measured_D10", "measured_D50", "measured_D90", "predicted_D10", "predicted_D50", "predicted_D90",
                                 "MAPE_D10", "MAPE_D50", "MAPE_D90", "concentration_MAPE", "count_shape_MAPE", "PBM_seconds", "status"))
                for record in self.result["records"]:
                    missing = [np.nan]*3
                    writer.writerow((int(record["experiment_index"])+1, *record.get("observed_percentiles", missing),
                                     *record.get("predicted_percentiles", missing), *record.get("size_errors", missing),
                                     record.get("concentration_mape", np.nan), record.get("count_shape_mape", np.nan),
                                     record["seconds"], record["error"]))
        except OSError as error:
            QMessageBox.warning(self, "Export model evaluation", str(error))
