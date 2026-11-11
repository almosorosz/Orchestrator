"""Monte Carlo reliability and probabilistic sensitivity windows."""
import csv
from math import ceil
from threading import Event

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from .control_panel import fill_table, table
from .probabilistic import (
    CONDITIONS, ERROR_LEVELS, assess_reliability, error_limits, model_intervals,
    partial_rank_correlations, prediction_errors, prcc_diagnostics, propagate, sample_intervals,
)
from .session import history, record_step, signature
from .simulated_profile import editable_groups, remember_active_doe
from .simulated_profile_dialog import SimulatedProfileDialog


class ProbabilisticJob(QThread):
    message = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, case, model_index, experiment_index, request, parent):
        super().__init__(parent)
        self.case, self.model_index, self.experiment_index = case, model_index, experiment_index
        self.request = request
        self.cancel_requested = Event()

    def run(self):
        try:
            result = propagate(self.case, self.model_index, self.experiment_index,
                               **self.request, cancelled=self.cancel_requested.is_set,
                               progress=self.message.emit)
        except InterruptedError:
            self.failed.emit("Probabilistic analysis stopped.")
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.completed.emit(result)


class ProbabilisticDialog(QDialog):
    def __init__(self, case, parent=None, *, sensitivity=False, workers=1):
        super().__init__(parent)
        self.sensitivity, self.workers = sensitivity, workers
        self.action = "probabilistic_sensitivity" if sensitivity else "model_reliability"
        title = "Probabilistic sensitivity analysis" if sensitivity else "Model reliability"
        self.setWindowTitle(f"IMED | {title}")
        if sensitivity:
            self.setSizeGripEnabled(True)
            available = self.screen().availableGeometry()
            height = min(580, available.height() - 48)
            if parent is not None and parent.window().isVisible():
                height = min(height, parent.window().height() - 24)
            self.resize(min(1100, available.width() - 24), height)
        else:
            self.resize(1100, 850)
        self.case, self.job, self.result = case, None, None
        self.model, self.experiment = QComboBox(), QComboBox()
        self.model.addItems(f"Model {i+1}" for i in range(len(case["ModelParameter"])))
        self.experiment.addItems(f"Experiment {i+1}" for i in range(len(case["ExpData"])))
        self.samples, self.seed = QSpinBox(), QSpinBox()
        self.samples.setRange(2, 10000)
        self.samples.setValue(100)
        self.seed.setRange(0, 2147483647)
        self.use_parameters = QCheckBox("Sample kinetic parameters jointly from their CI covariance")
        self.use_parameters.setChecked(not sensitivity)
        self.use_parameters.setEnabled(sensitivity)
        form = QFormLayout()
        if sensitivity:
            self.profile_drafts = {}
            self.profile_button = QPushButton("Define profile…")
            self.recorded_button = QPushButton("Use recorded profile")
            self.profile_status = QLabel()
            self.profile_status.setWordWrap(True)
            self.profile_button.clicked.connect(self.edit_profile)
            self.recorded_button.clicked.connect(self.use_recorded_profile)
            selectors = QHBoxLayout()
            selectors.addWidget(self.model, 1)
            selectors.addWidget(QLabel("Experiment"))
            selectors.addWidget(self.experiment, 1)
            form.addRow("Model", selectors)
            profile_row = QHBoxLayout()
            profile_row.addWidget(self.profile_button)
            profile_row.addWidget(self.recorded_button)
            profile_row.addWidget(self.profile_status, 1)
            form.addRow("Profile", profile_row)
        else:
            form.addRow("Model", self.model)
            form.addRow("Experiment", self.experiment)
        sampling = QHBoxLayout()
        sampling.addWidget(QLabel("Draws"))
        sampling.addWidget(self.samples)
        sampling.addWidget(QLabel("Random seed"))
        sampling.addWidget(self.seed)
        form.addRow("Sampling", sampling)
        form.addRow(self.use_parameters)
        self.error_threshold = QDoubleSpinBox()
        self.error_threshold.setRange(0, 1000000)
        self.error_threshold.setDecimals(3)
        self.error_threshold.setValue(5)
        self.error_threshold.setSuffix(" %")
        self.reliability_scope = QComboBox()
        self.reliability_scope.addItem("All samples", "all_times")
        self.reliability_scope.addItem("Final samples", "final")
        self.reliability_scope.setCurrentIndex(self.reliability_scope.findData("final"))
        self.reliability_scope.setToolTip("All samples uses every measured size time point. "
                                         "Final samples uses the last measured value of each size quantile.")
        assessment_controls = QWidget()
        assessment_row = QHBoxLayout(assessment_controls)
        assessment_row.setContentsMargins(0, 0, 0, 0)
        assessment_row.addWidget(QLabel("Allowed absolute relative error"))
        assessment_row.addWidget(self.error_threshold)
        assessment_row.addWidget(QLabel("Assess"))
        assessment_row.addWidget(self.reliability_scope, 1)
        assessment_controls.setVisible(not sensitivity)
        form.addRow(assessment_controls)
        self.plot_bounds = {"absolute": [np.nan, np.nan], "relative": [np.nan, np.nan]}
        self.limit_inputs = {}
        self.condition_controls = {}
        condition_group = QGroupBox("Operating conditions · independent uniform ranges")
        grid = QGridLayout(condition_group)
        for offset in (0, 4):
            for column, text in enumerate(("Vary", "Condition", "Lower", "Upper")):
                grid.addWidget(QLabel(text), 0, offset+column)
        for index, (key, label, low, high) in enumerate(CONDITIONS):
            row, offset = index % 3 + 1, 4*(index // 3)
            enabled, lower, upper = QCheckBox(), QDoubleSpinBox(), QDoubleSpinBox()
            enabled.setChecked(key == "temperature_offset")
            for control in (lower, upper):
                control.setDecimals(4)
                control.setRange(-200 if key == "temperature_offset" else .0001, 1000)
            lower.setValue(low)
            upper.setValue(high)
            self.condition_controls[key] = enabled, lower, upper
            for column, widget in enumerate((enabled, QLabel(label), lower, upper)):
                grid.addWidget(widget, row, offset+column)
        condition_group.setVisible(sensitivity)
        self.covariance_status = QLabel()
        self.covariance_status.setWordWrap(True)
        note = QLabel(
            "Joint normal draws use the displayed parameter scale, including log-rate parameters. "
            "Failed simulations are counted and retained with their draw values. "
            + ("PRCC measures monotonic association while controlling the other sampled inputs. "
               "Duration changes are aligned by profile progress in the evolution plot."
               if sensitivity else
               "Relative error = 100 × (predicted − measured) / predicted. "
               "Dashed limits use ± prctile(abs(error), [95, 99])."))
        note.setWordWrap(True)
        note.setVisible(not sensitivity)
        self.use_parameters.setToolTip(note.text())
        self.tabs, self.figures = QTabWidget(), {}
        self.reliability_label = QLabel("Run the analysis to calculate reliability.")
        self.reliability_label.setWordWrap(True)
        self.reliability_table = table(("Response", "Comparisons within limit", "Passing / total comparisons",
                                        "Draws passing every point", "Passing / total draws"))
        reliability_page = QWidget(self)
        reliability_page.setVisible(not sensitivity)
        assessment_layout = QVBoxLayout(reliability_page)
        assessment_layout.addWidget(self.reliability_label)
        assessment_layout.addWidget(self.reliability_table)
        explanation = QLabel(
            "Comparison reliability = 100 × fraction with |100 × (predicted − measured) / predicted| "
            "within the allowed error. Joint model reliability is the fraction of draws for which "
            "D10, D50 and D90 all pass at every selected measured point. Failed simulations and "
            "unavailable predicted sizes count as failures. The 95% sampling interval uses the "
            "number of independent parameter draws.")
        explanation.setWordWrap(True)
        assessment_layout.addWidget(explanation)
        self.reliability_export = QPushButton("Export reliability CSV")
        self.reliability_export.setEnabled(False)
        self.reliability_export.clicked.connect(self.export_reliability)
        assessment_layout.addWidget(self.reliability_export)
        if not sensitivity:
            self.tabs.addTab(reliability_page, "Reliability")
        names = (["Sensitivity (PRCC)", "Prediction evolution", "FBRM prediction", "Final prediction distribution", "Input draws"]
                 if sensitivity else ["Absolute error", "Relative error", "Prediction evolution", "FBRM prediction",
                                      "Final prediction distribution", "Parameter draws"])
        for name in names:
            figure = Figure(figsize=(10.5, 5.5), layout="constrained")
            canvas = FigureCanvasQTAgg(figure)
            page, scroll = QWidget(), QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(canvas)
            layout = QVBoxLayout(page)
            if name == "Sensitivity (PRCC)":
                description = QLabel("PRCC ranges from −1 to +1. Positive values mean larger inputs tend to "
                                     "produce larger predictions after accounting for other inputs. Negative "
                                     "values mean the reverse. Unavailable cells are labelled with a reason.")
                description.setWordWrap(True)
                self.prcc_status = QLabel()
                self.prcc_status.setWordWrap(True)
                layout.addWidget(description)
                layout.addWidget(self.prcc_status)
            if name in ("Absolute error", "Relative error"):
                kind = "relative" if name == "Relative error" else "absolute"
                unit = "%" if kind == "relative" else "µm"
                lower, upper = QLineEdit(), QLineEdit()
                for control in (lower, upper):
                    control.setPlaceholderText("Auto")
                    control.setMaximumWidth(145)
                self.limit_inputs[kind] = lower, upper
                bounds_row = QHBoxLayout()
                bounds_row.addWidget(QLabel(f"X lower ({unit})"))
                bounds_row.addWidget(lower)
                bounds_row.addWidget(QLabel(f"X upper ({unit})"))
                bounds_row.addWidget(upper)
                apply_button, auto_button = QPushButton("Apply x limits"), QPushButton("Auto")
                apply_button.clicked.connect(lambda _checked=False, key=kind: self.apply_plot_limits(key))
                auto_button.clicked.connect(lambda _checked=False, key=kind: self.reset_plot_limits(key))
                bounds_row.addWidget(apply_button)
                bounds_row.addWidget(auto_button)
                bounds_row.addStretch()
                layout.addLayout(bounds_row)
            layout.addWidget(NavigationToolbar2QT(canvas, page))
            layout.addWidget(scroll)
            self.tabs.addTab(page, name)
            self.figures[name] = figure, canvas
        self.status = QLabel("Choose a model and experiment, then run the analysis.")
        self.status.setWordWrap(True)
        self.run_button, self.stop_button = QPushButton("Run analysis"), QPushButton("Stop")
        self.export_button, self.close_button = QPushButton("Export samples CSV"), QPushButton("Close")
        self.intervals_export = QPushButton("Export model intervals CSV")
        self.intervals_export.setEnabled(False)
        self.intervals_export.clicked.connect(self.export_intervals)
        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.run_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.export_button.clicked.connect(self.export_csv)
        self.close_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        for button in (self.run_button, self.stop_button, self.export_button, self.intervals_export, self.close_button):
            buttons.addWidget(button)
        layout = QVBoxLayout(self)
        if sensitivity:
            body = QWidget()
            content = QVBoxLayout(body)
            content.setContentsMargins(0, 0, 0, 0)
            scroll = QScrollArea()
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidgetResizable(True)
            scroll.setWidget(body)
            self.tabs.setMinimumHeight(360)
            layout.addLayout(buttons)
            layout.addWidget(self.status)
            layout.addWidget(scroll, 1)
        else:
            content = layout
        content.addLayout(form)
        content.addWidget(condition_group)
        content.addWidget(self.covariance_status)
        content.addWidget(note)
        content.addWidget(self.tabs, 1)
        if not sensitivity:
            layout.addWidget(self.status)
            layout.addLayout(buttons)
        self.controls = [self.model, self.experiment, self.samples, self.seed]
        if sensitivity:
            self.controls += [self.use_parameters, *[widget for controls in self.condition_controls.values() for widget in controls]]
        for step in reversed(history(case)):
            if (step["action"] == self.action
                    and step["signature"] == signature(case, int(step["model_index"]), include_weights=False)):
                self.restore(step["result"])
                break
        self.update_covariance_status()
        self.model.currentIndexChanged.connect(self.update_covariance_status)
        if sensitivity:
            self.update_profile_status()
            self.model.currentIndexChanged.connect(self.update_profile_status)
            self.experiment.currentIndexChanged.connect(self.update_profile_status)
        for control in self.controls:
            signal = (control.currentIndexChanged if isinstance(control, QComboBox)
                      else control.toggled if isinstance(control, QCheckBox) else control.valueChanged)
            signal.connect(self.clear_result)
        self.error_threshold.valueChanged.connect(self.update_reliability)
        self.reliability_scope.currentIndexChanged.connect(self.update_reliability)

    def update_covariance_status(self):
        try:
            intervals = model_intervals(self.case, self.model.currentIndex())
        except ValueError as error:
            self.covariance_status.setText(str(error))
        else:
            self.covariance_status.setText(f"CI covariance available for {len(intervals.indices)} continuous parameters · "
                                            f"{intervals.source} · {100*intervals.confidence:g}% CIs")

    def profile_selection(self):
        return self.model.currentIndex(), self.experiment.currentIndex()

    def update_profile_status(self):
        model, experiment = self.profile_selection()
        recipe = self.profile_drafts.get((model, experiment))
        groups = editable_groups(self.case, model, recipe)
        self.profile_button.setEnabled(bool(groups))
        self.profile_button.setToolTip("" if groups else "No editable DoE factors were found for this case and model.")
        self.recorded_button.setEnabled(bool(recipe))
        self.profile_status.setText(recipe["name"] if recipe else "Recorded experiment")
        stir_enabled = "stirring" in groups
        for widget in self.condition_controls["stirring_factor"]:
            widget.setEnabled(stir_enabled)
            widget.setToolTip("" if stir_enabled else "Requires a DoE stirring factor and a power-dependent kinetic term.")

    def edit_profile(self):
        model, experiment = self.profile_selection()
        key = model, experiment
        dialog = SimulatedProfileDialog(self.case, model, experiment, self, previous=self.profile_drafts.get(key))
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.profile_drafts[key] = dialog.result_profile
            self.update_profile_status()
            self.clear_result()
        dialog.deleteLater()

    def use_recorded_profile(self):
        self.profile_drafts.pop(self.profile_selection(), None)
        self.update_profile_status()
        self.clear_result()

    def clear_result(self):
        self.result = None
        self.export_button.setEnabled(False)
        self.intervals_export.setEnabled(False)
        self.reliability_export.setEnabled(False)
        self.reliability_label.setText("Run the analysis to calculate reliability.")
        self.reliability_table.setRowCount(0)
        if self.sensitivity:
            self.prcc_status.clear()
        self.status.setText("Settings changed. Run the analysis to refresh the plots.")
        for figure, canvas in self.figures.values():
            figure.clear()
            canvas.draw_idle()

    def restore(self, result):
        request = result["request"]
        self.model.setCurrentIndex(int(result["model_index"]))
        self.experiment.setCurrentIndex(int(result["experiment_index"]))
        if self.sensitivity and request.get("simulated_profile"):
            self.profile_drafts[self.profile_selection()] = request["simulated_profile"]
        self.samples.setValue(int(request["samples"]))
        self.seed.setValue(int(request["seed"]))
        self.use_parameters.setChecked(bool(request["use_parameters"]))
        assessment = result.get("reliability", {})
        self.error_threshold.setValue(float(assessment.get("threshold_percent", 5)))
        self.reliability_scope.setCurrentIndex(self.reliability_scope.findData(assessment.get("scope", "final")))
        for kind, controls in self.limit_inputs.items():
            bounds = np.asarray(result.get("error_plot_limits", {}).get(kind, [np.nan, np.nan])).reshape(-1)
            self.plot_bounds[kind] = bounds.tolist()
            for control, value in zip(controls, bounds):
                control.setText(f"{value:.9g}" if np.isfinite(value) else "")
        conditions = request.get("conditions", {})
        for key, (enabled, lower, upper) in self.condition_controls.items():
            enabled.setChecked(key in conditions)
            if key in conditions:
                lower.setValue(float(conditions[key][0]))
                upper.setValue(float(conditions[key][1]))
        self.show_result(result, remember=False)

    def start(self):
        conditions = {}
        if self.sensitivity:
            conditions = {key: [lower.value(), upper.value()]
                          for key, (enabled, lower, upper) in self.condition_controls.items()
                          if enabled.isEnabled() and enabled.isChecked()}
        request = {"samples": self.samples.value(), "seed": self.seed.value(),
                   "use_parameters": self.use_parameters.isChecked(), "conditions": conditions,
                   "workers": self.workers}
        if self.sensitivity:
            request["simulated_profile"] = self.profile_drafts.get(self.profile_selection())
            self.profile_button.setEnabled(False)
            self.recorded_button.setEnabled(False)
        self.clear_result()
        self.job = ProbabilisticJob(self.case, self.model.currentIndex(), self.experiment.currentIndex(), request, self)
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
            self.job.cancel_requested.set()
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping analysis…")

    def finish(self):
        self.job.deleteLater()
        self.job = None
        for control in (*self.controls, self.run_button, self.close_button):
            control.setEnabled(True)
        if self.sensitivity:
            self.update_profile_status()
        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(self.result is not None)
        self.intervals_export.setEnabled(self.result is not None)

    def reject(self):
        if self.job is None:
            super().reject()

    def apply_plot_limits(self, kind):
        lower, upper = (control.text().strip() for control in self.limit_inputs[kind])
        try:
            bounds = [float(lower), float(upper)] if lower or upper else [np.nan, np.nan]
            if (lower or upper) and (not np.isfinite(bounds).all() or bounds[0] >= bounds[1]):
                raise ValueError
        except ValueError:
            self.status.setText("Enter finite lower and upper x limits with lower < upper. Scientific notation is accepted.")
            return
        self.plot_bounds[kind] = bounds
        self.update_reliability()

    def reset_plot_limits(self, kind):
        for control in self.limit_inputs[kind]:
            control.clear()
        self.plot_bounds[kind] = [np.nan, np.nan]
        self.update_reliability()

    def update_reliability(self):
        if self.result is None or self.sensitivity:
            return
        self.error_samples = prediction_errors(self.result, scope=self.reliability_scope.currentData())
        assessment = assess_reliability(self.error_samples, self.error_threshold.value())
        assessment["scope"] = self.reliability_scope.currentData()
        self.result["reliability"] = assessment
        self.result["error_plot_limits"] = {key: list(bounds) for key, bounds in self.plot_bounds.items()}
        rows = []
        for name, response in assessment["responses"].items():
            available = response["observation_count"] > 0
            rows.append((name,
                         f"{response['comparison_reliability']:.2f}%" if available else "No measured sizes",
                         f"{response['comparison_successes']}/{response['comparison_count']}",
                         f"{response['draw_reliability']:.2f}%" if available else "—",
                         f"{response['passing_draw_count']}/{assessment['draw_count']}" if available else "—"))
        fill_table(self.reliability_table, rows)
        if assessment["joint_available"]:
            lower, upper = assessment["joint_sampling_interval"]
            text = (f"Joint model reliability {assessment['joint_reliability']:.2f}% within ±{self.error_threshold.value():g}% error · "
                    f"{assessment['joint_passing_draw_count']}/{assessment['draw_count']} draws pass all three size quantiles. "
                    f"95% sampling interval {lower:.2f}–{upper:.2f}%.")
        else:
            text = "Joint reliability requires measured values for D10, D50 and D90. Available responses are assessed below."
        self.reliability_label.setText(text)
        self.reliability_export.setEnabled(True)
        self.status.setText(
            f"Model {int(self.result['model_index'])+1} · Experiment {int(self.result['experiment_index'])+1} · "
            f"{len(self.result['predictions'])}/{assessment['draw_count']} successful draws · "
            f"{self.reliability_scope.currentText()}. " + text)
        self.plot_errors(self.result, relative=False)
        self.plot_errors(self.result, relative=True)
        for name in ("Absolute error", "Relative error"):
            self.figures[name][1].draw_idle()

    def plot_errors(self, result, *, relative):
        name = "Relative error" if relative else "Absolute error"
        figure = self.figures[name][0]
        figure.clear()
        axes = figure.subplots(1, 3)
        kind = "relative" if relative else "absolute"
        for i, ax in enumerate(axes):
            key = f"D{(10, 50, 90)[i]}"
            data = self.error_samples[key]
            errors, baseline_error = data[kind], data["nominal_"+kind]
            values = errors[np.isfinite(errors)]
            unit = "%" if relative else "µm"
            samples = len(data["time"])
            sample_unit = "sample" if samples == 1 else "samples"
            ax.set_title(f"{key} · {name.lower()}\n{samples} measured {sample_unit} · {errors.shape[0]} draws\n"
                         f"{len(values)} finite error comparisons", fontsize=10)
            ax.set_xlabel(f"Error ({unit})")
            ax.set_ylabel("Probability density")
            if not len(data["time"]):
                ax.text(.5, .5, "No measured percentiles", transform=ax.transAxes, ha="center")
                continue
            if not values.size:
                ax.text(.5, .5, "No finite predicted sizes", transform=ax.transAxes, ha="center")
                continue
            bins = min(50, max(1, ceil(np.sqrt(len(values)))))
            ax.hist(values, bins=bins, density=True, color="royalblue", alpha=.5, edgecolor="black", linewidth=.1)
            if np.isfinite(baseline_error[-1]):
                ax.axvline(baseline_error[-1], ls="-.", color="black", label="Nominal final error")
            for level, limit, color in zip(ERROR_LEVELS, error_limits(values), ("red", "blue")):
                ax.axvline(-limit, ls="--", color=color, alpha=.6)
                ax.axvline(limit, ls="--", color=color, alpha=.6,
                           label=f"{level:g}% · ±{limit:.3g} {unit}")
            if relative:
                limits = ax.get_xlim()
                tolerance = self.error_threshold.value()
                ax.axvspan(-tolerance, tolerance, color="green", alpha=.08, label=f"Allowed ±{tolerance:g}%")
                ax.axvline(-tolerance, ls=":", color="green")
                ax.axvline(tolerance, ls=":", color="green")
                ax.set_xlim(limits)
            ax.legend(fontsize=8)
        bounds = self.plot_bounds[kind]
        for ax in axes:
            if np.isfinite(bounds).all():
                ax.set_xlim(*bounds)
            ax.grid(alpha=.2)

    def show_result(self, result, *, remember=True):
        self.result = result
        for figure, _ in self.figures.values():
            figure.clear()
        predictions = np.asarray(result["predictions"], dtype=float)
        concentrations = np.asarray(result["concentrations"], dtype=float)
        # A single successful draw loses its singleton dimension in a MAT file.
        predictions = predictions.reshape(-1, 3, len(result["time"]))
        concentrations = concentrations.reshape(-1, len(result["time"]))
        result["predictions"], result["concentrations"] = predictions, concentrations
        count = len(np.asarray(result["valid"]).reshape(-1))
        inputs = np.asarray(result["inputs"], dtype=float).reshape(count, -1)
        result["inputs"] = inputs
        names = np.asarray(result["input_names"]).reshape(-1).astype(str).tolist()
        names = [name.rstrip() for name in names]
        if self.sensitivity:
            ax = self.figures["Sensitivity (PRCC)"][0].subplots()
            outputs = np.column_stack((predictions[:, :, -1], concentrations[:, -1]))
            if "counts" in result:
                counts = np.asarray(result["counts"]).reshape(-1, len(result["time"]))
                calibration = result.get("fbrm", {})
                count_scale = float(calibration["scale"]) if calibration.get("scaled", False) else 1.
                outputs = np.column_stack((outputs, counts[:, -1]*count_scale))
            sampled_inputs = inputs[np.asarray(result["valid"], dtype=bool).reshape(-1)]
            prcc = partial_rank_correlations(sampled_inputs, outputs)
            diagnostics = prcc_diagnostics(sampled_inputs, outputs)
            result["prcc"], result["prcc_diagnostics"] = prcc, diagnostics
            image = ax.imshow(np.ma.masked_invalid(prcc), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            labels = ("Final D10", "Final D50", "Final D90", "Final concentration", "Final FBRM / count")
            ax.set_xticks(range(prcc.shape[1]), labels[:prcc.shape[1]],
                          rotation=20, ha="right")
            ax.set_yticks(range(len(names)), names)
            ax.set_title("Partial rank correlation", fontsize=11)
            self.figures["Sensitivity (PRCC)"][0].colorbar(image, ax=ax, label="PRCC")
            messages = []
            for column in range(prcc.shape[1]):
                if diagnostics["draw_counts"][column] < diagnostics["minimum_draws"]:
                    messages.append(f"{labels[column]} has {diagnostics['draw_counts'][column]} usable draws; "
                                    f"at least {diagnostics['minimum_draws']} are needed.")
                elif diagnostics["output_spans"][column] == 0:
                    messages.append(f"{labels[column]} does not vary across the sampled draws.")
            if np.any(diagnostics["input_spans"] == 0):
                messages.append("Some sampled inputs have identical values across all draws.")
            if not np.isfinite(prcc).any() and not messages:
                messages.append("The sampled inputs do not provide independently identifiable rank effects. "
                                "Check for perfectly correlated parameter draws.")
            self.prcc_status.setText(" ".join(messages) if messages else
                                     f"{len(sampled_inputs)} successful draws. Each output uses its own finite predictions.")
            for row, column in np.argwhere(~np.isfinite(prcc)):
                reason = ("Fixed input" if diagnostics["input_spans"][row] == 0 else
                          "Too few draws" if diagnostics["draw_counts"][column] < diagnostics["minimum_draws"] else
                          "No variation" if diagnostics["output_spans"][column] == 0 else "Not identifiable")
                ax.text(column, row, reason, ha="center", va="center", fontsize=7)
        else:
            self.update_reliability()
        axes = self.figures["Prediction evolution"][0].subplots(2, 2).ravel()
        x = 100*np.asarray(result["time"])/result["time"][-1] if self.sensitivity else result["time"]
        histories = [predictions[:, i] for i in range(3)] + [concentrations]
        nominal = [np.asarray(result["nominal"]["percentiles"])[i] for i in range(3)] + [result["nominal"]["concentration"]]
        result["prediction_intervals"] = {}
        for i, (ax, traces, center) in enumerate(zip(axes, histories, nominal)):
            bands = sample_intervals(traces)
            key = ("D10", "D50", "D90", "concentration")[i]
            result["prediction_intervals"][key] = bands
            ax.fill_between(x, bands["lower99"], bands["upper99"], alpha=.12, label="99% model uncertainty")
            ax.fill_between(x, bands["lower95"], bands["upper95"], alpha=.25, label="95% model uncertainty")
            ax.plot(x, bands["median"], label="Sample median")
            ax.plot(x, center, "k--", label="Nominal")
            ax.set_ylabel(f"D{(10,50,90)[i]} (µm)" if i < 3 else "Concentration (g/g)")
            ax.set_xlabel("Profile progress (%)" if self.sensitivity else "Time (s)")
        axes[0].legend(fontsize=8)
        self.plot_fbrm(result)
        axes = self.figures["Final prediction distribution"][0].subplots(2, 2).ravel()
        for i, (ax, traces, center) in enumerate(zip(axes, histories, nominal)):
            finite = traces[:, -1][np.isfinite(traces[:, -1])]
            if finite.size:
                ax.hist(finite, bins=50, density=True, alpha=.6)
            ax.axvline(center[-1], color="black", ls="-.", label="Nominal")
            ax.set_xlabel(f"Final D{(10,50,90)[i]} (µm)" if i < 3 else "Final concentration (g/g)")
            ax.set_ylabel("Probability density")
        axes[0].legend()
        name = "Input draws" if self.sensitivity else "Parameter draws"
        rows = ceil(len(names)/3)
        figure, canvas = self.figures[name]
        canvas.setMinimumHeight(max(420, rows*180))
        axes = figure.subplots(rows, min(len(names), 3), squeeze=False).ravel()
        for i, label in enumerate(names):
            axes[i].hist(inputs[:, i], bins=50, density=True, alpha=.6)
            if i < np.size(result["parameter_mean"]):
                axes[i].axvline(np.asarray(result["parameter_mean"]).reshape(-1)[i], color="black", ls="-.")
            axes[i].set_title(label)
            axes[i].set_ylabel("Probability density")
        for ax in axes[len(names):]:
            ax.set_visible(False)
        for figure, canvas in self.figures.values():
            for ax in figure.axes:
                ax.grid(alpha=.2)
            canvas.draw_idle()
        failed = count-len(predictions)
        if self.sensitivity:
            template = f"Experiment {int(result['experiment_index'])+1}"
            recipe = result["request"].get("simulated_profile")
            label = f"{recipe['name']} · Template {template}" if recipe else template
            self.status.setText(f"Model {int(result['model_index'])+1} · {label} · "
                                f"{len(predictions)}/{count} successful draws · {failed} failed.")
        self.export_button.setEnabled(True)
        self.intervals_export.setEnabled(True)
        if remember:
            record_step(self.case, self.action, result, model_index=int(result["model_index"]),
                        context=signature(self.case, int(result["model_index"]), include_weights=False))
            request = result["request"]
            if request.get("simulated_profile") or "stirring_factor" in request.get("conditions", {}):
                remember_active_doe(self.case)

    def plot_fbrm(self, result):
        ax = self.figures["FBRM prediction"][0].subplots()
        if "counts" not in result or "counts" not in result["nominal"]:
            ax.text(.5, .5, "Run the analysis to generate FBRM model uncertainty intervals.",
                    transform=ax.transAxes, ha="center", va="center")
            return
        time = np.asarray(result["time"])
        counts = np.asarray(result["counts"]).reshape(-1, len(time))
        calibration = result.get("fbrm", {})
        scaled = bool(calibration.get("scaled", False))
        scale = float(calibration["scale"]) if scaled else 1.
        bands = sample_intervals(counts*scale)
        result["prediction_intervals"]["FBRM" if scaled else "particle_number_density"] = bands
        x = 100*time/time[-1] if self.sensitivity else time
        ax.fill_between(x, bands["lower99"], bands["upper99"], alpha=.12, label="99% model uncertainty")
        ax.fill_between(x, bands["lower95"], bands["upper95"], alpha=.25, label="95% model uncertainty")
        ax.plot(x, bands["median"], label="Sample median")
        ax.plot(x, np.asarray(result["nominal"]["counts"])*scale, "k--", label="Nominal")
        if scaled:
            observed = np.asarray(calibration["observed"]).reshape(2, -1)
            coordinates = 100*observed[0]/time[-1] if self.sensitivity else observed[0]
            ax.plot(coordinates, observed[1], "r.", label="Measured FBRM")
            ax.set_ylabel("FBRM counts")
            ax.set_title(f"FBRM prediction · fixed proportional scale {scale:.5g}\n"
                         "Scale fitted to this experiment · intervals propagate parameter uncertainty", fontsize=10)
        else:
            ax.set_ylabel("PBM particle number density (#/mL)")
            ax.set_title("Count prediction and model uncertainty\nFBRM scaling requires usable measured counts", fontsize=10)
        ax.set_xlabel("Profile progress (%)" if self.sensitivity else "Time (s)")
        ax.legend(fontsize=9)

    def export_reliability(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export model reliability", "model_reliability.csv", "CSV (*.csv)")
        if not path:
            return
        assessment = self.result["reliability"]
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("response", "threshold_percent", "scope", "passing_comparisons", "total_comparisons",
                                 "comparison_reliability_percent", "passing_draws", "total_draws", "draw_reliability_percent",
                                 "joint_sampling_lower95", "joint_sampling_upper95"))
                for name, row in assessment["responses"].items():
                    writer.writerow((name, assessment["threshold_percent"], assessment["scope"], row["comparison_successes"],
                                     row["comparison_count"], row["comparison_reliability"], row["passing_draw_count"],
                                     assessment["draw_count"], row["draw_reliability"], "", ""))
                writer.writerow(("Joint D10 D50 D90", assessment["threshold_percent"], assessment["scope"], "", "", "",
                                 assessment["joint_passing_draw_count"], assessment["draw_count"], assessment["joint_reliability"],
                                 *assessment["joint_sampling_interval"]))
        except OSError as error:
            QMessageBox.warning(self, "Export reliability", str(error))

    def export_intervals(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export model uncertainty intervals", "model_prediction_intervals.csv", "CSV (*.csv)")
        if not path:
            return
        columns = {"time_s": self.result["time"]}
        for name, bands in self.result["prediction_intervals"].items():
            for bound, values in bands.items():
                columns[f"{name}_{bound}"] = values
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(columns)
                writer.writerows(zip(*columns.values()))
        except OSError as error:
            QMessageBox.warning(self, "Export model intervals", str(error))

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export probabilistic samples", f"{self.action}.csv", "CSV (*.csv)")
        if not path:
            return
        result = self.result
        names = np.asarray(result["input_names"]).reshape(-1).astype(str).tolist()
        valid = np.asarray(result["valid"], dtype=bool).reshape(-1)
        final = np.full((len(valid), 4), np.nan)
        final[valid] = np.column_stack((result["predictions"][:, :, -1], result["concentrations"][:, -1]))
        final_counts = np.full((len(valid), 2), np.nan)
        if "counts" in result:
            counts = np.asarray(result["counts"]).reshape(-1, len(result["time"]))
            final_counts[valid, 0] = counts[:, -1]
            if result.get("fbrm", {}).get("scaled", False):
                final_counts[valid, 1] = counts[:, -1]*float(result["fbrm"]["scale"])
        assessment = result.get("reliability", {})
        failures = np.asarray(result["failure_messages"]).reshape(-1).astype(str)
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["draw", *names, "success", "D10_um", "D50_um", "D90_um", "concentration",
                                 "particle_number_density", "FBRM_prediction", "D10_pass", "D50_pass", "D90_pass", "joint_pass",
                                 "threshold_percent", "assessment_scope", "status"])
                for i in range(len(valid)):
                    flags = [bool(assessment["responses"][name]["passing_draws"][i])
                             if assessment["responses"][name]["observation_count"] else ""
                             for name in ("D10", "D50", "D90")] if assessment else [""]*3
                    joint = bool(assessment["joint_passing_draws"][i]) if assessment.get("joint_available", False) else ""
                    writer.writerow([i+1, *result["inputs"][i], bool(valid[i]), *final[i], *final_counts[i], *flags, joint,
                                     assessment.get("threshold_percent", ""), assessment.get("scope", ""), failures[i].rstrip()])
        except OSError as error:
            QMessageBox.warning(self, "Export samples", str(error))
