
"""Inspect the proposal, selected inputs and Gaussian-process projections."""

import numpy as np
from PySide6.QtCharts import QAreaSeries, QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QTabWidget, QVBoxLayout, QWidget,
)

from .profiles import CHANNELS, TARGETS

MEAN = "#1e40af"
PROPOSAL = "#047857"
OBSERVED = "#a04c08"
COLORS = ("#1e40af", "#047857", "#a04c08", "#b21d49", "#7042b3", "#0b7285", "#6e5a18", "#334455")
BAND_SD = 1.96


def chart_view():
    view = QChartView()
    view.setRenderHint(QPainter.RenderHint.Antialiasing)
    view.setMinimumHeight(250)
    return view


def replace_chart(view, chart):
    previous = view.chart()
    view.setChart(chart)
    # QChartView releases the previous chart when its replacement is installed.
    previous.deleteLater()


def line(name, x, y, color=None, *, dashed=False, markers=False):
    series = QLineSeries()
    series.setName(name)
    for abscissa, ordinate in zip(x, y):
        series.append(float(abscissa), float(ordinate))
    if color:
        pen = QPen(QColor(color), 2)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        series.setPen(pen)
    series.setPointsVisible(markers)
    return series


def scatter(name, x, y, color, *, diameter=8):
    series = QScatterSeries()
    series.setName(name)
    series.setColor(QColor(color))
    series.setBorderColor(QColor(color))
    series.setMarkerSize(diameter)
    for abscissa, ordinate in zip(x, y):
        series.append(float(abscissa), float(ordinate))
    return series


def finish_chart(chart, horizontal, vertical, *, sample_count=None, show_legend=True):
    chart.createDefaultAxes()
    axes = chart.axes(Qt.Orientation.Horizontal)
    if axes:
        axis = axes[0]
        if sample_count is not None:
            axis.setRange(.75, sample_count + .25)
            axis.setTickAnchor(1)
            axis.setTickInterval(max(1, int(np.ceil((sample_count - 1) / 12))))
            axis.setTickType(QValueAxis.TickType.TicksDynamic)
            axis.setLabelFormat("%d")
        elif axis.max() > axis.min():
            pad = .025 * (axis.max() - axis.min())
            axis.setRange(axis.min() - pad, axis.max() + pad)
        axis.setTitleText(horizontal)
    axes = chart.axes(Qt.Orientation.Vertical)
    if axes:
        axis = axes[0]
        if axis.max() > axis.min():
            pad = .06 * (axis.max() - axis.min())
            axis.setRange(axis.min() - pad, axis.max() + pad)
        axis.setTitleText(vertical)
    chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
    chart.legend().setVisible(show_legend)


def profile_chart(channel, traces):
    chart = QChart()
    for index, (name, profile) in enumerate(traces):
        time, values = profile.time, profile.values
        if profile.step:
            time, values = np.repeat(time, 2)[1:], np.repeat(values, 2)[:-1]
        chart.addSeries(line(name, time, values, (OBSERVED, "#626b89", MEAN)[index % 3],
                             dashed=index == 0 and len(traces) > 1,
                             markers=index > 0 or len(time) < 10))
    label, unit, _ = CHANNELS[channel]
    finish_chart(chart, "Time (s)", f"{label} ({unit})")
    return chart


class ProposalView(QTabWidget):
    def __init__(self):
        super().__init__()
        self.result = None
        self.channels, self.sample_variables = QComboBox(), QComboBox()
        self.targets, self.dimensions = QComboBox(), QComboBox()
        self.show_observations = QCheckBox("Show measured responses")
        self.show_observations.setChecked(True)
        self.recipe = chart_view()
        self.samples = chart_view()
        self.projection = chart_view()
        self.recipe_note = self.page("Decoded recipe", self.recipe, (("Profile", self.channels),))
        self.sample_note = self.page("Observed inputs", self.samples, (("Variable", self.sample_variables),))
        self.sample_legend = QLabel(self.widget(1))
        self.sample_legend.setWordWrap(True)
        self.widget(1).layout().insertWidget(2, self.sample_legend)
        self.projection_note = self.page("GP projections", self.projection, (("Output", self.targets),
                                        ("Decision dimension", self.dimensions), ("", self.show_observations)))
        self.channels.currentIndexChanged.connect(self.draw_recipe)
        self.sample_variables.currentIndexChanged.connect(self.draw_samples)
        self.targets.currentIndexChanged.connect(self.draw_projection)
        self.dimensions.currentIndexChanged.connect(self.draw_projection)
        self.show_observations.toggled.connect(self.draw_projection)

    def page(self, name, chart, controls):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        row = QHBoxLayout()
        for label, control in controls:
            if label:
                row.addWidget(QLabel(label))
            row.addWidget(control)
        row.addStretch()
        layout.addLayout(row)
        layout.addWidget(chart, 1)
        note = QLabel()
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(note)
        self.addTab(widget, name)
        return note

    def set_result(self, result):
        self.result = result
        for box in (self.channels, self.sample_variables, self.targets, self.dimensions):
            box.blockSignals(True)
            box.clear()
        if result is not None:
            for key in result.profiles:
                self.channels.addItem(CHANNELS[key][0], key)
            self.sample_variables.addItem("All variables · standardized", None)
            for column, variable in enumerate(result.variables):
                label = variable.label if result.vary[column] else f"{variable.label} · held"
                self.sample_variables.addItem(f"{label} ({variable.unit})", column)
                self.dimensions.addItem(f"{label} ({variable.unit})", column)
            self.targets.addItem("Weighted objective loss", "loss")
            for key in TARGETS:
                if key in result.predictions:
                    self.targets.addItem(TARGETS[key][0], key)
        for box in (self.channels, self.sample_variables, self.targets, self.dimensions):
            box.blockSignals(False)
        self.draw_recipe()
        self.draw_samples()
        self.draw_projection()

    def draw_recipe(self):
        channel = self.channels.currentData()
        if channel is None:
            replace_chart(self.recipe, QChart())
            self.recipe_note.setText("")
            return
        profile = self.result.profiles[channel]
        _, unit, _ = CHANNELS[channel]
        replace_chart(self.recipe, profile_chart(channel, (("Proposed recipe", profile),)))
        detail = f"{len(profile.time)} knots · {profile.values.min():.5g} to {profile.values.max():.5g} {unit}"
        if channel == "feed":
            detail += f" · total antisolvent {profile.total():.5g} g"
        if channel == "temperature" and not profile.step:
            cooling = np.max(np.r_[0., -60 * np.diff(profile.values) / np.diff(profile.time)])
            detail += f" · peak cooling {cooling:.5g} °C/min"
        self.recipe_note.setText(detail)

    def draw_samples(self):
        chart = QChart()
        if self.result is None:
            replace_chart(self.samples, chart)
            self.sample_note.setText("")
            self.sample_legend.clear()
            return
        result = self.result
        selected = self.sample_variables.currentData()
        all_variables = selected is None
        x = np.arange(1, len(result.observed) + 1)
        proposed = len(x) + 1
        observed = result.scaler.transform(result.observed) if all_variables else result.observed
        point = result.scaler.transform(result.point[None, :])[0] if all_variables else result.point
        columns = range(observed.shape[1]) if all_variables else (selected,)
        legend = []
        for column in columns:
            label = result.variables[column].label
            color = COLORS[column % len(COLORS)]
            dashed = (column // len(COLORS)) % 2 == 1
            chart.addSeries(line(label if all_variables else "Selected experiments", x, observed[:, column], color,
                                 dashed=dashed, markers=True))
            if all_variables:
                swatch = "– –" if dashed else "——"
                legend.append(f'<span style="background-color: #ffffff; color: {color};"> {swatch} </span> {label}')
            series = scatter("Proposal", [proposed], [point[column]], color if all_variables else PROPOSAL, diameter=10)
            chart.addSeries(series)
        unit = "Standard deviations from the training mean" if all_variables else result.variables[selected].unit
        finish_chart(chart, "Selected experiment (load order)", unit,
                     sample_count=proposed, show_legend=not all_variables)
        replace_chart(self.samples, chart)
        self.sample_legend.setText(" &nbsp; ".join(legend))
        self.sample_legend.setVisible(all_variables)
        names = [result.state["runs"][index]["name"] for index in result.state["selected"]]
        samples = " · ".join(f"{i}: {name}" for i, name in enumerate(names[:8], start=1))
        if len(names) > 8:
            samples += f" · … {len(names)} experiments"
        detail = f"{samples} · {proposed}: proposal"
        if all_variables:
            detail += ". Select a variable for its original units."
        else:
            detail += f" · proposal {point[selected]:.6g} {unit}"
        self.sample_note.setText(detail)

    def draw_projection(self):
        chart = QChart()
        result = self.result
        column, output = self.dimensions.currentData(), self.targets.currentData()
        if result is None or column is None or output is None:
            replace_chart(self.projection, chart)
            self.projection_note.setText("")
            return
        x, mean, sigma = result.projection(column, output)
        upper = line("", x, mean + BAND_SD * sigma)
        lower = line("", x, mean - BAND_SD * sigma)
        if np.any(sigma > 0):
            area = QAreaSeries(upper, lower)
            upper.setParent(area)
            lower.setParent(area)
            area.setName("±1.96 SD")
            area.setBrush(QColor(51, 110, 203, 65))
            area.setPen(QPen(Qt.PenStyle.NoPen))
            chart.addSeries(area)
        is_loss = output == "loss"
        title, unit = ("Weighted objective loss", "dimensionless") if is_loss else TARGETS[output][:2]
        chart.addSeries(line("Predicted mean" if output != "duration" else "Exact duration", x, mean, MEAN))
        center = result.point[column]
        if is_loss:
            score = mean - result.state["exploration"] * sigma
            chart.addSeries(line("Acquisition", x, score, PROPOSAL, dashed=True))
            value, sd = result.mean_loss, result.loss_sd
            chart.addSeries(scatter("Proposal acquisition", [center], [result.acquisition], PROPOSAL, diameter=10))
        else:
            value, sd = result.predictions[output]
            limits = result.state["output_bounds"].get(output, ())
            for name, limit in zip(("Minimum output", "Maximum output"), limits):
                if limit is not None:
                    chart.addSeries(line(name, [x[0], x[-1]], [limit, limit], "#963b48", dashed=True))
        chart.addSeries(scatter("Proposal mean", [center], [value], MEAN, diameter=11))
        if self.show_observations.isChecked():
            filtered = result.observations(output)
            valid = np.isfinite(filtered)
            if valid.any():
                chart.addSeries(scatter("Measured points", result.observed[valid, column],
                                        filtered[valid], OBSERVED))
        variable = result.variables[column]
        finish_chart(chart, f"{variable.label} ({variable.unit})", f"{title} ({unit})")
        replace_chart(self.projection, chart)
        detail = (f"{variable.label} = {center:.6g} · proposal mean {value:.6g} ± {sd:.5g} SD. "
                  "The slice fixes other inputs at the proposal. Measured points can have different values for those inputs.")
        if is_loss:
            detail += f" Exploration bonus {result.state['exploration'] * sd:.5g} · acquisition {result.acquisition:.6g}."
        elif output in result.state["output_bounds"]:
            margin = result.state["output_margin"]
            lower, upper = result.state["output_bounds"][output]
            if lower is not None or upper is not None:
                detail += f" Screening interval at ±{margin:g} SD: {value - margin * sd:.5g} to {value + margin * sd:.5g}."
        self.projection_note.setText(detail)
