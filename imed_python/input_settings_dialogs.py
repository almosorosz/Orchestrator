"""Edit model inputs before the starting experiment batch exists."""

import numpy as np
from PySide6.QtCharts import QChart, QChartView, QLineSeries
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QVBoxLayout, QWidget,
)

from experiment.adaptive_loop import IMED_OBJECTIVES, imed_response_weights
from .experiment_editor import InputTable, number
from .solver import log_solubility


def decimal_field(minimum, maximum, value, decimals=4):
    field = QDoubleSpinBox()
    field.setDecimals(decimals)
    field.setRange(minimum, maximum)
    field.setValue(value)
    return field


class ObjectivesDialog(QDialog):
    def __init__(self, case, parent=None):
        super().__init__(parent)
        self.setWindowTitle("IMED objectives")
        layout = QVBoxLayout(self)
        description = QLabel("Set at least one positive weight. A weight of zero excludes that response and its recording channels.")
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        self.fields = []
        try:
            weights = imed_response_weights(case)
        except ValueError:
            weights = [0.] * len(IMED_OBJECTIVES)
        for (label, _), weight in zip(IMED_OBJECTIVES, weights):
            field = decimal_field(0., 10000., weight, decimals=8)
            self.fields.append(field)
            form.addRow(f"{label} weight", field)
        layout.addLayout(form)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        self.weights = [field.value() for field in self.fields]
        if not any(self.weights):
            self.status.setText("Enter a positive weight for at least one objective.")
            return
        super().accept()


class SolubilityDialog(QDialog):
    def __init__(self, saved, parent=None):
        super().__init__(parent)
        self.setWindowTitle("IMED | Solubility")
        available = self.screen().availableGeometry()
        self.resize(min(920, available.width() - 60), min(610, available.height() - 60))
        self.saved = saved if isinstance(saved, dict) else {}
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.coefficients = InputTable(("Value",), self.status.setText, tuple("abcdefgh"))
        self.coefficients.load([[self.saved.get(key)] for key in "abcdefgh"])
        self.chart = QChart()
        self.chart.legend().hide()
        self.view = QChartView(self.chart)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.min_temperature = decimal_field(-273.14, 1000., 5., decimals=2)
        self.max_temperature = decimal_field(-273.14, 1000., 60., decimals=2)
        self.antisolvent = decimal_field(0., 1., 0.)
        form = QFormLayout()
        form.addRow("Minimum temperature (°C)", self.min_temperature)
        form.addRow("Maximum temperature (°C)", self.max_temperature)
        form.addRow("Antisolvent mass fraction", self.antisolvent)
        plot = QWidget()
        plot_layout = QVBoxLayout(plot)
        plot_layout.addLayout(form)
        plot_layout.addWidget(self.view, 1)
        body = QHBoxLayout()
        body.addWidget(self.coefficients, 1)
        body.addWidget(plot, 3)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        update = buttons.addButton("Update plot", QDialogButtonBox.ButtonRole.ActionRole)
        update.clicked.connect(self.draw_plot)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if all(key in self.saved for key in "abcdefgh"):
            self.draw_plot()
        else:
            self.status.setText("Enter coefficients a–h and select Update plot to preview the curve.")

    def values(self):
        return {key: number(row[0], f"coefficient {key}")
                for key, row in zip("abcdefgh", self.coefficients.rows())}

    def draw_plot(self):
        try:
            coefficients = tuple(self.values().values())
            low, high = self.min_temperature.value(), self.max_temperature.value()
            if high <= low:
                raise ValueError("Set a maximum temperature above the minimum.")
            temperature = np.linspace(low, high, 200)
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.exp(log_solubility(temperature + 273.15, self.antisolvent.value(), coefficients))
            if not np.isfinite(values).all():
                raise ValueError("The coefficients produce nonfinite solubility over the selected temperatures.")
        except ValueError as error:
            self.status.setText(str(error))
            return
        self.chart.removeAllSeries()
        line = QLineSeries()
        for t, value in zip(temperature, values):
            line.append(float(t), float(value))
        self.chart.addSeries(line)
        self.chart.createDefaultAxes()
        self.chart.axes()[0].setTitleText("Temperature (°C)")
        self.chart.axes()[1].setTitleText("Solubility (g/g solution)")
        self.status.clear()

    def accept(self):
        try:
            self.solubility = {**self.saved, **self.values()}
        except ValueError as error:
            self.status.setText(str(error))
            return
        super().accept()
