"""Construct an unfitted IMED model with explicitly selected mechanisms."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .parameter_labels import PARAMETERS

MECHANISMS = (
    ("Primary nucleation", 0, 5, (5., 3., 7.)),
    ("Secondary nucleation", 5, 12, (4., 2., 6.)),
    ("Growth", 12, 18, (-.5, -2., 1.)),
    ("Dissolution", 18, 24, (-.5, -2., 1.)),
    ("Agglomeration", 24, 28, (-8., -10., -6.)),
    ("Breakage", 28, 35, (-2., -4., 0.)),
)
LOG_ACTIVATION = {3, 9, 16, 22}
SELECTORS = {8: range(2), 26: range(1, 6), 34: range(1, 3)}
DISCRETE_DEFAULTS = {8: 0, 26: 1, 34: 1}


def mechanism_defaults(start, stop, rate):
    rows = np.zeros((stop - start, 4))
    rows[0] = (1, *rate)
    for index in range(start + 1, stop):
        if index in DISCRETE_DEFAULTS:
            rows[index - start, 1:] = DISCRETE_DEFAULTS[index]
        elif index in LOG_ACTIVATION:
            rows[index - start, 1:] = -np.inf
        elif index == 33:
            rows[index - start, 1:] = 1.
    return rows


class CandidateModelDialog(QDialog):
    def __init__(self, names, parent=None):
        super().__init__(parent)
        self.names = names
        self.model = None
        self.setWindowTitle("IMED | Generate candidate model")
        self.resize(860, 520)
        self.name = QLineEdit()
        form = QFormLayout()
        form.addRow("Model name", self.name)
        self.tabs = QTabWidget()
        self.mechanisms = []
        for name, start, stop, rate in MECHANISMS:
            page = QWidget()
            layout = QVBoxLayout(page)
            enabled = QCheckBox(f"Enable {name.lower()}")
            enabled.setChecked(name in {"Primary nucleation", "Growth"})
            layout.addWidget(enabled)
            parameters = QTableWidget(stop - start, 5)
            parameters.setHorizontalHeaderLabels(("Parameter", "Nominal", "Minimum", "Maximum", "Estimate"))
            for row, (estimate, nominal, lower, upper) in enumerate(mechanism_defaults(start, stop, rate)):
                index = start + row
                label, meaning = PARAMETERS[index]
                item = QTableWidgetItem(label)
                item.setToolTip(meaning)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                parameters.setItem(row, 0, item)
                for column, value in enumerate((nominal, lower, upper), 1):
                    parameters.setItem(row, column, QTableWidgetItem(str(value)))
                flag = QTableWidgetItem()
                flag.setCheckState(Qt.CheckState.Checked if estimate else Qt.CheckState.Unchecked)
                flag.setFlags(flag.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if index in SELECTORS:
                    flag.setFlags(flag.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                parameters.setItem(row, 4, flag)
            parameters.resizeColumnsToContents()
            parameters.horizontalHeader().setStretchLastSection(True)
            enabled.toggled.connect(parameters.setEnabled)
            parameters.setEnabled(enabled.isChecked())
            layout.addWidget(parameters)
            self.tabs.addTab(page, name)
            self.mechanisms.append((enabled, parameters, start, stop))
        self.status = QLabel("Review the starting values and bounds before fitting. Log parameters use base 10.")
        self.status.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add candidate")
        buttons.accepted.connect(self.build)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.tabs)
        layout.addWidget(self.status)
        layout.addWidget(buttons)

    def build(self):
        name = self.name.text().strip()
        if not name or name.casefold() in self.names:
            self.status.setText("Enter a unique candidate name.")
            return
        if not any(enabled.isChecked() for enabled, *_ in self.mechanisms):
            self.status.setText("Enable at least one mechanism.")
            return
        table = np.zeros((len(PARAMETERS), 4))
        for tab, (enabled, editor, start, stop) in enumerate(self.mechanisms):
            if not enabled.isChecked():
                table[start, 1:] = -np.inf
                continue
            for row, index in enumerate(range(start, stop)):
                try:
                    estimated = editor.item(row, 4).checkState() == Qt.CheckState.Checked
                    values = [float(editor.item(row, column).text()) for column in (1, 2, 3)]
                    nominal, lower, upper = values
                    disabled_activation = index in LOG_ACTIVATION and not estimated and all(value == -np.inf for value in values)
                    if not disabled_activation and (not np.isfinite(values).all() or not lower <= nominal <= upper):
                        raise ValueError("Use finite, ordered bounds around the nominal value.")
                    if index in SELECTORS and nominal not in SELECTORS[index]:
                        raise ValueError(f"Choose one of {', '.join(map(str, SELECTORS[index]))}.")
                    if index == 34 and nominal == 2 and table[33, 2] <= 0:
                        raise ValueError("A Gaussian breakage distribution requires a positive width.")
                    table[index] = (estimated, *values)
                except ValueError as error:
                    self.tabs.setCurrentIndex(tab)
                    editor.selectRow(row)
                    self.status.setText(f"{PARAMETERS[index][0]}. {error}")
                    return
        self.model = {"Name": name, "paraset": table}
        self.accept()
