"""Construct an unfitted IMED model with explicitly selected mechanisms."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .kinetic_defaults import MATLAB_KINETIC_DEFAULTS
from .kernel_choice import add_kernel_choice, kernel_values
from .parameter_labels import BREAKAGE_DISTRIBUTION, KERNEL_OPTIONS, PARAMETERS

MECHANISMS = (
    ("Primary nucleation", 0, 5),
    ("Secondary nucleation", 5, 12),
    ("Growth", 12, 18),
    ("Dissolution", 18, 24),
    ("Agglomeration", 24, 28),
    ("Breakage", 28, 35),
)
LOG_ACTIVATION = {3, 9, 16, 22}
SELECTORS = {8: (1, 2), **KERNEL_OPTIONS}


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
        for name, start, stop in MECHANISMS:
            page = QWidget()
            layout = QVBoxLayout(page)
            enabled = QCheckBox(f"Enable {name.lower()}")
            enabled.setChecked(True)
            layout.addWidget(enabled)
            parameters = QTableWidget(stop - start, 5)
            parameters.setHorizontalHeaderLabels(("Parameter", "Nominal", "Minimum", "Maximum", "Estimate"))
            for row, (estimate, nominal, lower, upper) in enumerate(MATLAB_KINETIC_DEFAULTS[start:stop]):
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
                if index in KERNEL_OPTIONS:
                    add_kernel_choice(parameters, row, 1, index, nominal)
            parameters.resizeColumnsToContents()
            parameters.horizontalHeader().setStretchLastSection(True)
            enabled.toggled.connect(parameters.setEnabled)
            parameters.setEnabled(enabled.isChecked())
            layout.addWidget(parameters)
            self.tabs.addTab(page, name)
            self.mechanisms.append((enabled, parameters, start, stop))
        self.status = QLabel("Starting values and bounds follow MATLAB IMED. Log parameters use base 10.")
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
                    if index in KERNEL_OPTIONS:
                        values = kernel_values(editor.cellWidget(row, 1))
                        if index == BREAKAGE_DISTRIBUTION and values[1] == 2 and table[33, 2] <= 0:
                            raise ValueError("Gaussian fragment volume requires a positive σ.")
                        table[index] = values
                        continue
                    estimated = editor.item(row, 4).checkState() == Qt.CheckState.Checked
                    values = [float(editor.item(row, column).text()) for column in (1, 2, 3)]
                    nominal, lower, upper = values
                    disabled_activation = index in LOG_ACTIVATION and not estimated and all(value == -np.inf for value in values)
                    if not disabled_activation and (not np.isfinite(values).all() or not lower <= nominal <= upper):
                        raise ValueError("Use finite, ordered bounds around the nominal value.")
                    if index in SELECTORS and nominal not in SELECTORS[index]:
                        raise ValueError(f"Choose one of {', '.join(map(str, SELECTORS[index]))}.")
                    table[index] = (estimated, *values)
                except ValueError as error:
                    self.tabs.setCurrentIndex(tab)
                    editor.selectRow(row)
                    self.status.setText(f"{PARAMETERS[index][0]}. {error}")
                    return
        self.model = {"Name": name, "paraset": table}
        self.accept()
