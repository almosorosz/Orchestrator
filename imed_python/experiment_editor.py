"""Edit experimental and process records using IMED's MAT schema."""
import csv
from copy import deepcopy
from io import StringIO

import numpy as np
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .campaign_dialog import PROCESS_FIELDS
from .campaign_import import seed_distribution
from .session import display_name

TRACES = {
    "TemperatureData": ("Temperature", "Time (s)", "Temperature (°C)"),
    "FlowRateData": ("Antisolvent", "Time (s)", "Feed (g/s)"),
    "ConcentrationData": ("Concentration", "Time (s)", "Concentration (g/g solution)"),
    "StirringRateData": ("Stirring", "Time (s)", "Stirring rate (RPM)"),
    "CountsData": ("FBRM counts", "Time (s)", "Counts"),
    "D10Data": ("D10 trend", "Time (s)", "D10 (µm)"),
    "D50Data": ("D50 trend", "Time (s)", "D50 (µm)"),
    "D90Data": ("D90 trend", "Time (s)", "D90 (µm)"),
    "InitialCSDsData": ("Seed CSD", "Size (µm)", "Volume density"),
    "FinalCSDsData": ("Final CSD", "Size (µm)", "Volume density"),
}
PERCENTILES = ("InitialD10D50D90", "FinalD10D50D90")
DEFAULTS = {"AntiSolvMass": 0., "SeedMass": 0., "PowerRatio": 1.}


def number(text, label):
    try:
        value = float(text)
        if np.isfinite(value):
            return value
    except (ValueError, TypeError):
        pass
    raise ValueError(f"Enter a finite number for {label}.")


def clipboard_rows(text):
    delimiter = "\t" if "\t" in text else "," if "," in text else "\t"
    rows = [[value.strip() for value in row] for row in csv.reader(StringIO(text), delimiter=delimiter)
            if any(value.strip() for value in row)]
    if not rows:
        raise ValueError("Copy cells from Excel first.")
    if len(rows) > 1 and all(value and value[0].isalpha() for value in rows[0]):
        try:
            [float(value) for value in rows[0]]
        except ValueError:
            rows.pop(0)
    if len({len(row) for row in rows}) != 1:
        raise ValueError("Paste a rectangular range of cells.")
    for row in rows:
        for value in row:
            number(value, "each pasted cell")
    return rows


class InputTable(QWidget):
    def __init__(self, headers, report, row_labels=()):
        super().__init__()
        self.report = report
        self.fixed = bool(row_labels)
        self.table = QTableWidget(len(row_labels), len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setVerticalHeaderLabels(row_labels)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        actions = QHBoxLayout()
        paste = QPushButton("Paste from Excel")
        paste.clicked.connect(self.paste)
        actions.addWidget(paste)
        if not self.fixed:
            add = QPushButton("Add row")
            remove = QPushButton("Delete selected rows")
            add.clicked.connect(self.add_row)
            remove.clicked.connect(self.delete_rows)
            actions.addWidget(add)
            actions.addWidget(remove)
        actions.addStretch()
        layout.addLayout(actions)
        self.shortcut = QShortcut(QKeySequence.StandardKey.Paste, self.table)
        self.shortcut.activated.connect(self.paste)

    def add_row(self):
        current = self.table.currentRow()
        row = current + 1 if current >= 0 else self.table.rowCount()
        self.table.insertRow(row)
        self.table.setCurrentCell(row, 0)

    def delete_rows(self):
        rows = {index.row() for index in self.table.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)

    def paste(self):
        try:
            rows = clipboard_rows(QApplication.clipboard().text())
            first_row = max(0, self.table.currentRow())
            first_column = max(0, self.table.currentColumn())
            if first_column + len(rows[0]) > self.table.columnCount():
                raise ValueError("The copied cells exceed the available columns.")
            end = first_row + len(rows)
            if end > self.table.rowCount():
                if self.fixed:
                    raise ValueError("The copied cells exceed the available settings.")
                self.table.setRowCount(end)
            self.put(rows, first_row, first_column)
            self.report(f"Pasted {len(rows)} row(s).")
        except ValueError as error:
            self.report(str(error))

    def put(self, rows, first_row=0, first_column=0):
        for row, values in enumerate(rows, first_row):
            for column, value in enumerate(values, first_column):
                if value is not None:
                    self.table.setItem(row, column, QTableWidgetItem(str(value)))

    def load(self, rows):
        self.table.clearContents()
        if not self.fixed:
            self.table.setRowCount(len(rows))
        self.put(rows)
        self.table.setCurrentCell(-1, -1)

    def rows(self):
        return [[item.text().strip() if (item := self.table.item(row, column)) else ""
                 for column in range(self.table.columnCount())]
                for row in range(self.table.rowCount())]


def read_trace(rows, name):
    points = np.asarray([[number(cell, f"{name}, row {i}") for cell in row]
                         for i, row in enumerate(rows, 1) if any(row)]).reshape(-1, 2).T
    if points.size and (points[0, 0] < 0 or np.any(np.diff(points[0]) <= 0)):
        raise ValueError(f"{name} requires nonnegative, strictly increasing times or sizes.")
    return points


class ExperimentEditor(QDialog):
    def __init__(self, case, parent=None):
        super().__init__(parent)
        self.setWindowTitle("IMED | Edit experimental inputs")
        self.resize(1000, 620)
        processes = case.get("ProcInf", [])
        self.records = [(deepcopy(exp), deepcopy(processes[i]) if i < len(processes) else {})
                        for i, exp in enumerate(case.get("ExpData", []))]
        self.index = -1
        self.changed = False
        self.new_records = set()
        self.result = None
        self.choices = QComboBox()
        self.choices.activated.connect(self.change_record)
        add = QPushButton("Add experiment")
        add.clicked.connect(self.add_record)
        self.remove = QPushButton("Delete experiment")
        self.remove.clicked.connect(self.remove_record)
        actions = QHBoxLayout()
        actions.addWidget(self.choices, 1)
        actions.addWidget(add)
        actions.addWidget(self.remove)
        self.name = QLineEdit()
        self.initial = QLineEdit()
        form = QFormLayout()
        form.addRow("Experiment name", self.name)
        form.addRow("Initial concentration (g/g solution)", self.initial)
        self.status = QLabel("Paste Excel cells into the selected row and column. Optional profiles may be blank.")
        self.status.setWordWrap(True)
        self.tabs = QTabWidget()
        self.process = InputTable(("Value",), self.status.setText, tuple(PROCESS_FIELDS.values()))
        self.tabs.addTab(self.process, "Process settings")
        self.traces = {key: InputTable(spec[1:], self.status.setText) for key, spec in TRACES.items()}
        for key, editor in self.traces.items():
            self.tabs.addTab(editor, TRACES[key][0])
        self.sizes = InputTable(("D10 (µm)", "D50 (µm)", "D90 (µm)"), self.status.setText, ("Seed", "Final"))
        self.tabs.addTab(self.sizes, "Size percentiles")
        self.solubility = InputTable(("Value",), self.status.setText, tuple("abcdefgh"))
        self.tabs.addTab(self.solubility, "Solubility")
        coefficients = case.get("Solubility")
        self.coefficients = coefficients if isinstance(coefficients, dict) else {}
        self.solubility.load([[self.coefficients.get(key)] for key in "abcdefgh"])
        self.tabs.setCurrentWidget(self.traces["TemperatureData"])
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(actions)
        layout.addLayout(form)
        layout.addWidget(self.tabs)
        layout.addWidget(self.status)
        layout.addWidget(buttons)
        if self.records:
            self.select(0)
        else:
            self.add_record()

    def select(self, index):
        self.index = index
        self.choices.clear()
        self.choices.addItems([display_name(exp, f"Experiment {i + 1}")
                               for i, (exp, _) in enumerate(self.records)])
        self.choices.setCurrentIndex(index)
        for widget in (self.remove, self.name, self.initial, self.process, self.sizes, *self.traces.values()):
            widget.setEnabled(index >= 0)
        exp, process = self.records[index] if index >= 0 else ({}, {})
        self.name.setText(display_name(exp, f"Experiment {index + 1}") if index >= 0 else "")
        concentration = np.asarray(exp.get("ConcentrationData", [])).reshape(2, -1)
        initial = exp.get("InitialConcentration", concentration[1, 0] if concentration.size else "")
        self.initial.setText(str(initial) if np.size(initial) == 1 else "")
        self.process.load([[process.get(key, DEFAULTS.get(key))] for key in PROCESS_FIELDS])
        self.sizes.load([np.asarray(exp.get(key, [])).reshape(-1).tolist()[:3] for key in PERCENTILES])
        for key, editor in self.traces.items():
            editor.load(np.asarray(exp.get(key, [])).reshape(2, -1).T.tolist())
        self.loaded_fields = self.fields()

    def fields(self):
        return (self.name.text(), self.initial.text(), self.process.rows(), self.sizes.rows(),
                [editor.rows() for editor in self.traces.values()])

    def store_record(self):
        if self.index < 0:
            return
        if self.index not in self.new_records and self.fields() == self.loaded_fields:
            return
        exp, process = self.records[self.index]
        name = self.name.text().strip()
        if not name:
            raise ValueError("Enter an experiment name.")
        initial = number(self.initial.text(), "initial concentration")
        if not 0 <= initial < 1:
            raise ValueError("Initial concentration must be in [0, 1) g/g solution.")
        traces = {key: read_trace(editor.rows(), TRACES[key][0]) for key, editor in self.traces.items()}
        times, temperatures = traces["TemperatureData"]
        if len(times) < 2 or times[0] != 0 or np.any(temperatures <= -273.15):
            raise ValueError("Temperature needs two or more points, starting at 0 s, above absolute zero.")
        for key, trace in traces.items():
            if key == "TemperatureData" or not trace.size:
                continue
            label = TRACES[key][0]
            if key.endswith("CSDsData"):
                if trace.shape[1] < 2 or not np.any(trace[1] > 0):
                    raise ValueError(f"{label} needs two or more points and some positive density.")
            elif trace[0, -1] > times[-1]:
                raise ValueError(f"{label} extends beyond the temperature profile.")
            if np.any(trace[1] < 0):
                raise ValueError(f"{label} cannot contain negative values.")
        if np.any(traces["ConcentrationData"][1] >= 1):
            raise ValueError("Measured concentration must be below 1 g/g solution.")
        proc = {key: number(row[0], PROCESS_FIELDS[key])
                for key, row in zip(PROCESS_FIELDS, self.process.rows())}
        positive = {"SolvMass", "SolventDensity", "AntiSolventDensity", "SolidDensity", "ShapeFactor"}
        if any(value < 0 or (key in positive and value == 0) for key, value in proc.items()):
            raise ValueError("Enter positive solvent mass, densities and shape factor, and nonnegative other settings.")
        sizes = {}
        for key, row in zip(PERCENTILES, self.sizes.rows()):
            values = np.asarray([number(value, key) for value in row] if any(row) else [])
            if values.size and (values[0] <= 0 or np.any(np.diff(values) < 0)):
                raise ValueError("Enter ordered positive D10, D50 and D90 values.")
            sizes[key] = values
        if not sizes[PERCENTILES[1]].size:
            observed = [dict(traces[key].T) for key in ("D10Data", "D50Data", "D90Data")]
            for time in sorted(set(observed[0]).intersection(*observed[1:]), reverse=True):
                values = [series[time] for series in observed]
                if 0 < values[0] <= values[1] <= values[2]:
                    sizes[PERCENTILES[1]] = np.asarray(values)
                    break
        if proc["SeedMass"] and not traces["InitialCSDsData"].size:
            if not sizes[PERCENTILES[0]].size:
                raise ValueError("Positive seed mass requires a seed distribution or seed percentiles.")
            traces["InitialCSDsData"] = seed_distribution(sizes[PERCENTILES[0]])
        self.records[self.index] = ({**exp, **traces, **sizes, "Name": name, "InitialConcentration": initial},
                                    {**process, **proc})
        self.changed = True
        self.new_records.discard(self.index)

    def change_record(self, index):
        try:
            self.store_record()
        except ValueError as error:
            self.choices.setCurrentIndex(self.index)
            self.status.setText(str(error))
        else:
            self.select(index)

    def add_record(self):
        try:
            self.store_record()
        except ValueError as error:
            self.status.setText(str(error))
            return
        names = {display_name(exp, "") for exp, _ in self.records}
        number = next(i for i in range(1, len(names) + 2) if f"Experiment {i}" not in names)
        template = self.records[self.index][1] if self.index >= 0 else {}
        self.records.append(({"Name": f"Experiment {number}"}, {k: template[k] for k in PROCESS_FIELDS if k in template}))
        self.new_records.add(len(self.records) - 1)
        self.select(len(self.records) - 1)

    def remove_record(self):
        self.records.pop(self.index)
        self.new_records = {i - int(i > self.index) for i in self.new_records if i != self.index}
        self.changed = True
        self.select(min(self.index, len(self.records) - 1))

    def apply(self):
        try:
            self.store_record()
            original = [str(self.coefficients[key]) if key in self.coefficients else "" for key in "abcdefgh"]
            entered = [row[0] for row in self.solubility.rows()]
            solubility = self.coefficients
            if entered != original:
                solubility = {key: number(value, f"coefficient {key}")
                              for key, value in zip("abcdefgh", entered) if value}
                if solubility:
                    solubility = {**{key: value for key, value in self.coefficients.items() if key not in "abcdefgh"},
                                  **solubility}
                self.changed = True
            if solubility and any(key not in solubility for key in "abcdefgh"):
                raise ValueError("Enter all eight solubility coefficients or leave the table blank.")
        except ValueError as error:
            self.status.setText(str(error))
            return
        self.result = self.records, solubility
        self.accept()
