"""Editable physical reservoir contents, with blank meaning untracked."""
from copy import deepcopy
from uuid import uuid4

import math
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout)


class ReservoirDialog(QDialog):
    def __init__(self, topology, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reservoir amounts and reactor starting mass")
        self.resize(880, 440)
        self.definition = deepcopy(topology)
        layout = QVBoxLayout(self)
        note = QLabel("Enter available material in grams. Blank means untracked. Re-enter the amount after a refill. "
                      "Density converts volumetric pump delivery to grams when no mass calibration is loaded. "
                      "For a model reactor, use Starting reactor mass for material already present before the run; defined charging stages add their deliveries.")
        note.setWordWrap(True)
        layout.addWidget(note)
        units = self.definition["unit_operations"]
        self.table = QTableWidget(len(units), 4)
        self.table.setHorizontalHeaderLabels(["Container / unit", "Available material (g)", "Density (g/mL)", "Starting reactor mass (g)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setMinimumSectionSize(120)
        for row, unit in enumerate(units):
            for column, value in enumerate((f"{unit['code']}: {unit['name']}", unit.get("inventory_g", ""),
                                             unit.get("density_g_ml", ""), unit.get("initial_system_mass_g", ""))):
                self.table.setItem(row, column, QTableWidgetItem(str(value) if value is not None else ""))
            from PySide6.QtCore import Qt
            self.table.item(row, 0).setFlags(Qt.ItemFlag.ItemIsEnabled)
        layout.addWidget(self.table, 1)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        try:
            for row, unit in enumerate(self.definition["unit_operations"]):
                for column, key in ((1, "inventory_g"), (2, "density_g_ml"), (3, "initial_system_mass_g")):
                    text = self.table.item(row, column).text().strip()
                    old = unit.get(key)
                    value = float(text) if text else None
                    if value is not None and (not math.isfinite(value) or value < 0 or (key == "density_g_ml" and value == 0)):
                        raise ValueError(f"{unit['name']}: amounts must be nonnegative and density must be positive.")
                    if value is None:
                        unit.pop(key, None)
                    else:
                        unit[key] = value
                    if key == "inventory_g" and (old != value or not unit.get("inventory_id")):
                        unit["inventory_id"] = uuid4().hex
        except ValueError as error:
            self.error.setText(str(error))
            return
        super().accept()
