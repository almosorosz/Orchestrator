"""Named phases for manually defined Control Panel experiments."""
from dataclasses import asdict
from uuid import uuid4

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout,
                               QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from experiment.run_stages import PURPOSES, RunStage, stages_from_payload


class RunStagesDialog(QDialog):
    def __init__(self, stages=(), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Control Panel run stages")
        self.resize(850, 460)
        self.stages = []
        layout = QVBoxLayout(self)
        note = QLabel("Label preparation, dissolution, seeding, experiment and cleanup periods in seconds. "
                      "These stages annotate the next generated experiment; device profiles still define its commands. "
                      "BO/IMED can select the stages that form the modeling experiment period.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Stage name", "Purpose", "Start (s)", "End (s)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)
        row = QHBoxLayout()
        add = QPushButton("Add stage")
        add.clicked.connect(lambda: self.add())
        remove = QPushButton("Remove stage")
        remove.clicked.connect(lambda: self.table.removeRow(self.table.currentRow()))
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch()
        layout.addLayout(row)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #ef4444;")
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for stage in stages_from_payload(stages):
            self.add(asdict(stage))

    def add(self, stage=None):
        row = self.table.rowCount()
        start = self.table.cellWidget(row - 1, 3).value() if row else 0.
        stage = stage or {"id": uuid4().hex[:12], "name": f"Stage {row + 1}", "purpose": "Experiment", "start_s": start, "end_s": start + 60}
        self.table.insertRow(row)
        name = QTableWidgetItem(stage["name"])
        name.setData(Qt.ItemDataRole.UserRole, stage["id"])
        self.table.setItem(row, 0, name)
        purpose = QComboBox()
        purpose.addItems(PURPOSES)
        purpose.setCurrentText(stage["purpose"])
        self.table.setCellWidget(row, 1, purpose)
        for column, key in ((2, "start_s"), (3, "end_s")):
            field = QDoubleSpinBox()
            field.setRange(0., 1e9)
            field.setDecimals(6)
            field.setValue(stage[key])
            self.table.setCellWidget(row, column, field)
        self.table.resizeColumnsToContents()

    def accept(self):
        try:
            stages = [RunStage(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole), self.table.item(row, 0).text(),
                               self.table.cellWidget(row, 1).currentText(), self.table.cellWidget(row, 2).value(),
                               self.table.cellWidget(row, 3).value()) for row in range(self.table.rowCount())]
            stages = stages_from_payload(stages)
            self.stages = [asdict(stage) for stage in stages]
        except (ValueError, TypeError) as error:
            self.error.setText(str(error))
            return
        super().accept()
