"""Edit a device's sections independently of every other device's timeline."""
from copy import deepcopy

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton, QVBoxLayout, QWidget)

from experiment.doe_campaign import ROLES
from experiment.recipe import TIME_UNITS, parameter
from gui.recipe_editor import ParameterTable, identifier


class DeviceProfileDialog(QDialog):
    def __init__(self, lane, stages, parent=None, *, time_unit="min"):
        super().__init__(parent)
        self.setWindowTitle(f"Device profile · {lane['name']}")
        self.resize(1100, 560)
        self.lane = lane
        self.time_unit = time_unit
        self.sections = deepcopy(lane.get("profile", []))
        self.current = -1
        self.table = None
        self.default_duration = max(1., sum(stage["duration"]["value"] for stage in stages)) if stages else 60.
        layout = QVBoxLayout(self)
        self.enabled = QCheckBox("Use an independent profile for this device")
        self.enabled.setChecked(True)
        layout.addWidget(self.enabled)
        help_text = QLabel("Sections run in order, starting at time zero. Set one Hold for a constant device, or add as many sections as needed. "
                           "The longest device/shared-stage profile sets the experiment duration. Shorter profiles hold their final target. "
                           "Mass additions stop at the end of their section. Saving replaces this line's shared-stage actions.")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.body = QWidget()
        columns = QHBoxLayout(self.body)
        left = QVBoxLayout()
        self.section_list = QListWidget()
        self.section_list.currentRowChanged.connect(self.select)
        left.addWidget(self.section_list, 1)
        for title, callback in (("Add section", self.add_section), ("Remove section", self.remove_section),
                                ("Move up", lambda: self.move(-1)), ("Move down", lambda: self.move(1))):
            button = QPushButton(title)
            button.clicked.connect(callback)
            left.addWidget(button)
        columns.addLayout(left, 1)
        self.details = QVBoxLayout()
        form = QFormLayout()
        self.name = QLineEdit()
        self.kind = QComboBox()
        self.kind.addItem("Hold", "Hold")
        if lane["role"] != "valve_state":
            self.kind.addItem("Ramp", "Ramp")
        if ROLES[lane["role"]].unit == "g/h":
            self.kind.addItem("Mass addition", "Dose")
        self.kind.currentIndexChanged.connect(self.change_kind)
        self.control_mode = QComboBox()
        self.control_mode.addItem("Internal vessel (Process)", "Process")
        self.control_mode.addItem("Jacket", "Jacket")
        form.addRow("Section name", self.name)
        form.addRow("Action", self.kind)
        if lane["role"] == "T":
            form.addRow("Temperature control", self.control_mode)
        self.details.addLayout(form)
        columns.addLayout(self.details, 4)
        layout.addWidget(self.body, 1)
        self.enabled.toggled.connect(self.body.setEnabled)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if not self.sections:
            self.sections.append(self.new_section(self.default_duration))
        self.refresh(0)

    def new_section(self, duration=60.):
        section = {"id": identifier(), "name": f"Section {len(self.sections) + 1}", "kind": "Hold",
                   "duration": parameter(duration), "start": parameter(self.lane["initial"]),
                   "end": parameter(self.lane["initial"]), "control_mode": "Process"}
        if self.sections:
            section["start"]["equal_previous"] = True
            section["end"]["equal_previous"] = True
        return section

    def save_current(self):
        if 0 <= self.current < len(self.sections) and self.table is not None:
            self.table.save()
            self.sections[self.current].update(name=self.name.text().strip(), kind=self.kind.currentData(),
                                                control_mode=self.control_mode.currentData())
            section = self.sections[self.current]
            self.section_list.item(self.current).setText(self.section_text(section))

    def section_text(self, section):
        return f"{section['name']} · {section['duration']['value'] / TIME_UNITS[self.time_unit]:g} {self.time_unit}"

    def refresh(self, row):
        self.current = -1
        self.section_list.blockSignals(True)
        self.section_list.clear()
        for section in self.sections:
            self.section_list.addItem(self.section_text(section))
        self.section_list.setCurrentRow(row)
        self.section_list.blockSignals(False)
        self.select(row)

    def select(self, row):
        self.save_current()
        self.current = row
        if not 0 <= row < len(self.sections):
            return
        section = self.sections[row]
        self.name.setText(section["name"])
        self.kind.blockSignals(True)
        self.kind.setCurrentIndex(max(0, self.kind.findData(section["kind"])))
        self.kind.blockSignals(False)
        self.control_mode.setCurrentIndex(max(0, self.control_mode.findData(section.get("control_mode", "Process"))))
        self.make_table()

    def make_table(self):
        if self.table is not None:
            self.details.removeWidget(self.table)
            self.table.hide()
            self.table.deleteLater()
        section = self.sections[self.current]
        unit = "g" if section["kind"] == "Dose" else ROLES[self.lane["role"]].unit
        entries = [("Duration (s)", section["duration"])]
        if section["kind"] == "Ramp":
            entries.append((f"Start ({unit})", section["start"]))
        entries.append((f"End ({unit})", section["end"]))
        self.table = ParameterTable(entries, allow_previous=True, time_unit=self.time_unit, previous_hint=(
            "Keep the preceding pump flow; the mass for this section is calculated from its duration."
            if section["kind"] == "Dose" else None))
        self.details.addWidget(self.table, 1)
        self.summary.setText("Profile duration uses the sum of its section durations in each generated experiment. "
                             "Equal to previous uses this device's preceding final target; the first uses Initial target. "
                             "For mass additions, it keeps the previous flow and calculates the required mass.")

    def change_kind(self):
        if self.current < 0:
            return
        self.save_current()
        self.make_table()

    def add_section(self):
        self.save_current()
        self.sections.append(self.new_section())
        self.refresh(len(self.sections) - 1)

    def remove_section(self):
        if len(self.sections) > 1 and self.current >= 0:
            self.sections.pop(self.current)
            self.refresh(min(self.current, len(self.sections) - 1))

    def move(self, offset):
        row = self.current
        if 0 <= row + offset < len(self.sections):
            self.save_current()
            self.sections[row], self.sections[row + offset] = self.sections[row + offset], self.sections[row]
            self.refresh(row + offset)

    def accept(self):
        self.save_current()
        if self.enabled.isChecked() and any(not section["name"] for section in self.sections):
            self.summary.setText("Enter a name for every section.")
            return
        super().accept()
