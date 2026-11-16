"""Editable stage lists and decision controls for BO profiles."""
from dataclasses import asdict, replace
from math import fsum, isclose, isfinite

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .physical import Stage, control_values, grouped_controls, physical_controls, validate_stages
from .profiles import CHANNELS


CONTROL_LIMIT = 4
STAGE_FIELDS = ("duration_s", "start", "end")


def read_only(text):
    cell = QTableWidgetItem(text)
    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return cell


def available_controls(channel, stages):
    return grouped_controls(channel, stages) + physical_controls(channel, stages)


class ProfileStageEditor(QWidget):
    changed = Signal()

    def __init__(self, channel, parent=None):
        super().__init__(parent)
        self.channel = channel
        self.label, self.unit, _ = CHANNELS[channel]
        self.horizon = None
        self.selected_controls = set()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ("Stage", "Duration (s)", f"Start ({self.unit})", f"End ({self.unit})"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed
                                   | QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self.edit_stage)
        self.table.itemSelectionChanged.connect(self.update_buttons)

        self.controls = QTableWidget(0, 4)
        self.controls.setHorizontalHeaderLabels(("Vary", "Control", "Value", "Unit"))
        header = self.controls.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.controls.verticalHeader().hide()
        self.controls.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.controls.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.controls.itemChanged.connect(self.toggle_control)

        self.add_button = QPushButton("Add stage")
        self.delete_button = QPushButton("Delete stage")
        self.up_button = QPushButton("Move up")
        self.down_button = QPushButton("Move down")
        self.add_button.clicked.connect(self.add_stage)
        self.delete_button.clicked.connect(self.delete_stage)
        self.up_button.clicked.connect(lambda: self.move_stage(-1))
        self.down_button.clicked.connect(lambda: self.move_stage(1))
        self.time_note = QLabel()
        self.time_note.setWordWrap(True)
        self.control_note = QLabel()
        self.control_note.setWordWrap(True)

        left, right = QVBoxLayout(), QVBoxLayout()
        left.addWidget(self.table)
        actions = QHBoxLayout()
        for button in (self.add_button, self.delete_button, self.up_button, self.down_button):
            actions.addWidget(button)
        actions.addStretch()
        left.addLayout(actions)
        left.addWidget(self.time_note)

        right.addWidget(QLabel("Decision controls"))
        right.addWidget(self.controls)
        right.addWidget(self.control_note)
        layout = QHBoxLayout(self)
        layout.addLayout(left, 3)
        layout.addLayout(right, 2)
        self.show_stages(())

    def load(self, spec):
        self.selected_controls = set(spec.get("controls", ()))
        self.show_stages(tuple(Stage(**stage) for stage in spec.get("stages", ())))

    def spec(self):
        stages = self.read_stages()
        if not stages:
            return None
        stages = validate_stages(self.channel, stages, self.horizon)
        controls = [item.key for item in available_controls(self.channel, stages)
                    if item.key in self.selected_controls]
        if len(controls) > CONTROL_LIMIT:
            raise ValueError(f"Select at most {CONTROL_LIMIT} {self.label.lower()} decision controls.")
        return {"stages": [asdict(stage) for stage in stages], "controls": controls}

    def set_horizon(self, horizon):
        self.horizon = horizon
        self.refresh_controls()

    def read_stages(self):
        stages = []
        for row in range(self.table.rowCount()):
            try:
                values = [float(self.table.item(row, col).text()) for col in range(1, 4)]
            except ValueError as error:
                raise ValueError(f"Enter numeric values in {self.label.lower()} stage {row + 1}.") from error
            if not all(isfinite(value) for value in values):
                raise ValueError(f"Enter finite values in {self.label.lower()} stage {row + 1}.")
            stages.append(Stage(*values))
        return tuple(stages)

    def stage_type(self, stage):
        if self.channel == "feed" and stage.start == stage.end == 0:
            return "Off"
        return "Hold" if stage.start == stage.end else "Ramp"

    def show_stages(self, stages, select=-1):
        self.table.blockSignals(True)
        self.table.setRowCount(len(stages))
        for row, stage in enumerate(stages):
            self.table.setItem(row, 0, read_only(self.stage_type(stage)))
            for col, field in enumerate(STAGE_FIELDS, start=1):
                text = str(getattr(stage, field))
                if self.channel == "temperature" and row and col == 2:
                    item = read_only(text)
                    item.setToolTip("The previous stage’s end temperature sets this value.")
                else:
                    item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, col, item)
        if stages and select >= 0:
            self.table.selectRow(select)
        self.table.blockSignals(False)
        self.update_buttons()
        self.refresh_controls()

    def edit_stage(self, cell):
        self.table.blockSignals(True)
        if self.channel == "temperature" and cell.column() == 3 and cell.row() + 1 < self.table.rowCount():
            self.table.item(cell.row() + 1, 2).setText(cell.text())
        try:
            stages = self.read_stages()
            for row, stage in enumerate(stages):
                self.table.item(row, 0).setText(self.stage_type(stage))
        except ValueError:
            pass
        finally:
            self.table.blockSignals(False)
        self.refresh_controls()
        self.changed.emit()

    def refresh_controls(self):
        controls, values = (), ()
        try:
            stages = self.read_stages()
            if stages:
                # An over-budget sequence can still be edited and assigned decision controls.
                stages = validate_stages(self.channel, stages)
                controls = available_controls(self.channel, stages)
                values = control_values(stages, controls)
                self.selected_controls.intersection_update(control.key for control in controls)
                self.set_time_note(fsum(stage.duration_s for stage in stages))
            else:
                self.time_note.setText(f"Add stages to define a {self.label.lower()} profile.")
        except ValueError as error:
            self.time_note.setText(str(error))

        self.controls.blockSignals(True)
        self.controls.setRowCount(len(controls))
        for row, (definition, value) in enumerate(zip(controls, values)):
            cell = read_only("")
            cell.setData(Qt.ItemDataRole.UserRole, definition.key)
            cell.setFlags(cell.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            cell.setCheckState(Qt.CheckState.Checked if definition.key in self.selected_controls
                               else Qt.CheckState.Unchecked)
            self.controls.setItem(row, 0, cell)
            self.controls.setItem(row, 1, read_only(definition.label))
            self.controls.setItem(row, 2, read_only(f"{value:g}"))
            self.controls.setItem(row, 3, read_only(definition.unit))
        self.controls.blockSignals(False)
        self.update_control_note()

    def set_time_note(self, total):
        if self.horizon is None:
            self.time_note.setText(f"Total stage duration {total:g} s.")
            return
        remaining = self.horizon - total
        if isclose(total, self.horizon, rel_tol=0., abs_tol=1e-8):
            self.time_note.setText(f"Stages fill the {self.horizon:g} s batch horizon.")
        elif remaining < 0:
            self.time_note.setText(f"Stages exceed the {self.horizon:g} s batch horizon by {-remaining:g} s.")
        else:
            ending = "Hold the final temperature" if self.channel == "temperature" else "Keep feed off"
            self.time_note.setText(f"{ending} for the remaining {remaining:g} s.")

    def update_control_note(self, text=None):
        count = sum(self.controls.item(row, 0).checkState() == Qt.CheckState.Checked
                    for row in range(self.controls.rowCount()))
        self.control_note.setText(text or f"{count} of {CONTROL_LIMIT} decision controls selected.")

    def toggle_control(self, cell):
        if cell.column() != 0:
            return
        key = cell.data(Qt.ItemDataRole.UserRole)
        if cell.checkState() == Qt.CheckState.Checked:
            count = sum(self.controls.item(row, 0).checkState() == Qt.CheckState.Checked
                        for row in range(self.controls.rowCount()))
            if count > CONTROL_LIMIT:
                self.controls.blockSignals(True)
                cell.setCheckState(Qt.CheckState.Unchecked)
                self.controls.blockSignals(False)
                self.update_control_note(f"Select at most {CONTROL_LIMIT} decision controls.")
                return
            self.selected_controls.add(key)
        else:
            self.selected_controls.discard(key)
        self.update_control_note()
        self.changed.emit()

    def update_buttons(self):
        position = self.table.currentRow()
        self.delete_button.setEnabled(position >= 0)
        self.up_button.setEnabled(position > 0)
        self.down_button.setEnabled(0 <= position < self.table.rowCount() - 1)

    def change_rows(self, stages, order, select):
        previous = tuple(stages)
        stages = [previous[index] for index in order]
        new_positions = {old: new for new, old in enumerate(order)}
        controls = set()
        for key in self.selected_controls:
            if key.startswith("stage:"):
                prefix, number, kind = key.split(":")
                if int(number) in new_positions:
                    controls.add(f"{prefix}:{new_positions[int(number)]}:{kind}")
            else:
                controls.add(key)
        if self.channel == "temperature" and stages:
            stages[0] = replace(stages[0], start=previous[0].start)
            for index in range(1, len(stages)):
                stages[index] = replace(stages[index], start=stages[index - 1].end)
        self.selected_controls = controls
        self.show_stages(stages, select)
        self.changed.emit()

    def add_stage(self):
        try:
            stages = list(self.read_stages())
        except ValueError as error:
            self.time_note.setText(str(error))
            return
        remaining = self.horizon - fsum(stage.duration_s for stage in stages) if self.horizon else 60.
        duration = min(remaining, 60.) if remaining > 0 else min(self.horizon or 60., 60.)
        value = 0.
        if self.channel == "temperature":
            value = stages[-1].end if stages else 25.
        stages.append(Stage(duration, value, value))
        self.show_stages(stages, len(stages) - 1)
        self.changed.emit()

    def delete_stage(self):
        row = self.table.currentRow()
        if row < 0:
            return
        try:
            stages = self.read_stages()
        except ValueError:
            self.table.blockSignals(True)
            initial = self.table.item(0, 2).text()
            self.table.removeRow(row)
            if self.channel == "temperature" and self.table.rowCount():
                self.table.setItem(0, 2, QTableWidgetItem(initial))
                for index in range(1, self.table.rowCount()):
                    self.table.item(index, 2).setText(self.table.item(index - 1, 3).text())
            self.table.blockSignals(False)
            self.selected_controls.clear()
            self.refresh_controls()
            self.update_buttons()
            self.changed.emit()
            return
        order = [index for index in range(len(stages)) if index != row]
        self.change_rows(stages, order, min(row, len(order) - 1))

    def move_stage(self, offset):
        position = self.table.currentRow()
        target = position + offset
        if position < 0 or not 0 <= target < self.table.rowCount():
            return
        try:
            stages = self.read_stages()
        except ValueError as error:
            self.time_note.setText(str(error))
            return
        order = list(range(len(stages)))
        order[position], order[target] = order[target], order[position]
        self.change_rows(stages, order, target)


class StageEditor(QWidget):
    """Store an ordered editable stage list and selected controls for T and AS."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabs = QTabWidget()
        self.editors = {}
        for channel in ("temperature", "feed"):
            editor = ProfileStageEditor(channel, self)
            editor.changed.connect(self.changed)
            self.editors[channel] = editor
            self.tabs.addTab(editor, editor.label)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

    def load(self, specs, horizon=None):
        for channel, editor in self.editors.items():
            editor.horizon = horizon
            editor.load(specs.get(channel, {}))

    def set_horizon(self, horizon):
        for editor in self.editors.values():
            editor.set_horizon(horizon)

    def specs(self):
        return {channel: spec for channel, editor in self.editors.items()
                if (spec := editor.spec()) is not None}
