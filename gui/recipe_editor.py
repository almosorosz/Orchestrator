"""Drag-and-drop Experiment Definition with device columns and time rows."""
from copy import deepcopy
from uuid import uuid4
import numpy as np

from PySide6.QtCore import QByteArray, QMimeData, QPointF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QDrag, QBrush, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QInputDialog,
    QGraphicsItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QSizePolicy,
)

from experiment.doe_campaign import ROLES
from experiment.recipe import PURPOSES, SYSTEM_FIELDS, TIME_UNITS, parameter, validate_recipe
from hardware_devices.capabilities import is_monitor_only


MIME = "application/x-crystaloop-action"
ROW_HEIGHT, COLUMN_WIDTH, LABEL_WIDTH, HEADER_HEIGHT = 92, 172, 170, 54
COLORS = {"T": "#1d4ed8", "AS": "#7c3aed", "seed": "#0f766e", "valve_state": "#b45309"}


def identifier():
    return uuid4().hex[:12]


def number(value):
    field = QDoubleSpinBox()
    field.setDecimals(6)
    field.setRange(-1e9, 1e9)
    field.setValue(value)
    return field


class ParameterTable(QTableWidget):
    def __init__(self, entries, parent=None, temperature=False, time_unit="s"):
        super().__init__(len(entries), 6 if temperature else 5, parent)
        self.setHorizontalHeaderLabels(("Parameter", "Across experiments", "Value", "Minimum", "Maximum")
                                      + (("Equal to previous",) if temperature else ()))
        self.entries = entries
        self.links = {}
        self.scales, self.original_display, self.original_values = {}, {}, {}
        for row, (label, item) in enumerate(entries):
            self.scales[row] = TIME_UNITS[time_unit] if label.startswith("Duration") else 1.
            if label.startswith("Duration"):
                label = f"Duration ({time_unit})"
            title = QTableWidgetItem(label)
            title.setFlags(title.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row, 0, title)
            mode = QComboBox()
            mode.addItems(("Fixed", "DoE", "Alternate"))
            mode.setCurrentText(item["mode"])
            self.setCellWidget(row, 1, mode)
            for column, key in enumerate(("value", "min", "max"), 2):
                field = number(item[key] / self.scales[row])
                if label.startswith("Duration"):
                    field.setDecimals(9)
                    field.setValue(item[key] / self.scales[row])
                self.original_display[row, column] = field.value()
                self.original_values[row, column] = item[key]
                field.setMinimumWidth(65)
                field.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
                self.setCellWidget(row, column, field)
            if temperature and not label.startswith("Duration"):
                link = QCheckBox("Equal to previous")
                link.setToolTip("Use this device's preceding section's final temperature in each experiment; the first section uses Initial target.")
                self.setCellWidget(row, 5, link)
                self.links[row] = link
                def enable(checked, r=row):
                    for c in range(1, 5):
                        self.cellWidget(r, c).setEnabled(not checked)
                link.toggled.connect(enable)
                link.setChecked(item.get("equal_previous", False))
        self.resizeColumnsToContents()
        for column, width in enumerate((112, 145, 105, 105, 105, 165)[:self.columnCount()]):
            self.setColumnWidth(column, width)
        self.horizontalHeader().setStretchLastSection(True)

    def save(self):
        for row, (_, item) in enumerate(self.entries):
            item["mode"] = self.cellWidget(row, 1).currentText()
            for column, key in enumerate(("value", "min", "max"), 2):
                displayed = self.cellWidget(row, column).value()
                item[key] = (self.original_values[row, column] if displayed == self.original_display[row, column]
                             else displayed * self.scales[row])
            if row in self.links:
                item["equal_previous"] = self.links[row].isChecked()


class ActionPalette(QListWidget):
    def __init__(self):
        super().__init__()
        self.addItems(("Hold", "Ramp", "Mass addition", "Equilibration", "Open valve", "Close valve"))
        self.setDragEnabled(True)
        self.setMaximumHeight(130)

    def startDrag(self, _):
        if self.currentItem() is None:
            return
        mime = QMimeData()
        mime.setData(MIME, QByteArray(self.currentItem().text().encode()))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class ActionBlock(QGraphicsRectItem):
    def __init__(self, editor, stage, lane):
        super().__init__(0, 0, COLUMN_WIDTH - 12, ROW_HEIGHT - 12)
        self.editor, self.stage_id, self.lane_id = editor, stage["id"], lane["id"]
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setBrush(QBrush(QColor(COLORS.get(lane["role"], "#334155"))))
        self.setPen(QPen(QColor("#64748b")))
        action = stage["actions"][lane["id"]]
        value = action["end"]
        unit = "g" if action["kind"] == "Dose" else ROLES[lane["role"]].unit
        detail = ("Equal to previous" if value.get("equal_previous") else
                  f"{value['value']:g} {unit}" if value["mode"] == "Fixed" else f"{value['mode']} {value['min']:g}–{value['max']:g} {unit}")
        stop = (len(editor.recipe["stages"]) - 1 if action.get("until_end") else
                next((i for i, item in enumerate(editor.recipe["stages"]) if item["id"] == action.get("until_stage")),
                     editor.recipe["stages"].index(stage)))
        span = stop - editor.recipe["stages"].index(stage) + 1
        self.setRect(0, 0, COLUMN_WIDTH - 12, max(ROW_HEIGHT - 12, ROW_HEIGHT * span - 12))
        mode = "\n" + action["control_mode"] if lane["role"] == "T" else ""
        text = editor.scene.addText(f"{action.get('name', action['kind'])}\n{detail}{mode}")
        text.setDefaultTextColor(QColor("#f8fafc"))
        text.setTextWidth(COLUMN_WIDTH - 22)
        text.setParentItem(self)
        text.setPos(5, 3)
        self.setToolTip("Double-click to edit. Drag to move this action to another stage or compatible line.")

    def mouseDoubleClickEvent(self, event):
        self.editor.edit_action(self.stage_id, self.lane_id)
        event.accept()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        position = self.pos()
        editor, stage, lane = self.editor, self.stage_id, self.lane_id
        QTimer.singleShot(0, lambda: editor.move_block(stage, lane, position + QPointF(10, 10)))


class TimelineView(QGraphicsView):
    def __init__(self, editor):
        super().__init__(editor.scene)
        self.editor = editor
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setBackgroundBrush(QColor("#07111f"))

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(MIME):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(MIME):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasFormat(MIME):
            point = self.mapToScene(event.position().toPoint())
            self.editor.drop_action(bytes(event.mimeData().data(MIME)).decode(), point)
            event.acceptProposedAction()

    def mouseDoubleClickEvent(self, event):
        point = self.mapToScene(event.position().toPoint())
        if point.y() < HEADER_HEIGHT:
            lane = int((point.x() - LABEL_WIDTH) // COLUMN_WIDTH)
            if 0 <= lane < len(self.editor.recipe["lanes"]):
                self.editor.edit_line(lane)
                return
        super().mouseDoubleClickEvent(event)


class RecipeEditor(QWidget):
    changed = Signal()

    def __init__(self, recipe, configs, room):
        super().__init__()
        self.configs, self.room = configs, room
        self.recipe = deepcopy(recipe) if recipe else {"version": 1, "lanes": [], "stages": []}
        self.scene = QGraphicsScene(self)
        self.view = TimelineView(self)
        layout = QHBoxLayout(self)
        side = QVBoxLayout()
        side.addWidget(QLabel("Drag an action onto a device lane"))
        self.palette = ActionPalette()
        side.addWidget(self.palette)
        self.stage_list = QListWidget()
        self.stage_list.setMinimumHeight(140)
        self.stage_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.stage_list.model().rowsMoved.connect(self.reorder)
        self.stage_list.itemDoubleClicked.connect(lambda _: self.edit_stage())
        side.addWidget(QLabel("Stages · drag to reorder"))
        side.addWidget(self.stage_list, 1)
        for label, callback in (("Add stage", self.add_stage), ("Edit stage", self.edit_stage),
                                ("Delete stage", self.delete_stage), ("Add device line", self.add_line),
                                ("Edit device profile…", self.edit_device_profile),
                                ("Remove selected action", self.delete_action)):
            button = QPushButton(label)
            button.setMaximumHeight(32)
            button.setStyleSheet("QPushButton { padding: 4px 10px; }")
            button.clicked.connect(callback)
            side.addWidget(button)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.notice.setText("Use shared stages, or Edit device profile for separate sections per device. Shorter profiles hold their final target; mass additions stop when finished.")
        self.horizon = QLabel()
        self.horizon.setWordWrap(True)
        side.addWidget(self.horizon)
        side.addWidget(self.notice)
        left = QWidget()
        left.setLayout(side)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.view)
        split.setSizes([220, 850])
        layout.addWidget(split)
        self.refresh()

    def refresh(self):
        active = self.stage_list.currentRow()
        self.stage_list.blockSignals(True)
        self.stage_list.clear()
        for stage in self.recipe["stages"]:
            item = QListWidgetItem(f"{stage['name']} · {self.time_text(stage['duration']['value'])}", self.stage_list)
            item.setData(Qt.ItemDataRole.UserRole, stage["id"])
        self.stage_list.setCurrentRow(min(max(active, 0), self.stage_list.count() - 1))
        self.stage_list.blockSignals(False)
        self.scene.clear()
        width = LABEL_WIDTH + COLUMN_WIDTH * len(self.recipe["lanes"])
        for column, lane in enumerate(self.recipe["lanes"]):
            text = self.scene.addText(f"{lane['name']}\n{lane.get('device_id') or 'Assign device'}" + (" · own profile" if lane.get("profile") else ""))
            text.setDefaultTextColor(QColor("#e2e8f0"))
            text.setTextWidth(COLUMN_WIDTH - 10)
            text.setPos(LABEL_WIDTH + column * COLUMN_WIDTH, 0)
            text.setToolTip("Double-click the header to edit the line and its device assignment.")
        time = 0.
        blocks = []
        for row, stage in enumerate(self.recipe["stages"]):
            y = HEADER_HEIGHT + row * ROW_HEIGHT
            text = self.scene.addText(f"{self.time_text(time)}\n{stage['name']}\n{stage['purpose']}")
            text.setDefaultTextColor(QColor("#cbd5e1"))
            text.setTextWidth(LABEL_WIDTH - 8)
            text.setPos(0, y)
            for column, lane in enumerate(self.recipe["lanes"]):
                self.scene.addRect(LABEL_WIDTH + column * COLUMN_WIDTH, y, COLUMN_WIDTH, ROW_HEIGHT,
                                   QPen(QColor("#24364d")))
                if lane["id"] in stage["actions"]:
                    blocks.append((stage, lane, column, y))
            time += stage["duration"]["value"]
        rendered_blocks = []
        for stage, lane, column, y in blocks:
            block = ActionBlock(self, stage, lane)
            self.scene.addItem(block)
            block.setPos(LABEL_WIDTH + column * COLUMN_WIDTH + 6, y + 6)
            rendered_blocks.append((block, stage["actions"][lane["id"]], y))
        # Independent profiles show all of their own sections in their column.
        # The canvas uses the shared-stage nominal time scale; generation resolves
        # sampled durations separately for each experiment.
        total = max([time] + [sum(s["duration"]["value"] for s in lane.get("profile", [])) for lane in self.recipe["lanes"]])
        self.horizon.setText(f"Nominal experiment duration: {self.time_text(total)}")
        durations = [stage["duration"]["value"] for stage in self.recipe["stages"]]
        anchors = np.r_[0., np.cumsum(durations)]
        scale = (ROW_HEIGHT * len(durations) / max(time, 1.) if durations else
                 max(400., ROW_HEIGHT * max([1] + [len(lane.get('profile', [])) for lane in self.recipe["lanes"]])) / max(total, 1.))
        def position_at(elapsed):
            if durations and elapsed <= time:
                return HEADER_HEIGHT + float(np.interp(elapsed, anchors, np.arange(len(anchors)) * ROW_HEIGHT))
            return HEADER_HEIGHT + (ROW_HEIGHT * len(durations) + (elapsed - time) * scale if durations else elapsed * scale)
        bottom = HEADER_HEIGHT + ROW_HEIGHT * len(self.recipe["stages"])
        for block, action, y in rendered_blocks:
            if action.get("until_end"):
                block.setRect(0, 0, COLUMN_WIDTH - 12, max(ROW_HEIGHT - 12, position_at(total) - y - 12))
        for column, lane in enumerate(self.recipe["lanes"]):
            elapsed = 0.
            for section_index, section in enumerate(lane.get("profile", [])):
                y = position_at(elapsed)
                final_hold = section["kind"] == "Hold" and section_index == len(lane["profile"]) - 1
                until = total if final_hold else elapsed + section["duration"]["value"]
                height = max(26., position_at(until) - y)
                block = self.scene.addRect(LABEL_WIDTH + column * COLUMN_WIDTH + 6, y + 2, COLUMN_WIDTH - 12, height - 4,
                                           QPen(QColor("#64748b")), QBrush(QColor(COLORS.get(lane["role"], "#334155"))))
                value = section["end"]
                unit = "g" if section["kind"] == "Dose" else ROLES[lane["role"]].unit
                target = ("Equal to previous" if value.get("equal_previous") else f"{value['value']:g}" if value["mode"] == "Fixed"
                          else f"{value['mode']} {value['min']:g}–{value['max']:g}")
                if not value.get("equal_previous"):
                    target += f" {unit}"
                text = self.scene.addText(f"{section['name']} · {self.time_text(section['duration']['value'])}\n{section['kind']} · {target}"
                                          + ("\nHolds to experiment end" if final_hold and until > elapsed + section["duration"]["value"] else ""))
                text.setDefaultTextColor(QColor("#f8fafc"))
                text.setTextWidth(COLUMN_WIDTH - 20)
                text.setParentItem(block)
                text.setPos(LABEL_WIDTH + column * COLUMN_WIDTH + 9, y + 3)
                elapsed += section["duration"]["value"]
                bottom = max(bottom, y + height)
            if lane.get("profile") and elapsed < total and lane["profile"][-1]["kind"] != "Hold":
                y = position_at(elapsed)
                self.scene.addRect(LABEL_WIDTH + column * COLUMN_WIDTH + 6, y + 2, COLUMN_WIDTH - 12,
                                   max(2., position_at(total) - y - 4), QPen(QColor("#475569")), QBrush(QColor("#132033")))
                text = self.scene.addText("Off after mass addition" if lane["profile"][-1]["kind"] == "Dose" else
                                          "Hold final target\nto experiment end")
                text.setDefaultTextColor(QColor("#94a3b8"))
                text.setTextWidth(COLUMN_WIDTH - 20)
                text.setPos(LABEL_WIDTH + column * COLUMN_WIDTH + 9, y + 4)
            bottom = max(bottom, position_at(total))
        if not durations:
            ticks = sorted({0., total} | {sum(s["duration"]["value"] for s in lane.get("profile", [])[:i + 1])
                                         for lane in self.recipe["lanes"] for i in range(len(lane.get("profile", [])))})
            last_y = -100.
            for tick in ticks:
                y = position_at(tick)
                if y - last_y >= 22. or tick == total:
                    label = self.scene.addText(self.time_text(tick))
                    label.setDefaultTextColor(QColor("#94a3b8"))
                    label.setPos(4, y)
                    last_y = y
        self.scene.setSceneRect(0, 0, max(width, 700), max(bottom + 40, 400))
        self.changed.emit()

    @property
    def time_unit(self):
        return self.recipe.get("time_unit", "min")

    def time_text(self, seconds):
        return f"{seconds / TIME_UNITS[self.time_unit]:g} {self.time_unit}"

    def reorder(self):
        if self.stage_list.signalsBlocked():
            return
        by_id = {stage["id"]: stage for stage in self.recipe["stages"]}
        self.recipe["stages"] = [by_id[self.stage_list.item(i).data(Qt.ItemDataRole.UserRole)]
                                  for i in range(self.stage_list.count())]
        self.refresh()

    def stage_dialog(self, stage):
        dialog = QDialog(self)
        dialog.setWindowTitle("Stage settings")
        dialog.resize(710, 230)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        name = QLineEdit(stage["name"])
        purpose = QComboBox()
        purpose.addItems(PURPOSES)
        purpose.setCurrentText(stage["purpose"])
        form.addRow("Stage name", name)
        form.addRow("Purpose", purpose)
        layout.addLayout(form)
        table = ParameterTable([("Duration (s)", stage["duration"])], time_unit=self.time_unit)
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec():
            stage.update(name=name.text().strip(), purpose=purpose.currentText())
            table.save()
            return True
        return False

    def add_stage(self):
        stage = {"id": identifier(), "name": "New stage", "purpose": "Experiment", "duration": parameter(60.), "actions": {}}
        if self.stage_dialog(stage):
            row = self.stage_list.currentRow() + 1
            self.recipe["stages"].insert(row, stage)
            self.refresh()
            self.stage_list.setCurrentRow(row)

    def edit_stage(self):
        row = self.stage_list.currentRow()
        if row >= 0:
            stage = deepcopy(self.recipe["stages"][row])
            if self.stage_dialog(stage):
                self.recipe["stages"][row] = stage
                self.refresh()

    def delete_stage(self):
        row = self.stage_list.currentRow()
        if row >= 0:
            self.recipe["stages"].pop(row)
            self.refresh()

    def add_line(self):
        self.edit_line(None)

    def edit_line(self, index):
        old = self.recipe["lanes"][index] if index is not None else {}
        dialog = QDialog(self)
        dialog.setWindowTitle("Device line")
        form = QFormLayout(dialog)
        name = QLineEdit(old.get("name", ""))
        role = QComboBox()
        for key, details in ROLES.items():
            role.addItem(details.label, key)
        role.setCurrentIndex(max(0, role.findData(old.get("role"))))
        device = QComboBox()
        initial = number(old.get("initial", self.room if role.currentData() == "T" else 0.))
        def devices():
            selected = device.currentData() or old.get("device_id")
            device.clear()
            device.addItem("Assign when applying DoE", "")
            for key, config in self.configs.items():
                if config.get("type") in ROLES[role.currentData()].compatible_types and not is_monitor_only(config):
                    device.addItem(f"{key} · {config.get('name', key)}", key)
            device.setCurrentIndex(max(0, device.findData(selected)))
        role.currentIndexChanged.connect(devices)
        devices()
        for title, widget in (("Line name", name), ("Device role", role), ("Device assignment", device), ("Initial target", initial)):
            form.addRow(title, widget)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if index is not None:
            profile_button = QPushButton("Edit device profile…")
            profile_button.clicked.connect(lambda: (dialog.reject(), self.edit_device_profile(index)))
            form.addRow(profile_button)
            remove = QPushButton("Remove device line")
            form.addRow(remove)
            def delete():
                key = old["id"]
                self.recipe["lanes"].pop(index)
                for stage in self.recipe["stages"]:
                    stage["actions"].pop(key, None)
                dialog.reject()
                self.refresh()
            remove.clicked.connect(delete)
        if dialog.exec():
            key = old.get("id") or role.currentData()
            if not old and key in {lane["id"] for lane in self.recipe["lanes"]}:
                key += "_" + identifier()
            line = {**deepcopy(old), "id": key, "name": name.text().strip() or ROLES[role.currentData()].label,
                    "role": role.currentData(), "device_id": device.currentData(), "initial": initial.value()}
            if index is None:
                self.recipe["lanes"].append(line)
            else:
                self.recipe["lanes"][index] = line
            self.refresh()

    def edit_device_profile(self, index=None):
        # clicked(bool) is not a lane index.
        if isinstance(index, bool):
            index = None
        if index is None:
            if not self.recipe["lanes"]:
                self.notice.setText("Add a device line first.")
                return
            names = [f"{i + 1}. {lane['name']}" for i, lane in enumerate(self.recipe["lanes"])]
            chosen, ok = QInputDialog.getItem(self, "Device profile", "Device line", names, 0, False)
            if not ok:
                return
            index = names.index(chosen)
        from gui.device_profile_dialog import DeviceProfileDialog
        lane = self.recipe["lanes"][index]
        dialog = DeviceProfileDialog(lane, self.recipe["stages"], self, time_unit=self.time_unit)
        if dialog.exec():
            if dialog.enabled.isChecked():
                lane["profile"] = dialog.sections
                for stage in self.recipe["stages"]:
                    stage["actions"].pop(lane["id"], None)
            else:
                lane.pop("profile", None)
            self.refresh()

    def cell(self, point):
        return int((point.y() - HEADER_HEIGHT) // ROW_HEIGHT), int((point.x() - LABEL_WIDTH) // COLUMN_WIDTH)

    def drop_action(self, kind, point):
        row, column = self.cell(point)
        if not 0 <= row < len(self.recipe["stages"]) or not 0 <= column < len(self.recipe["lanes"]):
            self.notice.setText("Drop the action into an existing stage and device line.")
            return
        stage, lane = self.recipe["stages"][row], self.recipe["lanes"][column]
        if lane.get("profile"):
            self.notice.setText("This line uses its own profile. Edit device profile to change its sections, or disable it to use shared stages.")
            return
        if lane["id"] in stage["actions"]:
            self.notice.setText("This device already has an action in that stage. Double-click it to edit.")
            return
        if (kind in {"Open valve", "Close valve"} and lane["role"] != "valve_state"
                or kind == "Ramp" and lane["role"] == "valve_state"
                or kind == "Mass addition" and ROLES[lane["role"]].unit != "g/h"):
            self.notice.setText("Choose a compatible action for this device line.")
            return
        action = {"kind": "Dose" if kind == "Mass addition" else "Ramp" if kind == "Ramp" else "Hold",
                  "name": kind, "start": parameter(lane["initial"]),
                  "end": parameter(1. if kind == "Open valve" else 0. if kind == "Close valve" else lane["initial"]),
                  "control_mode": "Process"}
        if lane["role"] == "T":
            action["start"]["equal_previous"] = kind == "Ramp"
            action["end"]["equal_previous"] = kind == "Equilibration"
        stage["actions"][lane["id"]] = action
        self.edit_action(stage["id"], lane["id"], new=True)

    def edit_action(self, stage_id, lane_id, new=False):
        stage = next(stage for stage in self.recipe["stages"] if stage["id"] == stage_id)
        lane = next(lane for lane in self.recipe["lanes"] if lane["id"] == lane_id)
        action = deepcopy(stage["actions"][lane_id])
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{stage['name']} · {lane['name']}")
        dialog.resize(1000, 340)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        name = QLineEdit(action.get("name", action["kind"]))
        form.addRow("Action name", name)
        mode = QComboBox()
        mode.addItem("Internal vessel (Process)", "Process")
        mode.addItem("Jacket", "Jacket")
        mode.setCurrentIndex(max(0, mode.findData(action.get("control_mode"))))
        if lane["role"] == "T":
            form.addRow("Temperature control", mode)
        layout.addLayout(form)
        span = QComboBox()
        current = self.recipe["stages"].index(stage)
        for following in self.recipe["stages"][current:]:
            span.addItem(f"End of {following['name']}", following["id"])
        span.addItem("End of experiment (longest profile)", "__end__")
        span.setCurrentIndex(max(0, span.findData("__end__" if action.get("until_end") else action.get("until_stage", stage_id))))
        form.addRow("Action lasts until", span)
        unit = "g" if action["kind"] == "Dose" else ROLES[lane["role"]].unit
        entries = [(f"End ({unit})", action["end"])]
        if action["kind"] == "Ramp":
            entries.insert(0, (f"Start ({unit})", action["start"]))
        table = ParameterTable(entries, temperature=lane["role"] == "T")
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec():
            table.save()
            action.update(name=name.text().strip(), control_mode=mode.currentData())
            action["until_end"] = span.currentData() == "__end__"
            action["until_stage"] = "" if action["until_end"] else span.currentData()
            stage["actions"][lane_id] = action
        elif new:
            stage["actions"].pop(lane_id)
        self.refresh()

    def move_block(self, stage_id, lane_id, point):
        row, column = self.cell(point)
        if 0 <= row < len(self.recipe["stages"]) and 0 <= column < len(self.recipe["lanes"]):
            source = next(stage for stage in self.recipe["stages"] if stage["id"] == stage_id)
            target, lane = self.recipe["stages"][row], self.recipe["lanes"][column]
            original = next(item for item in self.recipe["lanes"] if item["id"] == lane_id)
            if lane["role"] == original["role"] and lane["id"] not in target["actions"]:
                target["actions"][lane["id"]] = source["actions"].pop(lane_id)
        self.refresh()

    def delete_action(self):
        for block in self.scene.selectedItems():
            if isinstance(block, ActionBlock):
                stage = next(stage for stage in self.recipe["stages"] if stage["id"] == block.stage_id)
                stage["actions"].pop(block.lane_id, None)
        self.refresh()


class ExperimentDefinitionDialog(QDialog):
    def __init__(self, profile, configs, parent=None):
        super().__init__(parent)
        from gui.experiment_profile_dialog import DataclassEditor

        self.setWindowTitle("Experiment Definition")
        self.resize(1230, 800)
        self.profile = profile
        self.draft = deepcopy(profile)
        self.limit_warnings = []
        layout = QVBoxLayout(self)
        self.source = QComboBox()
        self.source.addItems(("Device timeline", "Component design"))
        self.source.setCurrentIndex(0 if profile.timeline else 1)
        source_row = QHBoxLayout()
        source_row.addWidget(self.source, 1)
        source_row.addWidget(QLabel("Timeline time units"))
        self.time_units = QComboBox()
        for label, unit in (("Seconds (s)", "s"), ("Minutes (min)", "min"), ("Hours (h)", "h")):
            self.time_units.addItem(label, unit)
        self.time_units.setCurrentIndex(self.time_units.findData(profile.timeline.get("time_unit", "min")))
        self.time_units.setToolTip("Converts displayed durations and temperature-rate limits together. Existing experiment timing stays the same.")
        source_row.addWidget(self.time_units)
        layout.addLayout(source_row)
        units_help = QLabel("Durations and temperature-rate limits use the selected time unit. Drafts can be saved; rate limits are checked when generating experiments.")
        units_help.setWordWrap(True)
        layout.addWidget(units_help)
        tabs = QTabWidget()
        self.editor = RecipeEditor(profile.timeline, configs, profile.T_room)
        self.editor.changed.connect(lambda: self.source.setCurrentIndex(0))
        recipe_page = QWidget()
        recipe_layout = QVBoxLayout(recipe_page)
        import_button = QPushButton("Import current component profile")
        import_button.clicked.connect(lambda: self.import_components(configs))
        recipe_layout.addWidget(import_button)
        recipe_layout.addWidget(self.editor, 1)
        tabs.addTab(recipe_page, "Device timeline")
        system = QWidget()
        form = QFormLayout(system)
        self.system = {}
        self.system_labels = {}
        self.rate_unit = "min"
        self.rate_original = {}
        for key in SYSTEM_FIELDS:
            field = number(getattr(profile, key))
            if key not in {"v_heat_max", "v_cool_max", "AS_add_max", "T_room", "sol_add_max"}:
                field.setToolTip("Component-design default. In the device timeline, add explicit waiting or heating stages.")
            self.system[key] = field
            labels = {"v_heat_max": "Maximum heating rate (°C/min)", "v_cool_max": "Maximum cooling rate (°C/min)",
                      "AS_add_max": "Maximum antisolvent rate (g/min)", "T_room": "Room temperature (°C)",
                      "sol_add_max": "Maximum solution/solvent rate (g/min)",
                      "time_to_wait_prior_exp": "Wait before experiment (h)", "aceton_boil_out_T": "Boil-out temperature (°C)",
                      "wait_after_exp": "Wait after experiment (h)", "t_before_open_valve": "Wait before discharge (h)"}
            self.system_labels[key] = QLabel(labels[key])
            form.addRow(self.system_labels[key], field)
            if key in {"v_heat_max", "v_cool_max"}:
                field.setDecimals(9)
                field.setValue(getattr(profile, key))
                self.rate_original[key] = (getattr(profile, key), field.value())
        tabs.addTab(system, "System settings")
        legacy_scroll = QScrollArea()
        legacy_scroll.setWidgetResizable(True)
        self.legacy = DataclassEditor(self.draft)
        legacy_scroll.setWidget(self.legacy)
        tabs.addTab(legacy_scroll, "Component design")
        self.source.currentIndexChanged.connect(lambda index: tabs.setCurrentIndex(0 if index == 0 else 2))
        tabs.setCurrentIndex(0 if profile.timeline else 2)
        layout.addWidget(tabs, 1)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #ef4444")
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.time_units.currentIndexChanged.connect(self.update_time_units)
        self.update_time_units()

    def rate_value(self, key):
        stored, displayed = self.rate_original[key]
        value = self.system[key].value()
        return stored if value == displayed else value * 60 / TIME_UNITS[self.rate_unit]

    def update_time_units(self):
        values = {key: self.rate_value(key) for key in ("v_heat_max", "v_cool_max")}
        self.rate_unit = self.time_units.currentData()
        for key, title in (("v_heat_max", "Maximum heating rate"), ("v_cool_max", "Maximum cooling rate")):
            field = self.system[key]
            field.setValue(values[key] * TIME_UNITS[self.rate_unit] / 60)
            self.rate_original[key] = (values[key], field.value())
            self.system_labels[key].setText(f"{title} (°C/{self.rate_unit})")
            field.setToolTip(f"{values[key]:g} °C/min. This display converts with Timeline time units.")
        self.editor.recipe["time_unit"] = self.rate_unit
        blocked = self.editor.blockSignals(True)
        try:
            self.editor.refresh()
        finally:
            self.editor.blockSignals(blocked)

    def import_components(self, configs):
        from experiment.experiment_service import ExperimentService
        from experiment.recipe import recipe_from_frame

        try:
            self.legacy.save()
            for key, field in self.system.items():
                if key in self.rate_original:
                    value = getattr(self.draft, key)
                    field.setValue(value * TIME_UNITS[self.rate_unit] / 60)
                    self.rate_original[key] = (value, field.value())
                else:
                    field.setValue(getattr(self.draft, key))
            service = ExperimentService()
            service.exp_par = deepcopy(self.draft)
            service.exp_par.timeline = {}
            creator = service.generate_initial_doe(1, "LHS")
            frame = next(iter(creator.exp_storage.data.values()))
            self.editor.recipe = recipe_from_frame(frame, self.draft, configs)
            self.editor.recipe["time_unit"] = self.rate_unit
            self.editor.refresh()
            self.error.setText("Imported one component profile. Edit stage purposes to identify charging, experiment, equilibration and cleanup.")
        except (ValueError, TypeError, KeyError) as error:
            self.error.setText(str(error))

    def accept(self):
        try:
            if self.source.currentIndex() == 0:
                self.draft.timeline = deepcopy(validate_recipe(self.editor.recipe))
                for key, field in self.system.items():
                    setattr(self.draft, key, self.rate_value(key) if key in self.rate_original else field.value())
                from experiment.recipe import compile_recipe, recipe_parameters
                self.limit_warnings = []
                compile_recipe(self.draft.timeline, {key: value["value"] for key, value in recipe_parameters(self.draft.timeline)},
                               self.draft, check_limits=False, limit_warnings=self.limit_warnings)
            else:
                self.legacy.save()
                self.draft.timeline = {}
        except (ValueError, KeyError, TypeError) as error:
            self.error.setText(str(error))
            return
        self.profile.__dict__.update(deepcopy(self.draft.__dict__))
        super().accept()
