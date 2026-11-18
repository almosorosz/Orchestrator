"""Drag-and-drop Experiment Definition with device columns and time rows."""
from copy import deepcopy
from uuid import uuid4

from PySide6.QtCore import QByteArray, QMimeData, QPointF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QDrag, QBrush, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QGraphicsItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from experiment.doe_campaign import ROLES
from experiment.recipe import PURPOSES, SYSTEM_FIELDS, parameter, validate_recipe
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
    def __init__(self, entries, parent=None):
        super().__init__(len(entries), 5, parent)
        self.setHorizontalHeaderLabels(("Parameter", "Across experiments", "Value", "Minimum", "Maximum"))
        self.entries = entries
        for row, (label, item) in enumerate(entries):
            title = QTableWidgetItem(label)
            title.setFlags(title.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row, 0, title)
            mode = QComboBox()
            mode.addItems(("Fixed", "DoE", "Alternate"))
            mode.setCurrentText(item["mode"])
            self.setCellWidget(row, 1, mode)
            for column, key in enumerate(("value", "min", "max"), 2):
                self.setCellWidget(row, column, number(item[key]))
        self.resizeColumnsToContents()
        self.horizontalHeader().setStretchLastSection(True)

    def save(self):
        for row, (_, item) in enumerate(self.entries):
            item["mode"] = self.cellWidget(row, 1).currentText()
            for column, key in enumerate(("value", "min", "max"), 2):
                item[key] = self.cellWidget(row, column).value()


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
        detail = f"{value['value']:g} {unit}" if value["mode"] == "Fixed" else f"{value['mode']} {value['min']:g}–{value['max']:g} {unit}"
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
                                ("Remove selected action", self.delete_action)):
            button = QPushButton(label)
            button.setMaximumHeight(32)
            button.clicked.connect(callback)
            side.addWidget(button)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
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
            item = QListWidgetItem(f"{stage['name']} · {stage['duration']['value']:g} s", self.stage_list)
            item.setData(Qt.ItemDataRole.UserRole, stage["id"])
        self.stage_list.setCurrentRow(min(max(active, 0), self.stage_list.count() - 1))
        self.stage_list.blockSignals(False)
        self.scene.clear()
        width = LABEL_WIDTH + COLUMN_WIDTH * len(self.recipe["lanes"])
        for column, lane in enumerate(self.recipe["lanes"]):
            text = self.scene.addText(f"{lane['name']}\n{lane.get('device_id') or 'Assign device'}")
            text.setDefaultTextColor(QColor("#e2e8f0"))
            text.setTextWidth(COLUMN_WIDTH - 10)
            text.setPos(LABEL_WIDTH + column * COLUMN_WIDTH, 0)
            text.setToolTip("Double-click the header to edit the line and its device assignment.")
        time = 0.
        for row, stage in enumerate(self.recipe["stages"]):
            y = HEADER_HEIGHT + row * ROW_HEIGHT
            text = self.scene.addText(f"{time:g} s\n{stage['name']}\n{stage['purpose']}")
            text.setDefaultTextColor(QColor("#cbd5e1"))
            text.setTextWidth(LABEL_WIDTH - 8)
            text.setPos(0, y)
            for column, lane in enumerate(self.recipe["lanes"]):
                self.scene.addRect(LABEL_WIDTH + column * COLUMN_WIDTH, y, COLUMN_WIDTH, ROW_HEIGHT,
                                   QPen(QColor("#24364d")))
                if lane["id"] in stage["actions"]:
                    block = ActionBlock(self, stage, lane)
                    self.scene.addItem(block)
                    block.setPos(LABEL_WIDTH + column * COLUMN_WIDTH + 6, y + 6)
            time += stage["duration"]["value"]
        self.scene.setSceneRect(0, 0, max(width, 700), max(HEADER_HEIGHT + ROW_HEIGHT * len(self.recipe["stages"]), 400))
        self.changed.emit()

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
        table = ParameterTable([("Duration (s)", stage["duration"])])
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
            line = {"id": key, "name": name.text().strip() or ROLES[role.currentData()].label,
                    "role": role.currentData(), "device_id": device.currentData(), "initial": initial.value()}
            if index is None:
                self.recipe["lanes"].append(line)
            else:
                self.recipe["lanes"][index] = line
            self.refresh()

    def cell(self, point):
        return int((point.y() - HEADER_HEIGHT) // ROW_HEIGHT), int((point.x() - LABEL_WIDTH) // COLUMN_WIDTH)

    def drop_action(self, kind, point):
        row, column = self.cell(point)
        if not 0 <= row < len(self.recipe["stages"]) or not 0 <= column < len(self.recipe["lanes"]):
            self.notice.setText("Drop the action into an existing stage and device line.")
            return
        stage, lane = self.recipe["stages"][row], self.recipe["lanes"][column]
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
        stage["actions"][lane["id"]] = action
        self.edit_action(stage["id"], lane["id"], new=True)

    def edit_action(self, stage_id, lane_id, new=False):
        stage = next(stage for stage in self.recipe["stages"] if stage["id"] == stage_id)
        lane = next(lane for lane in self.recipe["lanes"] if lane["id"] == lane_id)
        action = deepcopy(stage["actions"][lane_id])
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{stage['name']} · {lane['name']}")
        dialog.resize(760, 270)
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
        unit = "g" if action["kind"] == "Dose" else ROLES[lane["role"]].unit
        entries = [(f"End ({unit})", action["end"])]
        if action["kind"] == "Ramp":
            entries.insert(0, (f"Start ({unit})", action["start"]))
        table = ParameterTable(entries)
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec():
            table.save()
            action.update(name=name.text().strip(), control_mode=mode.currentData())
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
        layout = QVBoxLayout(self)
        self.source = QComboBox()
        self.source.addItems(("Device timeline", "Component design"))
        self.source.setCurrentIndex(0 if profile.timeline else 1)
        layout.addWidget(self.source)
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
            form.addRow(labels[key], field)
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

    def import_components(self, configs):
        from experiment.experiment_service import ExperimentService
        from experiment.recipe import recipe_from_frame

        try:
            self.legacy.save()
            for key, field in self.system.items():
                field.setValue(getattr(self.draft, key))
            service = ExperimentService()
            service.exp_par = deepcopy(self.draft)
            service.exp_par.timeline = {}
            creator = service.generate_initial_doe(1, "LHS")
            frame = next(iter(creator.exp_storage.data.values()))
            self.editor.recipe = recipe_from_frame(frame, self.draft, configs)
            self.editor.refresh()
            self.error.setText("Imported one component profile. Edit stage purposes to identify charging, experiment, equilibration and cleanup.")
        except (ValueError, TypeError, KeyError) as error:
            self.error.setText(str(error))

    def accept(self):
        try:
            if self.source.currentIndex() == 0:
                self.draft.timeline = deepcopy(validate_recipe(self.editor.recipe))
                for key, field in self.system.items():
                    setattr(self.draft, key, field.value())
                from experiment.recipe import compile_recipe, recipe_parameters
                compile_recipe(self.draft.timeline, {key: value["value"] for key, value in recipe_parameters(self.draft.timeline)}, self.draft)
            else:
                self.legacy.save()
                self.draft.timeline = {}
        except (ValueError, KeyError, TypeError) as error:
            self.error.setText(str(error))
            return
        self.profile.__dict__.update(deepcopy(self.draft.__dict__))
        super().accept()
