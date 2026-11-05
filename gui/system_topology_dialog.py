"""Define modules/unit operations and generate MSI plus an offline graph."""
from copy import deepcopy
from pathlib import Path
import json
import html
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QWidget, QPlainTextEdit, QFileDialog, QApplication, QCheckBox)
from experiment.system_topology import validate_topology, system_graph
from gui.system_graph_view import SystemGraphView
from gui.flow_layout import FlowLayout


class SystemTopologyDialog(QDialog):
    def __init__(self, definition, configs, parent=None, generate=False, initial_tab=0):
        super().__init__(parent)
        self.setWindowTitle("Unit operations, modules and MSI")
        self.resize(1120, 820)
        self.configs = deepcopy(configs)
        self.definition = deepcopy(definition)
        self.graph_data = None
        layout = QVBoxLayout(self)
        help_text = QLabel("Define modules and physical unit operations here. Use the device cards in Control Panel to assign "
            "pumps' positive/clockwise source and target, or a device's connected unit and module. "
            "A pump may draw from or deliver to containers in other modules. "
            "Devices not included in the system definition are omitted from MSI. Types and names are user-defined.")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.tabs = QTabWidget()
        editor = QWidget()
        rows = QVBoxLayout(editor)
        rows.addWidget(QLabel("Modules"))
        self.modules = QTableWidget(0, 2)
        self.modules.setHorizontalHeaderLabels(["Identifier code", "Name"])
        self.modules.horizontalHeader().setStretchLastSection(True)
        rows.addWidget(self.modules)
        module_actions = QHBoxLayout()
        add_module, remove_module = QPushButton("Add module"), QPushButton("Remove selected module")
        module_actions.addWidget(add_module)
        module_actions.addWidget(remove_module)
        rows.addLayout(module_actions)
        rows.addWidget(QLabel("Unit operations — containers, crystallizers, distillers, MSMPR stages, or any other type"))
        self.units = QTableWidget(0, 4)
        self.units.setHorizontalHeaderLabels(["Identifier code", "Name", "Type (any text)", "Module code"])
        self.units.horizontalHeader().setStretchLastSection(True)
        rows.addWidget(self.units)
        unit_actions = QHBoxLayout()
        add_unit, remove_unit = QPushButton("Add unit operation"), QPushButton("Remove selected unit")
        unit_actions.addWidget(add_unit)
        unit_actions.addWidget(remove_unit)
        rows.addLayout(unit_actions)
        self.tabs.addTab(editor, "Physical setup")
        sensor_editor = QWidget()
        sensor_layout = QVBoxLayout(sensor_editor)
        sensor_help = QLabel("Define sensors belonging to a module, rather than an individual unit operation. "
            "Enter identifier codes without the S_ prefix; use unique codes across modules. "
            "These definitions generate the S_…[[…]] MSI section and graph nodes. "
            "They do not create hardware connections or measured values; configure OPC/read-only devices separately.")
        sensor_help.setWordWrap(True)
        sensor_layout.addWidget(sensor_help)
        self.sensors = QTableWidget(0, 3)
        self.sensors.setHorizontalHeaderLabels(["Sensor code (without S_)", "Name", "Module code"])
        self.sensors.horizontalHeader().setStretchLastSection(True)
        sensor_layout.addWidget(self.sensors)
        sensor_actions = QHBoxLayout()
        add_sensor = QPushButton("Add module sensor")
        remove_sensor = QPushButton("Remove selected sensor")
        sensor_actions.addWidget(add_sensor)
        sensor_actions.addWidget(remove_sensor)
        sensor_layout.addLayout(sensor_actions)
        self.tabs.addTab(sensor_editor, "Module sensors")
        output = QWidget()
        output_layout = QVBoxLayout(output)
        self.code = QPlainTextEdit()
        self.code.setReadOnly(True)
        self.code.setMaximumHeight(72)
        self.code.setVisible(False)
        output_layout.addWidget(self.code)
        legend = QLabel("Squares: equipment (module color) · red circles: read-only · yellow circles: control · triangles: module sensors. "
                        "Drag a node to rearrange it. Drag empty space to pan. Scroll to zoom.")
        legend.setToolTip("Same PyVis renderer as the notebook. Edges use the source module color; grey dashed links indicate stirrer placement.")
        legend.setWordWrap(True)
        output_layout.addWidget(legend)
        self.module_legend = QLabel()
        self.module_legend.setWordWrap(True)
        output_layout.addWidget(self.module_legend)
        self.graph = SystemGraphView()
        output_layout.addWidget(self.graph, 1)
        tools = FlowLayout()
        copy = QPushButton("Copy MSI")
        save = QPushButton("Save MSI + topology JSON…")
        fit = QPushButton("Fit graph")
        arrange = QPushButton("Auto layout")
        labels = QCheckBox("Show labels")
        labels.setChecked(True)
        show_code = QCheckBox("Show MSI code")
        show_code.toggled.connect(self.code.setVisible)
        svg = QPushButton("Export graph PNG…")
        for button in (copy, save, fit, arrange, labels, show_code, svg):
            tools.addWidget(button)
        output_layout.addLayout(tools)
        self.graph_tab_index = self.tabs.addTab(output, "MSI / connectivity graph")
        self.tabs.currentChanged.connect(lambda index: help_text.setVisible(index == 0))
        layout.addWidget(self.tabs, 1)
        self.status = QLabel("Save unit definitions, assign devices, then generate MSI.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        generate_button = QPushButton("Generate MSI + graph")
        apply = QPushButton("Apply definitions")
        cancel = QPushButton("Cancel")
        for button in (generate_button, apply, cancel):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        for module in definition["modules"]:
            self.add_row(self.modules, [module["code"], module.get("name", module["code"])])
            for sensor in module.get('sensors', []):
                self.add_row(self.sensors, [sensor['code'], sensor['name'], module['code']])
        for unit in definition["unit_operations"]:
            self.add_row(self.units, [unit["code"], unit["name"], unit["kind"], unit["module"]])
        add_module.clicked.connect(lambda: self.add_row(self.modules, [self.next_code(self.modules, "M"), "Module"]))
        add_unit.clicked.connect(lambda: self.add_row(self.units, [self.next_code(self.units, "U"), "Unit operation", "Container", self.text(self.modules, 0, 0)]))
        remove_module.clicked.connect(lambda: self.remove_rows(self.modules))
        remove_unit.clicked.connect(lambda: self.remove_rows(self.units))
        self.modules.itemChanged.connect(self.invalidate)
        self.units.itemChanged.connect(self.invalidate)
        add_sensor.clicked.connect(lambda: self.add_row(self.sensors, [self.next_code(self.sensors, 'sensor'), 'Module sensor', self.text(self.modules, 0, 0)]))
        remove_sensor.clicked.connect(lambda: self.remove_rows(self.sensors))
        self.sensors.itemChanged.connect(self.invalidate)
        generate_button.clicked.connect(self.generate)
        apply.clicked.connect(self.apply)
        cancel.clicked.connect(self.reject)
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.code.toPlainText()))
        fit.clicked.connect(self.graph.fit_graph)
        arrange.clicked.connect(self.graph.auto_layout)
        labels.toggled.connect(self.graph.show_labels)
        save.clicked.connect(self.save)
        svg.clicked.connect(self.export_svg)
        self.tabs.setCurrentIndex(initial_tab)
        if generate:
            QTimer.singleShot(0, self.generate)

    @staticmethod
    def text(table, row, column):
        item = table.item(row, column)
        return item.text().strip() if item else ""

    @staticmethod
    def add_row(table, values):
        row = table.rowCount()
        table.insertRow(row)
        for column, value in enumerate(values):
            table.setItem(row, column, QTableWidgetItem(value))

    def next_code(self, table, prefix):
        existing = {self.text(table, row, 0) for row in range(table.rowCount())}
        number = 1
        while f"{prefix}{number}" in existing:
            number += 1
        return f"{prefix}{number}"

    def remove_rows(self, table):
        for row in sorted({item.row() for item in table.selectedItems()}, reverse=True):
            table.removeRow(row)
        self.invalidate()

    def invalidate(self, *args):
        self.code.clear()
        self.graph.clear()
        self.module_legend.clear()
        self.graph_data = None
        self.status.setText("Definitions changed. Generate MSI again after completing device assignments.")

    def read_definition(self):
        definition = {"version": 1, "modules": [], "unit_operations": []}
        for row in range(self.modules.rowCount()):
            code = self.text(self.modules, row, 0)
            if code or self.text(self.modules, row, 1):
                definition["modules"].append({"code": code, "name": self.text(self.modules, row, 1) or code})
        for row in range(self.units.rowCount()):
            values = [self.text(self.units, row, col) for col in range(4)]
            if any(values):
                definition["unit_operations"].append(dict(zip(("code", "name", "kind", "module"), values)))
        by_module = {module['code']: module for module in definition['modules']}
        for row in range(self.sensors.rowCount()):
            code, name, module = [self.text(self.sensors, row, col) for col in range(3)]
            if not any((code, name, module)):
                continue
            if module not in by_module:
                raise ValueError(f'Sensor {code}: choose an existing module code')
            by_module[module].setdefault('sensors', []).append({'code': code, 'name': name})
        validate_topology(definition, self.configs, complete=False)
        return definition

    def generate(self):
        try:
            self.definition = self.read_definition()
            code, graph = system_graph(self.definition, self.configs)
            self.definition["msi_code"] = code
            self.graph_data = graph
            self.code.setPlainText(code)
            self.tabs.setCurrentIndex(self.graph_tab_index)
            self.graph.draw_graph(graph, code)
            self.module_legend.setText(' · '.join(
                f'<span style="color:{self.graph.module_colors[module["id"]]}">■ {html.escape(module.get("label", module["id"]))} '
                f'({html.escape(module["id"])})</span>' for module in graph['modules']))
            included = sum(c.get("connectivity", {}).get("role", "excluded") != "excluded" for c in self.configs.values())
            self.status.setText(f"MSI generated: {len(self.definition['modules'])} modules, {len(self.definition['unit_operations'])} unit operations, "
                                f"{included} included devices. Positive pump flow is source → pump → target.")
        except (ValueError, KeyError, TypeError) as error:
            self.invalidate()
            self.status.setText(f"Definition incomplete: {error}")

    def apply(self):
        try:
            definition = self.read_definition()
            try:
                definition["msi_code"] = system_graph(definition, self.configs)[0]
            except ValueError:
                pass  # Save a draft while the device cards are being assigned.
            self.definition = definition
            self.accept()
        except (ValueError, KeyError, TypeError) as error:
            self.status.setText(str(error))

    def save(self):
        if self.graph_data is None:
            self.generate()
        if self.graph_data is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save MSI", "system.msi.txt", "Text (*.txt)")
        if filename:
            try:
                path = Path(filename)
                path.write_text(self.code.toPlainText(), encoding="utf-8")
                devices = {key: {field: config[field] for field in ("name", "type", "connectivity") if field in config}
                           for key, config in self.configs.items()}
                path.with_suffix(".json").write_text(json.dumps({"system_topology": self.definition, "devices": devices}, indent=2), encoding="utf-8")
                self.status.setText(f"Saved MSI and topology JSON to {path.parent}")
            except OSError as error:
                self.status.setText(str(error))

    def export_svg(self):
        if self.graph_data is None:
            self.generate()
        if self.graph_data is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save connectivity graph", "system.png", "PNG (*.png)")
        if filename:
            if not self.graph.save_image(filename):
                self.status.setText(f"Could not save graph image: {filename}")
