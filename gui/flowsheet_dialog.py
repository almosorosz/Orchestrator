"""Read-only flowsheet generated from the Control Panel's current MSI graph."""

from collections import defaultdict

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QGridLayout, QHeaderView, QLabel, QPushButton, QScrollArea, QSlider, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from experiment.pump_calibration_files import delivered_flow, validate_calibration, validate_seed_calibration
from experiment.pump_curve_fits import seed_exponential_mass


def plain_label(text=""):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


class FlowsheetDialog(QDialog):
    def __init__(self, graph, configs, parent=None, *, msi_code):
        super().__init__(parent)
        self.setWindowTitle("System flowsheet")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        available = self.screen().availableGeometry()
        self.resize(min(930, available.width() - 48), min(630, available.height() - 48))
        self.nodes = {node["id"]: node for node in graph["nodes"]}
        self.modules = {module["id"]: module["label"] for module in graph["modules"]}
        self.configs = configs
        self.routes = {}
        incoming, outgoing = {}, {}
        for edge in graph["edges"]:
            if edge["kind"] == "flow":
                incoming[edge["target"]] = edge["source"]
                outgoing[edge["source"]] = edge["target"]
        for node in self.nodes.values():
            if node.get("equipment") == "pump":
                key = node["id"]
                self.routes[node["device_id"]] = (incoming[key], outgoing[key])

        self.tabs = QTabWidget()
        self.connectivity_page = QWidget()
        self.connectivity_view = None
        connectivity_layout = QVBoxLayout(self.connectivity_page)
        connectivity_layout.addWidget(plain_label("Drag equipment or sensors to rearrange them. Scroll to zoom."))
        if self.routes:
            self.tabs.addTab(self.material_tab(), "Material flow")
        else:
            self.tabs.addTab(self.connectivity_page, "Connectivity")
        self.tabs.addTab(self.instrument_tab(graph["edges"]), "Instrumentation")
        if self.routes:
            self.tabs.addTab(self.calibration_tab(), "Pump calibration")
            self.tabs.addTab(self.connectivity_page, "Connectivity")
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        if self.routes:
            self.select_pump(self.pump_picker.currentData())

        def load_connectivity():
            if self.tabs.currentWidget() is self.connectivity_page and self.connectivity_view is None:
                from gui.system_graph_view import SystemGraphView

                self.connectivity_view = SystemGraphView(self.connectivity_page)
                connectivity_layout.addWidget(self.connectivity_view, 1)
                self.connectivity_view.draw_graph(graph, msi_code)

        self.tabs.currentChanged.connect(load_connectivity)
        load_connectivity()

    def unit_label(self, key, *, multiline=False):
        node = self.nodes[key]
        separator = "\n" if multiline else " · "
        return separator.join((*node["label"].splitlines(), self.modules[node["module"]]))

    def material_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        canvas = QWidget()
        diagram = QGridLayout(canvas)
        diagram.setAlignment(Qt.AlignmentFlag.AlignTop)
        diagram.setHorizontalSpacing(12)
        diagram.setVerticalSpacing(8)
        diagram.setColumnStretch(0, 2)
        diagram.setColumnStretch(2, 2)
        diagram.setColumnStretch(4, 2)
        self.route_buttons = {}
        self.route_group = QButtonGroup(self)
        by_target = defaultdict(list)
        for device, (source, target) in self.routes.items():
            by_target[target].append((device, source))

        if by_target:
            for column, title in ((0, "Source"), (2, "Pump"), (4, "Destination")):
                heading = plain_label(title)
                font = heading.font()
                font.setBold(True)
                heading.setFont(font)
                diagram.addWidget(heading, 0, column)

        row = 1
        for target, feeds in by_target.items():
            destination = plain_label(self.unit_label(target, multiline=True))
            destination.setAlignment(Qt.AlignmentFlag.AlignCenter)
            destination.setMargin(12)
            destination.setStyleSheet("border: 1px solid palette(mid); border-radius: 8px;")
            diagram.addWidget(destination, row, 4, len(feeds), 1)
            for device, source in feeds:
                diagram.addWidget(plain_label(self.unit_label(source, multiline=True)), row, 0)
                button = QPushButton(device)
                button.setToolTip(self.configs[device]["name"])
                button.setAccessibleName(f"Inspect {self.configs[device]['name']} ({device})")
                button.setCheckable(True)
                button.setStyleSheet("QPushButton:checked { background: #2563eb; color: white; }")
                button.clicked.connect(lambda checked, device=device: self.select_pump(device))
                self.route_group.addButton(button)
                self.route_buttons[device] = button
                diagram.addWidget(plain_label("⟶"), row, 1)
                diagram.addWidget(button, row, 2)
                diagram.addWidget(plain_label("⟶"), row, 3)
                row += 1

        connected = {unit for route in self.routes.values() for unit in route}
        isolated = [self.unit_label(key) for key, node in self.nodes.items()
                    if node["role"] == "unit" and not node.get("equipment") and key not in connected]
        if isolated:
            diagram.addWidget(plain_label("Units without material-flow links\n" + "\n".join(isolated)),
                              row, 0, 1, 5)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(canvas)
        layout.addWidget(scroll, 1)
        self.route_detail = plain_label()
        self.inspect = QPushButton("Inspect pump calibration")
        self.inspect.clicked.connect(lambda: self.tabs.setCurrentWidget(self.calibration_page))
        layout.addWidget(self.route_detail)
        layout.addWidget(self.inspect, alignment=Qt.AlignmentFlag.AlignRight)
        return page

    def instrument_tab(self, edges):
        attachments = {edge["source"]: edge["target"] for edge in edges if edge["kind"] == "attachment"}
        roles = {"controlled": "Control", "read_only": "Measurement", "sensor": "Module sensor"}
        rows = []
        for edge in edges:
            kind = edge["kind"]
            if kind not in roles:
                continue
            source, target = edge["source"], edge["target"]
            instrument, attached = (source, target) if kind == "controlled" else (target, source)
            channel = self.nodes[instrument]["label"].replace("\n", " · ")
            if kind == "sensor":
                location = self.modules[self.nodes[attached]["module"]]
            else:
                location = self.unit_label(attachments.get(attached, attached))
            rows.append((channel, roles[kind], location))
        page = QWidget()
        layout = QVBoxLayout(page)
        if not rows:
            layout.addWidget(plain_label("No instruments are mapped in the current MSI."))
            layout.addStretch()
            return page
        table = QTableWidget(len(rows), 3)
        table.setHorizontalHeaderLabels(["Signal or device", "Role", "Connected to"])
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().hide()
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                table.setItem(row, column, item)
        table.setSortingEnabled(True)
        layout.addWidget(table)
        return page

    def calibration_tab(self):
        self.calibration_page = QWidget()
        layout = QVBoxLayout(self.calibration_page)
        form = QFormLayout()
        self.pump_picker = QComboBox()
        for device in self.routes:
            self.pump_picker.addItem(f"{self.configs[device]['name']} · {device}", device)
        self.calibration_kind = QComboBox()
        self.command = QDoubleSpinBox()
        self.command.setDecimals(6)
        self.command_label = plain_label("Pump setting")
        form.addRow("Pump", self.pump_picker)
        form.addRow("Calibration", self.calibration_kind)
        form.addRow(self.command_label, self.command)
        layout.addLayout(form)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setAccessibleName("Explore the calibrated range")
        layout.addWidget(self.slider)
        self.estimate = plain_label()
        self.calibration_detail = plain_label()
        layout.addWidget(self.estimate)
        layout.addWidget(self.calibration_detail)
        layout.addStretch()
        self.pump_picker.currentIndexChanged.connect(lambda: self.select_pump(self.pump_picker.currentData()))
        self.calibration_kind.currentIndexChanged.connect(self.load_calibration)
        self.command.valueChanged.connect(self.update_estimate)
        self.slider.valueChanged.connect(self.slider_changed)
        return self.calibration_page

    def select_pump(self, device):
        if device is not None:
            source, target = self.routes[device]
            self.route_buttons[device].setChecked(True)
            source_name = self.nodes[source]["label"].splitlines()[0]
            target_name = self.nodes[target]["label"].splitlines()[0]
            self.route_detail.setText(f"{source_name} → {self.configs[device]['name']} ({device}) → {target_name}")
        self.inspect.setEnabled(device is not None)
        self.route_detail.setVisible(device is not None)
        self.pump_picker.setEnabled(device is not None)
        with QSignalBlocker(self.pump_picker), QSignalBlocker(self.calibration_kind):
            self.pump_picker.setCurrentIndex(self.pump_picker.findData(device) if device is not None else -1)
            self.calibration_kind.clear()
            config = self.configs.get(device, {})
            for title, key in (("Liquid delivery", "pump_calibration"), ("Dry seed delivery", "seed_calibration")):
                if config.get(key):
                    self.calibration_kind.addItem(title, key)
        self.load_calibration()

    def load_calibration(self):
        device, kind = self.pump_picker.currentData(), self.calibration_kind.currentData()
        self.calibration_kind.setEnabled(self.calibration_kind.count() > 1)
        self.command.setEnabled(False)
        self.slider.setEnabled(False)
        self.estimate.clear()
        self.command_label.setText("Pumping time" if kind == "seed_calibration" else "Pump setting")
        self.command.setSuffix("")
        with QSignalBlocker(self.command), QSignalBlocker(self.slider):
            self.command.setValue(0)
            self.slider.setValue(0)
        if not kind:
            message = "No calibration is saved for this pump. Use Pump calibration files in the Control Panel."
            self.calibration_detail.setText(message if device else "No pumps are mapped in the current MSI.")
            return
        config = self.configs[device]
        calibration = config[kind]
        seed = kind == "seed_calibration"
        try:
            points = (validate_seed_calibration(calibration, config) if seed
                      else validate_calibration(calibration, config["type"]))
        except (ValueError, TypeError, KeyError) as error:
            self.calibration_detail.setText(f"Review the saved calibration for {device}. {error}")
            return
        self.calibration_points = points
        self.measured_end = float(points[-1, 0 if seed else 1])
        unit = "s" if seed else calibration["output_unit"]
        extrapolates = seed and "seed_fit" in calibration
        self.command.setSuffix(f" {unit}")
        with QSignalBlocker(self.command):
            self.command.setRange(0., max(self.measured_end, 1e6) if extrapolates else self.measured_end)
            self.command.setValue(self.measured_end if seed else float(config.get("default_target") or 0.))
        details = [f"Calibrated range 0–{self.measured_end:g} {unit}"]
        if seed:
            delay = points[points[:, 1] == 0, 0][-1]
            details.extend((f"{calibration['rpm']:g} RPM", f"{delay:g} s empty-tube delay"))
        if calibration.get("source_name"):
            details.append(calibration["source_name"])
        if extrapolates:
            details.append("Enter a longer time to explore extrapolation.")
        self.calibration_detail.setText(" · ".join(details))
        self.command.setEnabled(True)
        self.slider.setEnabled(True)
        self.update_estimate(self.command.value())

    def slider_changed(self, position):
        self.command.setValue(self.measured_end * position / self.slider.maximum())

    def update_estimate(self, value):
        if not self.command.isEnabled():
            return
        config = self.configs[self.pump_picker.currentData()]
        maximum = self.slider.maximum()
        with QSignalBlocker(self.slider):
            self.slider.setValue(round(maximum * min(1., value / self.measured_end)))
        try:
            if self.calibration_kind.currentData() == "seed_calibration":
                fit = config["seed_calibration"].get("seed_fit")
                if fit:
                    mass = float(seed_exponential_mass(fit, value))
                else:
                    time, grams = self.calibration_points.T
                    mass = np.interp(value, time, grams)
                note = " · extrapolated" if value > self.measured_end + 1e-6 else ""
                self.estimate.setText(f"{mass:.3f} g dry seed estimated{note}")
            else:
                flow = float(delivered_flow(config, min(value, self.measured_end))) / 60.
                self.estimate.setText(f"{flow:.3f} g/min estimated")
        except ValueError as error:
            self.estimate.setText(str(error))
