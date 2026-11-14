"""Physical placement fields embedded in each Control Panel device card."""
from copy import deepcopy
from PySide6.QtWidgets import QWidget, QFormLayout, QComboBox
from experiment.system_topology import PUMPS, READ_ONLY


class DeviceConnectivity(QWidget):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.definition = None
        self.loading = True
        form = QFormLayout(self)
        form.setContentsMargins(0, 5, 0, 5)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setHorizontalSpacing(5)
        form.setVerticalSpacing(3)
        self.role = QComboBox()
        self.role.addItem("Not included in system definition", "excluded")
        if config["type"] not in READ_ONLY:
            self.role.addItem("Controlled flow" if config["type"] in PUMPS else "Controlled device", "controlled")
        if config["type"] not in PUMPS:
            self.role.addItem("Read-only probe", "read_only")
        self.module = QComboBox()
        self.unit = QComboBox()
        self.source = QComboBox()
        self.target = QComboBox()
        form.addRow("System role:", self.role)
        form.addRow("Pump module:" if config["type"] in PUMPS else "Module:", self.module)
        if config["type"] in PUMPS:
            self.module.setToolTip("Module containing this pump. Source and target containers may belong to other modules.")
            form.addRow("From unit (+):", self.source)
            form.addRow("To unit (+):", self.target)
            self.source.setToolTip("Source container for positive / clockwise flow; may belong to any module.")
            self.target.setToolTip("Target container for positive / clockwise flow; may belong to any module.")
        else:
            form.addRow("Connected unit:", self.unit)
        for combo in (self.role, self.module, self.unit, self.source, self.target):
            combo.currentIndexChanged.connect(self.changed)
        self.role.setCurrentIndex(max(0, self.role.findData(config.get("connectivity", {}).get("role", "excluded"))))
        self.loading = False

    def set_definition(self, definition):
        if definition == self.definition:
            return
        self.loading = True
        self.definition = deepcopy(definition)
        saved = self.config.get("connectivity", {})
        for combo, key, rows in ((self.module, "module", definition["modules"]),
                                  (self.unit, "unit", definition["unit_operations"]),
                                  (self.source, "source", definition["unit_operations"]),
                                  (self.target, "target", definition["unit_operations"])):
            combo.clear()
            combo.addItem("Select…", "")
            for row in rows:
                suffix = f" [module {row['module']}]" if "module" in row else ""
                combo.addItem(f"{row['code']}: {row.get('name', row['code'])}{suffix}", row["code"])
            value = saved.get(key, "")
            if value and combo.findData(value) < 0:
                combo.addItem(f"{value} [missing — reassign]", value)
            combo.setCurrentIndex(max(0, combo.findData(value)))
        self.loading = False

    def changed(self, *args):
        if self.loading:
            return
        panel = self.parent()
        while panel is not None and not hasattr(panel, "hardware_manager"):
            panel = panel.parent()
        if panel is not None and (panel.hardware_manager.is_busy or panel.experiment_window_attached()):
            saved = self.config.get("connectivity", {})
            self.loading = True
            for combo, key in ((self.role, "role"), (self.module, "module"), (self.unit, "unit"),
                               (self.source, "source"), (self.target, "target")):
                combo.setCurrentIndex(max(0, combo.findData(saved.get(key, "excluded" if key == "role" else ""))))
            self.loading = False
            self.setToolTip("End the experiment session / connection operation before changing the physical setup.")
            return
        unit = self.source.currentData() if self.config["type"] in PUMPS else self.unit.currentData()
        # Default an unassigned pump to the source module, but preserve an
        # explicit owning module when its source container changes.
        infer_module = self.config["type"] not in PUMPS or not self.module.currentData()
        if self.sender() in (self.source, self.unit) and self.definition and infer_module:
            module = next((u["module"] for u in self.definition["unit_operations"] if u["code"] == unit), "")
            self.module.blockSignals(True)
            self.module.setCurrentIndex(max(0, self.module.findData(module)))
            self.module.blockSignals(False)
        self.config["connectivity"] = {"role": self.role.currentData(), "module": self.module.currentData() or "",
            "unit": self.unit.currentData() or "", "source": self.source.currentData() or "", "target": self.target.currentData() or ""}
