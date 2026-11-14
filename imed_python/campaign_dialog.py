"""Map and review a CrystalLoop recording before importing it into IMED."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QLabel, QListWidget, QListWidgetItem, QMessageBox, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

from .campaign_import import CHANNELS, channel_options, import_run


PROCESS_FIELDS = {
    "SolvMass": "Initial solvent mass (g)", "AntiSolvMass": "Initial antisolvent mass (g)",
    "SolventDensity": "Solvent density (g/cm³)", "AntiSolventDensity": "Antisolvent density (g/cm³)",
    "SolidDensity": "Crystal density (g/cm³)", "ShapeFactor": "Crystal shape factor",
    "SeedMass": "Seed mass (g)", "PowerRatio": "Power ratio",
}


class CampaignImportDialog(QDialog):
    def __init__(self, runs, case, parent):
        super().__init__(parent)
        self.setWindowTitle("IMED | Import CrystalLoop campaign")
        self.resize(650, 700)
        self.runs = runs
        self.experiments = []
        self.processes = []
        form = QFormLayout()
        layout = QVBoxLayout(self)
        notice = QLabel("Select completed runs and map recorded channels. "
                        "Use the crystallization window to exclude filling or cleaning.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.runs_list = QListWidget()
        for _, name, file in runs:
            item = QListWidgetItem(name, self.runs_list)
            item.setToolTip(str(file))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
        self.runs_list.setMaximumHeight(100)
        layout.addWidget(self.runs_list)

        columns, defaults = channel_options(runs)
        self.channels = {}
        for role in CHANNELS:
            box = QComboBox()
            box.addItem("— None —", "")
            for column in columns:
                box.addItem(column, column)
            box.setCurrentIndex(max(0, box.findData(defaults[role])))
            self.channels[role] = box
            form.addRow(role, box)
        self.multiplier = self.number(0.000001, 1000, 1)
        form.addRow("Concentration multiplier to g/g solution", self.multiplier)
        self.no_feed = QCheckBox()
        form.addRow("This campaign has no antisolvent feed", self.no_feed)
        self.direction = QComboBox()
        self.direction.addItem("Reservoir weight decreases", -1)
        self.direction.addItem("Added weight increases", 1)
        form.addRow("AS scale direction", self.direction)
        self.smoothing = QSpinBox()
        self.smoothing.setRange(1, 101)
        self.smoothing.setSingleStep(2)
        self.smoothing.setValue(5)
        form.addRow("Scale median window (samples)", self.smoothing)
        self.start = self.number(0, 1e9)
        self.end = self.number(0, 1e9)
        form.addRow("Start elapsed time (s)", self.start)
        form.addRow("End elapsed time (s, 0 = last sample)", self.end)
        self.initial_concentration = self.number(-1, 0.999999, -1)
        form.addRow("Initial concentration override (−1 = recorded at start)", self.initial_concentration)

        template = next(iter(case.get("ProcInf", [])), {})
        self.process = {}
        for key, label in PROCESS_FIELDS.items():
            value = template.get(key, 1 if key == "PowerRatio" else 0)
            self.process[key] = self.number(0, 1e9, float(value))
            form.addRow(label, self.process[key])
        existing = next(iter(case.get("ExpData", [])), {})
        seed = np.asarray(existing.get("InitialD10D50D90", [])).reshape(-1)
        self.seed = [self.number(0, 1e6, seed[i] if seed.size == 3 else 0) for i in range(3)]
        for label, box in zip(("Seed D10", "Seed D50", "Seed D90"), self.seed):
            form.addRow(label + " (µm, volume percentile)", box)

        panel = QWidget()
        panel.setLayout(form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        layout.addWidget(scroll)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                  | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import selected runs")
        buttons.accepted.connect(self.import_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def number(low, high, value=0):
        box = QDoubleSpinBox()
        box.setDecimals(6)
        box.setRange(low, high)
        box.setValue(value)
        return box

    def import_selection(self):
        mapping = {role: box.currentData() for role, box in self.channels.items()}
        process = {key: box.value() for key, box in self.process.items()}
        seed = tuple(box.value() for box in self.seed) if process["SeedMass"] else None
        if not mapping["Temperature"] or (not mapping["AS scale"] and not self.no_feed.isChecked()):
            QMessageBox.warning(self, "Channel mapping", "Map temperature and the AS scale, or select no AS feed.")
            return
        if any(process[key] <= 0 for key in
               ("SolvMass", "SolventDensity", "AntiSolventDensity", "SolidDensity", "ShapeFactor")):
            QMessageBox.warning(self, "Process inputs", "Enter positive solvent mass, densities, and shape factor.")
            return
        start = self.start.value()
        end = self.end.value() or None
        if end is not None and end <= start:
            QMessageBox.warning(self, "Run window", "End elapsed time must be later than the start.")
            return

        initial = self.initial_concentration.value()
        experiments, processes = [], []
        for index, (number, name, file) in enumerate(self.runs):
            if self.runs_list.item(index).checkState() != Qt.CheckState.Checked:
                continue
            try:
                data = import_run(file, mapping, start=start, end=end,
                                  concentration_multiplier=self.multiplier.value(),
                                  scale_direction=self.direction.currentData(),
                                  smoothing=self.smoothing.value(),
                                  initial_concentration=None if initial < 0 else initial,
                                  seed_percentiles=seed, no_feed=self.no_feed.isChecked())
            except (ValueError, OSError) as error:
                QMessageBox.warning(self, name, str(error))
                return
            data["Name"] = name
            data["RecordingFile"] = str(file)
            experiments.append(data)
            processes.append({**process, "No": number or index + 1})
        if not experiments:
            QMessageBox.warning(self, "Import campaign", "Select at least one completed run.")
            return
        self.experiments, self.processes = experiments, processes
        self.accept()
