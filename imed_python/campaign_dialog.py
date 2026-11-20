"""Map and review a CrystalLoop recording before importing it into IMED."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QLabel, QListWidget, QListWidgetItem, QMessageBox, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

from experiment.loop_process import PROCESS_FIELDS
from .campaign_import import CHANNELS, channel_options, import_run, read_json
from .size_data import FINAL_SIZE_SAMPLE_TOOLTIP, MAX_FINAL_SIZE_SAMPLES


class CampaignImportDialog(QDialog):
    def __init__(self, runs, case, parent, *, require_process=True):
        super().__init__(parent)
        self.require_process = require_process
        self.setWindowTitle("IMED | Import CrystalLoop campaign")
        self.resize(650, 700)
        self.runs = runs
        self.experiments = []
        self.processes = []
        form = QFormLayout()
        layout = QVBoxLayout(self)
        notice = QLabel("Select completed runs and map recorded channels. "
                        "Use the crystallization window to exclude filling or cleaning. "
                        "Saved per-run process inputs and defined charging stages supply the starting charges when available.")
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
        self.final_size_samples = QSpinBox()
        self.final_size_samples.setRange(1, MAX_FINAL_SIZE_SAMPLES)
        self.final_size_samples.setToolTip(FINAL_SIZE_SAMPLE_TOOLTIP)
        form.addRow("Final size samples to average", self.final_size_samples)
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
        definition_file = runs[0][2].parent.parent / "definition.json" if runs else None
        self.definition = read_json(definition_file) if definition_file and definition_file.is_file() else {}
        self.definitions = {item["experiment_id"]: item for item in self.definition.get("campaign", {}).get("experiments", [])}
        self.use_periods = QCheckBox("Use each run's defined experiment period")
        self.use_periods.setChecked(bool(self.definitions) and all(item.get("model_window") for item in self.definitions.values()))
        self.use_periods.setEnabled(any(item.get("model_window") for item in self.definitions.values()))
        configs = self.definition.get("devices", {})
        self.charge_roles = {key: lane["device_id"] for key, lane in self.definition.get("campaign", {}).get("line_roles", {}).items()
                             if lane.get("role") in {"AS", "solvent", "solution"}}
        self.charge_roles.update({role: device for device, config in configs.items()
                                  if (role := config.get("connectivity", {}).get("experiment_role")) in {"AS", "solvent", "solution"}})
        has_pump = any(configs.get(profile["device_id"], {}).get("type") in {"masterflex", "ismatec"}
                       and any(point["target"] for point in profile["points"])
                       for item in self.definitions.values() for profile in item.get("profiles", []))
        if not has_pump and not self.channels["AS scale"].currentData():
            self.no_feed.setChecked(True)
        self.use_periods.setToolTip("Uses each sampled run's own Experiment/Equilibration interval, excluding preparation and cleanup. Clear to enter common manual times.")
        self.use_periods.toggled.connect(lambda checked: (self.start.setEnabled(not checked), self.end.setEnabled(not checked)))
        self.start.setEnabled(not self.use_periods.isChecked())
        self.end.setEnabled(not self.use_periods.isChecked())
        form.addRow(self.use_periods)
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
        if not mapping["Temperature"] or (not mapping["AS scale"] and not self.no_feed.isChecked()):
            QMessageBox.warning(self, "Channel mapping", "Map temperature and the AS scale, or select no AS feed.")
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
                metadata = read_json(file.with_name("run.json"))
                spec = self.definitions.get(metadata.get("experiment_id"), {})
                run_start, run_end = (spec.get("model_window") or (start, end)) if self.use_periods.isChecked() else (start, end)
                run_process = dict(process)
                use_stored = self.use_periods.isChecked() or not spec.get("model_window")
                stored = (spec.get("model_process") or {}) if use_stored else {}
                run_process.update({key: value for key, value in stored.items() if key != "SeedPercentiles"})
                if "SeedMass" not in stored and spec.get("seed_mass_g") is not None:
                    run_process["SeedMass"] = spec["seed_mass_g"]
                run_seed = (stored.get("SeedPercentiles") or tuple(box.value() for box in self.seed)) if run_process["SeedMass"] else None
                if run_seed and any(value <= 0 for value in run_seed):
                    raise ValueError("Enter positive seed percentiles for this seeded run.")
                tubing = spec.get("pump_adjustments") or {}
                adjusted_pumps = tubing.get("pumps", {})
                lanes = self.definition.get("campaign", {}).get("line_roles", {})
                as_device = next((device for role, device in self.charge_roles.items()
                                  if lanes.get(role, {}).get("role", role) == "AS"), None)
                data = import_run(file, mapping, start=run_start, end=run_end,
                                  concentration_multiplier=self.multiplier.value(),
                                  scale_direction=self.direction.currentData(),
                                  smoothing=self.smoothing.value(),
                                  initial_concentration=run_process.get("InitialConcentration") if initial < 0 else initial,
                                  seed_percentiles=run_seed, no_feed=self.no_feed.isChecked(),
                                  require_initial_concentration=self.require_process,
                                  final_size_samples=self.final_size_samples.value(),
                                  pump_adjustment=adjusted_pumps.get(as_device))
                data.update(Name=name, RecordingFile=str(file))
                from bo.profiles import campaign_profiles
                from .doe_design import reference_rpm
                from .mixing import average_mixing

                have_densities = min(run_process["SolventDensity"], run_process["AntiSolventDensity"]) > 0
                if not stored and self.use_periods.isChecked() and self.charge_roles and run_start > 0 and run_end is not None and have_densities:
                    from experiment.material_balance import initial_charges
                    from experiment.models import DeviceProfile, ExperimentDefinition, ProfilePoint

                    shift = tubing.get("lead_in_s", 0.)
                    if run_start > shift:
                        profiles = []
                        for entry in spec["profiles"]:
                            if adjustment := adjusted_pumps.get(entry["device_id"]):
                                nominal = adjustment["nominal_profile"]
                                points = tuple(ProfilePoint(**point) for point in nominal["points"])
                            else:
                                points = tuple(ProfilePoint(point["time_s"] - shift, point["target"])
                                               for point in entry["points"] if point["time_s"] >= shift)
                            profiles.append(DeviceProfile(entry["device_id"], points, entry["interpolation"], entry["unit"]))
                        planned = ExperimentDefinition(name, profiles, model_window=(run_start - shift, run_end - shift))
                        if data.get("InitialConcentration") is not None:
                            run_process["InitialConcentration"] = data["InitialConcentration"]
                        run_process = initial_charges(planned, run_process, self.charge_roles, self.definition["devices"],
                                                       line_roles=self.definition["campaign"].get("line_roles"))
                if self.require_process and any(run_process[key] <= 0 for key in
                       ("SolvMass", "SolventDensity", "AntiSolventDensity", "SolidDensity", "ShapeFactor")):
                    raise ValueError("Enter positive solvent mass, densities, and shape factor.")
                data["CommandedProfiles"] = {key: value.payload() for key, value in
                                              campaign_profiles(data, run_process, self.definition).items()}
                rpm = reference_rpm(data, run_process)
                if rpm is not None:
                    run_process["StirringRPM"] = rpm
                if "MixingSettings" in run_process:
                    run_process = average_mixing(data, run_process, run_process["MixingSettings"], mean_rpm=rpm)
            except (ValueError, TypeError, KeyError, OSError) as error:
                QMessageBox.warning(self, name, str(error))
                return
            experiments.append(data)
            processes.append({**run_process, "No": number or index + 1})
        if not experiments:
            QMessageBox.warning(self, "Import campaign", "Select at least one completed run.")
            return
        self.experiments, self.processes = experiments, processes
        self.accept()
