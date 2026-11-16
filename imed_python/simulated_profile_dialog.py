"""Edit the DoE inputs used in a new PBM prediction."""

import re

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPlainTextEdit,
    QTabWidget, QVBoxLayout, QWidget,
)

from .simulated_profile import editable_groups, feed_for_duration, prediction_inputs, template_rpm


def _format_points(points):
    return "\n".join(f"{time:.16g}, {value:.16g}" for time, value in np.asarray(points).T)


def _parse_points(text, label):
    points = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        columns = re.split(r"[,\s]+", line.strip())
        try:
            time, value = map(float, columns)
        except ValueError as error:
            raise ValueError(f"{label} row {line_number} needs a time and a value.") from error
        points.append((time, value))
    return np.asarray(points, dtype=float).T


class SimulatedProfileDialog(QDialog):
    def __init__(self, case, model_index, experiment_index, parent=None, *, previous=None):
        super().__init__(parent)
        self.setWindowTitle("IMED | Define simulated profile")
        self.resize(660, 460)
        self.case, self.experiment_index = case, experiment_index
        self.result_profile = None
        self.schedules, self.masses = {}, {}
        self.seed_sizes = []
        self.rpm = self.reference_rpm = None
        previous = previous or {}
        experiment, process = prediction_inputs(case, experiment_index)
        groups = editable_groups(case, model_index, previous)
        temperature = np.asarray(experiment["TemperatureData"], dtype=float).reshape(2, -1)
        end = temperature[0, -1]
        self.name = QLineEdit(previous.get("name", "Custom profile"))
        self.name.setMaxLength(100)
        form = QFormLayout()
        form.addRow("Profile name", self.name)
        layout = QVBoxLayout(self)
        layout.addWidget(self.note(
            f"Template is Experiment {experiment_index+1}. Editable inputs follow the DoE. "
            "Other inputs come from the template."))
        layout.addLayout(form)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        if "temperature" in groups:
            self.add_schedule("temperature", "Temperature", temperature, previous,
                              "Enter time (s), temperature (°C) on each line. The last time sets the run duration.")
        if "feed" in groups:
            feed = feed_for_duration(experiment.get("FlowRateData", []), end)
            feed_layout = self.add_schedule(
                "feed", "Antisolvent", feed, previous,
                "Enter time (s), antisolvent feed (g/s) on each line. "
                "The last rate is held through the run. Enter a final zero rate to stop dosing.")
            mass = QFormLayout()
            self.add_mass(mass, "initial_antisolvent", "Initial antisolvent (g)", process["AntiSolvMass"], previous)
            feed_layout.insertLayout(1, mass)
        if "seed" in groups:
            page = QWidget()
            seed_layout = QFormLayout(page)
            self.add_mass(seed_layout, "seed_mass", "Initial seed mass (g)", process["SeedMass"], previous)
            sizes = previous.get("seed_percentiles", ("", "", ""))
            for percentile, value in zip((10, 50, 90), sizes):
                edit = QLineEdit(str(value))
                self.seed_sizes.append(edit)
                seed_layout.addRow(f"Seed D{percentile} (µm)", edit)
            seed_layout.addRow(self.note(
                "Leave all seed percentiles blank to reuse the template seed size distribution. "
                "A new seeded case without a template distribution needs all three values."))
            self.tabs.addTab(page, "Seed")
        if "stirring" in groups:
            page = QWidget()
            stirring_layout = QFormLayout(page)
            reference = previous.get("reference_rpm", template_rpm(process))
            self.reference_rpm = QLineEdit(f"{reference:g}" if np.isfinite(reference) else "")
            self.rpm = QLineEdit(str(previous.get("stirring_rpm", "")))
            self.rpm.setPlaceholderText("Use template speed")
            stirring_layout.addRow("Template stirring speed (RPM)", self.reference_rpm)
            stirring_layout.addRow("New stirring speed (RPM)", self.rpm)
            stirring_layout.addRow(self.note(
                f"Template power ratio is {process['PowerRatio']:g}. "
                "The PBM uses this ratio × (new RPM / template RPM)³, assuming the same vessel, "
                "impeller and power number."))
            self.tabs.addTab(page, "Stirring")
        self.status = self.note("")
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Use profile")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def note(text):
        label = QLabel(text)
        label.setWordWrap(True)
        return label

    def add_schedule(self, key, label, baseline, previous, description):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.note(description + " Start at 0. Values interpolate linearly between points."))
        text = QPlainTextEdit(_format_points(previous.get(key, baseline)))
        text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(text)
        self.schedules[key] = text, _format_points(baseline), label
        self.tabs.addTab(page, label)
        return layout

    def add_mass(self, form, key, label, baseline, previous):
        baseline_text = f"{baseline:.16g}"
        text = QLineEdit(f"{previous.get(key, baseline):.16g}")
        self.masses[key] = text, baseline_text
        form.addRow(label, text)

    def accept(self):
        recipe = {"name": self.name.text().strip() or "Custom profile"}
        try:
            for key, (text, baseline, label) in self.schedules.items():
                if text.toPlainText().strip() != baseline:
                    recipe[key] = _parse_points(text.toPlainText(), label)
            for key, (text, baseline) in self.masses.items():
                if text.text().strip() != baseline:
                    recipe[key] = float(text.text())
            sizes = [edit.text().strip() for edit in self.seed_sizes]
            if any(sizes):
                if not all(sizes):
                    raise ValueError("Enter all three seed percentiles.")
                recipe["seed_percentiles"] = [float(value) for value in sizes]
            if self.rpm is not None and self.rpm.text().strip():
                if not self.reference_rpm.text().strip():
                    raise ValueError("Enter the template stirring speed to convert the new RPM to a power ratio.")
                recipe["stirring_rpm"] = float(self.rpm.text())
                recipe["reference_rpm"] = float(self.reference_rpm.text())
            prediction_inputs(self.case, self.experiment_index, recipe)
        except ValueError as error:
            self.status.setText(str(error))
            return
        self.result_profile = recipe
        super().accept()
