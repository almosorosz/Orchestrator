"""User-facing instructions for the external MODEL handoff and experiment files."""
from copy import deepcopy
from html import escape
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton, QTabWidget, QTextBrowser, QVBoxLayout)

from app.presets import validate_manual_experiments
from experiment.recording import atomic_json
from hardware_devices.limits import target_limits


def experiment_template(context):
    """Export existing, already defined experiments in the loadable CP format."""
    payload = {"version": 3, "devices": deepcopy(context.get("devices", {})), "profiles": {},
               "experiments": deepcopy(context.get("experiments", [])),
               "experiment_metadata": deepcopy(context.get("experiment_metadata", [{} for _ in context.get("experiments", [])])),
               "simulation_speed": context.get("simulation_speed", 1.),
               "recording_folder": context.get("experiment_root", "")}
    if context.get("system_topology") is not None:
        payload["system_topology"] = deepcopy(context["system_topology"])
    if not payload["devices"] or not payload["experiments"]:
        raise ValueError("First generate and assign a DoE, or generate an experiment in the Control Panel. "
                         "The template copies those device IDs, profiles and units for your external program to edit.")
    validate_manual_experiments(payload)
    return payload


def ai_review_prompt(context):
    return f"""Review my external-model approach for the currently implemented CrystalLoop MODEL-phase file handoff.

I will supply campaign_input.json, a loadable next-experiment template, relevant execution.csv recordings, and my modeling method. Ask for missing files or assumptions; do not invent measurements, device IDs, calibration factors or capabilities.

Experiment output root: {context.get('experiment_root') or '[set in Settings]'}
Model output root: {context.get('model_root') or '[set in Settings]'}
Current MODEL iteration folder: {context.get('active_model_folder') or '[read from CrystalLoop after the batch finishes]'}

Current interface:
1. CrystalLoop executes the starting batch and publishes campaign_input.json in its MODEL iteration folder. My external program runs separately and reads the manifest and each completed run's execution.csv.
2. My program writes next_experiments.json as a FULL Control Panel configuration using CrystalLoop's exported template. experiments is a list of pandas split-format tables (columns, index, data). Each table uses time_s in seconds and actual configured actuator IDs. Targets are numeric and finite; times are nonnegative and strictly increasing. Preserve device configuration and required read-only monitors. Command units are temperature °C, stirrer and pump RPM, valve 0/1. Convert mass flows using the actual pump calibration.
3. Preserve or update experiment_metadata for each experiment when changing timing: run_stages have id/name/purpose/start_s/end_s and model_window is [start_s,end_s]. The modeling period is continuous and must be inside the full run. Distinguish measured outputs from commanded targets; preparation and cleanup data must not be treated as the modeling experiment.
4. When all calculations and files are complete, publish a nonempty MODEL.json beside campaign_input.json, writing a temporary file and renaming it last. This marker signals completion; its contents do not execute recipes. next_experiments.json is not automatically imported.
5. The user ends the previous experiment session and stops LOOP monitoring, uses Control Panel → Load Configuration for next_experiments.json, connects devices, reviews View Experiment Profiles, then Dashboard Start → Control Panel profiles to execute the new batch.

Evaluate whether my method uses the recorded data correctly, handles uncertainty and failed/incomplete runs, and produces feasible next experiments within configured device limits and stated scientific constraints. Check the full file format, units, pump calibration, stage timing, modeling window and required assignments. Report concrete issues and corrections supported by the supplied files, then describe the exact file-write and user load/review/start sequence. Work only with data/configuration files. Do not request, disclose or modify CrystalLoop's codebase, and do not assume an API, MCP service or automatic recipe-import connector exists.
"""


class ExternalModelInfoDialog(QDialog):
    def __init__(self, context_provider, parent=None):
        super().__init__(parent)
        self.setWindowTitle("External model → CrystalLoop experiment engine")
        self.resize(980, 670)
        self.context_provider = context_provider
        layout = QVBoxLayout(self)
        intro = QLabel("Use your own MATLAB, Python or other model with CrystalLoop's experiment execution and recording. "
                       "Follow the MODEL-phase file handoff below to prepare and start the next experiments.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.tabs = QTabWidget()
        self.pages = {}
        for title in ("Workflow", "Files and folders", "New experiments", "AI review prompt"):
            page = QTextBrowser()
            page.setOpenExternalLinks(False)
            self.tabs.addTab(page, title)
            self.pages[title] = page
        layout.addWidget(self.tabs, 1)
        paths = QFormLayout()
        self.path_fields = {}
        for key, label in (("experiment_root", "Experiment output folder (all starts)"), ("model_root", "Model output folder"),
                           ("active_model_folder", "Active / most recent MODEL folder")):
            field = QLineEdit()
            field.setReadOnly(True)
            self.path_fields[key] = field
            paths.addRow(label, field)
        layout.addLayout(paths)
        actions = QHBoxLayout()
        refresh = QPushButton("Refresh paths")
        refresh.clicked.connect(self.refresh)
        self.open_model_button = QPushButton("Open MODEL folder")
        self.open_model_button.clicked.connect(self.open_model_folder)
        self.export_button = QPushButton("Save next-experiment template…")
        self.export_button.clicked.connect(self.export_template)
        self.copy_prompt_button = QPushButton("Copy AI review prompt")
        self.copy_prompt_button.clicked.connect(self.copy_prompt)
        for button in (refresh, self.open_model_button, self.export_button, self.copy_prompt_button):
            actions.addWidget(button)
        actions.addStretch()
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        actions.addWidget(close)
        layout.addLayout(actions)
        self.refresh()

    def refresh(self):
        context = self.context_provider()
        for key, field in self.path_fields.items():
            field.setText(str(context.get(key) or "Not created / configured yet"))
        self.open_model_button.setEnabled(bool(context.get("active_model_folder") or context.get("model_root")))
        self.pages["Workflow"].setHtml("""
            <h3>When CrystalLoop enters the MODEL phase</h3>
            <ol><li>With <b>External MODEL file</b> selected, the completed batch has a unique MODEL subfolder
            containing <code>campaign_input.json</code>. Find it using the active MODEL folder below.
            Run your external modeling program separately.</li>
            <li>Your program reads that manifest, finds the recorded campaign, and analyzes each completed run's
            <code>execution.csv</code>. Keep measured outputs such as FBRM and D10/D50/D90 separate from commanded inputs.</li>
            <li>Write the next experiments as <code>next_experiments.json</code> using the template on this window.
            Save it beside <code>campaign_input.json</code> for easy tracking.</li>
            <li>When calculations and files are complete, publish a nonempty <code>MODEL.json</code> in that same MODEL
            subfolder. Write to a temporary file first and rename it when finished.</li>
            <li>CrystalLoop detects completion. End the previous experiment session and stop LOOP monitoring,
            then use <b>Control Panel → Load Configuration</b> to load <code>next_experiments.json</code>.
            Connect the required devices, review <b>View Experiment Profiles</b>, then use <b>Dashboard Start</b>
            and choose Control Panel profiles. Repeat after the next batch.</li></ol>
            <p><b>Current limitation:</b> MODEL detection is a completion signal. Its contents are not interpreted as
            an executable recipe. New experiments require the explicit load/review/start step above.</p>
        """)
        root = escape(str(context.get("model_root") or "[MODEL root from Settings]"))
        active = escape(str(context.get("active_model_folder") or "[created after the first batch finishes]"))
        self.pages["Files and folders"].setHtml(f"""
            <h3>Where to put the files</h3>
            <p>MODEL root: <code>{root}</code><br>Active / most recent iteration: <code>{active}</code></p>
            <pre>[MODEL root]/_model_&lt;iteration&gt;_&lt;campaign&gt;/
    campaign_input.json       ← CrystalLoop's input manifest
    next_experiments.json     ← your next batch, loaded through Control Panel
    MODEL.json                ← your completion marker, written last</pre>
            <p>The marker belongs in the specific iteration subfolder, not directly in the MODEL root or
            Experiment folder. Accepted examples are <code>MODEL.json</code>, <code>MODEL1.csv</code>,
            <code>MODEL1.mat</code>, <code>MODEL.txt</code> and a nonempty extensionless <code>MODEL</code>.
            Names such as <code>MODEL_done.json</code> do not match the completion monitor.</p>
            <p>Only the completion marker is watched in the current version. The next-experiment JSON can be stored
            elsewhere if you select it using Load Configuration. Dropping a CSV or JSON into the Experiment folder
            does not queue or execute a new experiment.</p>
            <p>The Settings Experiment output folder contains the full recordings. The manifest identifies the
            completed campaign so your external program can locate the correct data.</p>
        """)
        device_rows = []
        for device_id, config in context.get("devices", {}).items():
            if config.get("type") in {"huber", "masterflex", "ismatec", "jkem", "nidaq_valve"}:
                limits = target_limits(config)
                device_rows.append(f"<tr><td>{escape(device_id)}</td><td>{escape(config.get('name', device_id))}</td>"
                                   f"<td>{escape(limits['unit'])}</td><td>{limits['min']:g} … {limits['max']:g}</td></tr>")
        self.pages["New experiments"].setHtml("""
            <h3>Use the loadable Control Panel JSON format</h3>
            <p><b>Save next-experiment template…</b> exports your currently assigned DoE or generated Control Panel batch,
            together with its existing device configuration. Your external program edits the <code>experiments</code>
            array and preserves the device IDs and configuration. This file is directly loadable with Load Configuration.</p>
            <pre>"experiments": [{
  "columns": ["time_s", "T1", "S1"],
  "index": [0, 1, 2],
  "data": [[0, 25, 300], [300, 20, 300], [600, 20, 300]]
}]</pre>
            <p>This is an example of the experiments array within the full exported configuration; an isolated array
            is not a complete configuration. Add one object per experiment. Times are seconds, nonnegative and
            strictly increasing. Use actual configured device IDs as columns and numeric, finite targets.</p>
            <p>Targets use <b>device command units</b> such as temperature °C, Masterflex and Ismatec RPM,
            stirrer RPM, valve 0/1. These differ from the mass-flow g/h used in Experiment Definition.
            Apply your pump calibration before writing device targets. Continuous targets interpolate linearly;
            valves hold the previous binary state. Shorter device profiles hold their final target.</p>
            <p>Read-only FBRM, D-value probes and balances remain in the devices configuration and are recorded as
            monitors. Do not add fake measured values as experiment target columns.</p>
            <p><code>experiment_metadata</code> has one entry per experiment. Keep stage IDs and update each
            <code>run_stages</code> start/end and <code>model_window</code> when changing the timeline. Stage times
            are seconds; the modeling window must remain inside the full run. Preparation and cleanup stay recorded.</p>
            <h3>Configured command devices</h3><table border="1" cellpadding="5"><tr>
            <th>ID</th><th>Name</th><th>Target unit</th><th>Configured limits</th></tr>
            """ + "".join(device_rows) + "</table>")
        self.pages["AI review prompt"].setPlainText(ai_review_prompt(context))

    def copy_prompt(self):
        prompt = ai_review_prompt(self.context_provider())
        self.pages["AI review prompt"].setPlainText(prompt)
        QApplication.clipboard().setText(prompt)
        self.tabs.setCurrentWidget(self.pages["AI review prompt"])

    def open_model_folder(self):
        context = self.context_provider()
        folder = context.get("active_model_folder") or context.get("model_root")
        if folder and Path(folder).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(folder).resolve())))
        else:
            QMessageBox.information(self, "MODEL folder", "Configure an existing MODEL folder, then start a batch to create its iteration subfolder.")

    def export_template(self):
        try:
            context = self.context_provider()
            payload = experiment_template(context)
        except (ValueError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "Experiment template", str(error))
            return None
        folder = context.get("active_model_folder") or context.get("model_root") or ""
        filename, _ = QFileDialog.getSaveFileName(self, "Save loadable next-experiment template",
                                                 str(Path(folder) / "next_experiments.json"), "JSON (*.json)")
        if not filename:
            return None
        path = Path(filename)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        try:
            atomic_json(path, payload)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Experiment template", str(error))
            return None
        QMessageBox.information(self, "Experiment template saved", f"Saved {len(payload['experiments'])} experiment(s) to {path}. "
                                "Edit the experiment data in your external program, then load this file through Control Panel → Load Configuration.")
        return path
