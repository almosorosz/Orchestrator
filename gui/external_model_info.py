"""User-facing instructions for the external MODEL handoff and experiment files."""
from copy import deepcopy
from html import escape
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from app.presets import validate_manual_experiments
from experiment.recording import atomic_json
from hardware_devices.limits import target_limits


def experiment_template(context):
    """Export existing, already defined experiments in the loadable CP format."""
    payload = {"version": 3, "devices": deepcopy(context.get("devices", {})), "profiles": {},
               "experiments": deepcopy(context.get("experiments", [])),
               "simulation_speed": context.get("simulation_speed", 1.),
               "recording_folder": context.get("experiment_root", "")}
    if context.get("system_topology") is not None:
        payload["system_topology"] = deepcopy(context["system_topology"])
    if not payload["devices"] or not payload["experiments"]:
        raise ValueError("First generate and assign a DoE, or generate an experiment in the Control Panel. "
                         "The template copies those device IDs, profiles and units for your external program to edit.")
    validate_manual_experiments(payload)
    return payload


class ExternalModelInfoDialog(QDialog):
    def __init__(self, context_provider, parent=None):
        super().__init__(parent)
        self.setWindowTitle("External model → CrystalLoop experiment engine")
        self.resize(980, 670)
        self.context_provider = context_provider
        layout = QVBoxLayout(self)
        intro = QLabel("Use your own MATLAB, Python or other model with CrystalLoop's experiment execution and recording. "
                       "This guide distinguishes the current file handoff from a proposed fully automatic connector.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.tabs = QTabWidget()
        self.pages = {}
        for title in ("Workflow", "Files and folders", "New experiments", "Automatic connector"):
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
        for button in (refresh, self.open_model_button, self.export_button):
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
            <h3>Current external-model workflow</h3>
            <ol><li>Set existing Experiment and MODEL folders in Settings. Choose <b>External MODEL file</b>.
            Prepare a DoE and apply device assignments, or generate Control Panel experiments. Connect the devices
            and the read-only monitors you want recorded. Start from the Dashboard.</li>
            <li>CrystalLoop executes and saves the batch, then creates a unique MODEL subfolder and
            <code>campaign_input.json</code>. Run your external modeling program separately.</li>
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
            <h3>One set of output locations</h3><ul>
            <li><b>Control Panel Start:</b> Settings → Experiment output folder.</li>
            <li><b>Start DoE independent from loop:</b> the same Experiment output folder.</li>
            <li><b>Dashboard LOOP:</b> recordings in Experiment output folder; fits, stages and handoff files in Model output folder.</li>
            <li><b>BO/IMED Execute recipe:</b> recordings in Experiment output folder; recipe/dataset snapshots in Model output folder.</li></ul>
            <p>Each campaign has its own subfolder. A run without modeling does not need to create a modeling stage.
            Legacy per-mode folders in older configurations do not override the general Settings locations.</p>
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
            <p>Targets use <b>device command units</b>: temperature °C, Masterflex RPM, Ismatec mL/min,
            stirrer RPM, valve 0/1. These differ from the mass-flow g/h used in Experiment Definition.
            Apply your pump calibration before writing device targets. Continuous targets interpolate linearly;
            valves hold the previous binary state. Shorter device profiles hold their final target.</p>
            <p>Read-only FBRM, D-value probes and balances remain in the devices configuration and are recorded as
            monitors. Do not add fake measured values as experiment target columns.</p>
            <h3>Configured command devices</h3><table border="1" cellpadding="5"><tr>
            <th>ID</th><th>Name</th><th>Target unit</th><th>Configured limits</th></tr>
            """ + "".join(device_rows) + "</table>")
        self.pages["Automatic connector"].setHtml("""
            <h3>Recommended next integration: a watched exchange folder</h3>
            <p><b>This automatic adapter is a proposed extension, not an enabled feature in this version.</b></p>
            <ol><li>CrystalLoop publishes <code>campaign_input.json</code> with the campaign ID and recorded-data paths.</li>
            <li>Your program writes <code>next_experiments.json</code> in the exported format, then atomically publishes
            <code>MODEL.json</code> with a unique response ID, source campaign ID, and a run/stop decision.</li>
            <li>The adapter checks the IDs to reject duplicate or stale responses, validates device assignments,
            units, limits and connectivity, then submits the batch through the existing experiment runner.</li>
            <li>CrystalLoop records the new batch, publishes the next manifest and repeats until stop or the
            configured experiment limit. Errors and every handoff are logged.</li></ol>
            <p>This is convenient for Python, MATLAB and other tools: they only read and write data files.
            A REST or MCP interface can be added later using the same validated request format.</p>
        """)

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
