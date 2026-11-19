"""Choose the continuous run interval sent to BO or IMED."""
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import (QComboBox, QHeaderView, QLabel, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from experiment.run_stages import EXPERIMENT_PURPOSES, modeling_campaign, stage_window


class ExperimentPeriodWidget(QWidget):
    def __init__(self, period, parent=None):
        super().__init__(parent)
        self.context = {}
        self.saved_ids = set(period["stage_ids"])
        self.initialized = False
        layout = QVBoxLayout(self)
        self.mode = QComboBox()
        for caption, key in (("Use experiment definition", "definition"),
                             ("Select run stages", "stages"), ("Manual start / end in Recording", "manual")):
            self.mode.addItem(caption, key)
        self.mode.setCurrentIndex(self.mode.findData(period["mode"]))
        layout.addWidget(self.mode)
        note = QLabel("All stages execute and remain in the full recording. BO/IMED only models the selected "
                      "continuous period, starting at model time zero. Select adjacent stages; separated periods "
                      "cannot be joined. Durations resolve separately for every generated experiment. "
                      "Set process masses, concentration and seed information for the beginning of this period.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Use", "Stage", "Purpose", "Start (s)", "End (s)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.summary)
        self.table.itemChanged.connect(self.update_summary)
        self.mode.currentIndexChanged.connect(self.update_summary)

    def selected_ids(self):
        return [self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
                for row in range(self.table.rowCount())
                if self.table.item(row, 0).checkState() == Qt.CheckState.Checked]

    def set_context(self, context):
        selected = set(self.selected_ids()) if self.initialized else self.saved_ids
        use_defaults = not self.initialized and not selected
        self.context = context
        stages = context.get("run_stages", ())
        experiments = context["campaign"].experiments if context.get("campaign") else ()
        with QSignalBlocker(self.table):
            self.table.setRowCount(len(stages))
            for row, stage in enumerate(stages):
                check = QTableWidgetItem()
                check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                check.setData(Qt.ItemDataRole.UserRole, stage.id)
                check.setCheckState(Qt.CheckState.Checked if stage.id in selected or
                                    (use_defaults and stage.purpose in EXPERIMENT_PURPOSES) else Qt.CheckState.Unchecked)
                self.table.setItem(row, 0, check)
                variants = [other for exp in experiments for other in exp.run_stages if other.id == stage.id] or [stage]
                def span(key):
                    low, high = min(getattr(s, key) for s in variants), max(getattr(s, key) for s in variants)
                    return f"{low:g}" if low == high else f"{low:g}–{high:g}"
                for col, value in enumerate((stage.name, stage.purpose, span("start_s"), span("end_s")), 1):
                    item = QTableWidgetItem(value)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    item.setToolTip(value)
                    self.table.setItem(row, col, item)
        # Keep a saved selection visible if the current source has no phases yet.
        if stages:
            self.initialized = True
        self.update_summary()

    def values(self):
        ids = self.selected_ids() if self.initialized else list(self.saved_ids)
        if self.mode.currentData() == "stages":
            stage_window(self.context.get("run_stages", ()), ids)
        return {"mode": self.mode.currentData(), "stage_ids": ids}

    def update_summary(self, *_):
        self.table.setEnabled(self.mode.currentData() == "stages")
        if not self.context.get("run_stages"):
            self.summary.setText("No named run stages. Add stages in Experiment Definition, or Control Panel → Run stages… "
                                 "and regenerate the experiments. Existing profiles can use manual start/end times.")
        elif self.mode.currentData() == "manual":
            self.summary.setText("Set start/end times in Recording. These absolute seconds apply to each starting run. "
                                 "Follow-ups retain the surrounding preparation and cleanup.")
        else:
            try:
                window = stage_window(self.context["run_stages"], self.selected_ids() if self.mode.currentData() == "stages" else None)
                text = (f"First run: {window[0]:g}–{window[1]:g} s. Start/end columns show the range across the starting batch."
                        if window else "No Experiment/Equilibration stages. Choose Select run stages and select the modeling period.")
            except ValueError as error:
                text = str(error)
            self.summary.setText(text)

    def validate(self, setup):
        modeling_campaign(self.context.get("campaign"), setup)
