"""Choose the continuous run interval sent to BO or IMED."""
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import (QComboBox, QFormLayout, QHeaderView, QLabel, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from experiment.run_stages import EXPERIMENT_PURPOSES, modeling_campaign, stage_window


class ExperimentPeriodWidget(QWidget):
    def __init__(self, period, *, manual_start, manual_end, parent=None):
        super().__init__(parent)
        self.context = {}
        self.saved_ids = set(period["stage_ids"])
        self.initialized = False
        layout = QVBoxLayout(self)
        self.mode = QComboBox()
        for caption, key in (("Use experiment definition", "definition"),
                             ("Select run stages", "stages"), ("Manual start / end", "manual")):
            self.mode.addItem(caption, key)
        self.mode.setCurrentIndex(self.mode.findData(period["mode"]))
        layout.addWidget(self.mode)
        note = QLabel("The full recipe runs and is recorded. BO and IMED use the selected continuous period for "
                      "modeling and follow-up recommendations. Select adjacent stages. In a follow-up, all "
                      "unselected stages keep their programmed targets and durations. Enter process inputs "
                      "for the beginning of the selected period.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.manual_form = QWidget()
        form = QFormLayout(self.manual_form)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Start elapsed time (s)", manual_start)
        form.addRow("End elapsed time (s)", manual_end)
        layout.addWidget(self.manual_form)
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
        self.update_summary()

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
        mode = self.mode.currentData()
        stages = self.context.get("run_stages", ())
        self.manual_form.setVisible(mode == "manual")
        self.table.setVisible(bool(stages) and mode != "manual")
        self.table.setEnabled(mode == "stages")
        if mode == "manual":
            self.summary.setText("Times apply to each starting run. Leave the end blank to use the end of the programmed run.")
        elif not stages:
            campaign = self.context.get("campaign")
            first = campaign.experiments[0] if campaign and campaign.experiments else None
            if mode == "definition" and first is not None and first.model_window is not None:
                start, end = first.model_window
                self.summary.setText(f"First run uses its defined period of {start:g}–{end:g} s.")
            else:
                self.summary.setText("No named run stages are available. Add stages in Experiment Definition or Control Panel → Run stages…, "
                                     "or choose Manual start / end to set the period here.")
        else:
            try:
                window = stage_window(stages, self.selected_ids() if mode == "stages" else None)
                text = (f"First run: {window[0]:g}–{window[1]:g} s. Start/end columns show the range across the starting batch."
                        if window else "No Experiment/Equilibration stages. Choose Select run stages and select the modeling period.")
            except ValueError as error:
                text = str(error)
            self.summary.setText(text)

    def validate(self, setup):
        modeling_campaign(self.context.get("campaign"), setup)
