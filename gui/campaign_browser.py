"""Campaign browsing with file I/O and rendering on a dedicated worker."""
from pathlib import Path
from math import isfinite

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, Qt, QUrl
from PySide6.QtGui import QImage, QPixmap, QDesktopServices
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QAbstractItemView, QSplitter,
    QScrollArea, QComboBox, QCheckBox, QSpinBox)

from experiment.archive import scan_campaigns, campaign_details, export_runs, plot_traces
from gui.flow_layout import FlowLayout

PLOT_HINT = "Ctrl/Shift-select runs to overlay them. Queued runs have no samples."


def render_recorded(details, run_ids, device_id, dark, time_range=(None, None)):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from gui.render_lock import RENDER_LOCK
    traces = []
    for run in details["runs"]:
        if run.get("run_id") in run_ids:
            traces.append((run, plot_traces(run, device_id, device_configs=details["devices"], time_range=time_range)))
    devices = sorted({col.rsplit(".", 1)[0] for _, data in traces for col in data})
    if not devices:
        raise ValueError("No recorded numeric samples for the selected experiments, device and time range.")
    with RENDER_LOCK:
        fig = Figure(figsize=(12, max(3.5, 2.8 * len(devices))), dpi=100, layout="constrained")
        canvas = FigureCanvasAgg(fig)
        bg, fg = ("#111827", "#e5e7eb") if dark else ("#ffffff", "#111827")
        fig.set_facecolor(bg)
        palette = ['#60a5fa', '#fb923c', '#4ade80', '#e879f9', '#facc15', '#22d3ee', '#f87171', '#c4b5fd']
        for index, device in enumerate(devices, 1):
            axis = fig.add_subplot(len(devices), 1, index)
            axis.set_facecolor(bg)
            config = details["devices"].get(device, {})
            from hardware_devices.limits import DEFAULT_LIMITS
            primary_unit = DEFAULT_LIMITS.get(config.get("type"), {}).get("unit", "value")
            units = {"speed_rpm": "RPM", "flow": "mL/min", "setpoint": "°C", "temp_internal": "°C",
                     "temp_external": "°C", "weight_g": "g", "measured_voltage": "V", "torque_ozin": "oz·in",
                     "target_state": "0/1", "target": primary_unit}
            secondary = None
            for experiment, (run, data) in enumerate(traces):
                for column, values in data.items():
                    if column.rsplit(".", 1)[0] != device:
                        continue
                    field = column.rsplit(".", 1)[1]
                    secondary_field = field == "torque_ozin" or (config.get("type") == "nidaq_valve" and field in {"target", "target_state"})
                    if secondary_field and secondary is None:
                        secondary = axis.twinx()
                        secondary.tick_params(colors=fg, labelsize=6)
                        secondary.set_ylabel(units.get(field, "value"), color=fg, fontsize=7)
                    plot_axis = secondary if secondary_field else axis
                    plot_axis.plot([v[0] for v in values], [v[1] for v in values],
                              "--" if field == "target" else ":" if field in {"temp_external", "torque_ozin"} else "-",
                              color=palette[experiment % len(palette)], linewidth=1,
                              label=f"{run['index']}: {field} ({units.get(field, 'value')})")
            axis.set_ylabel(f"{config.get('name', device)}\n{device}", color=fg, fontsize=7)
            axis.set_xlabel("Experiment elapsed time (s)", color=fg, fontsize=7)
            if time_range != (None, None):
                axis.set_xlim(*time_range)
            axis.tick_params(colors=fg, labelsize=6)
            axis.grid(alpha=.2)
            handles, labels = axis.get_legend_handles_labels()
            if secondary is not None:
                extra_handles, extra_labels = secondary.get_legend_handles_labels()
                handles += extra_handles
                labels += extra_labels
            if handles:
                legend = axis.legend(handles, labels, fontsize=6, facecolor=bg, edgecolor="#64748b", loc="best")
                for text in legend.get_texts():
                    text.set_color(fg)
        canvas.draw()
        width, height = canvas.get_width_height()
        return QImage(canvas.buffer_rgba(), width, height, width * 4, QImage.Format.Format_RGBA8888).copy()


class ArchiveWorker(QObject):
    finished = Signal(object, object, str)

    @Slot(object)
    def perform(self, job):
        result, error = None, ""
        try:
            kind = job["kind"]
            roots = job.get("roots", [])
            if job.get("known"):
                from app.campaign_roots import registered_roots
                roots = roots + registered_roots()
            if kind == "scan":
                result = scan_campaigns(roots)
            elif kind == "recover":
                from experiment.recovery import recover_roots
                result = recover_roots(roots, allow_legacy=job.get("legacy", False))
            elif kind == "plot":
                details = campaign_details(job["folder"])
                result = render_recorded(details, job["runs"], job["device"], job["dark"], job["time_range"])
            elif kind == "export":
                result = export_runs(campaign_details(job["folder"]), job["runs"], job["destination"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.finished.emit(job, result, error)


class CampaignBrowser(QWidget):
    requested = Signal(object)
    message = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.roots = []
        self.rows = []
        self.details = None
        self.dark_mode = True
        self._busy = False
        self._pending = None
        self._closed = False
        self._image = QImage()
        self._plotted_folder = None
        self._plot_revision = 0
        layout = QVBoxLayout(self)
        self.setObjectName('campaignBrowser')
        self.setStyleSheet('''
            QWidget#campaignBrowser QPushButton { padding: 3px 6px; font-size: 12px; }
            QWidget#campaignBrowser QLineEdit, QWidget#campaignBrowser QComboBox,
            QWidget#campaignBrowser QSpinBox { padding: 3px 5px; font-size: 12px; }
            QWidget#campaignBrowser QTableWidget { font-size: 11px; }
            QWidget#campaignBrowser QHeaderView::section { padding: 1px 3px; font-size: 11px; }
            QWidget#campaignBrowser QCheckBox { font-size: 11px; }
        ''')
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        paths = QHBoxLayout()
        self.root_input = QLineEdit()
        self.root_input.setPlaceholderText("Campaign root or a campaign folder")
        browse = QPushButton("Browse…")
        refresh = QPushButton("Refresh")
        self.recover_button = QPushButton("Recover recordings")
        self.recover_button.setToolTip("Recover interrupted recordings")
        paths.addWidget(self.root_input, 1)
        for button in (browse, refresh, self.recover_button):
            paths.addWidget(button)
        layout.addLayout(paths)
        self.status = QLabel("Select a campaign to inspect recorded and queued experiments.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-size: 11px;")
        status_row = QHBoxLayout()
        status_row.addWidget(self.status, 1)
        self.legacy_recovery = QCheckBox("Legacy recovery (older apps closed)")
        self.legacy_recovery.setToolTip("Include legacy recordings in recovery; close all older app instances first.")
        status_row.addWidget(self.legacy_recovery)
        layout.addLayout(status_row)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.selection_split = split
        split.setMinimumHeight(56)
        split.setMaximumHeight(78)
        self.campaigns = QTableWidget(0, 4)
        self.campaigns.setHorizontalHeaderLabels(["Campaign", "Status", "Completed", "Recorded (UTC)"])
        self.runs = QTableWidget(0, 4)
        self.runs.setHorizontalHeaderLabels(["#", "Experiment", "Status", "Run ID"])
        for table in (self.campaigns, self.runs):
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.setMinimumHeight(56)
            table.verticalHeader().setDefaultSectionSize(20)
            table.horizontalHeader().setStretchLastSection(True)
            split.addWidget(table)
        self.campaigns.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.runs.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(split)
        actions = FlowLayout(spacing=4)
        self.device = QComboBox()
        self.device.setMaximumWidth(220)
        self.device.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.device.setMinimumContentsLength(12)
        self.device.addItem("All devices", None)
        self.plot_button = QPushButton("Plot selected")
        clear = QPushButton("Clear plots")
        clear.clicked.connect(self.clear_plots)
        self.time_start = QLineEdit()
        self.time_end = QLineEdit()
        for field, placeholder in ((self.time_start, "From (s)"), (self.time_end, "To (s)")):
            field.setPlaceholderText(placeholder)
            field.setToolTip("Elapsed time in seconds. Leave blank for an open bound.")
            field.setMaximumWidth(85)
            field.returnPressed.connect(self.plot_selected)
        self.export_button = QPushButton("Export CSVs…")
        self.open_button = QPushButton("Open folder")
        self.auto = QCheckBox("Refresh list")
        self.auto.setToolTip("Refresh experiment and campaign lists while running")
        self.auto.setChecked(True)
        self.zoom = QSpinBox()
        self.zoom.setRange(50, 200)
        self.zoom.setValue(100)
        self.zoom.setSuffix("%")
        for widget in (self.device, self.time_start, self.time_end, self.plot_button, clear, self.export_button, self.open_button):
            actions.addWidget(widget)
        fit = QPushButton("Fit width")
        for widget in (self.auto, QLabel("Plot zoom:"), self.zoom, fit):
            actions.addWidget(widget)
        self.selection_toggle = QPushButton("Hide lists")
        self.selection_toggle.setCheckable(True)
        self.selection_toggle.toggled.connect(lambda hidden: split.setVisible(not hidden))
        self.selection_toggle.toggled.connect(lambda hidden: self.selection_toggle.setText("Show lists" if hidden else "Hide lists"))
        actions.addWidget(self.selection_toggle)
        layout.addLayout(actions)
        self.caption = QLabel(PLOT_HINT)
        self.caption.setStyleSheet("font-size: 11px;")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        scroll = QScrollArea()
        self.image_scroll = scroll
        scroll.setWidgetResizable(True)
        self.image_label = QLabel("Recorded experiment plots appear here.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll, 1)
        self._thread = QThread(self)
        self._worker = ArchiveWorker()
        self._worker.moveToThread(self._thread)
        self.requested.connect(self._worker.perform, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._finished, Qt.ConnectionType.QueuedConnection)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()
        self.timer = QTimer(self)
        self.timer.setInterval(3000)
        self.timer.timeout.connect(self._auto_refresh)
        self.timer.start()
        browse.clicked.connect(self._browse)
        refresh.clicked.connect(self.refresh)
        self.root_input.returnPressed.connect(self.refresh)
        self.recover_button.clicked.connect(self.recover)
        self.campaigns.itemSelectionChanged.connect(self._selection_changed)
        self.plot_button.clicked.connect(self.plot_selected)
        self.export_button.clicked.connect(self.export_selected)
        self.open_button.clicked.connect(self.open_folder)
        self.zoom.valueChanged.connect(self._show_image)
        fit.clicked.connect(self.fit_image)

    def set_roots(self, roots, recover=False):
        self.roots = list(dict.fromkeys(str(Path(root).resolve()) for root in roots if root))
        if recover:
            self.recover()
        else:
            self.refresh()

    def _all_roots(self):
        custom = self.root_input.text().strip()
        return [custom] if custom else self.roots

    def _submit(self, job, background=False):
        if self._closed:
            return
        if self._busy:
            if not background:
                self._pending = job  # One latest request, never an unbounded queue.
            return
        self._busy = True
        self.status.setText(f"{job['kind'].capitalize()} in progress…")
        self.requested.emit(job)

    def refresh(self):
        self._submit(self._root_job("scan"))

    def _root_job(self, kind):
        return {"kind": kind, "roots": self._all_roots(), "known": not bool(self.root_input.text().strip())}

    def recover(self):
        self._submit({**self._root_job("recover"), "legacy": self.legacy_recovery.isChecked()})

    def _auto_refresh(self):
        if self.isVisible() and self.auto.isChecked():
            self._submit(self._root_job("scan"), background=True)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Campaign recording folder")
        if folder:
            self.root_input.setText(folder)
            self.refresh()

    def _fill(self, table, rows):
        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for col, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()

    @Slot(object, object, str)
    def _finished(self, job, result, error):
        self._busy = False
        if self._closed:
            return
        stale_plot = job["kind"] == "plot" and job["revision"] != self._plot_revision
        if stale_plot:
            pass
        elif error:
            self.status.setText(error + " — refresh to retry; acquisition is unaffected.")
        elif job["kind"] == "scan":
            previous = self.details["folder"] if self.details else None
            self.rows = result["campaigns"]
            self.campaigns.blockSignals(True)
            self._fill(self.campaigns, [(r["name"], r["status"], f"{r['completed']}/{r['total']}", r["recorded_at"]) for r in self.rows])
            for row, campaign in enumerate(self.rows):
                for column in range(self.campaigns.columnCount()):
                    self.campaigns.item(row, column).setToolTip(campaign["folder"])
            index = next((i for i, r in enumerate(self.rows) if r["folder"] == previous), 0)
            if self.rows:
                self.campaigns.selectRow(index)
            self.campaigns.blockSignals(False)
            self._selection_changed()
            self.status.setText(f"{len(self.rows)} campaigns. " + ("Some files are temporarily unreadable; refresh to retry." if result["errors"] else ""))
        elif job["kind"] == "plot":
            if self.details and self.details["folder"] == job["folder"]:
                self._image = result
                self._plotted_folder = job["folder"]
                self.fit_image()
                self.caption.setText(f"{len(job['runs'])} experiments overlaid • elapsed time (s) • full-resolution data retained in CSVs")
                self.caption.setToolTip(f"Recorded CSV snapshot: {job['folder']}. Numeric fields show measurement units. "
                                        "Display limited to 4,000 points/trace; exported CSVs retain all samples.")
                self.status.setText("Recorded experiments plotted. Other experiments continue independently.")
        elif job["kind"] == "export":
            self.status.setText(f"Exported to {result}")
        elif job["kind"] == "recover":
            recovered = sum(r["status"] == "recovered" for r in result)
            waiting = [r for r in result if r["status"] == "waiting"]
            text = f"Recovery: {recovered} campaigns recovered; {len(waiting)} waiting. Hardware was not restarted."
            if waiting:
                text += " " + waiting[0]["message"]
            self.message.emit(text, "WARNING" if waiting else "INFO")
            self.status.setText(text)
            self._pending = self._pending or self._root_job("scan")
        pending, self._pending = self._pending, None
        if pending:
            self._submit(pending)

    def _selection_changed(self):
        selected = self.campaigns.selectionModel().selectedRows()
        previous = self.details["folder"] if self.details else None
        chosen = {r.get("run_id") for r in self.selected_runs()}
        self.details = self.rows[selected[0].row()] if selected and selected[0].row() < len(self.rows) else None
        self.runs.blockSignals(True)
        self._fill(self.runs, [(r["index"], r["name"], r["status"], r.get("run_id", "—")) for r in self.details["runs"]] if self.details else [])
        self.runs.clearSelection()
        if self.details and previous == self.details["folder"]:
            from PySide6.QtCore import QItemSelectionModel
            for i, run in enumerate(self.details["runs"]):
                if run.get("run_id") in chosen:
                    self.runs.selectionModel().select(self.runs.model().index(i, 0),
                        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        self.runs.blockSignals(False)
        if previous != (self.details["folder"] if self.details else None):
            self.device.clear()
            self.device.addItem("All devices", None)
            for key, config in (self.details["devices"].items() if self.details else ()):
                self.device.addItem(f"{key}: {config.get('name', key)}", key)
            self.clear_plots()

    def clear_plots(self):
        self._plot_revision += 1
        if self._pending and self._pending["kind"] == "plot":
            self._pending = None
        self._image = QImage()
        self._plotted_folder = None
        self.image_label.setText("Recorded experiment plots appear here.")
        self.image_label.setMinimumSize(0, 0)
        self.caption.setText(PLOT_HINT)
        self.caption.setToolTip("")
        self.status.setText("Plot area cleared.")

    def selected_runs(self):
        if not self.details:
            return []
        return [self.details["runs"][r.row()] for r in self.runs.selectionModel().selectedRows()
                if r.row() < len(self.details["runs"])]

    def plot_selected(self):
        selected = [r for r in self.selected_runs() if r.get("run_id")]
        if not selected or not self.details:
            self.status.setText("Select at least one executed experiment; queued experiments have no recorded data.")
            return
        if len(selected) > 8:
            self.status.setText("Select up to eight experiments per plot to keep comparisons responsive. CSV export retains all selected experiments.")
            return
        try:
            start, end = (float(field.text()) if field.text().strip() else None
                          for field in (self.time_start, self.time_end))
            if any(value is not None and (not isfinite(value) or value < 0) for value in (start, end)):
                raise ValueError
            if start is not None and end is not None and start >= end:
                raise ValueError
        except ValueError:
            self.status.setText("Enter nonnegative elapsed seconds with From less than To, or leave either field blank.")
            return
        self._submit({"kind": "plot", "folder": self.details["folder"], "runs": [r["run_id"] for r in selected],
                      "device": self.device.currentData(), "dark": self.dark_mode,
                      "time_range": (start, end), "revision": self._plot_revision})

    def export_selected(self):
        if not self.details:
            return
        selected = self.selected_runs()
        if not selected or any(r["status"] != "completed" for r in selected):
            self.status.setText("Select fully saved, completed experiments to export.")
            return
        destination = QFileDialog.getExistingDirectory(self, "Export selected experiment CSVs")
        if destination:
            self._submit({"kind": "export", "folder": self.details["folder"], "runs": [r["run_id"] for r in selected], "destination": destination})

    def open_folder(self):
        if self.details:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.details["folder"]))

    def _show_image(self):
        if not self._image.isNull():
            pixmap = QPixmap.fromImage(self._image)
            size = pixmap.size() * (self.zoom.value() / 100)
            self.image_label.setPixmap(pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.image_label.setMinimumSize(size)

    def fit_image(self):
        if not self._image.isNull():
            self.zoom.setValue(max(50, min(200, int(100 * (self.image_scroll.viewport().width() - 10) / self._image.width()))))
            self._show_image()

    def shutdown(self):
        self._closed = True
        self.timer.stop()
        self._pending = None
        self._thread.requestInterruption()
        self._thread.quit()
        return self._thread.wait(5000)
