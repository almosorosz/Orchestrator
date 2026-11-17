"""Campaign browsing with file I/O and rendering on a dedicated worker."""
from pathlib import Path
from math import isfinite

from PySide6.QtCore import QItemSelectionModel, QObject, QSignalBlocker, QThread, QTimer, Signal, Slot, Qt, QUrl
from PySide6.QtGui import QImage, QPixmap, QDesktopServices
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QAbstractItemView, QSplitter,
    QScrollArea, QCheckBox, QSpinBox, QDialog, QDialogButtonBox, QListWidget, QListWidgetItem)

from experiment.archive import scan_campaigns, campaign_details, export_campaign_runs, plot_traces
from gui.flow_layout import FlowLayout

PLOT_HINT = "Ctrl/Shift-select campaigns or runs. With no runs selected, Plot selected uses all executed runs."


def render_recorded(campaigns, device_keys, dark, time_range=(None, None)):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from gui.render_lock import RENDER_LOCK
    from hardware_devices.limits import DEFAULT_LIMITS

    devices = {}
    series = 0
    for campaign_number, (details, run_ids) in enumerate(campaigns, 1):
        configs = details["devices"]
        run_ids = set(run_ids)
        included = None if device_keys is None else {
            key for key, kind in device_keys if configs.get(key, {}).get("type", "") == kind
        }
        if included is not None and not included:
            continue
        for run in details["runs"]:
            if run.get("run_id") not in run_ids:
                continue
            label = f"C{campaign_number} #{run['index']}" if len(campaigns) > 1 else f"#{run['index']}"
            try:
                data = plot_traces(run, device_ids=included, device_configs=configs, time_range=time_range)
            except FileNotFoundError:
                continue  # A running experiment may have a run marker before its first sample.
            for column, values in data.items():
                device, field = column.rsplit(".", 1)
                config = configs.get(device, {})
                _, traces = devices.setdefault((device, config.get("type", "")), (config, []))
                traces.append((series, label, field, values))
            series += 1
    if not devices:
        raise ValueError("No recorded numeric samples for the selected experiments, devices and time range.")
    with RENDER_LOCK:
        fig = Figure(figsize=(12, max(3.5, 2.8 * len(devices))), dpi=100, layout="constrained")
        canvas = FigureCanvasAgg(fig)
        bg, fg = ("#111827", "#e5e7eb") if dark else ("#ffffff", "#111827")
        fig.set_facecolor(bg)
        palette = ['#60a5fa', '#fb923c', '#4ade80', '#e879f9', '#facc15', '#22d3ee', '#f87171', '#c4b5fd',
                   '#a3e635', '#f472b6', '#818cf8', '#fbbf24', '#2dd4bf', '#fda4af', '#a78bfa', '#38bdf8']
        for index, ((device, kind), (config, traces)) in enumerate(sorted(devices.items()), 1):
            axis = fig.add_subplot(len(devices), 1, index)
            axis.set_facecolor(bg)
            primary_unit = DEFAULT_LIMITS.get(kind, {}).get("unit", "value")
            units = {"speed_rpm": "RPM", "flow": "mL/min", "setpoint": "°C", "temp_internal": "°C",
                     "temp_external": "°C", "weight_g": "g", "measured_voltage": "V", "torque_ozin": "oz·in",
                     "target_state": "0/1", "target": primary_unit}
            secondary = None
            for experiment, label, field, values in traces:
                secondary_field = field == "torque_ozin" or (kind == "nidaq_valve" and field in {"target", "target_state"})
                if secondary_field and secondary is None:
                    secondary = axis.twinx()
                    secondary.tick_params(colors=fg, labelsize=6)
                    secondary.set_ylabel(units.get(field, "value"), color=fg, fontsize=7)
                plot_axis = secondary if secondary_field else axis
                plot_axis.plot([v[0] for v in values], [v[1] for v in values],
                               "--" if field == "target" else ":" if field in {"temp_external", "torque_ozin"} else "-",
                               color=palette[experiment % len(palette)], linewidth=1,
                               label=f"{label}: {field} ({units.get(field, 'value')})")
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
                legend = axis.legend(handles, labels, fontsize=6, facecolor=bg, edgecolor="#64748b",
                                     ncols=min(5, (len(handles) + 7) // 8), loc="upper right")
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
            elif kind in {"plot", "export"}:
                campaigns = [(campaign_details(folder), ids) for folder, ids in job["runs"].items()]
                if kind == "plot":
                    result = render_recorded(campaigns, job["devices"], job["dark"], job["time_range"])
                else:
                    result = export_campaign_runs(campaigns, job["destination"])
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
        self.selected_campaigns = []
        self.run_rows = []
        self.device_choices = {}
        self.device_filter = None
        self.dark_mode = True
        self._busy = False
        self._pending = None
        self._closed = False
        self._image = QImage()
        self._plot_revision = 0
        layout = QVBoxLayout(self)
        self.setObjectName('campaignBrowser')
        self.setStyleSheet('''
            QWidget#campaignBrowser QPushButton { padding: 3px 6px; font-size: 12px; }
            QWidget#campaignBrowser QLineEdit,
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
        self.status = QLabel("Select campaigns to inspect their experiments.")
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
        split.setMaximumHeight(145)
        self.campaigns = QTableWidget(0, 4)
        self.campaigns.setHorizontalHeaderLabels(["Campaign", "Status", "Completed", "Recorded (UTC)"])
        self.runs = QTableWidget(0, 5)
        self.runs.setHorizontalHeaderLabels(["#", "Experiment", "Campaign", "Status", "Run ID"])
        for table in (self.campaigns, self.runs):
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.setMinimumHeight(56)
            table.verticalHeader().setDefaultSectionSize(20)
            table.horizontalHeader().setStretchLastSection(True)
            split.addWidget(table)
        self.campaigns.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.campaigns.setToolTip("Ctrl-click to select several campaigns. Shift-click to select a range.")
        self.runs.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(split)
        actions = FlowLayout(spacing=4)
        self.device = QPushButton("All devices ▾")
        self.device.setToolTip("Choose which device plots to show")
        self.device.setEnabled(False)
        self.device.clicked.connect(self.choose_devices)
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
        self.open_button.setToolTip("Open the current campaign’s recording folder.")
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
            self._update_campaigns(result["campaigns"])
            self.status.setText(f"{len(self.rows)} campaigns. " + ("Some files are temporarily unreadable; refresh to retry." if result["errors"] else ""))
        elif job["kind"] == "plot":
            self._image = result
            self.fit_image()
            count = sum(len(ids) for ids in job["runs"].values())
            caption = f"{count} runs selected across {len(job['runs'])} campaign(s) · elapsed time (s)"
            self.caption.setText(caption + (" · " + job["legend"] if job["legend"] else ""))
            self.caption.setToolTip("\n".join(job["runs"]) + "\nDisplay limited to 4,000 points per trace.")
            self.status.setText("Recorded experiment plots are ready.")
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

    def _update_campaigns(self, rows):
        previous = {campaign["folder"] for campaign in self.selected_campaigns}
        current_row = self.campaigns.currentRow()
        current = self.rows[current_row]["folder"] if 0 <= current_row < len(self.rows) else None
        first_scan = not self.rows
        self.rows = rows
        with QSignalBlocker(self.campaigns):
            self._fill(self.campaigns, [(r["name"], r["status"], f"{r['completed']}/{r['total']}", r["recorded_at"])
                                        for r in rows])
            self.campaigns.clearSelection()
            selected = []
            for index, campaign in enumerate(rows):
                for column in range(self.campaigns.columnCount()):
                    self.campaigns.item(index, column).setToolTip(campaign["folder"])
                if campaign["folder"] in previous:
                    selected.append(index)
                if campaign["folder"] == current:
                    self.campaigns.setCurrentCell(index, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
            if rows and not selected and (first_scan or previous):
                self.campaigns.selectRow(0)
            else:
                for index in selected:
                    self._select_row(self.campaigns, index)
        self._selection_changed()

    @staticmethod
    def _select_row(table, row):
        table.selectionModel().select(table.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    def _selection_changed(self):
        previous = {campaign["folder"] for campaign in self.selected_campaigns}
        chosen = {(campaign["folder"], run["experiment_id"]) for campaign, run in self.selected_runs()}
        self.selected_campaigns = [self.rows[index] for index in sorted(
            row.row() for row in self.campaigns.selectionModel().selectedRows()) if index < len(self.rows)]
        self.run_rows = [(campaign, run) for campaign in self.selected_campaigns for run in campaign["runs"]]
        with QSignalBlocker(self.runs):
            self._fill(self.runs, [(run["index"], run["name"], campaign["name"], run["status"], run.get("run_id", "—"))
                                   for campaign, run in self.run_rows])
            self.runs.clearSelection()
            for index, (campaign, run) in enumerate(self.run_rows):
                self.runs.item(index, 2).setToolTip(f"{campaign['recorded_at']} UTC\n{campaign['folder']}")
                if (campaign["folder"], run["experiment_id"]) in chosen:
                    self._select_row(self.runs, index)
        self.device_choices = {(key, config.get("type", "")): f"{key}: {config.get('name', key)}"
                               for campaign in reversed(self.selected_campaigns) for key, config in campaign["devices"].items()}
        self._update_device_button()
        if previous != {campaign["folder"] for campaign in self.selected_campaigns}:
            self.clear_plots()

    def _update_device_button(self):
        self.device.setEnabled(bool(self.selected_campaigns))
        if self.device_filter is None:
            label = "All devices"
        else:
            count = len(self.device_filter.intersection(self.device_choices))
            label = f"{count} of {len(self.device_choices)} devices"
        self.device.setText(f"{label} ▾")

    def choose_devices(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Device plots")
        dialog.resize(350, 420)
        layout = QVBoxLayout(dialog)
        devices = QListWidget()
        layout.addWidget(devices, 1)
        choices = sorted(self.device_choices.items())
        for key, label in choices:
            if any(other[0] == key[0] and other != key for other in self.device_choices):
                label += f" ({key[1] or 'unknown type'})"
            item = QListWidgetItem(label, devices)
            enabled = self.device_filter is None or key in self.device_filter
            item.setCheckState(Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked)

        def check_all(state):
            for index in range(devices.count()):
                devices.item(index).setCheckState(state)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        check = buttons.addButton("Select all", QDialogButtonBox.ButtonRole.ActionRole)
        clear = buttons.addButton("Clear selection", QDialogButtonBox.ButtonRole.ActionRole)
        check.clicked.connect(lambda: check_all(Qt.CheckState.Checked))
        clear.clicked.connect(lambda: check_all(Qt.CheckState.Unchecked))
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            checked = {key for index, (key, _) in enumerate(choices)
                       if devices.item(index).checkState() == Qt.CheckState.Checked}
            self.device_filter = None if len(checked) == len(choices) else checked
            self._update_device_button()

    def clear_plots(self):
        self._plot_revision += 1
        if self._pending and self._pending["kind"] == "plot":
            self._pending = None
        self._image = QImage()
        self.image_label.setText("Recorded experiment plots appear here.")
        self.image_label.setMinimumSize(0, 0)
        self.caption.setText(PLOT_HINT)
        self.caption.setToolTip("")
        self.status.setText("Plot area cleared.")

    def selected_runs(self):
        return [self.run_rows[index] for index in sorted(row.row() for row in self.runs.selectionModel().selectedRows())
                if index < len(self.run_rows)]

    @staticmethod
    def _group_runs(rows):
        grouped = {}
        for campaign, run in rows:
            grouped.setdefault(campaign["folder"], []).append(run["run_id"])
        return grouped

    def plot_selected(self):
        selected = [(campaign, run) for campaign, run in self.selected_runs() or self.run_rows if run.get("run_id")]
        if not selected:
            self.status.setText("Select at least one executed experiment; queued experiments have no recorded data.")
            return
        devices = None
        if self.device_filter is not None:
            devices = self.device_filter.intersection(self.device_choices)
            if not devices:
                self.status.setText("Choose at least one device plot.")
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
        runs = self._group_runs(selected)
        names = {campaign["folder"]: campaign["name"] for campaign, _ in selected}
        legend = " / ".join(f"C{i}: {names[folder]}" for i, folder in enumerate(runs, 1)) if len(runs) > 1 else ""
        self._plot_revision += 1
        self._submit({"kind": "plot", "runs": runs, "devices": devices, "dark": self.dark_mode, "legend": legend,
                      "time_range": (start, end), "revision": self._plot_revision})

    def export_selected(self):
        selected = self.selected_runs()
        if not selected or any(run["status"] != "completed" or not run.get("recording_complete") for _, run in selected):
            self.status.setText("Select fully saved, completed experiments to export.")
            return
        destination = QFileDialog.getExistingDirectory(self, "Export selected experiment CSVs")
        if destination:
            self._submit({"kind": "export", "runs": self._group_runs(selected), "destination": destination})

    def open_folder(self):
        if self.selected_campaigns:
            current = self.campaigns.currentRow()
            campaign = self.rows[current] if 0 <= current < len(self.rows) else None
            if campaign not in self.selected_campaigns:
                campaign = self.selected_campaigns[0]
            QDesktopServices.openUrl(QUrl.fromLocalFile(campaign["folder"]))

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
