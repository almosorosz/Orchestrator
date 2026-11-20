"""Full-width Settings workflow using the existing service-connected controls."""
from pathlib import Path
from textwrap import fill

import numpy as np
from PySide6.QtCore import QSignalBlocker, QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QPushButton, QScrollArea, QTabWidget, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from gui.setup_checklist import SetupChecklist, SetupItem
from gui.setup_readiness import loop_readiness
from gui.flow_layout import FlowLayout


# Labels for older component-design presets; timeline presets provide their own
# stage and device names through parameter_context.
LEGACY_CONTROLS = {
    "t_h_open": ("Valve closed hold", "h"), "be_opened_until": ("Valve open duration", "h"),
    "clean_sol": ("Cleaning solvent", "g"), "t_h_clean": ("Cleaning hold", "h"),
    "solvent": ("Rinse solvent", "g"), "t_h_solv": ("Rinse hold", "h"),
    "solution": ("Initial solution", "g"), "t_h_solution": ("Solution addition hold", "h"),
    "stirring_rate": ("Main stirring rate", "RPM"), "stirring_rate_seed": ("Seed stirring rate", "RPM"),
    "stirring_rate_sol": ("Solution stirring rate", "RPM"), "T_diss": ("Dissolution temperature", "°C"),
    "t_diss": ("Dissolution hold", "h"), "T_0": ("Initial experiment temperature", "°C"),
    "t_h_0": ("Initial temperature hold", "h"), "AS_initial": ("Initial antisolvent", "g"),
    "t_h_as_initial": ("Initial antisolvent hold", "h"), "seed": ("Seed mass", "g"),
    "t_h_seed": ("Seed addition hold", "h"), "AS_am": ("Antisolvent addition", "g"),
    "AS_add_time": ("Antisolvent addition duration", "h"), "t_h_AS": ("Antisolvent hold", "h"),
    "t_cool": ("Cooling duration", "h"), "T_final": ("Final temperature", "°C"),
    "t_h_cool": ("Final temperature hold", "h"), "t_eq": ("Equilibration duration", "h"),
    "t_boil": ("Boil-out duration", "h"), "t_h_boil": ("Boil-out hold", "h"),
}


class SettingsWorkflow:
    def _organize_workflow(self):
        layout = self.layout()
        old = layout.takeAt(0).widget()
        title = QLabel("Experiment setup")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(title)
        self.workflow_checklist = SetupChecklist()
        self.workflow_checklist.activated.connect(self._open_workflow_item)
        layout.addWidget(self.workflow_checklist)
        self.session_notice = QLabel()
        self.session_notice.setWordWrap(True)
        self.session_notice.hide()
        layout.addWidget(self.session_notice)
        layout.addWidget(self.status_label)
        layout.addWidget(self.error_label)
        self.workflow_tabs = QTabWidget()
        layout.addWidget(self.workflow_tabs, 1)

        def page(title):
            content = QWidget()
            rows = QVBoxLayout(content)
            rows.setContentsMargins(16, 12, 16, 12)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(content)
            self.workflow_tabs.addTab(scroll, title)
            return rows

        def actions(rows, *buttons):
            row = FlowLayout()
            for button in buttons:
                row.addWidget(button)
            rows.addLayout(row)

        design = page("1. Experiment design")
        design.addWidget(QLabel("Define device timelines, constants and variables, then generate the starting experiments."))
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.addRow("Experiment definition", self.profile_button)
        form.addRow("Sampling method", self.doe_type_combo)
        form.addRow("Initial experiments", self.exp_number_input)
        design.addLayout(form)
        actions(design, self.generate_button)
        self.variable_table = QTableWidget(0, 6)
        self.variable_table.setHorizontalHeaderLabels(["Stage / control", "Across experiments", "Value", "Minimum", "Maximum", "Unit"])
        self.variable_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.variable_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 6):
            self.variable_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        design.addWidget(QLabel("Defined controls · inherited targets are resolved separately for each experiment"))
        design.addWidget(self.variable_table, 1)
        counts = QHBoxLayout()
        for caption, widget in (("Varied controls:", self.changed_dimensions_count_label),
                                ("Dimensions:", self.changed_dimensions_label), ("Experiments:", self.experiment_count_label)):
            counts.addWidget(QLabel(caption))
            counts.addWidget(widget, 1)
        design.addLayout(counts)

        preview = page("2. Preview & devices")
        actions(preview, self.plot_doe_button, self.assign_doe_button, self.export_doe_button)
        self.preview_tabs = QTabWidget()
        preview.addWidget(self.preview_tabs, 1)
        profiles = QWidget()
        self.profile_preview_layout = QVBoxLayout(profiles)
        self.experiment_choice = QComboBox()
        self.experiment_choice.setPlaceholderText("Generate experiments to preview their profiles")
        self.experiment_choice.currentIndexChanged.connect(self._show_workflow_profile)
        self.profile_preview_layout.addWidget(self.experiment_choice)
        self.profile_placeholder = QLabel("Generate a DoE, or use Control Panel → View experiment profiles for manually defined experiments.")
        self.profile_placeholder.setWordWrap(True)
        self.profile_preview_layout.addWidget(self.profile_placeholder)
        self.profile_preview_layout.addStretch(1)
        self.profile_canvas = None
        self.preview_tabs.addTab(profiles, "Experiment profiles")
        diagnostics = QWidget()
        charts = QVBoxLayout(diagnostics)
        header = QHBoxLayout()
        header.addWidget(QLabel("Normalized sampling diagnostics"), 1)
        self.sampling_3d_button = QPushButton("3D view and factors…")
        self.sampling_3d_button.setEnabled(False)
        self.sampling_3d_button.clicked.connect(self._show_sampling_3d)
        header.addWidget(self.sampling_3d_button)
        charts.addLayout(header)
        sampling_scroll = QScrollArea()
        sampling_scroll.setWidgetResizable(True)
        sampling_scroll.setWidget(self.canvas_2d)
        sampling_scroll.setMinimumHeight(240)
        charts.addWidget(sampling_scroll, 1)
        self.preview_tabs.addTab(diagnostics, "Design diagnostics")
        self.sampling_3d_dialog = QDialog(self)
        self.sampling_3d_dialog.setWindowTitle("DoE samples and factors")
        detail = QVBoxLayout(self.sampling_3d_dialog)
        selectors = QFormLayout()
        self.factor_choices = []
        for name in ("X", "Y", "Z"):
            choice = QComboBox()
            choice.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            choice.currentIndexChanged.connect(self._redraw_workflow_3d)
            self.factor_choices.append(choice)
            selectors.addRow(f"{name} factor", choice)
        detail.addLayout(selectors)
        self.sampling_summary = QTableWidget(0, 3)
        self.sampling_summary.setHorizontalHeaderLabels(["Factor", "Minimum separation", "Mean separation"])
        self.sampling_summary.setToolTip("Pairwise distances within each normalized factor, as in the original sampling diagnostics.")
        self.sampling_summary.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sampling_summary.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2):
            self.sampling_summary.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.sampling_summary.setMaximumHeight(170)
        detail.addWidget(self.sampling_summary)
        detail.addWidget(self.canvas_3d, 1)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.sampling_3d_dialog.close)
        detail.addWidget(close)
        actions(preview, self.start_independent_doe_button)
        self.start_independent_doe_button.setText("Run DoE only")
        self.start_independent_doe_button.setToolTip("Execute the assigned starting experiments without automatic modeling or follow-ups.")

        loop = page("3. Automatic loop")
        loop_form = QFormLayout()
        loop_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        loop_form.addRow("Modeling method", self.loop_backend)
        loop_form.addRow("PBM size measurements", self.pbm_size_samples)
        loop_form.addRow("Final size samples to average", self.pbm_final_size_samples)

        def show_size_options():
            loop_form.setRowVisible(self.pbm_size_samples, self.loop_backend.currentData() != "bo")
            loop_form.setRowVisible(self.pbm_final_size_samples, self.uses_final_size_samples())

        self.loop_backend.currentIndexChanged.connect(show_size_options)
        self.pbm_size_samples.currentIndexChanged.connect(show_size_options)
        show_size_options()
        loop.addLayout(loop_form)
        self.loop_sequence = QLabel()
        self.loop_sequence.setWordWrap(True)
        loop.addWidget(self.loop_sequence)
        actions(loop, self.loop_setup_button, self.external_model_info_button)
        self.open_model_button = QPushButton("Open modeling settings")
        def open_model_settings():
            if self.loop_backend.currentData() == "external":
                self.show_external_model_info()
            elif self.loop_setup_button.isEnabled():
                if dialog := self.edit_loop_setup():
                    dialog.tabs.setCurrentWidget(dialog.modeling_tab)

        self.open_model_button.clicked.connect(open_model_settings)
        actions(loop, self.open_model_button)
        loop.addWidget(QLabel("Save the automatic loop setup, then start the loop from Dashboard. DoE-only execution is on Preview & devices."))
        loop.addStretch()

        files = page("4. Files & presets")
        self.folder_browse_buttons = []
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for caption, field in (("Experiment recordings", self.exp_path_input), ("Model results", self.MODEL_path_input)):
            container = QWidget()
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(field, 1)
            browse = QPushButton("Browse…")
            self.folder_browse_buttons.append(browse)
            browse.clicked.connect(lambda checked=False, field=field: self._browse_workflow_folder(field))
            row.addWidget(browse)
            open_folder = QPushButton("Open")
            open_folder.clicked.connect(lambda checked=False, field=field: self._open_workflow_folder(field))
            row.addWidget(open_folder)
            form.addRow(caption, container)
            field.textChanged.connect(field.setToolTip)
        files.addLayout(form)
        actions(files, self.define_button)
        actions(files, self.recording_button, self.reservoirs_button)
        actions(files, self.save_preset_button, self.load_preset_button)
        files.addStretch()
        old.deleteLater()
        self._workflow_creator = None
        self._workflow_normalized = np.empty((0, 0))
        self._workflow_timer = QTimer(self)
        self._workflow_timer.setInterval(900)
        self._workflow_timer.timeout.connect(self.update_workflow_checklist)
        self._workflow_timer.start()
        for control in (self.exp_path_input, self.MODEL_path_input, self.exp_number_input):
            control.textChanged.connect(lambda: self.update_workflow_checklist(use_context=False))
        self.loop_backend.currentIndexChanged.connect(self.update_workflow_checklist)
        self.update_workflow_checklist(use_context=False)

    def show_definition(self, parameters):
        from types import SimpleNamespace
        self._populate_workflow_variables(SimpleNamespace(recipe=parameters.timeline, exp_par=parameters))

    def _browse_workflow_folder(self, field):
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", field.text())
        if folder:
            field.setText(folder)
            field.editingFinished.emit()

    @staticmethod
    def _open_workflow_folder(field):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        if Path(field.text()).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(field.text()).resolve())))

    def _open_workflow_item(self, key):
        tab = {"files": 3, "batch": 0, "assignments": 1, "loop": 2}.get(key, 2)
        self.workflow_tabs.setCurrentIndex(tab)
        if key == "assignments" and self.assign_doe_button.isEnabled():
            self.assign_doe_button.click()
        elif key == "loop" and self.loop_setup_button.isEnabled():
            self.edit_loop_setup()

    def update_workflow_checklist(self, *_, use_context=True):
        if not hasattr(self, "workflow_checklist"):
            return
        roots = (self.exp_path_input.text().strip(), self.MODEL_path_input.text().strip())
        files_ready = all(value and Path(value).is_dir() for value in roots)
        items = [SetupItem("files", "Output folders", files_ready, "Choose experiment and model output folders under Files & presets."),
                 SetupItem("batch", "Starting experiments", getattr(self, "_doe_available", False), "Generate a Settings DoE or load Control Panel profiles for a manual starting batch."),
                 SetupItem("assignments", "DoE assignments", getattr(self, "_assignment_available", False), "Preview and apply DoE assignments, including required connected read-only monitors.")]
        mode = self.loop_backend.currentData()
        self.loop_sequence.setText("Initial experiments → collect data → " +
            ("BO recommendation → execute → repeat" if mode == "bo" else
             "PE → model discrimination or D-optimal design → execute → refit" if mode == "imed" else
             "external model phase → receive experiment files → execute") +
            (f". Configured follow-up limit: {self.loop_setup['max_followups']}." if mode != "external"
             else ". Supply new experiments during the external model phase; stop from Dashboard."))
        if use_context and self.isVisible() and self.loop_preview and mode != "external":
            try:
                from experiment.adaptive_loop import validate_setup

                validate_setup(self.loop_setup)
                context = self.loop_preview(mode, self.loop_setup, None)
                checks = loop_readiness(self.loop_setup, mode, context)
                pending = [check for check in checks if check.required and not check.ready]
                items.append(SetupItem("loop", "Automatic loop", not pending,
                                       pending[0].detail if pending else "Loop inputs are configured. Start from Dashboard; execution still performs full preflight."))
                if context.get("source") == "manual":
                    items[1] = SetupItem("batch", "Control Panel batch", bool(context.get("campaign")), "Starting experiments are loaded in Control Panel.")
                    items[2] = SetupItem("assignments", "Control Panel assignments", next((c.ready for c in checks if c.key == "devices"), False),
                                         next((c.detail for c in checks if c.key == "devices"), "Review Control Panel assignments."))
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                items.append(SetupItem("loop", "Automatic loop", False, str(error)))
        else:
            items.append(SetupItem("loop", "Automatic loop", False, "Open Automatic loop to review model inputs and the follow-up limit.", required=False))
        self.workflow_checklist.set_items(items)

    def _populate_workflow_variables(self, creator):
        from dataclasses import fields
        from experiment.profiles.profile_gen import DefineProcessParameter
        rows = []
        self._factor_labels = {}
        recipe = getattr(creator, "recipe", None)
        if recipe:
            from experiment.recipe import parameter_context, recipe_parameters
            for key, value in recipe_parameters(recipe):
                label, unit, scale, _ = parameter_context(recipe, key)
                self._factor_labels[key] = f"{label} ({unit})" if unit else label
                rows.append((label, "Equal to previous" if value.get("equal_previous") else value["mode"],
                             value["value"] / scale, value["min"] / scale, value["max"] / scale, unit))
        else:
            for field in fields(creator.exp_par):
                value = getattr(creator.exp_par, field.name)
                if isinstance(value, DefineProcessParameter):
                    label, unit = LEGACY_CONTROLS.get(field.name, (field.name.replace("_", " "), ""))
                    self._factor_labels[field.name] = f"{label} ({unit})" if unit else label
                    mode = "Alternate" if value.if_in_doe and value.if_only_minmax else "DoE" if value.if_in_doe else "Fixed"
                    rows.append((label, mode, value.value, value.min_val if value.if_in_doe else "—",
                                 value.max_val if value.if_in_doe else "—", unit))
        self.variable_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for col, value in enumerate(values):
                text = f"{value:.6g}" if isinstance(value, (float, int)) else str(value)
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.variable_table.setItem(row, col, item)

    def _show_workflow_profile(self):
        if self.profile_canvas:
            self.profile_preview_layout.removeWidget(self.profile_canvas)
            self.profile_canvas.hide()
            self.profile_canvas.deleteLater()
            self.profile_canvas = None
        creator = self._workflow_creator
        index = self.experiment_choice.currentIndex()
        if creator is None or index < 0:
            self.profile_placeholder.show()
            return
        figures = getattr(self, "_workflow_profile_figures", [])
        if index < len(figures):
            self.profile_canvas = FigureCanvas(figures[index])
            self.profile_canvas.setMinimumHeight(220)
            self.profile_canvas.setMaximumHeight(320)
            self.profile_preview_layout.insertWidget(self.profile_preview_layout.count() - 1, self.profile_canvas)
            self.profile_placeholder.hide()

    def _redraw_workflow_diagnostics(self):
        creator = self._workflow_creator
        if creator is None:
            return
        samples = np.asarray(creator.DoE_samples)
        keys = self._workflow_factor_keys
        self.fig_2d.clear()
        dimensions = min(samples.shape[1], len(keys)) if samples.ndim == 2 else 0
        self.sampling_3d_button.setEnabled(dimensions > 0)
        if not dimensions:
            self._workflow_normalized = np.empty((0, 0))
            self.sampling_summary.setRowCount(0)
            self.sampling_3d_dialog.close()
            for canvas in (self.canvas_2d, self.canvas_3d):
                self._show_plot_unavailable(canvas, getattr(self, "_workflow_dark", True))
            return
        data = creator.doe_sampler.scaler(samples)[:, :dimensions]
        self._workflow_normalized = data
        self.sampling_summary.setRowCount(dimensions)
        self.canvas_2d.setMinimumWidth(max(500, dimensions * 185))
        axes = self.fig_2d.add_subplot(111)
        for index in range(dimensions):
            label = self._factor_labels.get(keys[index], keys[index])
            axes.scatter(np.full(len(data), index), data[:, index], label=label)
            ordered = np.sort(data[:, index])
            count = len(ordered)
            minimum = float(np.min(np.diff(ordered))) if count > 1 else None
            mean = float(2 * np.dot(2 * np.arange(count) - count + 1, ordered) / (count * (count - 1))) if count > 1 else None
            for column, text in enumerate((label, f"{minimum:.3f}" if minimum is not None else "—",
                                            f"{mean:.3f}" if mean is not None else "—")):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.sampling_summary.setItem(index, column, item)
        axes.set_xticks(range(dimensions))
        axes.set_xticklabels([fill(self._factor_labels.get(k, k), 23) for k in keys[:dimensions]], fontsize=8)
        axes.set_ylabel("Normalized sampled value")
        axes.set_title(f"Sampling method: {creator.TYPE}")
        self.fig_2d.subplots_adjust(bottom=.35)
        if self.sampling_3d_dialog.isVisible():
            self._redraw_workflow_3d()
        else:
            self.update_plot_theme(getattr(self, "_workflow_dark", True))

    def _show_sampling_3d(self):
        dialog = self.sampling_3d_dialog
        if not dialog.isVisible():
            available = self.screen().availableGeometry()
            dialog.resize(min(1000, int(available.width() * .9)), min(740, int(available.height() * .88)))
        self._redraw_workflow_3d()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _redraw_workflow_3d(self, *_):
        data = self._workflow_normalized
        keys = self._workflow_factor_keys
        self.fig_3d.clear()
        if data.shape[1] >= 3:
            indices = [choice.currentIndex() for choice in self.factor_choices]
            axes = self.fig_3d.add_subplot(111, projection="3d")
            axes.scatter(*(data[:, index] for index in indices))
            for choice, setter, index in zip(self.factor_choices, (axes.set_xlabel, axes.set_ylabel, axes.set_zlabel), indices):
                label = self._factor_labels.get(keys[index], keys[index])
                choice.setToolTip(label)
                setter(fill(label, 27), fontsize=8)
        else:
            axes = self.fig_3d.add_subplot(111)
            axes.text(.5, .5, "3D diagnostics need at least three sampled factors.", ha="center", va="center", wrap=True)
            axes.axis("off")
        self.update_plot_theme(getattr(self, "_workflow_dark", True))

    def render_workflow(self, creator, dark):
        self._workflow_creator, self._workflow_dark = creator, dark
        self._populate_workflow_variables(creator)
        keys = list(creator.DoE_val)
        if getattr(creator, "recipe", None):
            from experiment.recipe import recipe_parameters, sampled_parameter
            keys = [key for key, value in recipe_parameters(creator.recipe) if sampled_parameter(value)]
        self._workflow_factor_keys = keys
        samples = np.asarray(creator.DoE_samples)
        dimensions = samples.shape[1] if samples.ndim == 2 else 0
        for index, choice in enumerate(self.factor_choices):
            with QSignalBlocker(choice):
                choice.clear()
                for key in keys[:dimensions]:
                    choice.addItem(self._factor_labels.get(key, key), key)
                choice.setCurrentIndex(min(index, choice.count() - 1))
                choice.setEnabled(dimensions >= 3)
        self._workflow_profile_figures = creator.plot_doe_experiments()
        with QSignalBlocker(self.experiment_choice):
            self.experiment_choice.clear()
            self.experiment_choice.addItems(list(creator.exp_storage.data))
            if self.experiment_choice.count():
                self.experiment_choice.setCurrentIndex(0)
        self._show_workflow_profile()
        self._redraw_workflow_diagnostics()
