"""Edit automatic LOOP settings with the same schema used by setup presets."""
from copy import deepcopy
from math import isfinite

from PySide6.QtCore import QLocale, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QScrollArea,
    QSpinBox, QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from experiment import adaptive_loop
from experiment.adaptive_loop import DEVICE_RECORDING_COLUMNS, validate_setup
from experiment.doe_campaign import ROLES
from hardware_devices.capabilities import is_monitor_only
from bo.profiles import make_profile
from experiment.loop_process import PROCESS_FIELDS
from imed_python.campaign_import import CHANNELS
from imed_python.demo_campaign import MONITOR_TYPES
from imed_python.size_data import FINAL_SIZE_SAMPLE_TOOLTIP, FINAL_SIZE_TERM, MAX_FINAL_SIZE_SAMPLES, SIZE_SAMPLE_OPTIONS
from gui.loop_setup_preview import PREVIEW_AMOUNTS, starting_input_rows, used_loop_values, uses_followup_offset, value_text
from gui.experiment_period_widget import ExperimentPeriodWidget
from gui.setup_checklist import SetupChecklist, SetupItem
from gui.setup_readiness import loop_readiness


PROCESS_INPUTS = {**PROCESS_FIELDS, "InitialConcentration": "Initial concentration (g/g solution)"}
PROCESS_OVERRIDES = {**PROCESS_INPUTS, "InitialVolume_ml": PREVIEW_AMOUNTS["InitialVolume_ml"]}
DESIGN_VALUES = {"batch": "Batch duration (s)", "concentration": "Initial concentration (g/g solution)",
                 "final_anti": "Final antisolvent mass fraction"}
DESIGN_PAIRS = {
    "temperature": ("Temperature (°C)", "Initial", "Final"),
    "temperature_rates": ("Temperature ramp rate (°C/s)", "Minimum", "Maximum"),
    "liquid_masses": ("Initial liquid masses (g)", "Solvent", "Antisolvent"),
    "antisolvent_rates": ("Antisolvent rate (g/s)", "Minimum", "Maximum"),
    "seed_bounds": ("Seed mass (g)", "Minimum", "Maximum"),
}


def number_field(value=None, hint=""):
    field = QLineEdit("" if value is None else str(value))
    validator = QDoubleValidator(field)
    locale = QLocale.c()
    locale.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
    validator.setLocale(locale)
    field.setValidator(validator)
    field.setPlaceholderText(hint)
    return field


def entered_number(field, name, *, optional=False):
    text = field.text().strip()
    if optional and not text:
        return None
    try:
        value = float(text)
        if isfinite(value):
            return value
    except ValueError:
        pass
    raise ValueError(f"Enter a finite number for {name}.")


def integer_field(value, minimum, maximum):
    field = QSpinBox()
    field.setRange(minimum, maximum)
    field.setValue(value)
    return field


def text_choice(choices, value):
    field = QComboBox()
    field.setEditable(False)
    field.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    field.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    field.setMinimumContentsLength(12)
    field.addItems(choices)
    if value and value not in choices:
        field.addItem(value)
        field.setItemData(field.count() - 1, "Saved selection is unavailable. Choose a currently configured device/channel.", Qt.ItemDataRole.ToolTipRole)
    field.setCurrentText(value)
    return field


def explanation(layout, text):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setStyleSheet("color: #94a3b8; font-size: 11px;")
    if isinstance(layout, QFormLayout):
        layout.addRow(label)
    else:
        layout.addWidget(label)
    return label


def toggle_group(title, enabled):
    group = QGroupBox(title)
    group.setCheckable(True)
    group.setChecked(enabled)
    return group


def edit_table(labels, add_text, add_row):
    table = QTableWidget(0, len(labels))
    table.setHorizontalHeaderLabels(labels)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    add = QPushButton(add_text)
    add.clicked.connect(add_row)
    remove = QPushButton("Remove selected")
    remove.setEnabled(False)
    table.currentCellChanged.connect(lambda row, *_: remove.setEnabled(row >= 0))
    remove.clicked.connect(lambda: table.removeRow(table.currentRow()))
    buttons = QHBoxLayout()
    buttons.addWidget(add)
    buttons.addWidget(remove)
    buttons.addStretch()
    return table, buttons


class SupportProfiles(QWidget):
    def __init__(self, profiles):
        super().__init__()
        self.editors = {}
        self.unknown = {role: profile for role, profile in profiles.items() if role not in ROLES or role in {"T", "AS"}}
        self.roles = QComboBox()
        self.pages = QStackedWidget()
        layout = QVBoxLayout(self)
        explanation(layout, "Support profiles supply fixed device schedules in simple profile-based follow-ups. "
                    "Stage-based runs inherit preparation, seeding, stirring and cleanup from the starting timeline.")
        header = QFormLayout()
        header.addRow("Profile role", self.roles)
        layout.addLayout(header)
        layout.addWidget(self.pages, 1)
        self.roles.currentIndexChanged.connect(self.pages.setCurrentIndex)
        for role, details in ROLES.items():
            if role in {"T", "AS"}:
                continue
            profile = profiles.get(role, {})
            if not isinstance(profile, dict):
                raise ValueError(f"Invalid {details.label.lower()} support profile.")
            group = toggle_group("Use support profile", role in profiles)
            editor = QVBoxLayout(group)
            interpolation = QComboBox()
            interpolation.addItem("Linear", False)
            interpolation.addItem("Step", True)
            interpolation.setCurrentIndex(int(bool(profile.get("step", role == "valve_state"))))
            form = QFormLayout()
            form.addRow("Interpolation", interpolation)
            editor.addLayout(form)
            table, buttons = edit_table(("Time (s)", f"Target ({details.unit})"), "Add point",
                                        lambda checked=False, role=role: self.add_point(role))
            editor.addWidget(table, 1)
            editor.addLayout(buttons)
            self.editors[role] = group, interpolation, table
            points = profile.get("points")
            if points is not None:
                trace = make_profile(points, "Support profile")
                if trace is None:
                    raise ValueError(f"Invalid {details.label.lower()} support timeline.")
                for time, target in zip(trace.time, trace.values):
                    self.add_point(role, time, target)
            self.pages.addWidget(group)
            self.roles.addItem(details.label, role)
        first_enabled = next((role for role in self.editors if role in profiles), None)
        if first_enabled:
            self.roles.setCurrentIndex(self.roles.findData(first_enabled))

    def add_point(self, role, time=None, value=None):
        _, _, table = self.editors[role]
        row = table.rowCount()
        table.insertRow(row)
        table.setCellWidget(row, 0, number_field(time if time is not None else 0 if row == 0 else None))
        table.setCellWidget(row, 1, number_field(value))
        table.setCurrentCell(row, 0)

    def values(self):
        result = dict(self.unknown)
        for role, (group, interpolation, table) in self.editors.items():
            if not group.isChecked():
                continue
            label = ROLES[role].label
            times, values = [], []
            for row in range(table.rowCount()):
                times.append(entered_number(table.cellWidget(row, 0), f"{label} time in row {row + 1}"))
                values.append(entered_number(table.cellWidget(row, 1), f"{label} target in row {row + 1}"))
            if not times or times[0] != 0 or any(a >= b for a, b in zip(times, times[1:])):
                raise ValueError(f"{label} needs a timeline starting at zero with strictly increasing times.")
            result[role] = {"points": [times, values], "step": interpolation.currentData()}
        return result


class LoopSetupDialog(QDialog):
    navigate_requested = Signal(str)

    def __init__(self, setup, backend, device_configs=None, parent=None, *, preview=None, size_preferences=None):
        # The dialog is imported on first use. If the app was open during a
        # source update, the already loaded loop module may be older.
        setup_for_editing = getattr(adaptive_loop, "setup_for_editing", None)
        if setup_for_editing is None:
            raise ValueError("CrystalLoop was updated while this window was open. Close and reopen the app to load the updated loop settings.")
        super().__init__(parent)
        self.setWindowTitle("Automatic loop setup")
        available = self.screen().availableGeometry()
        self.resize(min(1000, int(available.width() * .94)), min(800, int(available.height() * .90)))
        self.setMinimumSize(550, 460)
        supported, unsupported = setup_for_editing(setup)
        self.setup = deepcopy(supported)
        self.configs = device_configs or {}
        self.backend = backend
        self.preview = preview
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(12)
        if unsupported:
            fields = ", ".join(unsupported)
            self.compatibility_note = QLabel(f"This version does not support the saved fields {fields}. "
                                             "Review the settings below. Save removes those fields from the active setup.")
            self.compatibility_note.setTextFormat(Qt.TextFormat.PlainText)
            self.compatibility_note.setWordWrap(True)
            layout.addWidget(self.compatibility_note)
        self.readiness = SetupChecklist(parent=self)
        self.readiness.activated.connect(self._navigate_check)
        layout.addWidget(self.readiness)
        self._refreshing_checklist = False
        run = QGroupBox("Bayesian optimization" if backend == "bo" else "IMED")
        form = QFormLayout(run)
        self.followup_form = form
        self.max_followups = integer_field(self.setup["max_followups"], 1, 1000)
        self.max_followups.setMaximumWidth(140)
        self.max_followups.setToolTip("Number of recommended experiments to execute after the starting batch.")
        self.followup_start = number_field(self.setup["followup_start_s"])
        self.followup_start.setToolTip("Used for profiles without a defined experiment period. Stage-based follow-ups keep the preparation duration from the starting run.")
        form.addRow("Follow-up experiments", self.max_followups)
        form.addRow("Start delay for unstaged profiles (s)", self.followup_start)
        explanation(form, "The loop runs this many recommendations after the starting batch. Recommendations update the selected experiment period.")
        layout.addWidget(run)
        self.tabs = QTabWidget()
        self.modeling_tab = self.modeling_page(size_preferences)
        self.tabs.addTab(self.modeling_tab, "BO settings" if backend == "bo" else "IMED settings")
        self.tabs.addTab(self.recording_page(), "Recording")
        self.tabs.addTab(self.process_page(), "Process inputs")
        self.tabs.addTab(self.devices_page(), "Devices")
        self.support = SupportProfiles(self.setup["support_profiles"])
        advanced = QWidget()
        advanced_layout = QVBoxLayout(advanced)
        explanation(advanced_layout, "Optional overrides and simulation controls. Normal runs inherit their defined preparation and cleanup profiles.")
        advanced_layout.addWidget(self.overrides_group)
        advanced_layout.addWidget(self.support)
        advanced_layout.addWidget(self.simulation_group)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(advanced)
        self.tabs.addTab(scroll, "Advanced")
        if backend == "imed":
            self.tabs.addTab(self.design_page(), "IMED design")
        recording = self.setup["recording"]
        self.manual_start = number_field(recording["start_s"])
        self.manual_end = number_field(recording["end_s"], "End of programmed run")
        self.period = ExperimentPeriodWidget(self.setup["experiment_period"],
                                             manual_start=self.manual_start, manual_end=self.manual_end)
        self.period.mode.currentIndexChanged.connect(self.update_followup_start)
        self.update_followup_start()
        self.tabs.insertTab(0, self.period, "Experiment period")
        if self.preview is not None:
            self.used_values_tab = self.preview_page()
            self.tabs.insertTab(0, self.used_values_tab, "Used values")
            self.tabs.setCurrentWidget(self.used_values_tab)
            self.tabs.currentChanged.connect(self._tab_changed)
            self.refresh_preview()
        layout.addWidget(self.tabs, 1)
        self.error = QLabel()
        self.error.setTextFormat(Qt.TextFormat.PlainText)
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #dc2626;")
        self.error.hide()
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        navigation = QHBoxLayout()
        for title, key in (("Open Control Panel", "control"),
                           ("Open BO" if backend == "bo" else "Open IMED", backend),
                           ("Output folders", "folders")):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, key=key: self.navigate_requested.emit(key))
            navigation.addWidget(button)
        layout.insertLayout(1, navigation)
        notice = QLabel("Save applies these settings. You can navigate to another page and return to this draft from Automatic loop setup.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.check_timer = QTimer(self)
        self.check_timer.setSingleShot(True)
        self.check_timer.setInterval(180)
        self.check_timer.timeout.connect(self.refresh_checklist)
        for field in self.findChildren(QLineEdit):
            field.textChanged.connect(lambda *_: self.check_timer.start())
        for field in self.findChildren(QComboBox):
            field.currentIndexChanged.connect(lambda *_: self.check_timer.start())
        for field in self.findChildren(QCheckBox):
            field.toggled.connect(lambda *_: self.check_timer.start())
        for field in self.findChildren(QSpinBox):
            field.valueChanged.connect(lambda *_: self.check_timer.start())
        for field in self.findChildren(QTableWidget):
            field.itemChanged.connect(lambda *_: self.check_timer.start())
        self.refresh_checklist()
        self.live_checks = QTimer(self)
        self.live_checks.setInterval(1000)
        self.live_checks.timeout.connect(lambda: self.refresh_checklist() if self.isVisible() else None)
        self.live_checks.start()

    def _navigate_check(self, key):
        tab = {"batch": "Used values", "period": "Experiment period", "devices": "Devices",
               "recording": "Recording", "inputs": "Process inputs", "design": "IMED design", "advanced": "Advanced"}.get(key)
        if key == "limits":
            self.max_followups.setFocus()
            return
        if key == "pe":
            self.navigate_requested.emit("objectives")
            return
        if key == "model":
            self.navigate_requested.emit("bo_settings" if self.backend == "bo" else "imed_models")
            return
        if tab:
            for index in range(self.tabs.count()):
                if self.tabs.tabText(index) == tab:
                    self.tabs.setCurrentIndex(index)
                    break
        else:
            self.navigate_requested.emit(key)

    def refresh_checklist(self):
        if self._refreshing_checklist:
            return
        self._refreshing_checklist = True
        try:
            draft = self.collect()
            if self.preview is None:
                items = [SetupItem("batch", "Starting batch", False, "Open this dialog from Settings to check the assigned campaign.")]
            else:
                context = self.preview(self.backend, draft, self.preview_source.currentData())
                items = loop_readiness(draft, self.backend, context)
            self.readiness.set_items(items)
            if self.preview is not None:
                self.refresh_preview(draft=draft, context=context)
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.readiness.set_items([SetupItem("inputs", "Complete current values", False, str(error))])
            if self.preview is not None:
                self.process_value_note.setText(str(error))
                self.process_values.setEnabled(False)
                self.reset_starting_value.setEnabled(False)
        finally:
            self._refreshing_checklist = False

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "check_timer"):
            self.check_timer.start()

    def update_followup_start(self):
        self.followup_form.setRowVisible(self.followup_start,
                                        uses_followup_offset(self.period.mode.currentData(), self.period.context))

    def preview_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.preview_source = QComboBox()
        self.preview_source.addItem("Settings DoE", "doe")
        self.preview_source.addItem("Control Panel profiles", "manual")
        self.preview_source.setCurrentIndex(-1)
        self.preview_source.currentIndexChanged.connect(self.refresh_preview)
        row.addWidget(QLabel("Starting batch"))
        row.addWidget(self.preview_source, 1)
        refresh = QPushButton("Refresh values")
        refresh.clicked.connect(self.refresh_preview)
        row.addWidget(refresh)
        layout.addLayout(row)
        self.preview_note = QLabel()
        self.preview_note.setTextFormat(Qt.TextFormat.PlainText)
        self.preview_note.setWordWrap(True)
        layout.addWidget(self.preview_note)
        self.used_values = QTreeWidget()
        self.used_values.setHeaderLabels(["Setting", "Used value", "Source"])
        self.used_values.setUniformRowHeights(True)
        self.used_values.setColumnWidth(0, 210)
        self.used_values.setColumnWidth(1, 220)
        layout.addWidget(self.used_values, 1)
        return page

    def _tab_changed(self, index):
        if self.tabs.widget(index) in (self.used_values_tab, self.process_tab):
            self.refresh_preview()

    def refresh_preview(self, *, draft=None, context=None):
        try:
            if context is None:
                try:
                    draft = self.collect()
                except ValueError:
                    draft = self.collect(validate_period=False)
                context = self.preview(self.backend, draft, self.preview_source.currentData())
            self._refresh_device_choices(context.get("configs", self.configs))
            self.period.set_context(context)
            self.update_followup_start()
            if hasattr(self, "reactor"):
                selected = self.reactor.currentData()
                with QSignalBlocker(self.reactor):
                    self.reactor.clear()
                    self.reactor.addItem("Automatic from MSI / assignments", "")
                    for unit in context.get("topology", {}).get("unit_operations", []):
                        self.reactor.addItem(f"{unit['code']}: {unit['name']}", unit["code"])
                    if selected and self.reactor.findData(selected) < 0:
                        self.reactor.addItem(f"{selected} [missing]", selected)
                    self.reactor.setCurrentIndex(max(0, self.reactor.findData(selected)))
            current = self.collect()
            if current != draft:
                draft = current
                context = self.preview(self.backend, draft, context["source"])
            labels = {**DESIGN_VALUES, **{key: values[0] for key, values in DESIGN_PAIRS.items()}}
            groups = used_loop_values(draft, self.backend, context, labels)
            self.refresh_starting_values(draft, context)
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.preview_note.setText(f"Used values could not be refreshed. {error}")
            self.preview_note.show()
            self.process_value_note.setText(str(error))
            self.process_values.setEnabled(False)
            return
        with QSignalBlocker(self.preview_source):
            self.preview_source.setCurrentIndex(self.preview_source.findData(context["source"]))
        self.preview_note.setText("\n".join(context["messages"]))
        self.preview_note.setVisible(bool(context["messages"]))
        expanded = {self.used_values.topLevelItem(i).text(0): self.used_values.topLevelItem(i).isExpanded()
                    for i in range(self.used_values.topLevelItemCount())}
        self.used_values.clear()
        for title, settings in groups.items():
            if not settings:
                continue
            group = QTreeWidgetItem(self.used_values, [title])
            group.setFirstColumnSpanned(True)
            for name, value, origin in settings:
                row = QTreeWidgetItem(group, [name, value_text(value), origin])
                for column in range(3):
                    row.setToolTip(column, row.text(column))
            group.setExpanded(expanded.get(title, title in {"DoE decision variables", "Device assignments", "Starting process inputs"}))

    def _refresh_device_choices(self, configs):
        """Keep a draft usable after adding a device on another page."""
        self.configs = configs
        def update(field, choices):
            selected = field.currentText()
            available = list(choices)
            if selected not in available:
                available.append(selected)
            if available == [field.itemText(index) for index in range(field.count())]:
                return
            with QSignalBlocker(field):
                field.clear()
                field.addItems(available)
                field.setCurrentText(selected)
        choices = sorted(f"{device}.{name}" for device, cfg in configs.items()
                         for name in DEVICE_RECORDING_COLUMNS.get(cfg.get("type"), ()))
        for field in self.channels.values():
            update(field, ["Automatic", "Not recorded", *choices])
        for role, field in self.assignments.items():
            update(field, ["Automatic", *sorted(device for device, cfg in configs.items()
                    if cfg.get("type") in ROLES[role].compatible_types and not is_monitor_only(cfg))])
        for role, field in self.simulated_monitors.items():
            update(field, ["", *sorted(device for device, cfg in configs.items()
                    if cfg.get("type") == MONITOR_TYPES[role] and cfg.get("simulated"))])

    @staticmethod
    def scroll_form():
        content = QWidget()
        form = QFormLayout(content)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        scroll = QScrollArea()
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll, form

    def recording_page(self):
        page, form = self.scroll_form()
        explanation(form, "Experiment period selects the data used for modeling; the complete run is still recorded. "
                    "Channel menus select actual device signals. Automatic uses MSI measurement labels, then conventional channel IDs.")
        recording = self.setup["recording"]
        self.recording_numbers = {}
        for key, label, hint in (
            ("concentration_multiplier", "Concentration multiplier to g/g solution", ""),
            ("initial_concentration", "Initial concentration (g/g solution)", "Use process inputs or the recorded starting value"),
        ):
            field = number_field(recording[key], hint)
            self.recording_numbers[key] = field, label
            form.addRow(label, field)
        self.no_feed = QCheckBox("This campaign has no antisolvent feed")
        self.no_feed.setChecked(recording["no_feed"])
        form.addRow(self.no_feed)
        if self.backend == "imed":
            objectives = QPushButton("IMED objectives…")
            objectives.clicked.connect(lambda: self.navigate_requested.emit("objectives"))
            form.addRow(objectives)
            explanation(form, "Only objectives with positive weights require a measurement channel. Enter an initial "
                        "concentration for runs without a concentration trace. When AS is added, record the scale "
                        "or use a calibrated AS pump. Cooling-only runs can leave both unassigned.")
        self.direction = QComboBox()
        self.direction.addItem("Reservoir weight decreases", -1)
        self.direction.addItem("Added weight increases", 1)
        self.direction.setCurrentIndex(self.direction.findData(recording["scale_direction"]))
        form.addRow("Antisolvent scale direction", self.direction)
        self.smoothing = integer_field(recording["smoothing"], 1, 101)
        form.addRow("Scale median window (samples)", self.smoothing)
        measurements = QGroupBox("Recording channel mappings")
        channels = QFormLayout(measurements)
        choices = sorted(f"{device}.{field}" for device, config in self.configs.items()
                         for field in DEVICE_RECORDING_COLUMNS.get(config.get("type"), ()))
        self.channels = {}
        for role in CHANNELS:
            saved = self.setup["channels"].get(role)
            field = text_choice(["Automatic", "Not recorded", *choices],
                                "Automatic" if saved is None else saved or "Not recorded")
            self.channels[role] = field
            channels.addRow(role, field)
        form.addRow(measurements)
        return page

    def modeling_page(self, size_preferences):
        page, form = self.scroll_form()
        explanation(form, "These editors share the settings on the BO and IMED pages. Save this window to apply the loop configuration.")
        if self.backend == "bo":
            buttons = (("BO targets, bounds and profiles…", "bo_settings"),
                       ("Solubility for theoretical yield…", "solubility"))
        else:
            buttons = (("IMED objectives…", "objectives"), ("Solubility…", "solubility"),
                       ("Add candidate model…", "imed_candidate"), ("Model configurations…", "imed_models"),
                       ("PBM solver and CPU workers…", "imed_runtime"), ("PE grid and optimizer…", "imed_fit"))
        for title, key in buttons:
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, key=key: self.navigate_requested.emit(key))
            form.addRow(button)
        if size_preferences is not None:
            main_term, main_count = size_preferences
            count = integer_field(main_count.value(), 1, MAX_FINAL_SIZE_SAMPLES)
            count.setToolTip(FINAL_SIZE_SAMPLE_TOOLTIP)
            if self.backend == "imed":
                terms = QComboBox()
                for label, value in SIZE_SAMPLE_OPTIONS:
                    terms.addItem(label, value)
                terms.setCurrentIndex(main_term.currentIndex())
                form.addRow("PBM size measurements", terms)
                terms.currentIndexChanged.connect(main_term.setCurrentIndex)
                main_term.currentIndexChanged.connect(terms.setCurrentIndex)
            form.addRow("Final size samples to average", count)
            count.valueChanged.connect(main_count.setValue)
            main_count.valueChanged.connect(count.setValue)
            if self.backend == "imed":
                def show_count():
                    form.setRowVisible(count, terms.currentData() == FINAL_SIZE_TERM)

                terms.currentIndexChanged.connect(show_count)
                show_count()
        return page

    def process_page(self):
        page, form = self.scroll_form()
        self.process_tab = page
        explanation(form, "Process values describe the beginning of the selected experiment period, after preparation. "
                    "Edits here change model inputs. Pump commands come from the experiment definition.")
        self.derive_charges = QCheckBox("Derive initial liquid charges from preparation pump profiles")
        self.derive_charges.setChecked(self.setup["derive_initial_charges"])
        form.addRow(self.derive_charges)
        if self.preview is not None:
            row = QHBoxLayout()
            self.starting_experiment = QComboBox()
            self.starting_experiment.currentIndexChanged.connect(self.show_starting_values)
            row.addWidget(self.starting_experiment, 1)
            self.reset_starting_value = QPushButton("Use automatic value")
            self.reset_starting_value.setEnabled(False)
            self.reset_starting_value.clicked.connect(self.clear_starting_override)
            row.addWidget(self.reset_starting_value)
            form.addRow("Starting experiment", row)
            self.starting_values = {}
            self.process_values = QTableWidget(0, 3)
            self.process_values.setHorizontalHeaderLabels(["Process input", "Used value", "Source"])
            self.process_values.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.process_values.verticalHeader().hide()
            self.process_values.setFixedHeight(160)
            self.process_values.itemChanged.connect(self.edit_starting_value)
            self.process_values.currentCellChanged.connect(self.update_reset_starting_value)
            form.addRow(self.process_values)
            self.process_value_note = explanation(form, "Apply device assignments to preview starting amounts.")
        explanation(form, "Common inputs apply across the batch. Blank fields can inherit values from the case or recording settings.")
        self.process = {}
        for key, title in PROCESS_INPUTS.items():
            self.process[key] = number_field(self.setup["process"].get(key), "Use case or recording setting")
            form.addRow(title, self.process[key])
        if self.backend == "imed":
            self.process["PowerRatio"].setEnabled(False)
            self.process["PowerRatio"].setToolTip("IMED uses calculated average mixing power, or 1 when that option is unchecked.")
        seed = self.setup["recording"]["seed_percentiles"]
        self.seed_sizes = [number_field(value, label) for label, value in zip(("D10", "D50", "D90"), seed or [None] * 3)]
        row = QHBoxLayout()
        for field in self.seed_sizes:
            row.addWidget(field)
        form.addRow("Seed D10 / D50 / D90 (µm)", row)
        explanation(form, "An unseeded experiment uses Seed mass = 0 g and seed D10/D50/D90 = 0/0/0 (or blank). Positive seed mass requires a positive seed distribution.")
        group = QGroupBox("Starting experiment overrides")
        self.overrides_group = group
        overrides = QVBoxLayout(group)
        explanation(overrides, "Override a common model input for a named starting experiment, for example exp3 → Seed mass. "
                    "The physical experiment must deliver the matching charge; this table changes modeling inputs only.")
        self.overrides, buttons = edit_table(("Experiment name", "Process input", "Value"), "Add override",
                                             lambda: self.add_override())
        self.overrides.setMinimumHeight(125)
        overrides.addWidget(self.overrides)
        overrides.addLayout(buttons)
        form.addRow(group)
        for name, values in self.setup["initial_processes"].items():
            for key, value in values.items():
                if key in PROCESS_OVERRIDES:
                    self.add_override(name, key, value)
        return page

    def refresh_starting_values(self, draft, context):
        stale = context.get("assignments_stale")
        self.starting_values = {} if stale else starting_input_rows(draft, context)
        selected = self.starting_experiment.currentText()
        with QSignalBlocker(self.starting_experiment):
            self.starting_experiment.clear()
            self.starting_experiment.addItems(self.starting_values)
            if selected in self.starting_values:
                self.starting_experiment.setCurrentText(selected)
        case_record = next(iter(context["case"].get("ProcInf", [])), {})
        for key, field in self.process.items():
            saved = case_record.get(key)
            if key == "InitialConcentration" and saved is None:
                saved = draft["recording"]["initial_concentration"]
            field.setPlaceholderText(f"Use {value_text(saved)}" if saved is not None else "Use case or recording setting")
        self.show_starting_values()
        if stale:
            self.process_value_note.setText("Reapply the DoE device assignments to refresh these amounts.")

    def show_starting_values(self, *_):
        if self.process_values.state() == QAbstractItemView.State.EditingState:
            return
        data = self.starting_values.get(self.starting_experiment.currentText())
        self.process_values.setEnabled(bool(data))
        with QSignalBlocker(self.process_values):
            self.process_values.setRowCount(len(PREVIEW_AMOUNTS) if data else 0)
            if data:
                values, sources, issues, editable = data
                for row, (key, label) in enumerate(PREVIEW_AMOUNTS.items()):
                    value = values.get(key)
                    source = "Needs input" if key in issues else sources.get(key, "Needs input")
                    for column, text in enumerate((label, "" if value is None else value_text(value), source)):
                        item = QTableWidgetItem(text)
                        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                        if column == 1 and editable:
                            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                            item.setData(Qt.ItemDataRole.UserRole, value)
                        detail = issues.get(key, text)
                        if column == 2 and source == "Preparation pump profiles":
                            charges = values.get("PreparationCharges_g", {})
                            materials = ("AS",) if key == "AntiSolvMass" else ("solvent", "solution")
                            detail = "\n".join(f"{role.capitalize()} charge · {charges[role]:.8g} g"
                                               for role in materials if role in charges)
                        item.setToolTip(detail)
                        self.process_values.setItem(row, column, item)
        if data:
            self.process_value_note.setText("\n".join(issues.values()) if issues else
                                            "Edit a value to override it for this experiment." if editable else
                                            "The PBM validation simulator supplies these values.")
        else:
            self.process_value_note.setText("Apply device assignments to preview starting amounts.")
        self.update_reset_starting_value()

    def starting_override_row(self, row):
        if not 0 <= row < len(PREVIEW_AMOUNTS):
            return None
        key = tuple(PREVIEW_AMOUNTS)[row]
        name = self.starting_experiment.currentText()
        return next((i for i in range(self.overrides.rowCount())
                     if self.overrides.cellWidget(i, 0).text().strip() == name
                     and self.overrides.cellWidget(i, 1).currentData() == key), None)

    def update_reset_starting_value(self, *_):
        data = self.starting_values.get(self.starting_experiment.currentText())
        row = self.starting_override_row(self.process_values.currentRow())
        self.reset_starting_value.setEnabled(bool(data and data[3] and row is not None))

    def clear_starting_override(self):
        row = self.starting_override_row(self.process_values.currentRow())
        if row is not None:
            self.overrides.removeRow(row)
            self.refresh_checklist()

    def edit_starting_value(self, item):
        if item.column() != 1 or not item.flags() & Qt.ItemFlag.ItemIsEditable:
            return
        key = tuple(PREVIEW_AMOUNTS)[item.row()]
        text = item.text().strip()
        index = self.starting_override_row(item.row())
        if not text:
            if index is not None:
                self.overrides.removeRow(index)
            self.check_timer.start()
            return
        positive = key in {"SolvMass", "InitialVolume_ml"}
        try:
            value = float(text)
            if not isfinite(value) or value < 0 or positive and value == 0:
                raise ValueError
        except ValueError:
            with QSignalBlocker(self.process_values):
                prior = item.data(Qt.ItemDataRole.UserRole)
                item.setText("" if prior is None else value_text(prior))
            self.process_value_note.setText(f"Enter a {'positive' if positive else 'nonnegative'} value for {PREVIEW_AMOUNTS[key]}.")
            return
        if index is None:
            self.add_override(self.starting_experiment.currentText(), key, value)
        else:
            self.overrides.cellWidget(index, 2).setText(str(value))
        self.check_timer.start()

    def add_override(self, name="", key="SolvMass", value=None):
        row = self.overrides.rowCount()
        self.overrides.insertRow(row)
        self.overrides.setCellWidget(row, 0, QLineEdit(name))
        field = QComboBox()
        for candidate, label in PROCESS_OVERRIDES.items():
            field.addItem(label, candidate)
        field.setCurrentIndex(field.findData(key))
        self.overrides.setCellWidget(row, 1, field)
        self.overrides.setCellWidget(row, 2, number_field(value))
        self.overrides.setCurrentCell(row, 0)

    def devices_page(self):
        page, form = self.scroll_form()
        self.reactor = QComboBox()
        self.reactor.addItem("Automatic from MSI / assignments", "")
        if self.setup["reactor_unit"]:
            self.reactor.addItem(self.setup["reactor_unit"], self.setup["reactor_unit"])
            self.reactor.setCurrentIndex(1)
        form.addRow("Model reactor", self.reactor)
        saved = self.setup["manual_assignments"]
        manual = QGroupBox("Control Panel profile assignments")
        controls = QFormLayout(manual)
        explanation(controls, "Automatic follows MSI vessel connections, source-container material roles and device experiment roles. "
                    "Matching applied DoE assignments are reused. Choose a device explicitly only for an unresolved or ambiguous role.")
        self.assignments = {}
        for role, details in ROLES.items():
            devices = sorted(device for device, cfg in self.configs.items()
                             if cfg.get("type") in details.compatible_types and not is_monitor_only(cfg))
            self.assignments[role] = text_choice(["Automatic", *devices], saved.get(role) or "Automatic")
            controls.addRow(details.label, self.assignments[role])
        form.addRow(manual)
        pump = self.setup["seed_pump"]
        self.pump_group = toggle_group("Override seed pump settings", pump is not None)
        pump_form = QFormLayout(self.pump_group)
        explanation(pump_form, "A dry-seed CSV loaded on the assigned pump supplies its fixed RPM and converts grams into pulse duration. "
                    "Leave this group unchecked to use the dry-seed CSV automatically. Use these controls for a manual dry-seed rate or an older mass-flow calibration.")
        self.use_seed_calibration = QCheckBox("Use the seed pump's loaded calibration")
        self.use_seed_calibration.setChecked((pump or {}).get("use_device_calibration", pump is None))
        pump_form.addRow(self.use_seed_calibration)
        self.pump_rpm = number_field((pump or {}).get("rpm"))
        self.pump_rate = number_field((pump or {}).get("grams_per_second"))
        pump_form.addRow("Seed pump speed (RPM)", self.pump_rpm)
        pump_form.addRow("Dry seed delivery rate (g/s)", self.pump_rate)
        self.pump_rate.setEnabled(not self.use_seed_calibration.isChecked())
        self.use_seed_calibration.toggled.connect(lambda checked: self.pump_rate.setEnabled(not checked))
        form.addRow(self.pump_group)
        simulation = self.setup["manual_simulation"]
        self.simulation_group = toggle_group("Generate PBM telemetry for simulated Control Panel runs", simulation is not None)
        simulated_form = QFormLayout(self.simulation_group)
        self.simulated_hold = number_field((simulation or {}).get("hold_start_s"))
        self.simulated_end = number_field((simulation or {}).get("end_s"))
        simulated_form.addRow("Equilibration start (s)", self.simulated_hold)
        simulated_form.addRow("Recording end (s)", self.simulated_end)
        explanation(simulated_form, "Only for the built-in PBM test simulator: the modeled run starts at 0 s; equilibration start begins "
                    "the final constant-temperature hold; recording end finishes that hold. Discharge and cleaning follow afterward. Leave off for real measurements.")
        self.simulated_monitors = {}
        for role, kind in MONITOR_TYPES.items():
            devices = sorted(device for device, cfg in self.configs.items()
                             if cfg.get("type") == kind and cfg.get("simulated"))
            field = text_choice(["", *devices], (simulation or {}).get("monitors", {}).get(role, ""))
            self.simulated_monitors[role] = field
            simulated_form.addRow(role, field)
        form.addRow(self.simulation_group)
        return page

    def design_page(self):
        page, form = self.scroll_form()
        imed = self.setup["imed"]
        self.mixing_group = toggle_group("Calculate average mixing power for IMED", self.setup["mixing"]["enabled"])
        mixing = QFormLayout(self.mixing_group)
        explanation(mixing, "Uses average RPM and average liquid volume over the selected experiment period. "
                    "One scalar Np × (average RPM/60)³ × diameter⁵ / average volume is passed as PowerRatio to each PBM; unchecked = 1. "
                    "No stirring or volume profile is added to the numerical solver.")
        self.power_number = number_field(self.setup["mixing"]["power_number"])
        self.impeller_diameter = number_field(self.setup["mixing"]["diameter_m"] * 1000)
        self.mixing_volume = number_field(self.setup["mixing"]["volume_ml"])
        self.volume_mode = QComboBox()
        self.volume_mode.addItem("Fixed campaign volume", "fixed")
        self.volume_mode.addItem("From charges and antisolvent additions", "charged")
        self.volume_mode.setCurrentIndex(self.volume_mode.findData(self.setup["mixing"]["volume_mode"]))
        for label, field in (("Impeller power number Np", self.power_number), ("Impeller diameter (mm)", self.impeller_diameter),
                             ("Volume basis", self.volume_mode), ("Fixed liquid volume (mL)", self.mixing_volume)):
            mixing.addRow(label, field)
        self.volume_mode.currentIndexChanged.connect(lambda: self.mixing_volume.setEnabled(self.volume_mode.currentData() == "fixed"))
        self.mixing_volume.setEnabled(self.volume_mode.currentData() == "fixed")
        explanation(mixing, "Charged volume prefers the AS scale; otherwise calibrated pump delivery is used. "
                    "Component densities supply an estimated additive liquid volume. Power number and diameter are constant per campaign.")
        explanation(mixing, "The existing PBM uses this scalar in secondary nucleation and agglomeration through their power exponents. "
                    "Primary nucleation has no power term. An exponent fixed at zero gives no mixing dependence; review candidate bounds in IMED Model Configurations.")
        form.addRow(self.mixing_group)
        self.delta_aic = number_field(imed["delta_aic"])
        self.iterations = integer_field(imed["iterations"], 0, 1000)
        self.samples = integer_field(imed["covariance_samples"], 2, 10000)
        self.equilibration = number_field(imed["equilibration_s"])
        form.addRow("Model survival threshold (ΔAIC)", self.delta_aic)
        form.addRow("Design search iterations", self.iterations)
        form.addRow("Covariance samples", self.samples)
        form.addRow("Final equilibration (s)", self.equilibration)
        self.use_doe_decisions = QCheckBox("Use decision variables from the starting DoE for MD and DO")
        self.use_doe_decisions.setChecked(imed["use_doe_decisions"])
        form.addRow(self.use_doe_decisions)
        explanation(form, "DoE design uses the starting profiles and scalar ranges. Clear this option to enter manual IMED design conditions.")
        conditions = imed["conditions"]
        self.design_group = toggle_group("Use these IMED design conditions", bool(conditions))
        design = QFormLayout(self.design_group)
        self.design_numbers = {}
        for key, label in DESIGN_VALUES.items():
            self.design_numbers[key] = number_field(conditions.get(key))
            design.addRow(label, self.design_numbers[key])
        self.design_pairs = {}
        for key, (label, first, second) in DESIGN_PAIRS.items():
            row = QHBoxLayout()
            controls = []
            for name, value in zip((first, second), conditions.get(key, (None, None))):
                row.addWidget(QLabel(name))
                field = number_field(value)
                row.addWidget(field)
                controls.append(field)
            self.design_pairs[key] = controls
            design.addRow(label, row)
        def enable_feed_conditions(no_feed):
            for field in (self.design_numbers["final_anti"], *self.design_pairs["antisolvent_rates"]):
                field.setEnabled(not no_feed)

        self.no_feed.toggled.connect(enable_feed_conditions)
        enable_feed_conditions(self.no_feed.isChecked())
        form.addRow(self.design_group)
        return page

    def collect(self, *, validate_period=True):
        draft = deepcopy(self.setup)
        draft["reactor_unit"] = self.reactor.currentData()
        draft["derive_initial_charges"] = self.derive_charges.isChecked()
        draft["experiment_period"] = self.period.values() if validate_period else {"mode": "definition", "stage_ids": []}
        draft["max_followups"] = self.max_followups.value()
        if uses_followup_offset(self.period.mode.currentData(), self.period.context):
            draft["followup_start_s"] = entered_number(self.followup_start, "follow-up crystallization start")
        recording = draft["recording"]
        if self.period.mode.currentData() == "manual":
            recording["start_s"] = entered_number(self.manual_start, "experiment start time")
            recording["end_s"] = entered_number(self.manual_end, "experiment end time", optional=True)
        for key, (field, label) in self.recording_numbers.items():
            recording[key] = entered_number(field, label, optional=key == "initial_concentration")
        recording.update(no_feed=self.no_feed.isChecked(), scale_direction=self.direction.currentData(),
                         smoothing=self.smoothing.value())
        sizes = [entered_number(field, f"seed {label}", optional=True)
                 for field, label in zip(self.seed_sizes, ("D10", "D50", "D90"))]
        if any(size is not None for size in sizes) and any(size is None for size in sizes):
            raise ValueError("Enter all three seed percentiles together.")
        recording["seed_percentiles"] = sizes if sizes[0] is not None else []
        draft["channels"] = {role: "" if text == "Not recorded" else text for role, field in self.channels.items()
                             if (text := field.currentText().strip()) not in {"Automatic", ""}}
        for key, field in self.process.items():
            number = entered_number(field, PROCESS_INPUTS[key], optional=True)
            if number is None:
                draft["process"].pop(key, None)
            else:
                draft["process"][key] = number
        overrides = {name: {key: value for key, value in values.items() if key not in PROCESS_OVERRIDES}
                     for name, values in draft["initial_processes"].items()}
        for row in range(self.overrides.rowCount()):
            name = self.overrides.cellWidget(row, 0).text().strip()
            key = self.overrides.cellWidget(row, 1).currentData()
            if not name:
                raise ValueError(f"Enter the experiment name for override {row + 1}.")
            fields = overrides.setdefault(name, {})
            if key in fields:
                raise ValueError(f"{name} has more than one override for {PROCESS_OVERRIDES[key]}.")
            value = entered_number(self.overrides.cellWidget(row, 2), PROCESS_OVERRIDES[key])
            if key == "InitialVolume_ml" and value <= 0:
                raise ValueError(f"{name}: Enter a positive initial liquid volume (mL).")
            fields[key] = value
        draft["initial_processes"] = {name: values for name, values in overrides.items() if values}
        draft["manual_assignments"] = self.selected_text(self.assignments)
        draft["seed_pump"] = None
        if self.pump_group.isChecked():
            draft["seed_pump"] = {"rpm": entered_number(self.pump_rpm, "seed pump speed"),
                                  "grams_per_second": entered_number(self.pump_rate, "seed delivery rate", optional=self.use_seed_calibration.isChecked())}
            if self.use_seed_calibration.isChecked():
                draft["seed_pump"]["use_device_calibration"] = True
        draft["manual_simulation"] = {
            "hold_start_s": entered_number(self.simulated_hold, "simulated equilibration start"),
            "end_s": entered_number(self.simulated_end, "simulated recording end"),
            "monitors": self.selected_text(self.simulated_monitors),
        } if self.simulation_group.isChecked() else None
        draft["support_profiles"] = self.support.values()
        if self.backend == "imed":
            draft["mixing"] = {**self.setup["mixing"], "enabled": False}
            if self.mixing_group.isChecked():
                draft["mixing"] = {"enabled": True, "power_number": entered_number(self.power_number, "impeller power number"),
                               "diameter_m": entered_number(self.impeller_diameter, "impeller diameter") / 1000,
                               "volume_mode": self.volume_mode.currentData(), "volume_ml": entered_number(self.mixing_volume, "fixed liquid volume")}
            draft["imed"].update(delta_aic=entered_number(self.delta_aic, "model survival threshold"),
                                 iterations=self.iterations.value(), covariance_samples=self.samples.value(),
                                 equilibration_s=entered_number(self.equilibration, "final equilibration"),
                                 use_doe_decisions=self.use_doe_decisions.isChecked())
            conditions = {}
            if self.design_group.isChecked():
                conditions = {key: entered_number(field, DESIGN_VALUES[key])
                              for key, field in self.design_numbers.items()
                              if key != "final_anti" or not recording["no_feed"]}
                for key, controls in self.design_pairs.items():
                    if key == "antisolvent_rates" and recording["no_feed"]:
                        continue
                    label, *names = DESIGN_PAIRS[key]
                    conditions[key] = [entered_number(field, f"{label} {name.lower()}")
                                       for field, name in zip(controls, names)]
                if recording["no_feed"]:
                    previous = self.setup["imed"]["conditions"]
                    conditions["final_anti"] = previous.get("final_anti", 0.)
                    conditions["antisolvent_rates"] = previous.get("antisolvent_rates", [0., 0.])
            draft["imed"]["conditions"] = conditions
        return validate_setup(draft)

    @staticmethod
    def selected_text(controls):
        return {key: text for key, field in controls.items() if (text := field.currentText().strip()) and text != "Automatic"}

    def accept(self):
        try:
            draft = self.collect()
            if self.preview is not None:
                context = self.preview(self.backend, draft, self.preview_source.currentData())
                self.period.set_context(context)
                draft["doe_decisions"] = context.get("doe_decisions")
            self.period.validate(draft)
            self.setup = validate_setup(draft)
        except ValueError as error:
            self.error.setText(str(error))
            self.error.show()
            return
        super().accept()
