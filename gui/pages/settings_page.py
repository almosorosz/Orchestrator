from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QFormLayout,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QDialog,
    QMessageBox,
    QHBoxLayout,
    QStyle,
)

from PySide6.QtCore import Qt, Signal
from imed_python.size_data import SIZE_SAMPLE_OPTIONS

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from gui.settings_workflow import SettingsWorkflow


class SettingsPage(SettingsWorkflow, QWidget):

    define_requested = Signal()
    generate_requested = Signal()
    profile_requested = Signal()
    plot_doe_requested = Signal()
    assign_doe_requested = Signal()
    start_independent_doe_requested = Signal()
    export_doe_requested = Signal()
    recording_requested = Signal()
    reservoirs_requested = Signal()
    save_preset_requested = Signal()
    load_preset_requested = Signal()
    loop_navigation_requested = Signal(str)

    def __init__(self, parent=None, *, loop_devices=None, loop_preview=None, external_model_context=None):
        super().__init__(parent)
        from experiment.adaptive_loop import default_setup

        self.loop_setup = default_setup()
        self.loop_devices = loop_devices
        self.loop_preview = loop_preview
        self.external_model_context = external_model_context
        self.external_model_dialog = None
        self.loop_setup_dialog = None
        self._build_ui()

    def _build_ui(self):

        # ======================================================
        # 1. ENVIRONMENT CONFIGURATION
        # ======================================================

        config_box = QGroupBox(
            "Experiment and modeling setup"
        )

        config_layout = QGridLayout()
        config_layout.setContentsMargins(
            10, 10, 10, 10
        )
        config_layout.setHorizontalSpacing(8)
        config_layout.setVerticalSpacing(6)

        self.status_label = QLabel("")
        self.error_label = QLabel("")
        for label in (self.status_label, self.error_label):
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.error_label.setStyleSheet(
            "color: #ef4464;"
        )

        self.exp_path_input = QLineEdit()
        self.exp_path_input.setPlaceholderText(
            "General experiment recording folder"
        )

        self.MODEL_path_input = QLineEdit()
        self.MODEL_path_input.setPlaceholderText(
            "General model output folder"
        )

        self.exp_number_input = QLineEdit()
        self.exp_number_input.setPlaceholderText(
            "5"
        )
        self.exp_number_input.setToolTip("Initial batch size for non-full-factorial DoE generation. Follow-up experiments are configured in Automatic loop setup.")
        self.pbm_size_samples = QComboBox()
        for label, value in SIZE_SAMPLE_OPTIONS:
            self.pbm_size_samples.addItem(label, value)
        self.pbm_size_samples.setToolTip("Choose which recorded size measurements the LOOP MODEL stage uses for PBM fitting.")

        self.define_button = QPushButton(
            "Apply output locations"
        )
        self.define_button.setObjectName(
            "define_button"
        )
        self.define_button.setToolTip("Apply the general output locations and initial experiment count.")

        self.define_button.clicked.connect(
            lambda checked=False:
            self.define_requested.emit()
        )

        config_layout.addWidget(
            self.status_label, 0, 0, 1, 2
        )

        config_layout.addWidget(
            self.error_label, 1, 0, 1, 2
        )

        config_layout.addWidget(
            QLabel("Experiment output folder:"),
            2,
            0,
        )

        config_layout.addWidget(
            self.exp_path_input,
            2,
            1,
        )

        config_layout.addWidget(
            QLabel("Model output folder:"),
            3,
            0,
        )

        config_layout.addWidget(
            self.MODEL_path_input,
            3,
            1,
        )

        config_layout.addWidget(
            QLabel("Initial DoE experiments:"),
            4,
            0,
        )

        config_layout.addWidget(
            self.exp_number_input,
            4,
            1,
        )

        config_layout.addWidget(
            self.define_button,
            5,
            0,
            1,
            2,
        )

        config_layout.setColumnStretch(0, 0)
        self.reservoirs_button = QPushButton("Reservoir amounts…")
        self.reservoirs_button.clicked.connect(self.reservoirs_requested.emit)
        config_layout.addWidget(self.reservoirs_button, 6, 0, 1, 2)
        config_layout.setColumnStretch(1, 1)

        config_box.setLayout(config_layout)

        # ======================================================
        # 2. DOE WORKFLOW WITHIN ENVIRONMENT CONFIGURATION
        # ======================================================

        doe_controls = QWidget()

        form = QFormLayout()
        form.setContentsMargins(
            0, 0, 0, 0
        )
        form.setFieldGrowthPolicy(
            QFormLayout.AllNonFixedFieldsGrow
        )

        self.profile_button = QPushButton(
            "Edit Experiment Profile"
        )

        self.generate_button = QPushButton(
            "Generate experiments"
        )
        self.generate_button.setObjectName(
            "generate_button"
        )

        self.generate_button.clicked.connect(
            lambda checked=False:
            self.generate_requested.emit()
        )

        self.profile_button.clicked.connect(
            lambda checked=False:
            self.profile_requested.emit()
        )

        self.doe_type_combo = QComboBox()

        self.doe_type_combo.addItems([
            "Greedy minimizataion upon sobol sampling",
            "Latin Hypercube Sampling",
            "Latin Hypercube Sampling + Lloyd-Max optimization",
            "Random sampling",
            "Sobol sampling",
            "Full Factorial DoE generation",
        ])

        form.addRow(
            "Process settings:",
            self.profile_button,
        )

        form.addRow(
            "Sampling strategy:",
            self.doe_type_combo,
        )

        form.addRow(
            self.generate_button
        )
        self.assign_doe_button = QPushButton("Assign devices / preview DoE")
        self.assign_doe_button.setObjectName("assign_doe_button")
        self.start_independent_doe_button = QPushButton("Start DoE independent from loop")
        self.start_independent_doe_button.setEnabled(False)
        self.start_independent_doe_button.clicked.connect(lambda checked=False: self.start_independent_doe_requested.emit())
        self.export_doe_button = QPushButton("Export generated DoE CSV")
        self.assign_doe_button.setEnabled(False)
        self.export_doe_button.setEnabled(False)
        self.assign_doe_button.clicked.connect(lambda checked=False: self.assign_doe_requested.emit())
        self.export_doe_button.clicked.connect(lambda checked=False: self.export_doe_requested.emit())
        config_layout.addWidget(doe_controls, 6, 0, 1, 2)
        config_layout.addWidget(self.assign_doe_button, 7, 0, 1, 2)
        config_layout.addWidget(self.start_independent_doe_button, 8, 0, 1, 2)
        self.recording_button = QPushButton("Recording / SQL settings")
        self.recording_button.clicked.connect(lambda checked=False: self.recording_requested.emit())
        config_layout.addWidget(self.recording_button, 9, 0, 1, 2)
        self.save_preset_button = QPushButton("Save setup preset…")
        self.load_preset_button = QPushButton("Load setup preset…")
        self.save_preset_button.clicked.connect(self.save_preset_requested.emit)
        self.load_preset_button.clicked.connect(self.load_preset_requested.emit)
        config_layout.addWidget(self.save_preset_button, 10, 0, 1, 2)
        config_layout.addWidget(self.load_preset_button, 11, 0, 1, 2)
        config_layout.addWidget(QLabel("PBM size samples in LOOP"), 12, 0)
        config_layout.addWidget(self.pbm_size_samples, 12, 1)
        from experiment.adaptive_loop import MODES

        self.loop_backend = QComboBox()
        for label, mode in MODES:
            self.loop_backend.addItem(label, mode)
        self.loop_setup_button = QPushButton("Automatic loop setup…")
        self.loop_setup_button.setEnabled(False)
        self.loop_setup_button.clicked.connect(self.edit_loop_setup)
        self.loop_backend.currentIndexChanged.connect(self.update_loop_setup_button)
        config_layout.addWidget(QLabel("Dashboard LOOP modeling"), 13, 0)
        backend_row = QWidget()
        backend_layout = QHBoxLayout(backend_row)
        backend_layout.setContentsMargins(0, 0, 0, 0)
        backend_layout.addWidget(self.loop_backend, 1)
        self.external_model_info_button = QPushButton()
        self.external_model_info_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
        self.external_model_info_button.setFixedSize(32, 32)
        self.external_model_info_button.setStyleSheet("padding: 3px;")
        self.external_model_info_button.setAccessibleName("External model connection information")
        self.external_model_info_button.setToolTip("How to connect your own model: folders, completion files and new experiments")
        self.external_model_info_button.clicked.connect(self.show_external_model_info)
        backend_layout.addWidget(self.external_model_info_button)
        config_layout.addWidget(backend_row, 13, 1)
        config_layout.addWidget(self.loop_setup_button, 14, 0, 1, 2)

        doe_controls.setLayout(form)

        # ======================================================
        # 3. DOE INFORMATION
        # ======================================================

        doe_info_group = QWidget()
        form.addRow(QLabel("🧪 DoE information"))
        form.addRow(doe_info_group)

        doe_info_layout = QFormLayout()
        doe_info_layout.setContentsMargins(
            10, 10, 10, 10
        )

        self.changed_dimensions_count_label = QLabel("")
        self.changed_dimensions_label = QLabel("")
        self.experiment_count_label = QLabel("")

        self.changed_dimensions_label.setWordWrap(
            True
        )

        self.plot_doe_button = QPushButton(
            "📊 View DoE Experiment Profiles"
        )

        self.plot_doe_button.setObjectName(
            "plot_doe_button"
        )

        self.plot_doe_button.clicked.connect(
            lambda checked=False:
            self.plot_doe_requested.emit()
        )

        doe_info_layout.addRow(
            "Changed dimensions:",
            self.changed_dimensions_count_label,
        )

        doe_info_layout.addRow(
            "Dimensions:",
            self.changed_dimensions_label,
        )

        doe_info_layout.addRow(
            "Experiments:",
            self.experiment_count_label,
        )

        doe_info_layout.addRow(
            self.plot_doe_button
        )
        doe_info_layout.addRow(self.export_doe_button)

        doe_info_group.setLayout(
            doe_info_layout
        )

        # ======================================================
        # 4. VISUALIZATION AREA
        # ======================================================

        plots_group = QGroupBox(
            "📊 Visualizations"
        )

        plots_group.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )

        plots_layout = QVBoxLayout()
        plots_layout.setSpacing(15)
        plots_layout.setContentsMargins(
            10, 10, 10, 10
        )

        self.fig_2d = Figure(
            figsize=(4, 4),
            facecolor="none",
        )

        self.fig_3d = Figure(
            figsize=(4, 4),
            facecolor="none",
        )

        self.canvas_2d = FigureCanvas(
            self.fig_2d
        )

        self.canvas_3d = FigureCanvas(
            self.fig_3d
        )

        self.canvas_2d.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self.canvas_3d.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self.canvas_2d.setAttribute(
            Qt.WA_OpaquePaintEvent,
            True,
        )

        self.canvas_3d.setAttribute(
            Qt.WA_OpaquePaintEvent,
            True,
        )

        self.canvas_2d.setObjectName(
            "plotCanvas"
        )

        self.canvas_3d.setObjectName(
            "plotCanvas"
        )

        plots_layout.addWidget(
            self.canvas_2d,
            1,
        )

        plots_layout.addWidget(
            self.canvas_3d,
            1,
        )

        plots_group.setLayout(
            plots_layout
        )

        # ======================================================
        # 5. LEFT COLUMN
        # ======================================================

        inputs_container = QWidget()
        inputs_container.setMinimumWidth(0)
        inputs_container.setMaximumWidth(350)

        inputs_container.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Maximum,
        )

        inputs_layout = QVBoxLayout(
            inputs_container
        )

        inputs_layout.setContentsMargins(
            0, 0, 0, 0
        )
        inputs_layout.setSpacing(10)

        inputs_layout.addWidget(
            config_box
        )

        inputs_layout.addStretch()

        # ======================================================
        # 6. MAIN CONTENT
        # ======================================================

        container_widget = QWidget()
        container_widget.setMinimumWidth(0)

        container_layout = QGridLayout(
            container_widget
        )

        container_layout.setContentsMargins(
            10, 10, 10, 10
        )

        container_layout.setHorizontalSpacing(
            10
        )

        container_layout.setVerticalSpacing(
            10
        )

        container_layout.addWidget(
            inputs_container,
            0,
            0,
            Qt.AlignTop,
        )

        container_layout.addWidget(
            plots_group,
            0,
            1,
        )

        container_layout.setColumnMinimumWidth(
            0,
            250,
        )

        container_layout.setColumnStretch(
            0,
            0,
        )

        container_layout.setColumnMinimumWidth(
            1,
            0,
        )

        container_layout.setColumnStretch(
            1,
            1,
        )

        container_layout.setRowStretch(
            0,
            0,
        )

        # ======================================================
        # 7. SCROLL AREA
        # ======================================================

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(
            QFrame.NoFrame
        )
        scroll_area.setMinimumWidth(0)

        scroll_area.setWidget(
            container_widget
        )

        # ======================================================
        # 8. PAGE LAYOUT
        # ======================================================

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            0, 0, 0, 0
        )
        main_layout.setSpacing(0)

        main_layout.addWidget(
            scroll_area
        )
        self._organize_workflow()

    def get_environment_inputs(self):
        return (
            self.exp_path_input.text().strip(),
            self.MODEL_path_input.text().strip(),
            self.exp_number_input.text().strip(),
        )


    def get_sampling_strategy(self):
        return self.doe_type_combo.currentText()


    def show_doe_generating(self):
        text = "------⏳ Generating DoE ------"

        self.changed_dimensions_count_label.setText(text)
        self.changed_dimensions_label.setText(text)
        self.experiment_count_label.setText(text)


    def set_doe_info(
        self,
        changed_dimensions,
        dimensions,
        experiment_count,
    ):
        self.changed_dimensions_count_label.setText(
            str(dimensions)
        )
        self.changed_dimensions_count_label.setToolTip(str(changed_dimensions))

        self.changed_dimensions_label.setText(
            str(dimensions)
        )

        self.experiment_count_label.setText(
            str(experiment_count)
        )


    def set_generate_success(self):
        self.generate_button.setStyleSheet(
            """
            QPushButton#generate_button {
                background: #3aa662;
                border: 1px solid #2CFF05;
                color: white;
                text-align: center;
            }

            QPushButton#generate_button:hover {
                background: #88E788;
            }
            """
        )


    def set_define_success(self):
        self.define_button.setStyleSheet(
            """
            QPushButton#define_button {
                background: #3aa662;
                border: 1px solid #2CFF05;
                color: white;
            }

            QPushButton#define_button:hover {
                background: #88E788;
            }
            """
        )


    def set_message(self, message="", error=""):
        self.status_label.setText(message)
        self.error_label.setText(error)
        self.error_label.setToolTip(error)

        self.status_label.setVisible(bool(message))
        self.error_label.setVisible(bool(error))


    def clear_inputs(self):
        self.exp_path_input.clear()
        self.MODEL_path_input.clear()
        self.exp_number_input.clear()

    def edit_loop_setup(self):
        from gui.loop_setup_dialog import LoopSetupDialog

        if self.loop_setup_dialog is not None:
            if self.loop_setup_dialog.backend != self.loop_backend.currentData():
                self.loop_setup_dialog.reject()
                return self.edit_loop_setup()
            self.loop_setup_dialog.show()
            self.loop_setup_dialog.raise_()
            self.loop_setup_dialog.refresh_checklist()
            return self.loop_setup_dialog

        try:
            configs = self.loop_devices() if self.loop_devices is not None else {}
            dialog = LoopSetupDialog(self.loop_setup, self.loop_backend.currentData(), configs, self,
                                     preview=self.loop_preview)
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "Check loop setup", str(error))
            return
        self.loop_setup_dialog = dialog
        def finished(result):
            if result == QDialog.DialogCode.Accepted:
                self.loop_setup = dialog.setup
            self.loop_setup_dialog = None
            dialog.deleteLater()
            self.update_workflow_checklist()
        dialog.finished.connect(finished)
        def navigate(key):
            dialog.hide()
            if key == "folders":
                self.workflow_tabs.setCurrentIndex(3)
            self.loop_navigation_requested.emit(key)
        dialog.navigate_requested.connect(navigate)
        dialog.show()
        return dialog

    def external_model_info_context(self):
        context = self.external_model_context() if self.external_model_context is not None else {}
        return {**context, "experiment_root": self.exp_path_input.text().strip(),
                "model_root": self.MODEL_path_input.text().strip()}

    def show_external_model_info(self):
        from gui.external_model_info import ExternalModelInfoDialog
        if self.external_model_dialog is None:
            self.external_model_dialog = ExternalModelInfoDialog(self.external_model_info_context, self)
        else:
            self.external_model_dialog.refresh()
        self.external_model_dialog.show()
        self.external_model_dialog.raise_()
        self.external_model_dialog.activateWindow()
        return self.external_model_dialog

    def update_loop_setup_button(self):
        self.loop_setup_button.setEnabled(self.loop_backend.isEnabled() and self.loop_backend.currentData() != "external")


    def reset_ui(self):
        if self.loop_setup_dialog is not None:
            self.loop_setup_dialog.reject()
        self.set_doe_available(False)
        self.clear_inputs()
        from experiment.adaptive_loop import default_setup

        self.loop_backend.setCurrentIndex(0)
        self.loop_setup = default_setup()
        self.variable_table.setRowCount(0)

        self.changed_dimensions_count_label.clear()
        self.changed_dimensions_label.clear()
        self.experiment_count_label.clear()

        self.generate_button.setStyleSheet("")
        self.define_button.setStyleSheet("")

        self.status_label.clear()
        self.error_label.clear()

        self.status_label.setVisible(False)
        self.error_label.setVisible(False)


    def set_controls_enabled(self, enabled):
        self._controls_enabled = bool(enabled)
        for button in self.folder_browse_buttons:
            button.setEnabled(enabled)
        if self.loop_setup_dialog is not None:
            self.loop_setup_dialog.setEnabled(enabled)
        self.load_preset_button.setEnabled(enabled)
        self.define_button.setEnabled(enabled)
        for control in (self.exp_path_input, self.MODEL_path_input, self.exp_number_input, self.pbm_size_samples,
                        self.loop_backend):
            control.setEnabled(enabled)
        self.update_loop_setup_button()
        self.generate_button.setEnabled(enabled)
        self.profile_button.setEnabled(enabled)
        self.plot_doe_button.setEnabled(enabled)
        self.assign_doe_button.setEnabled(enabled and getattr(self, "_doe_available", False))
        self.export_doe_button.setEnabled(enabled and getattr(self, "_doe_available", False))
        self.start_independent_doe_button.setEnabled(enabled and getattr(self, "_assignment_available", False))

    def set_doe_available(self, available):
        self._doe_available = bool(available)
        if not available:
            self._workflow_creator = None
            self.experiment_choice.clear()
            self._workflow_profile_figures = []
        enabled = available and getattr(self, "_controls_enabled", True)
        self.assign_doe_button.setEnabled(enabled)
        self.export_doe_button.setEnabled(enabled)
        self.set_assignment_available(False)

    def set_assignment_available(self, available):
        self._assignment_available = bool(available)
        self.update_workflow_checklist(use_context=False)
        self.start_independent_doe_button.setEnabled(available and getattr(self, "_controls_enabled", True))
        self.assign_doe_button.setStyleSheet("""
            QPushButton#assign_doe_button {
                background: #3aa662;
                border: 1px solid #2CFF05;
                color: white;
            }
            QPushButton#assign_doe_button:hover:enabled {
                background: #88E788;
            }
            QPushButton#assign_doe_button:disabled {
                background: #2d6a43;
                border-color: #3aa662;
                color: #cbd5e1;
            }
        """ if available else "")

    def reset_plots(self, dark_mode):
        if dark_mode:
            bg_color = "#111827"
        else:
            bg_color = "#ffffff"

        for canvas in (
            self.canvas_2d,
            self.canvas_3d,
        ):
            fig = canvas.figure

            if fig:
                fig.clear()
                fig.patch.set_facecolor(
                    bg_color
                )

            canvas.draw()


    def update_plot_theme(self, dark_mode):
        if dark_mode:
            bg = "#111827"
            fg = "#e5e7eb"
            grid = "#334155"
        else:
            bg = "#ffffff"
            fg = "#1e293b"
            grid = "#cbd5e1"

        for canvas in (
            self.canvas_2d,
            self.canvas_3d,
        ):
            fig = canvas.figure

            fig.patch.set_facecolor(bg)

            for ax in fig.axes:
                ax.set_facecolor(bg)

                ax.tick_params(
                    colors=fg
                )

                ax.xaxis.label.set_color(fg)
                ax.yaxis.label.set_color(fg)
                ax.title.set_color(fg)

                for spine in ax.spines.values():
                    spine.set_color(grid)

                ax.grid(
                    True,
                    color=grid,
                    alpha=0.35,
                )

            canvas.draw_idle()


    def _show_plot_unavailable(
        self,
        canvas,
        dark_mode,
    ):
        msg = (
            "With these settings "
            "plotting is unavailable"
        )

        if dark_mode:
            bg = "#111827"
            ax_bg = "#1f2937"
        else:
            bg = "#ffffff"
            ax_bg = "#f8fafc"

        canvas.figure.clear()

        canvas.figure.patch.set_facecolor(
            bg
        )

        ax = canvas.figure.add_subplot(111)

        ax.set_facecolor(ax_bg)
        ax.axis("off")

        ax.text(
            0.5,
            0.5,
            msg,
            color="#ef4444",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            wrap=True,
        )

        canvas.draw()


    def render_doe_plots(
        self,
        experiment_creator,
        dark_mode,
    ):
        self.render_workflow(experiment_creator, dark_mode)
