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
)

from PySide6.QtCore import Qt, Signal

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class SettingsPage(QWidget):

    define_requested = Signal()
    generate_requested = Signal()
    profile_requested = Signal()
    plot_doe_requested = Signal()

    def __init__(self,parent=None):
        super().__init__(parent)

        self._build_ui()

    def _build_ui(self):

        # ======================================================
        # 1. ENVIRONMENT CONFIGURATION
        # ======================================================

        config_box = QGroupBox(
            "🌎 Environmental variables' settings"
        )

        config_layout = QGridLayout()
        config_layout.setContentsMargins(
            10, 10, 10, 10
        )
        config_layout.setHorizontalSpacing(8)
        config_layout.setVerticalSpacing(6)

        self.status_label = QLabel("")
        self.error_label = QLabel("")

        self.error_label.setStyleSheet(
            "color: #ef4464;"
        )

        self.exp_path_input = QLineEdit()
        self.exp_path_input.setPlaceholderText(
            r"C:\Desktop\Current_Experiment"
        )

        self.MODEL_path_input = QLineEdit()
        self.MODEL_path_input.setPlaceholderText(
            r"C:\Desktop\Model_Outputs"
        )

        self.exp_number_input = QLineEdit()
        self.exp_number_input.setPlaceholderText(
            "5"
        )

        self.define_button = QPushButton(
            "Define Variables"
        )
        self.define_button.setObjectName(
            "define_button"
        )

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
            QLabel("Experiment Path"),
            2,
            0,
        )

        config_layout.addWidget(
            self.exp_path_input,
            2,
            1,
        )

        config_layout.addWidget(
            QLabel("MODEL Path"),
            3,
            0,
        )

        config_layout.addWidget(
            self.MODEL_path_input,
            3,
            1,
        )

        config_layout.addWidget(
            QLabel("Number of experiments"),
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
        config_layout.setColumnStretch(1, 1)

        config_box.setLayout(config_layout)

        # ======================================================
        # 2. INITIAL DOE GENERATION
        # ======================================================

        exp_group = QGroupBox(
            "⚙ Initial DoE Generation"
        )

        form = QFormLayout()
        form.setContentsMargins(
            10, 10, 10, 10
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
            "Process conditions:",
            self.profile_button,
        )

        form.addRow(
            "Sampling strategy:",
            self.doe_type_combo,
        )

        form.addRow(
            self.generate_button
        )

        exp_group.setLayout(form)

        # ======================================================
        # 3. DOE INFORMATION
        # ======================================================

        doe_info_group = QGroupBox(
            "🧪 DoE Information"
        )

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

        inputs_layout.addWidget(
            exp_group
        )

        inputs_layout.addWidget(
            doe_info_group
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
            str(changed_dimensions)
        )

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

        self.status_label.setVisible(bool(message))
        self.error_label.setVisible(bool(error))


    def clear_inputs(self):
        self.exp_path_input.clear()
        self.MODEL_path_input.clear()
        self.exp_number_input.clear()


    def reset_ui(self):
        self.clear_inputs()

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
        self.define_button.setEnabled(enabled)
        self.generate_button.setEnabled(enabled)
        self.profile_button.setEnabled(enabled)
        self.plot_doe_button.setEnabled(enabled)

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
        try:
            experiment_creator.plot_sampled_points(
                fig_2d=self.canvas_2d.figure,
                fig_3d=self.canvas_3d.figure,
            )

            dims = getattr(
                experiment_creator,
                "dimensions",
                1,
            )

            min_width = max(
                350,
                dims * 85,
            )

            self.canvas_2d.setMinimumWidth(
                min_width
            )

            self.canvas_2d.draw()

        except Exception:
            self._show_plot_unavailable(
                self.canvas_2d,
                dark_mode,
            )

        try:
            experiment_creator.plot_sampled_points(
                fig_2d=self.canvas_2d.figure,
                fig_3d=self.canvas_3d.figure,
            )

            self.canvas_3d.setMinimumSize(
                300,
                300,
            )

            self.canvas_3d.draw()

        except Exception:
            self._show_plot_unavailable(
                self.canvas_3d,
                dark_mode,
            )