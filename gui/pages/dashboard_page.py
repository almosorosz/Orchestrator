from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QScrollArea,
    QStyle,
)

from gui.system_monitor_widget import SystemMonitorWidget
from gui.stylish_widgets import StatusBadge, LogViewer

from PySide6.QtCore import Signal

class DashboardPage(QWidget):

    start_requested = Signal()
    stop_requested  = Signal()
    reset_requested = Signal()
    plot_requested  = Signal()
    
    def __init__(
        self,
        dark_mode=True,
        spacing_1=8,
        spacing_2=16,
        parent=None,
    ):
        super().__init__(parent)

        self.dark_mode = dark_mode
        self.spacing_1 = spacing_1
        self.spacing_2 = spacing_2

        self._build_ui()

    def _build_ui(self):
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            "QScrollArea {"
            "border: none;"
            "background-color: transparent;"
            "}"
        )

        scroll_content = QWidget()
        scroll_content.setObjectName("dashboard_content")

        main_grid = QGridLayout(scroll_content)
        main_grid.setContentsMargins(20, 20, 20, 20)
        main_grid.setSpacing(15)

        # ----------------------------------------------------------
        # EXPERIMENT CONTROLS
        # ----------------------------------------------------------

        control_box = QGroupBox("🧪 Experiment Controls")

        control_layout = QGridLayout()
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(self.spacing_1)

        self.start_button = QPushButton("Start Experiment")
        self.start_button.setObjectName("start_button")

        self.stop_button = QPushButton("Stop Experiment")
        self.stop_button.setObjectName("stop_button")

        self.reset_button = QPushButton("♻️ Reset System")
        self.reset_button.setObjectName("reset_button")

        self.plot_button = QPushButton("📈 Plot Data")
        self.plot_button.setObjectName("plot_button")

        self.start_button.clicked.connect(
            lambda checked=False:
            self.start_requested.emit()
        )


        self.stop_button.clicked.connect(
            lambda checked=False:
            self.stop_requested.emit()
        )

        self.reset_button.clicked.connect(
            lambda checked=False:
            self.reset_requested.emit()
        )

        self.plot_button.clicked.connect(
            lambda checked=False:
            self.plot_requested.emit()
        )

        self.start_button.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_MediaPlay
            )
        )

        self.stop_button.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_MediaStop
            )
        )

        control_layout.addWidget(
            self.start_button, 0, 0
        )

        control_layout.addWidget(
            self.stop_button, 0, 1
        )

        control_layout.addWidget(
            self.reset_button, 1, 0
        )

        control_layout.addWidget(
            self.plot_button, 1, 1
        )

        control_box.setLayout(control_layout)

        # ----------------------------------------------------------
        # PROGRESS TRACK
        # ----------------------------------------------------------

        self.exp_status_box = QGroupBox(
            "📈 Progress Track"
        )
        self.exp_status_box.setObjectName(
            "exp_status_box"
        )

        exp_status_layout = QGridLayout()
        exp_status_layout.setContentsMargins(
            5, 5, 5, 5
        )
        exp_status_layout.setSpacing(
            self.spacing_2
        )

        self.omni_label = QLabel("Omni")
        self.MODEL_label = QLabel("MODEL")
        self.expts_label = QLabel("Experiments")

        self.omni_count_label = QLabel(
            "0 OMNI  runs"
        )

        self.model_count_label = QLabel(
            "0 MODEL runs"
        )

        self.expts_count_label = QLabel(
            "0 EXPTS runs"
        )

        self.dashboard_omni = StatusBadge(
            "Inactive"
        )

        self.dashboard_model = StatusBadge(
            "Inactive"
        )

        self.dashboard_runs = QLabel(
            "Runs: 0 / 0"
        )

        self.dashboard_current = QLabel(
            "Status: Not running"
        )

        self.variables_label = QLabel("")

        exp_status_layout.addWidget(
            self.omni_label, 0, 0
        )
        exp_status_layout.addWidget(
            self.MODEL_label, 0, 1
        )
        exp_status_layout.addWidget(
            self.expts_label, 0, 2
        )

        exp_status_layout.addWidget(
            self.omni_count_label, 1, 0
        )
        exp_status_layout.addWidget(
            self.model_count_label, 1, 1
        )
        exp_status_layout.addWidget(
            self.expts_count_label, 1, 2
        )

        exp_status_layout.addWidget(
            self.dashboard_omni, 2, 0
        )
        exp_status_layout.addWidget(
            self.dashboard_model, 2, 1
        )

        exp_status_layout.addWidget(
            self.dashboard_current,
            3,
            0,
            1,
            2,
        )

        self.exp_status_box.setLayout(
            exp_status_layout
        )

        # ----------------------------------------------------------
        # SYSTEM STATUS
        # ----------------------------------------------------------

        system_box = QGroupBox(
            "💻 System Status"
        )

        system_layout = QVBoxLayout()

        self.system_monitor = (
            SystemMonitorWidget()
        )

        self.system_monitor.set_theme(
            self.dark_mode
        )

        system_layout.addWidget(
            self.system_monitor
        )

        system_box.setLayout(system_layout)

        # ----------------------------------------------------------
        # STATUS LOG
        # ----------------------------------------------------------

        log_box = QGroupBox(
            "📜 Status Messages"
        )

        self.dashboard_log = LogViewer()
        self.dashboard_log.set_theme(
            self.dark_mode
        )

        log_layout = QVBoxLayout()
        log_layout.addWidget(
            self.dashboard_log
        )

        log_box.setLayout(log_layout)

        # ----------------------------------------------------------
        # CONNECTED DEVICES
        # ----------------------------------------------------------

        devices_box = QGroupBox(
            "🔌 Connected Devices"
        )

        devices_layout = QVBoxLayout()
        devices_layout.setContentsMargins(
            10, 10, 10, 10
        )
        devices_layout.setSpacing(
            self.spacing_2
        )

        self.device_omni_label = QLabel(
            "❌ OMNIBUS Controller: Inactive"
        )

        self.device_model_label = QLabel(
            "❌ MODEL Worker: Not Ready"
        )

        devices_layout.addWidget(
            self.device_omni_label
        )

        devices_layout.addWidget(
            self.device_model_label
        )

        devices_box.setLayout(
            devices_layout
        )

        # ----------------------------------------------------------
        # GRID
        # ----------------------------------------------------------

        main_grid.addWidget(
            self.exp_status_box, 0, 1
        )

        main_grid.addWidget(
            system_box, 0, 0
        )

        main_grid.addWidget(
            control_box, 1, 0
        )

        main_grid.addWidget(
            log_box, 2, 0
        )

        main_grid.addWidget(
            devices_box, 1, 1, 1, 2
        )

        scroll_area.setWidget(
            scroll_content
        )

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(
            0, 0, 0, 0
        )

        page_layout.addWidget(
            scroll_area
        )

    def add_log(self, message, level="INFO"):
        self.dashboard_log.add_log(
            message,
            level,
        )
    
    
    def set_theme(self, dark_mode):
        self.dark_mode = dark_mode
    
        self.dashboard_log.set_theme(
            dark_mode
        )
    
        self.system_monitor.set_theme(
            dark_mode
        )
    
    
    def set_omni_status(self, status):
        self.dashboard_omni.set_status(
            status
        )
    
    
    def set_model_status(self, status):
        self.dashboard_model.set_status(
            status
        )
    
    
    def set_omni_count(self, count):
        self.omni_count_label.setText(
            f"{count} runs"
        )
    
    
    def set_model_count(self, count):
        self.model_count_label.setText(
            f"{count} runs"
        )
    
    
    def set_experiment_count(self, count):
        self.expts_count_label.setText(
            f"{count} runs"
        )
    
    
    def set_total_runs(self, count):
        self.dashboard_runs.setText(
            f"Runs: {count}"
        )
    
    
    def set_current_stage(self, stage):
        self.dashboard_current.setText(
            f"Stage: {stage}"
        )
    
    
    def set_monitoring_state(
        self,
        active=None,
    ):
        if active is True:
            self.exp_status_box.setStyleSheet(
                """
                QGroupBox#exp_status_box {
                    background: #3aa662;
                    border: 2px solid #2CFF05;
                    border-radius: 12px;
                }
                """
            )
    
            self.exp_status_box.setTitle(
                "🟢 Experiment/Modeling monitoring active"
            )
    
        elif active is False:
            self.exp_status_box.setStyleSheet(
                """
                QGroupBox#exp_status_box {
                    background: #ff5b5b;
                    border: 2px solid #FF3131;
                    border-radius: 12px;
                }
                """
            )
    
            self.exp_status_box.setTitle(
                "🔴 Experiment/Modeling monitoring stopped"
            )
    
        else:
            self.exp_status_box.setStyleSheet("")
            self.exp_status_box.setTitle(
                "📈 Progress Track"
            )
    

    def set_experiment_controls_enabled(
        self,
        enabled,
    ):
        self.start_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
    
    
    def reset_status(self):
        self.set_omni_status("Inactive")
        self.set_model_status("Inactive")
    
        self.omni_count_label.setText(
            "0 OMNI  runs"
        )
    
        self.model_count_label.setText(
            "0 MODEL runs"
        )
    
        self.expts_count_label.setText(
            "0 EXPTS runs"
        )
    
        self.dashboard_runs.setText(
            "Runs: 0 / 0"
        )
    
        self.dashboard_current.setText(
            "Status: Not running"
        )
    
        self.variables_label.setText("")
    
        self.set_monitoring_state(None)