#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys            
import os
import time

from matplotlib.figure                    import Figure
from matplotlib.backends.backend_qtagg    import FigureCanvasQTAgg as FigureCanvas
from datetime                             import datetime

from PySide6.QtGui                        import QPainterPath, QRegion
from PySide6.QtCore                       import QTimer, Qt, Slot, QRect
from PySide6.QtWidgets                    import (QApplication,QMainWindow,QLabel, QScrollArea,QFormLayout,QSizePolicy,
                                                   QPushButton,QLineEdit,QMessageBox,QProgressBar,QFrame,
                                                   QWidget,QDialog,QVBoxLayout,QHBoxLayout,QStyle,QComboBox,
                                                   QStackedWidget,QGroupBox,QTextEdit,QGridLayout)

from gui_design.system_monitor_widget     import SystemMonitorWidget
from gui_design.experiment_profile_dialog import ExperimentProfileDialog, DOEPlotsDialog
from gui_design.stylish_widgets           import MetricCard, LogViewer, StatusBadge, LogViewer

from doe_sampling.doe_sampler             import DoESampler
from doe_sampling.doe_creator             import ExperimentCreator

from exp_generation.profile_gen           import DefineProcessParameter
from exp_generation.profile_gen           import ProfileCreator
from exp_generation.profile_gen           import ExperimentProfileParameters
from exp_generation.profile_gen           import ExperimentProfileStorage
from exp_generation.profile_gen           import ExpComponentGen
from exp_generation.profile_gen           import ProfileGenerator
from exp_generation.pump_calibration      import PumpRateConverter

from app_control.pure_functions           import counter
from app_control.Controller               import Controller, Stage
from app_control.DataPlotter              import DataPlotter


# In[2]:


#gui qt


# In[14]:


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowFlags( Qt.WindowType.Window |  Qt.WindowType.FramelessWindowHint |  Qt.WindowType.WindowSystemMenuHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.SPACING_1 = 8
        self.SPACING_2 = 16
        self.SPACING_3 = 24
        self.SPACING_4 = 32

        self.resize(1100, 500)
        self.drag_position = None
        self.resize_position = None
        self.resize_edges = None
        self.is_resizing = False

        self.resize_margin = 8

        self.dark_mode = True

        central_widget = QFrame()
        central_widget.setObjectName("centralContainer") 
        self.setCentralWidget(central_widget)

        self.setMouseTracking(True)
        self.centralWidget().setMouseTracking(True)

        # 1. Root layout is a VERTICAL layout to stack the Header on top
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ----------------------------------------------------------------- #
        # TOP HEADER BAR                                                    #
        # ----------------------------------------------------------------- #
        header_widget = QWidget()
        header_widget.setObjectName("topHeader")
        header_widget.setFixedHeight(50)
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(15, 0, 10, 0)
        header_layout.setSpacing(8)

        # App Logo & Title
        logo_label = QLabel("⚛  ECC 1.0 - Experiment Control Center")
        logo_label.setStyleSheet("font-weight: bold; font-size: 15px; color: white; background: transparent;")

        # Theme toggle
        self.theme_toggle = QPushButton("☀")
        self.theme_toggle.setObjectName("themeToggle")
        self.theme_toggle.setCheckable(True)
        self.theme_toggle.setChecked(False)
        self.theme_toggle.setFixedSize(36, 30)
        self.theme_toggle.setToolTip("Switch to light theme")
        self.theme_toggle.setCursor(Qt.PointingHandCursor)
        self.theme_toggle.clicked.connect(self.toggle_theme)
        self.theme_toggle.setObjectName("theme_toggle")

        # System Ready Status Pill
        status_pill = QLabel("● System Ready")
        status_pill.setObjectName("statusPill")
        status_pill.setFixedSize(120, 30)

        self.btn_minimize = QPushButton("—")
        self.btn_minimize.setObjectName("windowControlBtn")
        self.btn_minimize.setFixedSize(30, 30)
        self.btn_minimize.clicked.connect(self.showMinimized)

        self.btn_maximize = QPushButton("□")
        self.btn_maximize.setObjectName("windowControlBtn")
        self.btn_maximize.setFixedSize(30, 30)
        self.btn_maximize.clicked.connect(self.toggle_maximize)

        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("windowCloseBtn")
        self.btn_close.setFixedSize(30, 30)
        self.btn_close.clicked.connect(self.close)

        header_layout.addWidget(logo_label)
        header_layout.addStretch()

        # Theme toggle immediately left of System Ready
        header_layout.addWidget(self.theme_toggle)
        header_layout.addWidget(status_pill)

        header_layout.addSpacing(15)
        header_layout.addWidget(self.btn_minimize)
        header_layout.addWidget(self.btn_maximize)
        header_layout.addWidget(self.btn_close)


        header_layout.addSpacing(15)
        header_layout.addWidget(self.btn_minimize)
        header_layout.addWidget(self.btn_maximize)
        header_layout.addWidget(self.btn_close)

        root_layout.addWidget(header_widget)

        # ----------------------------------------------------------------- #
        # BODY CONTAINER (Holds Sidebar on left, Content Stack on right)      #
        # ----------------------------------------------------------------- #
        body_widget = QWidget()
        body_widget.setObjectName("bodyContainer")
        body_layout = QHBoxLayout(body_widget)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # ----------------------------------------------------------------- #
        # 1. LEFT SIDEBAR NAVIGATION PANEL                                  #
        # ----------------------------------------------------------------- #
        sidebar_widget = QWidget()
        sidebar_widget.setObjectName("sidebar")
        sidebar_widget.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(15, 20, 15, 20)
        sidebar_layout.setSpacing(self.SPACING_2)

        # Create navigation buttons
        self.btn_dashboard = QPushButton("🏠  Dashboard")
        self.btn_experiment = QPushButton("🧪  Experiment")
        self.btn_plots = QPushButton("📊  Data & Plots")
        self.btn_brain = QPushButton("🧠  Brain")
        self.btn_settings = QPushButton("⚙️  Settings")

        nav_buttons = [self.btn_dashboard,self.btn_experiment,self.btn_plots,self.btn_brain,self.btn_settings]

        for btn in nav_buttons:
            btn.setCheckable(True)
            sidebar_layout.addWidget(btn)

        sidebar_layout.addSpacing(15)

        quick_label = QLabel("Quick Actions")
        quick_label.setStyleSheet("color: #64748b; font-size: 11px; font-weight: bold; background: transparent;")
        sidebar_layout.addWidget(quick_label)

        self.quick_start_btn = QPushButton("▶ Start")
        self.quick_start_btn.setObjectName("start_button")
        self.quick_stop_btn = QPushButton("⏹ Stop")
        self.quick_stop_btn.setObjectName("stop_button")
        self.quick_reset_btn = QPushButton("🔄 Reset")
        self.quick_reset_btn.setObjectName("quick_reset_button")
        self.quick_quit_btn = QPushButton("🔐 Quit")
        self.quick_quit_btn.setObjectName("quit_button")

        quick_actions = [ self.quick_start_btn, self.quick_stop_btn, self.quick_reset_btn, self.quick_quit_btn]

        for qbtn in quick_actions:
            sidebar_layout.addWidget(qbtn)

        self.quick_start_btn.clicked.connect(self.start_command)
        self.quick_stop_btn.clicked.connect(self.stop_command)
        self.quick_reset_btn.clicked.connect(self.reset_command)
        self.quick_quit_btn.clicked.connect(self.close)

        sidebar_layout.addStretch()

        # ----------------------------------------------------------------- #
        # 2. RIGHT SIDE: STACKED WIDGET (CONTENT VIEWS)                     #
        # ----------------------------------------------------------------- #
        self.stack = QStackedWidget()

        self.dashboard_page = QWidget()
        self.setup_dashboard_view()

        self.experiment_page = QWidget()
        self.setup_experiment_view()

        self.plots_page = QWidget()
        self.setup_plots_view()

        self.brain_page = QWidget()
        self.setup_brain_view()

        self.settings_page = QWidget()
        self.setup_settings_view()

        self.stack.addWidget(self.dashboard_page)
        self.stack.addWidget(self.experiment_page)
        self.stack.addWidget(self.plots_page)
        self.stack.addWidget(self.brain_page)
        self.stack.addWidget(self.settings_page)

        self.btn_dashboard.clicked.connect(lambda: self.switch_page(0))
        self.btn_experiment.clicked.connect(lambda: self.switch_page(1))
        self.btn_plots.clicked.connect(lambda: self.switch_page(2))
        self.btn_brain.clicked.connect(lambda: self.switch_page(3))
        self.btn_settings.clicked.connect(lambda: self.switch_page(4))

        self.btn_dashboard.setChecked(True)
        self.stack.setCurrentIndex(0)

        body_layout.addWidget(sidebar_widget)
        body_layout.addWidget(self.stack)

        # Add the body container right below the header in the root layout
        root_layout.addWidget(body_widget)

        # ----------------------------------------------------------------- #
        # CONSTANTS, MODULES, AND INTERNAL VARIABLES                        #
        # ----------------------------------------------------------------- #

        as_pump_cal_location   = r"FILES\pump_calibration\expttsd1_AS.csv"
        solu_pump_cal_location = r"FILES\pump_calibration\expttsd1_SOLU.csv"
        solv_pump_cal_location = r"FILES\pump_calibration\expttsd1_SOLV.csv"

        self.as_rate_converter           = PumpRateConverter(as_pump_cal_location,  'File  W: AS ',      'File R: Mass ')  
        self.solution_rate_converter     = PumpRateConverter(solu_pump_cal_location,'P01:Value (Setpt)', 'File R: Mass ')  
        self.solvent_rate_converter      = PumpRateConverter(solv_pump_cal_location,'P01:Value (Setpt)', 'File R: Mass ')  
        self.cleaning_sol_rate_converter = PumpRateConverter(solv_pump_cal_location,'P01:Value (Setpt)', 'File R: Mass ') 

        self.exp_par       = ExperimentProfileParameters() # ...storing the parameters for DoE profile generation
        self.doe_sampler   = DoESampler()      # ...generating the settings for Experiments
        self.component_gen = ExpComponentGen() # ...functions for experiment section generation
        self.profiles      = ProfileGenerator(self.exp_par, self.component_gen)       
        self.exp_storage   = ExperimentProfileStorage(self.as_rate_converter, self.solution_rate_converter, self.solvent_rate_converter, self.cleaning_sol_rate_converter) 

        self.omni_filename  = "expttsd"
        self.MODEL_filename = "MODEL"

        self.variables_set               = False
        self.process_stop                = False
        self.CONTROLLER                  = None
        self.DATAPROCESSOR               = None
        self.create_experiments          = None

        self.initial_exp_count           = 0
        self.conducted_experiments_count = 0
        self.previous_exp_count          = 0
        self.MODEL_run_count             = 0
        self.omni_run_count              = 0

        self.omni_folder                 = ""
        self.model_folder                = ""

        self.timer                       = QTimer(self)

        self.create_style()

        self.timer.timeout.connect(self.monitoring_loop)

        self.monitor_mode = None
        self.monitor_path = None
        self.monitor_num  = None
        self.log_message("🌟 App opened up successfully.", "INFO")

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self.btn_maximize.setText("□")
            self.btn_maximize.setToolTip("Maximize")
        else:
            self.showMaximized()
            self.btn_maximize.setText("❐")
            self.btn_maximize.setToolTip("Restore")

    def switch_page(self, index):
        """Switches the active page in the stack and manages button states."""
        self.stack.setCurrentIndex(index)
        buttons = [self.btn_dashboard,self.btn_experiment,self.btn_plots,self.btn_brain,self.btn_settings]

        for i, btn in enumerate(buttons):
          if i < self.stack.count():
            btn.setChecked(i == index)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Keep rounded corners
        path = QPainterPath()
        path.addRoundedRect(self.rect(), 12, 12)

        region = QRegion(
            path.toFillPolygon().toPolygon()
        )

        self.setMask(region)


    def get_resize_edges(self, pos):
        """Determine which edges/corners the mouse is touching."""

        margin = self.resize_margin

        rect = self.rect()

        left = pos.x() <= margin
        right = pos.x() >= rect.width() - margin

        top = pos.y() <= margin
        bottom = pos.y() >= rect.height() - margin

        return left, right, top, bottom


    def update_resize_cursor(self, pos):
        """Change cursor when hovering over a resize edge."""

        left, right, top, bottom = self.get_resize_edges(pos)

        if (left and top) or (right and bottom):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)

        elif (right and top) or (left and bottom):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)

        elif left or right:
            self.setCursor(Qt.CursorShape.SizeHorCursor)

        elif top or bottom:
            self.setCursor(Qt.CursorShape.SizeVerCursor)

        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)


    def mousePressEvent(self, event):

        if event.button() == Qt.MouseButton.LeftButton:

            pos = event.position().toPoint()

            # ------------------------------------------------
            # 1. Check resize edges FIRST
            # ------------------------------------------------

            left, right, top, bottom = self.get_resize_edges(pos)

            if left or right or top or bottom:

                self.is_resizing = True

                self.resize_edges = (
                    left,
                    right,
                    top,
                    bottom
                )

                self.resize_position = (
                    event.globalPosition().toPoint()
                )

                self.resize_geometry = self.geometry()

                event.accept()
                return

            # ------------------------------------------------
            # 2. Otherwise check header for window dragging
            # ------------------------------------------------

            if pos.y() <= 50:

                self.drag_position = (
                    event.globalPosition().toPoint()
                    - self.frameGeometry().topLeft()
                )

                event.accept()
                return

        super().mousePressEvent(event)


    def mouseMoveEvent(self, event):

        pos = event.position().toPoint()

        # ====================================================
        # RESIZING
        # ====================================================
        if self.is_resizing:

            current_pos = event.globalPosition().toPoint()
            delta = current_pos - self.resize_position

            left, right, top, bottom = self.resize_edges

            geometry = QRect(self.resize_geometry)

            if left:
                new_left = self.resize_geometry.left() + delta.x()
                max_left = (
                    self.resize_geometry.right()
                    - self.minimumWidth()
                    + 1
                )
                geometry.setLeft(min(new_left, max_left))

            if right:
                new_right = self.resize_geometry.right() + delta.x()
                min_right = (
                    self.resize_geometry.left()
                    + self.minimumWidth()
                    - 1
                )
                geometry.setRight(max(new_right, min_right))

            if top:
                new_top = self.resize_geometry.top() + delta.y()
                max_top = (
                    self.resize_geometry.bottom()
                    - self.minimumHeight()
                    + 1
                )
                geometry.setTop(min(new_top, max_top))

            if bottom:
                new_bottom = self.resize_geometry.bottom() + delta.y()
                min_bottom = (
                    self.resize_geometry.top()
                    + self.minimumHeight()
                    - 1
                )
                geometry.setBottom(max(new_bottom, min_bottom))

            # IMPORTANT:
            # setGeometry must be INSIDE the resizing block
            self.setGeometry(geometry)

            event.accept()
            return

        # ====================================================
        # WINDOW DRAGGING
        # ====================================================
        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and self.drag_position is not None
        ):
            self.move(
                event.globalPosition().toPoint()
                - self.drag_position
            )

            event.accept()
            return

        # ====================================================
        # RESIZE CURSOR
        # ====================================================
        self.update_resize_cursor(pos)

        event.accept()

    def mouseReleaseEvent(self, event):

        if event.button() == Qt.MouseButton.LeftButton:

            self.is_resizing = False
            self.resize_edges = None
            self.resize_position = None
            self.resize_geometry = None
            self.drag_position = None

            self.setCursor(Qt.CursorShape.ArrowCursor)

            event.accept()
            return

        super().mouseReleaseEvent(event)

    # --- Page layout builders ---
    def setup_dashboard_view(self):

        # Wrap everything in a scroll area to maintain responsiveness
        scroll_area = QScrollArea(self.dashboard_page)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        scroll_content = QWidget()
        scroll_content.setObjectName("dashboard_content")
        # Use a QGridLayout for the main content area to support columns & rows
        main_grid = QGridLayout(scroll_content)
        main_grid.setContentsMargins(20, 20, 20, 20)
        main_grid.setSpacing(15)

        # ===================================================== #
        # 2. EXPERIMENT CONTROLS BOX (Middle-Left)            #
        # ===================================================== #
        control_box = QGroupBox("🧪 Experiment Controls")
        control_layout = QGridLayout()

        # Tighten inner margins and spacing to make it shorter
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(self.SPACING_1)

        self.start_button = QPushButton("Start Experiment")
        self.start_button.setObjectName("start_button")
        self.stop_button = QPushButton("Stop Experiment")
        self.stop_button.setObjectName("stop_button")
        self.reset_button = QPushButton("♻️ Reset System")
        self.reset_button.setObjectName("reset_button")
        self.plot_button = QPushButton("📈 Plot Data")
        self.plot_button.setObjectName("plot_button")

        self.start_button.clicked.connect(self.start_command)
        self.stop_button.clicked.connect(self.stop_command)
        self.reset_button.clicked.connect(self.reset_command)
        self.plot_button.clicked.connect(self.plot_expts)

        self.start_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.stop_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))

        # Place in a compact 2x2 grid configuration
        control_layout.addWidget(self.start_button, 0, 0)
        control_layout.addWidget(self.stop_button, 0, 1)
        control_layout.addWidget(self.reset_button, 1, 0)
        control_layout.addWidget(self.plot_button, 1, 1)

        control_box.setLayout(control_layout)

        # ===================================================== #
        # 3. Progress track STATUS BOX (Top-Center)           #
        # ===================================================== #
        #exp_status_box = QGroupBox("📈 Progress Track")
        #exp_status_box.setObjectName("exp_status_box")
        exp_status_layout = QGridLayout()
        self.exp_status_box = QGroupBox("📈 Progress Track")
        self.exp_status_box.setObjectName("exp_status_box")  
        exp_status_layout.setContentsMargins(5, 5, 5, 5)
        exp_status_layout.setSpacing(self.SPACING_2)

        self.omni_label = QLabel("Omni")
        self.MODEL_label = QLabel("MODEL")
        self.expts_label = QLabel("Experiments")

        self.omni_count_label = QLabel( "0 OMNI  runs")
        self.model_count_label = QLabel("0 MODEL runs")
        self.expts_count_label = QLabel("0 EXPTS runs")

        self.dashboard_omni = StatusBadge("Inactive")
        self.dashboard_model = StatusBadge("Inactive")
        self.dashboard_runs = QLabel("Runs: 0 / 0")
        self.dashboard_current = QLabel("Status: Not running")
        self.variables_label = QLabel("")

        exp_status_layout.addWidget(self.omni_label, 0, 0)
        exp_status_layout.addWidget(self.MODEL_label, 0, 1)
        exp_status_layout.addWidget(self.expts_label, 0, 2)
        exp_status_layout.addWidget(self.omni_count_label, 1, 0)
        exp_status_layout.addWidget(self.model_count_label, 1, 1)
        exp_status_layout.addWidget(self.expts_count_label, 1, 2)
        exp_status_layout.addWidget(self.dashboard_omni, 2, 0)
        exp_status_layout.addWidget(self.dashboard_model, 2, 1)
        exp_status_layout.addWidget(self.dashboard_current, 3, 0, 1, 2)

        self.exp_status_box.setLayout(exp_status_layout)

        # ===================================================== #
        # 4. SYSTEM HEALTH / STATUS BOX (Top-Right)             #
        # ===================================================== #
        system_box = QGroupBox("💻 System Status")
        system_layout = QVBoxLayout()
        self.system_monitor = SystemMonitorWidget()
        #self.system_monitor.set_theme(self.dark_mode)
        #self.system_monitor.worker.update.connect(self.update_system_cards)
        self.system_monitor.setMaximumHeight(100)
        system_layout.addWidget(self.system_monitor)
        system_box.setLayout(system_layout)

        # ===================================================== #
        # 5. LIVE LOG WINDOW (Middle-Right / Bottom)            #
        # ===================================================== #
        log_box = QGroupBox("📜 Status Messages")
        self.dashboard_log = LogViewer()
        self.dashboard_log.set_theme(self.dark_mode)

        log_layout = QVBoxLayout()
        log_layout.addWidget(self.dashboard_log)
        log_box.setLayout(log_layout)

        # ===================================================== #
        # 6. CONNECTED DEVICES BOX (Bottom-Left)                #
        # ===================================================== #
        devices_box = QGroupBox("🔌 Connected Devices")
        devices_layout = QVBoxLayout()
        devices_layout.setContentsMargins(10, 10, 10, 10)
        devices_layout.setSpacing(self.SPACING_2)

        # Example labels for devices (you can connect these to real status logic later)
        self.device_omni_label = QLabel("❌ OMNIBUS Controller: Inactive")
        self.device_model_label = QLabel("❌ MODEL Worker: Not Ready")

        devices_layout.addWidget(self.device_omni_label)
        devices_layout.addWidget(self.device_model_label)
        devices_box.setLayout(devices_layout)

        # ===================================================== #
        # GRID PLACEMENT (Assembling the Multi-Column Layout) #
        # ===================================================== #
        #main_grid.addWidget(config_box, 0, 0)
        main_grid.addWidget(self.exp_status_box, 0, 1)
        main_grid.addWidget(system_box, 0, 0)

        # Row 1: Experiment Controls (col 0), Status Messages / Logs (col 1 & 2)
        main_grid.addWidget(control_box, 1, 0)
        main_grid.addWidget(log_box, 2, 0)

        # Row 2: Connected Devices (col 0 underneath controls)
        main_grid.addWidget(devices_box, 1, 1, 1, 2)

        scroll_area.setWidget(scroll_content)
        page_layout = QVBoxLayout(self.dashboard_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll_area)   

    def setup_experiment_view(self):
        layout = QVBoxLayout(self.experiment_page)
        layout.addWidget(QLabel("<h1>Live experiment</h1><p>Experimental info plotted here.</p>"))

    def setup_plots_view(self):
        """Set up Data & Plots page with a scrollable Matplotlib container."""

        layout = QVBoxLayout(self.plots_page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Title
        title_label = QLabel("Data & Plots View")
        title_label.setStyleSheet(
            "font-size: 18px; "
            "font-weight: bold; "
            "color: #f8fafc;"
        )
        layout.addWidget(title_label)

        # ---------------------------------------------------------
        # Scroll area
        # ---------------------------------------------------------
        self.plot_scroll_area = QScrollArea()
        self.plot_scroll_area.setWidgetResizable(True)
        self.plot_scroll_area.setFrameShape(QFrame.NoFrame)

        # Optional: allow both horizontal and vertical scrolling
        self.plot_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        self.plot_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )

        # ---------------------------------------------------------
        # Widget inside the scroll area
        # ---------------------------------------------------------
        self.plot_container = QWidget()

        self.plot_container_layout = QVBoxLayout(
            self.plot_container
        )

        self.plot_container_layout.setContentsMargins(
            5, 5, 5, 5
        )
        self.plot_container_layout.setSpacing(10)

        # Placeholder
        self.plot_placeholder_label = QLabel("No experiment data plotted yet. ""Click 'Plot data' to load.")
        self.plot_placeholder_label.setStyleSheet("color: #64748b; ""font-size: 13px;")
        self.plot_container_layout.addWidget(self.plot_placeholder_label)

        # Important: allow contents to expand
        self.plot_container_layout.addStretch()

        # Put container inside scroll area
        self.plot_scroll_area.setWidget(self.plot_container)

        # Add scroll area to page
        layout.addWidget(self.plot_scroll_area, 1)

    def setup_brain_view(self):
        layout = QVBoxLayout(self.brain_page)
        layout.addWidget(QLabel("<h1>brain</h1><p>Log info plotted here.</p>"))

    def setup_settings_view(self):
        # ===================================================== #
        # 1. EXPERIMENT CONFIGURATION BOX
        # ===================================================== #
        config_box = QGroupBox("🌎 Environmental variables' settings")
        config_layout = QGridLayout()
        config_layout.setContentsMargins(10, 10, 10, 10)
        config_layout.setHorizontalSpacing(8)
        config_layout.setVerticalSpacing(6)

        self.status_label = QLabel("")
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #ef4464;")

        self.exp_path_input = QLineEdit()
        self.exp_path_input.setPlaceholderText(r"C:\Desktop\Current_Experiment")

        self.MODEL_path_input = QLineEdit()
        self.MODEL_path_input.setPlaceholderText(r"C:\Desktop\Model_Outputs")

        self.exp_number_input = QLineEdit()
        self.exp_number_input.setPlaceholderText("5")

        self.define_button = QPushButton("Define Variables")
        self.define_button.setObjectName("define_button")
        self.define_button.clicked.connect(self.define_command)

        config_layout.addWidget(self.status_label, 0, 0, 1, 2)
        config_layout.addWidget(self.error_label, 1, 0, 1, 2)

        config_layout.addWidget(QLabel("Experiment Path"), 2, 0)
        config_layout.addWidget(self.exp_path_input, 2, 1)

        config_layout.addWidget(QLabel("MODEL Path"), 3, 0)
        config_layout.addWidget(self.MODEL_path_input, 3, 1)

        config_layout.addWidget(QLabel("Number of experiments"), 4, 0)
        config_layout.addWidget(self.exp_number_input, 4, 1)

        config_layout.addWidget(self.define_button, 5, 0, 1, 2)

        # Input widgets take available width
        config_layout.setColumnStretch(0, 0)
        config_layout.setColumnStretch(1, 1)

        config_box.setLayout(config_layout)

        # ===================================================== #
        # 2. EXPERIMENT SETTINGS BOX
        # ===================================================== #
        exp_group = QGroupBox("⚙ Initial DoE Generation")

        form = QFormLayout()
        form.setContentsMargins(10, 10, 10, 10)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.profile_button = QPushButton("Edit Experiment Profile")

        self.generate_button = QPushButton("Generate experiments")
        self.generate_button.setObjectName("generate_button")

        self.generate_button.clicked.connect(self.define_doe)
        self.profile_button.clicked.connect(self.open_profile_dialog)

        self.doe_type_combo = QComboBox()
        self.doe_type_combo.addItems([
            'Greedy minimizataion upon sobol sampling',
            'Latin Hypercube Sampling',
            'Latin Hypercube Sampling + Lloyd-Max optimization',
            'Random sampling',
            'Sobol sampling',
            'Factorial DoE generation'
        ])

        form.addRow("Profile", self.profile_button)
        form.addRow("DoE type", self.doe_type_combo)
        form.addRow(self.generate_button)

        exp_group.setLayout(form)

        # ===================================================== #
        # 3. DoE INFORMATION
        # ===================================================== #
        doe_info_group = QGroupBox("🧪 DoE Information")

        doe_info_layout = QFormLayout()
        doe_info_layout.setContentsMargins(10, 10, 10, 10)

        self.changed_dimensions_count_label = QLabel("")
        self.changed_dimensions_label = QLabel("")
        self.experiment_count_label = QLabel("")

        self.plot_doe_button = QPushButton("📊 View DoE Experiment Profiles")
        self.plot_doe_button.setObjectName("plot_doe_button")
        self.plot_doe_button.clicked.connect(self.show_doe_plots)

        self.changed_dimensions_label.setWordWrap(True)

        doe_info_layout.addRow("Changed dimensions:",self.changed_dimensions_count_label)
        doe_info_layout.addRow("Dimensions:",self.changed_dimensions_label)
        doe_info_layout.addRow("Experiments:",self.experiment_count_label)

        doe_info_layout.addRow(self.plot_doe_button)

        doe_info_group.setLayout(doe_info_layout)

        doe_info_group.setLayout(doe_info_layout)

        # ===================================================== #
        # 4. PLOTS AREA
        # ===================================================== #
        plots_group = QGroupBox("📊 Visualizations")
        plots_group.setMaximumHeight(700)
        plots_group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Preferred )
        plots_layout = QVBoxLayout()
        plots_layout.setSpacing(15)
        plots_layout.setContentsMargins(10, 10, 10, 10)

        self.fig_2d = Figure(figsize=(4, 4), facecolor='none')
        self.fig_3d = Figure(figsize=(4, 4), facecolor='none')

        self.canvas_2d = FigureCanvas(self.fig_2d)
        self.canvas_3d = FigureCanvas(self.fig_3d)

        # Allow canvases to consume all available space
        self.canvas_2d.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding    )

        self.canvas_3d.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding    )

        self.canvas_2d.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.canvas_3d.setAttribute(Qt.WA_OpaquePaintEvent, True)

        self.canvas_2d.setObjectName("plotCanvas")
        self.canvas_3d.setObjectName("plotCanvas")

        self.canvas_2d.setStyleSheet("background-color: #111827;")

        self.canvas_3d.setStyleSheet( "background-color: #111827;")

        plots_layout.addWidget(self.canvas_2d, 1)
        plots_layout.addWidget(self.canvas_3d, 1)

        plots_group.setLayout(plots_layout)

        # ===================================================== #
        # 5. LEFT COLUMN
        # ===================================================== #
        inputs_container = QWidget()

        # IMPORTANT:
        # This makes the left side capable of being narrow.
        inputs_container.setMinimumWidth(0)
        inputs_container.setMaximumWidth(350)

        inputs_container.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Maximum    )

        inputs_layout = QVBoxLayout(inputs_container)
        inputs_layout.setContentsMargins(0, 0, 0, 0)
        inputs_layout.setSpacing(10)

        inputs_layout.addWidget(config_box)
        inputs_layout.addWidget(exp_group)
        inputs_layout.addWidget(doe_info_group)
        inputs_layout.addStretch()

        # ===================================================== #
        # 6. SCROLLABLE CONTENT
        # ===================================================== #
        container_widget = QWidget()
        container_widget.setMinimumWidth(0)

        container_layout = QGridLayout(container_widget)


        container_layout.setContentsMargins(10, 10, 10, 10)
        container_layout.setHorizontalSpacing(10)
        container_layout.setVerticalSpacing(10)

        # LEFT
        container_layout.addWidget(inputs_container,0,0,Qt.AlignTop    )

        # RIGHT
        container_layout.addWidget(plots_group,0,1    )

        # ===================================================== #
        # THIS IS THE IMPORTANT PART
        # ===================================================== #

        # Left column gets a controlled width.
        # Right column gets ALL remaining width.
        container_layout.setColumnMinimumWidth(0, 250)
        container_layout.setColumnStretch(0, 0)

        container_layout.setColumnMinimumWidth(1, 0)
        container_layout.setColumnStretch(1, 1)

        # Make the plot area expand vertically
        container_layout.setRowStretch(0, 0)

        # ===================================================== #
        # 7. SCROLL AREA
        # ===================================================== #
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        # Don't force the scroll area's child to have a huge minimum width
        scroll_area.setMinimumWidth(0)

        scroll_area.setWidget(container_widget)

        # ===================================================== #
        # 8. MAIN LAYOUT
        # ===================================================== #
        main_layout = QVBoxLayout(self.settings_page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_layout.addWidget(scroll_area)

    # ====================================================== #
    # STYLE                                                  #
    # ====================================================== #

    def create_style(self, dark=True):
        if dark:
            self.setStyleSheet("""
            QGroupBox QLabel {
                background: transparent;
                border: none;
            }

            QLabel#dashboard_current {
                background: transparent;
            }

            QWidget#metricCard {
                background: #111d31;     
                border: 1px solid #1e293b;  
                border-radius: 12px;
            }

            QWidget {
                background: #07111f;
                color: #e5e7eb;
                font-family: "Segoe UI";
                font-size: 13px;
            }

            QGroupBox {
                font-size: 13px;
                font-weight: 800;
                background: #111d31;
                border: 1px solid #16263d;
                border-radius: 12px;
                margin-top: 16px;
                padding: 16px;
            }

            QGroupBox#exp_status_box {
                background: #111d31;
                border: 1px solid #16263d;
                border-radius: 12px;
                margin-top: 16px;
                padding: 16px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 26px;
                top: -2px;
                padding: 0px 6px;
                color: white;
                font-size: 16px;
                font-weight: 700;
            }

            QPushButton {
                background: transparent;
                color: #d6dde8;
                border: 1px solid #16263d;
                border-radius: 8px;
                padding: 11px 15px;
                text-align: left;
                font-weight: 600;
            }

                QPushButton:hover {
                    background: #162846;
                    border: 1px solid #2f5fa7;
                    color: white;
                }

                QPushButton:pressed {
                    background: #10203a;
                }

                QPushButton:checked {
                    background: #2563eb;
                    border: 1px solid #3b82f6;
                    color: white;
                }

                QPushButton#plot_button {
                    background: transparent;
                    color: #d6dde8;
                    border: 1px solid #16263d;
                    border-radius: 8px;
                    padding: 11px 15px;
                    text-align: center;
                    font-weight: 600;
                }

                QPushButton#plot_button:hover {
                    background: #162846;
                    border: 1px solid #2f5fa7;
                    color: white;
                }

                QPushButton#plot_button:pressed {
                    background: #10203a;
                }

                QPushButton#plot_button:checked {
                    background: #2563eb;
                    border: 1px solid #3b82f6;
                    color: white;
                }
                QWidget#plotCanvas {
                    background: #111827;
                }
                QPushButton#reset_button {
                    background: transparent;
                    color: #d6dde8;
                    border: 1px solid #16263d;
                    border-radius: 8px;
                    padding: 11px 15px;
                    text-align: center;
                    font-weight: 600;
                }

                QPushButton#reset_button:hover {
                    background: #162846;
                    border: 1px solid #2f5fa7;
                    color: white;
                }

                QPushButton#reset_button:pressed {
                    background: #10203a;
                }

                QPushButton#reset_button:checked {
                    background: #2563eb;
                    border: 1px solid #3b82f6;
                    color: white;
                }

                QPushButton#generate_button {
                    background: #3b82f6;
                    border: 1px solid #60a5fa;
                    color: white;
                    text-align: center;
                    font-weight: 700;
                    padding: 11px;
                }

                QPushButton#generate_button:hover {
                    background: #2563eb;
                }

                QPushButton#generate_button:pressed {
                    background: #10203a;
                }

                QPushButton#generate_button:checked {
                    background: #2563eb;
                    border: 1px solid #3b82f6;
                    color: white;
                }

                QPushButton#start_button {
                    background: #22c55e;
                    border: 1px solid #37d774;
                    color: white;
                    text-align: center;
                    font-size: 12px;
                    font-weight: 700;
                    padding: 12px;
                }

                QPushButton#start_button:hover {
                    background: #16a34a;
                }

                QPushButton#stop_button {
                    background: #ef4444;
                    border: 1px solid #f87171;
                    color: white;
                    text-align: center;
                    font-size: 12px;
                    font-weight: 700;
                    padding: 12px;
                }

                QPushButton#stop_button:hover {
                    background: #dc2626;
                }

                QPushButton#define_button {
                    background: #3b82f6;
                    border: 1px solid #60a5fa;
                    color: white;
                    text-align: center;
                    font-weight: 700;
                    padding: 11px;
                }

                QPushButton#define_button:hover {
                    background: #2563eb;
                }

                QPushButton#define_button:pressed {
                    background: #10203a;
                }

                QLineEdit {
                    background: #0b1728;
                    border: 1px solid #24364d;
                    border-radius: 8px;
                    padding: 8px 10px;
                    color: white;
                }

                QLineEdit:focus {
                    border: 1px solid #3b82f6;
                }

                QTextEdit {
                    background: #081321;
                    border: 1px solid #16263d;
                    border-radius: 10px;
                    padding: 8px;
                    color: #dbe5ef;
                    selection-background-color: #2563eb;
                }

                QCheckBox {
                    spacing: 8px;
                    color: #e2e8f0;
                    font-weight: 500;
                }

                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border-radius: 4px;
                    border: 2px solid #5b6c84;
                    background: #111827;
                }

                QCheckBox::indicator:hover {
                    border: 2px solid #60a5fa;
                }

                QCheckBox::indicator:checked {
                    background: #2563eb;
                    border: 2px solid #3b82f6;
                }

                QScrollBar:vertical {
                    background: transparent;
                    width: 10px;
                    margin: 0px;
                }

                QScrollBar::handle:vertical {
                    background: #2d415d;
                    border-radius: 5px;
                    min-height: 30px;
                }

                QScrollBar::handle:vertical:hover {
                    background: #456182;
                }

                QScrollBar::add-line:vertical,
                QScrollBar::sub-line:vertical {
                    height: 0px;
                }

                QScrollBar::add-page:vertical,
                QScrollBar::sub-page:vertical {
                    background: transparent;
                }

                QTableWidget {
                    background: #081321;
                    border: 1px solid #16263d;
                    gridline-color: #16263d;
                    border-radius: 10px;
                    alternate-background-color: #0d1b2d;
                }

                QHeaderView::section {
                    background: #111d31;
                    color: white;
                    padding: 6px;
                    border: none;
                    border-bottom: 1px solid #16263d;
                    font-weight: 700;
                }

                QComboBox {
                    background: #0b1728;
                    border: 1px solid #24364d;
                    border-radius: 8px;
                    padding: 6px 10px;
                }

                QComboBox:hover {
                    border: 1px solid #3b82f6;
                }

                QComboBox::drop-down {
                    border: none;
                    width: 24px;
                }

                QToolTip {
                    background: #162846;
                    color: white;
                    border: 1px solid #3b82f6;
                    padding: 6px;
                    border-radius: 6px;
                }

                QLabel#statusCard {
                    background: #111d31;
                    border: 1px solid #16263d;
                    border-radius: 12px;
                    color: #e5e7eb;
                    padding: 16px;
                    font-size: 14px;
                    font-weight: 700;
                }

                QWidget#centralContainer {
                    background: #07111f;
                    border: 1px solid #ffffff;
                    border-radius: 12px;
                }

                QWidget#bodyContainer {
                    background: transparent;
                }

                QStackedWidget {
                    background: transparent;
                    border-bottom-right-radius: 12px;
                }

                QWidget#sidebar {
                    background: #081426;
                    border-right: 1px solid #17283f;
                    border-bottom-left-radius: 12px;
                }

                QWidget#topHeader {
                    background: #081426;
                    border-top-left-radius: 12px;
                    border-top-right-radius: 12px;
                    border-bottom: 1px solid #17283f;
                }

                QLabel#statusPill {
                    background: #064e3b;
                    color: #34d399;
                    border-radius: 10px;
                    padding: 4px 12px;
                    font-weight: 600;
                }

                QPushButton#windowControlBtn {
                    background: transparent;
                    color: #94a3b8;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                    text-align: center;
                    padding: 0px;
                }

                QPushButton#windowControlBtn:hover {
                    background: #1e293b;
                    color: white;
                }

                QPushButton#windowCloseBtn {
                    background: transparent;
                    color: #94a3b8;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                    text-align: center;
                    padding: 0px;
                }

                QPushButton#windowCloseBtn:hover {
                    background: #ef4444;
                    color: white;
                }
                QScrollBar:horizontal {
                background: #111827;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
                }

                QScrollBar::handle:horizontal {
                    background: #374151;
                    min-width: 30px;
                    border-radius: 6px;
                }

                QScrollBar::handle:horizontal:hover {
                    background: #4B5563;
                }

                QScrollBar::add-line:horizontal,
                QScrollBar::sub-line:horizontal {
                    width: 0px;
                }

                QScrollBar::add-page:horizontal,
                QScrollBar::sub-page:horizontal {
                    background: transparent;
                }
                QPushButton#theme_toggle {
                    background-color: #2b2f36;
                    color: #ffd166;
                    border: 1px solid #4a505a;
                    border-radius: 8px;
                    font-size: 20px;
                    font-weight: bold;
                    padding: 0px;
                }

                QPushButton#theme_toggle:hover {
                    background-color: #363b44;
                    border: 1px solid #6b7280;
                }

                QPushButton#theme_toggle:pressed {
                    background-color: #22262c;
                }
                """)
        else:
            self.setStyleSheet("""
                /* ==================================================
                   LIGHT THEME
                   ================================================== */
                QPushButton#theme_toggle {
                    background-color: #f8fafc;
                    color: #1e293b;
                    border: 1px solid #6b7280;
                    border-radius: 8px;
                    font-size: 20px;
                    font-weight: bold;
                    padding: 0px;
                    text-align: right;
                }

                QPushButton#theme_toggle:hover {
                    background-color:#e8f0fe;
                    border: 1px solid #6b7280;
                }

                QPushButton#theme_toggle:pressed {
                    background-color: #dbeafe;
                }
                QWidget {
                    background: #f8fafc;
                    color: #1e293b;
                    font-family: "Segoe UI";
                    font-size: 13px;
                }

                QWidget#centralContainer {
                    background: #f8fafc;
                    border: 1px solid #000000;
                    border-radius: 12px;
                }
                QWidget#plotCanvas {
                    background: #ffffff;
                }
                QWidget#bodyContainer {
                    background: transparent;

                }

                QWidget#sidebar {
                    background: #eef2f7;
                    border-right: 1px solid #d5dce5;
                    border-bottom-left-radius: 12px;
                }

                QWidget#topHeader {
                    background: #ffffff;
                    border-top-left-radius: 12px;
                    border-top-right-radius: 12px;
                    border-bottom: 1px solid #d5dce5;
                }

                QGroupBox {
                    font-size: 13px;
                    font-weight: 800;
                    background: #ffffff;
                    border: 1px solid #d8dee8;
                    border-radius: 12px;
                    margin-top: 16px;
                    padding: 16px;
                }

                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    left: 26px;
                    top: -2px;
                    padding: 0px 6px;
                    color: #1e293b;
                    font-size: 16px;
                    font-weight: 700;
                }

                QPushButton {
                    background: transparent;
                    color: #334155;
                    border: 1px solid #d8dee8;
                    border-radius: 8px;
                    padding: 11px 15px;
                    text-align: left;
                    font-weight: 600;
                }

                QPushButton:hover {
                    background: #e8f0fe;
                    border: 1px solid #93b4e8;
                    color: #1e40af;
                }

                QPushButton:pressed {
                    background: #dbeafe;
                }

                QPushButton:checked {
                    background: #2563eb;
                    border: 1px solid #3b82f6;
                    color: white;
                }

                QLabel#statusPill {
                    background: #dcfce7;
                    color: #15803d;
                    border-radius: 10px;
                    padding: 4px 12px;
                    font-weight: 600;
                }

                QPushButton#themeToggle {
                    background: #f1f5f9;
                    color: #475569;
                    border: 1px solid #cbd5e1;
                    border-radius: 7px;
                    padding: 0px;
                    font-size: 16px;
                    font-weight: 600;
                }

                QPushButton#themeToggle:hover {
                    background: #e2e8f0;
                    border: 1px solid #94a3b8;
                }

                QPushButton#windowControlBtn {
                    background: transparent;
                    color: #64748b;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                    text-align: center;
                    padding: 0px;
                }

                QPushButton#windowControlBtn:hover {
                    background: #e2e8f0;
                    color: #0f172a;
                }

                QPushButton#windowCloseBtn {
                    background: transparent;
                    color: #64748b;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                    text-align: center;
                    padding: 0px;
                }

                QPushButton#windowCloseBtn:hover {
                    background: #ef4444;
                    color: white;
                }

                QLineEdit {
                    background: #ffffff;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    padding: 8px 10px;
                    color: #1e293b;
                }

                QLineEdit:focus {
                    border: 1px solid #3b82f6;
                }

                QTextEdit {
                    background: #ffffff;
                    border: 1px solid #d8dee8;
                    border-radius: 10px;
                    padding: 8px;
                    color: #334155;
                    selection-background-color: #2563eb;
                }

                QComboBox {
                    background: #ffffff;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    padding: 6px 10px;
                    color: #334155;
                }

                QComboBox:hover {
                    border: 1px solid #3b82f6;
                }

                QScrollBar:vertical {
                    background: transparent;
                    width: 10px;
                }

                QScrollBar::handle:vertical {
                    background: #cbd5e1;
                    border-radius: 5px;
                    min-height: 30px;
                }

                QScrollBar::handle:vertical:hover {
                    background: #94a3b8;
                }

                QScrollBar::add-line:vertical,
                QScrollBar::sub-line:vertical {
                    height: 0px;
                }

                QScrollBar::add-page:vertical,
                QScrollBar::sub-page:vertical {
                    background: transparent;
                }
            """)


      # ====================================================== #
      # HELPERS & COMMANDS (Unchanged Logic)                   #
      # ====================================================== #    
    def update_plot_theme(self):

        if self.dark_mode:
            bg = "#111827"
            fg = "#e5e7eb"
            grid = "#334155"
        else:
            bg = "#ffffff"
            fg = "#1e293b"
            grid = "#cbd5e1"

        for fig in [self.canvas_2d.figure, self.canvas_3d.figure]:

            fig.patch.set_facecolor(bg)

            for ax in fig.axes:
                ax.set_facecolor(bg)
                ax.tick_params(colors=fg)

                ax.xaxis.label.set_color(fg)
                ax.yaxis.label.set_color(fg)
                ax.title.set_color(fg)

                for spine in ax.spines.values():
                    spine.set_color(grid)

                ax.grid(
                    True,
                    color=grid,
                    alpha=0.35
                )

        self.canvas_2d.draw_idle()
        self.canvas_3d.draw_idle()

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode

        if self.dark_mode:
            self.theme_toggle.setText("☀")
            self.theme_toggle.setToolTip("Switch to light theme")
        else:
            self.theme_toggle.setText("☾")
            self.theme_toggle.setToolTip("Switch to dark theme")

        self.create_style(self.dark_mode)
        self.update_plot_theme()

    def show_doe_plots(self):

        figures = self.create_experiments.plot_doe_experiments()

        self.doe_plot_dialog = DOEPlotsDialog(figures,self)

        self.doe_plot_dialog.show()

    def open_profile_dialog(self):
        self.create_experiments = ExperimentCreator(self.doe_sampler, self.component_gen, self.exp_par, self.profiles, self.exp_storage)
        dlg = ExperimentProfileDialog(self.exp_par,self)
        if dlg.exec():
            self.log_message("Experiment profile dialog opened/updated.", "INFO")

    def define_doe(self):

        if self.CONTROLLER is None:
            QMessageBox.warning(self, "Error", "Please define environment variables first!")
            self.log_message("Environment variables not set. Set environment variables first.","ERROR",)
            return

        self.doe_sampler   = DoESampler()                                             # ...generating the settings for Experiments
        self.component_gen = ExpComponentGen() # ...functions for experiment section generation
        self.profiles      = ProfileGenerator(self.exp_par, self.component_gen)       
        self.exp_storage   = ExperimentProfileStorage(self.as_rate_converter, self.solution_rate_converter, self.solvent_rate_converter, self.cleaning_sol_rate_converter) 

        self.define_command()

        self.changed_dimensions_count_label.setText("------⏳ Generating DoE ------")
        self.changed_dimensions_label.setText(      "------⏳ Generating DoE ------")
        self.experiment_count_label.setText(        "------⏳ Generating DoE ------")

        QApplication.processEvents()

        dict_for_strategy_selection = {

            "Greedy minimizataion upon sobol sampling"         : "GREEDY" ,
            "Latin Hypercube Sampling"                         : "LHS"    ,
            "Latin Hypercube Sampling + Lloyd-Max optimization": "LHS_OPT",
            "Random sampling"                                  : "RAND"   ,
            "Sobol sampling"                                   : "SOB"    ,
            "Factorial DoE generation"                         : "FACT"   ,
        }

        SAMPLING_STRATEGY_SELECTED = self.doe_type_combo.currentText()
        strategy_code = dict_for_strategy_selection.get(SAMPLING_STRATEGY_SELECTED, "GREEDY")

        # Create experiments generator
        self.create_experiments = ExperimentCreator(self.doe_sampler,self.component_gen,self.exp_par,self.profiles,self.exp_storage)
        self.create_experiments.set_doe_settings(self.exp_par, self.initial_exp_count, Type = strategy_code)

        self.create_experiments.generate_experiments()

        if self.create_experiments.DoE_val.keys():

            displayed_text = f"{',\n'.join(self.create_experiments.DoE_val.keys())}"
            self.changed_dimensions_count_label.setText(displayed_text)

        else:

            self.changed_dimensions_count_label.setText("0 dimensions are selected.")

        self.changed_dimensions_label.setText(f"{self.create_experiments.dimensions}")
        self.initial_exp_count = len(self.create_experiments.exp_storage.data)

        if strategy_code == "FACT":

            self.experiment_count_label.setText(f"{self.initial_exp_count}*\n(*Full Factorial DoE sample number\nis calculated automatically.)")
        else:

            self.experiment_count_label.setText(f"{self.initial_exp_count}")

        self.log_message(f"{self.initial_exp_count} samples in the initial DoE.", "INFO")
        self.log_message( f"{self.create_experiments.dimensions} changed dimensions in the initial DoE.","INFO",)
        self.log_message( f"Sampling strategy selected: {self.create_experiments.TYPE}.", "INFO",)
        self.reset_plots()

        try:
            self.create_experiments.plot_sampled_points(fig_2d=self.canvas_2d.figure, fig_3d=self.canvas_3d.figure)
            dims = getattr(self.create_experiments, "dimensions", 1)
            min_width = max(350, dims * 85)
            self.canvas_2d.setMinimumWidth(min_width)
            self.canvas_2d.draw()

        except Exception as e:

            msg = "With these settings plotting is unavailable"
            self.canvas_2d.figure.clear()
            self.canvas_2d.figure.patch.set_facecolor("#111827")  # Dark background
            ax2d = self.canvas_2d.figure.add_subplot(111)
            ax2d.set_facecolor("#1f2937")
            ax2d.axis("off")  # Hide axis borders & ticks
            ax2d.text(0.5,0.5,msg,color="#ef4444", fontsize=10,fontweight="bold",ha="center",va="center",wrap=True)
            self.canvas_2d.draw()

        try:

            self.create_experiments.plot_sampled_points(fig_2d=self.canvas_2d.figure, fig_3d=self.canvas_3d.figure)
            self.canvas_3d.setMinimumSize(300, 300)
            self.canvas_3d.draw()

        except Exception as e:

            msg = "With these settings plotting is unavailable"
            self.canvas_3d.figure.clear()
            self.canvas_3d.figure.patch.set_facecolor("#111827")
            ax3d = self.canvas_3d.figure.add_subplot(111)
            ax3d.set_facecolor("#1f2937")
            ax3d.axis("off")
            ax3d.text(0.5,0.5,msg,color="#ef4444",fontsize=10,fontweight="bold",ha="center",va="center",wrap=True)
            self.canvas_3d.draw()

        self.log_message("Initial DoE generated.", "SUCCESS")

        self.generate_button.setStyleSheet("""QPushButton#generate_button {background: #3aa662;
                                                border: 1px solid #2CFF05;color: white;text-align: center;}
                                              QPushButton#generate_button:hover {background: #88E788;}""")


    def log_message(self,message,level="INFO"):
        if hasattr(self,"dashboard_log"):
            self.dashboard_log.add_log(message,level)

    def upon_omni_finished(self):
        self.dashboard_omni.set_status("Inactive")
        self.dashboard_model.set_status("Running")
        self.omni_run_count = self.CONTROLLER.omni_count
        self.omni_count_label.setText(f'{self.omni_run_count} runs')

    def upon_MODEL_finished(self):
        self.dashboard_omni.set_status("Running")
        self.dashboard_model.set_status("Inactive")
        self.MODEL_run_count = self.CONTROLLER.model_count
        self.model_count_label.setText(f'{self.MODEL_run_count} runs')

    def monitoring_MODEL(self, path, num):
        if self.CONTROLLER.process_stop:
          return True

        self.CONTROLLER.current_stage = Stage.MODEL
        if self.CONTROLLER.if_model_finished(path):
          self.CONTROLLER.model_count += 1
          self.upon_MODEL_finished()
          PATH, exp_count = self.CONTROLLER.pathcreator_omni(self.CONTROLLER.model_count, self.CONTROLLER.omni_folder )
          self.CONTROLLER.current_omni_folder = PATH
          self.monitor_mode = Stage.OMNI
          self.monitor_path = PATH
          self.monitor_num = exp_count
          self.timer.start(3000)
          self.log_message(f"MODEL stage finished. Run count: {self.CONTROLLER.model_count}", "SUCCESS")

          return True

        else:

          return False

    def monitoring_omni(self, path, num):
        if self.CONTROLLER.process_stop:
            return True
        self.CONTROLLER.current_stage = Stage.OMNI
        if self.CONTROLLER.if_omni_finished(path, num):
            PATH = self.CONTROLLER.pathcreator_model(self.CONTROLLER.omni_count, self.CONTROLLER.model_folder)
            self.CONTROLLER.current_model_folder = PATH
            self.CONTROLLER.omni_count += 1
            self.upon_omni_finished()
            self.DATAPROCESSOR.read_data_from = self.CONTROLLER.current_omni_folder
            self.DATAPROCESSOR.data_reader()
            self.monitor_mode = Stage.MODEL
            self.monitor_path = PATH
            self.timer.start(3000)
            self.log_message(f"OMNI stage finished. Run count: {self.CONTROLLER.omni_count}", "SUCCESS")
            return True
        else:
            return False

    def update_dashboard_status(self):
        if self.CONTROLLER is None:
            self.dashboard_omni.setText("OMNI: -")
            self.dashboard_model.setText("MODEL: -")
            self.dashboard_runs.setText("Runs: 0 / 0")
            self.dashboard_current.setText("Stage: None")
        else:
            #self.dashboard_omni.setText(f"OMNI runs: {self.omni_run_count}")
            #self.dashboard_model.setText(f"MODEL runs: {self.MODEL_run_count}")
            self.dashboard_runs.setText(f"Runs: {self.omni_run_count + self.MODEL_run_count}")
            self.dashboard_current.setText(f"Stage: {self.CONTROLLER.current_stage.value}")


    def set_buttons_enabled(self, enabled):
        buttons = [
          self.start_button,
          self.define_button,
          self.reset_button,
          self.quick_start_btn,
          self.quick_reset_btn,
          self.generate_button,
          self.profile_button,
        ]
        for button in buttons:
            button.setEnabled(enabled)

    @Slot(dict)
    def update_system_cards(self, data):
        # Extract values directly from the dictionary emitted by the worker
        cpu  = data["cpu" ]
        ram  = data["ram" ]
        down = data["down"]
        up   = data["up"  ]
        self.log_message(f"CPU:{cpu}%, RAM:{ram}%, Net Down:{down:.2f} MB/s")

    def monitoring_loop(self):

        self.update_dashboard_status()
        CURR_EXP_NO = counter(self.monitor_path, self.omni_filename)

        if CURR_EXP_NO != 0 and self.monitor_mode == Stage.OMNI and self.previous_exp_count != CURR_EXP_NO:
            self.conducted_experiments_count += 1

        self.expts_count_label.setText(f'{self.conducted_experiments_count} runs')

        if self.CONTROLLER is None:
          return

        if self.process_stop:
          self.timer.stop()
          return

        if self.monitor_mode == Stage.OMNI:
          self.monitoring_omni(self.monitor_path, self.monitor_num)

        elif self.monitor_mode == Stage.MODEL:
          self.monitoring_MODEL(self.monitor_path, self.monitor_num)
        self.previous_exp_count = counter(self.monitor_path, self.omni_filename)

    def define_command(self):

        self.status_label.clear()
        self.error_label.clear()

        folder_loc = self.exp_path_input.text().strip()
        folder_MODEL_loc = self.MODEL_path_input.text().strip()
        no_exp = self.exp_number_input.text().strip()

        if not folder_loc or not folder_MODEL_loc or not no_exp:

          self.status_label.setText("Please fill all fields")
          self.status_label.setVisible(True)
          return
        if not os.path.isdir(folder_loc):

          self.error_label.setText("Invalid OMNI folder")
          self.error_label.setVisible(True)
          return
        if not os.path.isdir(folder_MODEL_loc):

          self.error_label.setText("Invalid MODEL folder")
          self.error_label.setVisible(True)
          return

        try:
          no_exp = int(no_exp)

          if no_exp <= 0:

            raise ValueError

        except ValueError:

          self.error_label.setText("Enter a positive integer")
          self.error_label.setVisible(True)
          return

        self.status_label.setVisible(False)
        self.error_label.setVisible(False)

        self.omni_folder       = folder_loc
        self.model_folder      = folder_MODEL_loc
        self.initial_exp_count = no_exp
        self.CONTROLLER        = Controller(self.omni_folder,self.model_folder,self.initial_exp_count,self.omni_filename,self.MODEL_filename)

        self.CONTROLLER.exp_num(no_exp)

        self.variables_set     = True
        self.define_button.setStyleSheet("""QPushButton#define_button {background: #3aa662;border: 1px solid #2CFF05;color: white;}
                                            QPushButton#define_button:hover {background: #88E788;}""")

        self.log_message("🌎 Environment variables set.", "SUCCESS")

    def start_command(self):

        if not self.variables_set:
          QMessageBox.warning(self, "Error", "Please define environment variables first!")
          self.log_message("Controller not initialized. Define environment variables first.", "ERROR")
          return
        if counter(self.omni_folder, self.omni_filename) != 0:
          QMessageBox.warning(self, "Error", "Files already in experiment folder")
          self.log_message("Files already in experiment folder.", "ERROR")
          return
        if counter(self.model_folder, self.MODEL_filename) != 0:
          QMessageBox.warning(self, "Error", "Files already in MODEL folder")
          self.log_message("Files already in MODEL folder.", "ERROR")
          return

        if self.create_experiments is None:
          QMessageBox.warning(self, "Error", "Generate the initial DoE first!")
          self.log_message("Initial DoE not defined.", "ERROR")
          return

        self.set_buttons_enabled(False)
        self.CONTROLLER.process_stop = False
        self.DATAPROCESSOR = DataPlotter(self.omni_filename, self.MODEL_filename)
        self.monitor_mode = Stage.OMNI

        PATH, exp_count = self.CONTROLLER.pathcreator_omni( self.CONTROLLER.model_count, self.CONTROLLER.omni_folder )
        self.CONTROLLER.current_omni_folder = PATH
        self.monitor_path = PATH
        self.monitor_num = self.initial_exp_count

        self.create_experiments.export_OMNI_expts(self.monitor_path)

        self.CONTROLLER.start()
        self.exp_status_box.setStyleSheet("""QGroupBox#exp_status_box {background:  #3aa662;border: 2px solid #2CFF05;border-radius: 12px;}""")
        self.exp_status_box.setTitle("🟢 Experiment/Modeling monitoring active")
        self.timer.start(5000)
        self.log_message("Experiment started", "INFO")
        self.dashboard_omni.set_status("Running")
        self.dashboard_model.set_status("Inactive")
        self.switch_page(0)

    def stop_command(self):
        if self.CONTROLLER:
          self.CONTROLLER.stop()
          self.timer.stop()
          self.set_buttons_enabled(True)
          self.dashboard_omni.set_status("Stopped")
          self.dashboard_model.set_status("Stopped")
          self.log_message("Experiment process stopped by user.", "WARNING")
          self.exp_status_box.setStyleSheet("""QGroupBox#exp_status_box {background:  #ff5b5b;border: 2px solid #FF3131;border-radius: 12px;}""")
          self.exp_status_box.setTitle("🔴 Experiment/Modeling monitoring stopped")

    def reset_plots(self):
        BG_COLOR = "#111827"  # Match your dark UI container hex color

        fig_2d = self.canvas_2d.figure
        fig_3d = self.canvas_3d.figure

        if fig_2d:
            fig_2d.clear()
            fig_2d.patch.set_facecolor(BG_COLOR)  # Solid fill kills artifacts

        if fig_3d:
            fig_3d.clear()
            fig_3d.patch.set_facecolor(BG_COLOR)  # Solid fill kills artifacts

        self.canvas_2d.draw()
        self.canvas_3d.draw() 

    def reset_command(self):
        if self.CONTROLLER:
          self.CONTROLLER.stop()    
        self.exp_profile_parameters = ExperimentProfileParameters()  

        self.create_experiments          = None

        self.variables_set               = False
        self.CONTROLLER                  = None
        self.DATAPROCESSOR               = None
        self.conducted_experiments_count = 0
        self.previous_exp_count          = 0
        self.MODEL_run_count             = 0
        self.omni_run_count              = 0

        self.exp_path_input.clear()
        self.MODEL_path_input.clear()
        self.exp_number_input.clear()
        self.dashboard_omni.set_status("Inactive")
        self.dashboard_model.set_status("Inactive")
        self.omni_count_label.setText( "0 OMNI  runs")
        self.model_count_label.setText("0 MODEL runs")
        self.expts_count_label.setText("0 EXPTS runs")

        self.changed_dimensions_count_label.setText("")
        self.changed_dimensions_label.setText("")
        self.experiment_count_label.setText("")

        self.generate_button.setStyleSheet("")
        self.define_button.setStyleSheet("")
        self.variables_label.setText("")
        self.exp_status_box.setTitle("📈 Progress Track")
        self.exp_status_box.setStyleSheet("")
        self.reset_plots()
        self.exp_par = ExperimentProfileParameters()   

        try:
            self.status_label.clear()
            self.error_label.clear()
        except: 
            print('')

        while self.plot_container_layout.count():
            item = self.plot_container_layout.takeAt(0)
            widget = item.widget()
            if widget and widget not in (self.canvas_2d, self.canvas_3d):
                widget.deleteLater()

        self.log_message("Reset settings", "SUCCESS")

    def plot_expts(self):
        if self.DATAPROCESSOR is None:
            QMessageBox.warning(self, "Error", "No data available yet!")
            return

        self.DATAPROCESSOR.read_data_from = self.CONTROLLER.current_omni_folder

        # --------------------------------------------------------------------------------------------------------------#
        # FOR TESTING #
        #self.DATAPROCESSOR = DataPlotter(self.omni_filename, self.MODEL_filename)
        #self.DATAPROCESSOR.read_data_from = r"C:\Users\abotondo\OneDrive - purdue.edu\ECNDATA\Desktop\ORCHESTRATOR_test"
        # --------------------------------------------------------------------------------------------------------------#

        self.DATAPROCESSOR.data_reader()

        fig = self.DATAPROCESSOR.plot_generator()

        if fig:
            self.log_message("Data plotted successfully.", "SUCCESS")

        # Remove existing plot
        while self.plot_container_layout.count():
            item = self.plot_container_layout.takeAt(0)
            widget = item.widget()

            if widget:
                widget.setParent(None)
                widget.deleteLater()

        # Create Matplotlib canvas
        self.canvas_embedded = FigureCanvas(fig)

        # Allow canvas to expand, but set a sensible minimum height based on row count
        num_rows = len(self.DATAPROCESSOR.EVAL)
        min_height = num_rows * 220  # 220px per experiment row

        self.canvas_embedded.setMinimumSize(800, min_height)
        self.canvas_embedded.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed) # Fixed height allows QScrollArea to handle overflow)

        self.canvas_embedded.setMinimumSize(0, 0)

        # Transparent Qt background
        self.canvas_embedded.setStyleSheet("background: transparent;")

        # Add to layout
        self.plot_container_layout.addWidget(self.canvas_embedded,1)

        # Draw after adding to layout
        self.canvas_embedded.draw_idle()

        self.switch_page(2)

    def closeEvent(self, event):
        self.reset_command()
        if hasattr(self, "system_monitor"):
            self.system_monitor.closeEvent(event)
            super().closeEvent(event)

# ========================================================== #
#                       MAIN                                 #
# ========================================================== #

if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


