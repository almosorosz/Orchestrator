#!/usr/bin/env python
# coding: utf-8

# In[1]:






# In[2]:


import sys            
import os
import time

from pathlib import Path

from matplotlib.figure                    import Figure
from matplotlib.backends.backend_qtagg    import FigureCanvasQTAgg as FigureCanvas
from datetime                             import datetime

from PySide6.QtGui                        import QPainterPath, QRegion, QIcon, QPixmap
from PySide6.QtCore                       import QTimer, Qt, Slot, QRect
from PySide6.QtWidgets                    import (QApplication,QMainWindow,QLabel, QScrollArea,QFormLayout,QSizePolicy,
                                                   QPushButton,QLineEdit,QMessageBox,QProgressBar,QFrame,
                                                   QWidget,QDialog,QVBoxLayout,QHBoxLayout,QStyle,QComboBox,
                                                   QStackedWidget,QGroupBox,QTextEdit,QGridLayout)

from gui.system_monitor_widget            import SystemMonitorWidget
from gui.experiment_profile_dialog        import ExperimentProfileDialog, DOEPlotsDialog
from gui.stylish_widgets                  import MetricCard, LogViewer, StatusBadge

from gui.pages.dashboard_page             import DashboardPage
from gui.pages.experiment_page            import ExperimentPage
from gui.pages.plots_page                 import PlotsPage
from gui.pages.brain_page                 import BrainPage
from gui.pages.modeling_page              import ModelingPage
from gui.pages.control_panel_page         import ControlPanelPage
from gui.pages.settings_page              import SettingsPage


from experiment.experiment_service        import ExperimentService

from experiment.execution_coordinator     import (ExperimentCoordinator,
                                                 )
from hardware_devices.hardware_manager    import HardwareBusyError

from experiment.profiles.profile_gen      import (
                                                 DefineProcessParameter,
                                                 ProfileCreator,
                                                 )

from app.paths                            import (
                                                 BASE_DIR,
                                                 ASSETS_DIR,
                                                 DATA_DIR,
                                                 LOGO_PATH,
                                                 TEXT_PATH,
                                                 ICON_PATH,
                                                 )


from app.controller                       import Controller, Stage
from app.data_plotter                     import DataPlotter

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowFlags( Qt.WindowType.Window |  Qt.WindowType.FramelessWindowHint |  Qt.WindowType.WindowSystemMenuHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)


        self.SPACING_1 = 8
        self.SPACING_2 = 16
        self.SPACING_3 = 24
        self.SPACING_4 = 32

        self.resize(1100, 800)
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

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Header
        header_widget = self._build_header()
        root_layout.addWidget(header_widget)        

        # ----------------------------------------------------------------- #
        # BODY
        # ----------------------------------------------------------------- #
        body_widget = QWidget()
        body_widget.setObjectName("bodyContainer")
        body_layout = QHBoxLayout(body_widget)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Sidebar
        sidebar_widget = self._build_sidebar()

        # Main content stack
        self.stack = self._build_page_stack()
        self._connect_page_signals()
        self._connect_navigation()

        body_layout.addWidget(sidebar_widget)
        body_layout.addWidget(self.stack, 1)

        root_layout.addWidget(body_widget)

        # ----------------------------------------------------------------- #
        # CONSTANTS, MODULES, AND INTERNAL VARIABLES                        #
        # ----------------------------------------------------------------- #

        self.create_style()

        self._initialize_experiment_services()

        self._initialize_runtime_state()

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
        """Switch the active page and update navigation button states."""

        self.stack.setCurrentIndex(index)

        for i, btn in enumerate(self.nav_buttons):
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

    @Slot(dict)
    def on_experiment_progress(self, progress):
        """
        Update the GUI when new OMNI experiment files
        have been detected.
        """

        new_experiments = progress["new_experiments"]
        total_experiments = progress["total_experiments"]

        self.log_message(
            f"{new_experiments} new experiment(s) "
            f"detected. Total conducted: {total_experiments}",
            "INFO",
        )

        self.dashboard_page.set_experiment_count(
            total_experiments
        )

    def _initialize_runtime_state(self):

        self.CONTROLLER = None
        self.DATAPROCESSOR = None
        self.create_experiments = None

        self.execution = ExperimentCoordinator(parent=self)

        self.execution.omni_completed.connect(self.on_omni_completed)
        self.execution.model_completed.connect(self.on_model_completed)
        self.execution.stage_changed.connect(self.on_stage_changed)
        self.execution.experiment_progress.connect(self.on_experiment_progress)
        self.execution.error_occurred.connect(self.on_execution_error)

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

    @Slot(dict)
    def on_execution_error(self, error_info):
        """
        Display a backend monitoring failure and restore
        user access to Reset.
        """

        stage = error_info["stage"]
        path = error_info["path"]
        error_type = error_info["error_type"]
        message = error_info["message"]

        self.dashboard_page.set_omni_status("Error")
        self.dashboard_page.set_model_status("Error")

        self.dashboard_page.set_current_stage("None")

        self.dashboard_page.set_monitoring_state(False)

        self.set_buttons_enabled(True)

        self.log_message(
            f"Execution error during {stage}: "
            f"{error_type}: {message}. "
            f"Monitoring path: {path}",
            "ERROR",
        )

        QMessageBox.critical(
            self,
            "Experiment Execution Error",
            f"An error occurred during {stage} monitoring.\n\n"
            f"{error_type}: {message}\n\n"
            f"Monitoring directory:\n{path}\n\n"
            f"Execution has been stopped. "
            f"Reset before starting another experiment.",
        )

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

    def _initialize_experiment_services(self):

        self.experiment_service = ExperimentService()

        self.omni_filename = "expttsd"
        self.MODEL_filename = "MODEL"

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
                    color: white;
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
                    color: #f7f7f5;
                    border: 1px solid #4a505a;
                    border-radius: 8px;
                    font-size: 20px;
                    font-weight: bold;
                    padding: 0px;
                }

                QPushButton[sidebarCompact="true"] {
                    padding-top: 2px;
                    padding-bottom: 2px;
                    padding-left: 10px;
                    padding-right: 10px;
                }

                QPushButton#theme_toggle:hover {
                    background-color: #363b44;
                    border: 1px solid #6b7280;
                }

                QPushButton#theme_toggle:pressed {
                    background-color: #22262c;
                }
                """)
            #self.logo_label.setStyleSheet("font-weight: bold; font-size: 15px; color: white; background: transparent;")

        else:
            self.setStyleSheet("""
                /* ==================================================
                   LIGHT THEME
                   ================================================== */

                QPushButton#plot_button {
                    text-align: center;
                }

                QPushButton[sidebarCompact="true"] {
                    padding-top: 2px;
                    padding-bottom: 2px;
                    padding-left: 10px;
                    padding-right: 10px;
                }

                QPushButton#reset_button {
                    text-align: center;
                }

                QPushButton#stop_button {
                    text-align: center;
                }

                QPushButton#start_button {
                    text-align: center;
                }

                QGroupBox QLabel {
                    background: transparent;
                    border: none;
                }
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

                QLabel#statusCard {
                    background: #111d31;
                    border: 1px solid #16263d;
                    border-radius: 12px;
                    color: #000000;
                    padding: 16px;
                    font-size: 14px;
                    font-weight: 700;
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
            #self.logo_label.setStyleSheet("font-weight: bold; font-size: 15px; color: black; background: transparent;")


    # ====================================================== #
    # HELPERS & COMMANDS (Unchanged Logic)                   #
    # ====================================================== #    



    def _build_header(self):
        header_widget = QWidget()
        header_widget.setObjectName("topHeader")
        header_widget.setFixedHeight(50)

        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(15, 0, 10, 0)
        header_layout.setSpacing(10)

        # ----------------------------------------------------------
        # LOGO
        # ----------------------------------------------------------

        self.logo_icon_label = QLabel()
        self.logo_icon_label.setStyleSheet(
            "background: transparent;"
        )

        if LOGO_PATH.exists():
            logo_pixmap = QPixmap(str(LOGO_PATH))

            self.logo_icon_label.setPixmap(
                logo_pixmap.scaledToHeight(
                    30,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        self.logo_text_label = QLabel()
        self.logo_text_label.setStyleSheet(
            "background: transparent;"
        )

        if TEXT_PATH.exists():
            text_pixmap = QPixmap(str(TEXT_PATH))

            self.logo_text_label.setPixmap(
                text_pixmap.scaledToHeight(
                    22,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        header_layout.addWidget(self.logo_icon_label)
        header_layout.addWidget(self.logo_text_label)

        header_layout.addStretch()

        # ----------------------------------------------------------
        # THEME
        # ----------------------------------------------------------

        self.theme_toggle = QPushButton("☀")
        self.theme_toggle.setObjectName("theme_toggle")
        self.theme_toggle.setCheckable(True)
        self.theme_toggle.setChecked(False)
        self.theme_toggle.setFixedSize(36, 30)
        self.theme_toggle.setToolTip("Switch to light theme")
        self.theme_toggle.setCursor(Qt.PointingHandCursor)

        self.theme_toggle.clicked.connect(
            self.toggle_theme
        )

        header_layout.addWidget(self.theme_toggle)

        # ----------------------------------------------------------
        # SYSTEM STATUS
        # ----------------------------------------------------------

        self.status_pill = QLabel("● 0/0 Connected")
        self.status_pill.setObjectName("statusPill")
        self.status_pill.setFixedSize(155, 30)

        header_layout.addWidget(self.status_pill)

        # ----------------------------------------------------------
        # WINDOW CONTROLS
        # ----------------------------------------------------------

        header_layout.addSpacing(15)

        self.btn_minimize = QPushButton("—")
        self.btn_minimize.setObjectName("windowControlBtn")
        self.btn_minimize.setFixedSize(30, 30)
        self.btn_minimize.setToolTip("Minimize")
        self.btn_minimize.clicked.connect(
            self.showMinimized
        )

        self.btn_maximize = QPushButton("□")
        self.btn_maximize.setObjectName("windowControlBtn")
        self.btn_maximize.setFixedSize(30, 30)
        self.btn_maximize.setToolTip("Maximize")
        self.btn_maximize.clicked.connect(
            self.toggle_maximize
        )

        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("windowCloseBtn")
        self.btn_close.setFixedSize(30, 30)
        self.btn_close.setToolTip("Close")
        self.btn_close.clicked.connect(
            self.close
        )

        header_layout.addWidget(self.btn_minimize)
        header_layout.addWidget(self.btn_maximize)
        header_layout.addWidget(self.btn_close)

        return header_widget

    def update_hardware_status(self, all_connected, connected, total):

        self._hardware_status = (
            all_connected,
            connected,
            total
        )

        if all_connected:
            self.status_pill.setText("● All Connected")
            background = "#064e3b" if self.dark_mode else "#dcfce7"
            foreground = "#34d399" if self.dark_mode else "#15803d"

        else:
            self.status_pill.setText(
                f"● {connected}/{total} Connected"
            )

            background = "#7f1d1d" if self.dark_mode else "#fee2e2"
            foreground = "#fca5a5" if self.dark_mode else "#b91c1c"

        self.status_pill.setStyleSheet(f"""
            QLabel#statusPill {{
                background: {background};
                color: {foreground};
                border-radius: 10px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 600;
            }}
        """)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode

        if self.dark_mode:
            self.theme_toggle.setText("☀")
            self.theme_toggle.setToolTip("Switch to light theme")

        else:
            self.theme_toggle.setText("☾")
            self.theme_toggle.setToolTip("Switch to dark theme")

        self.create_style(self.dark_mode)
        if hasattr(self, "_hardware_status"):
            self.update_hardware_status(
                *self._hardware_status
            )
        self.settings_page.update_plot_theme(self.dark_mode)
        self.dashboard_page.set_theme(self.dark_mode)
        self.control_panel_page.control_panel.set_theme(self.dark_mode)

        # Update embedded ExperimentDashboard
        dashboard = getattr( self.experiment_page, "dashboard", None )

        if dashboard is not None:
            dashboard.set_theme(self.dark_mode)

    def show_doe_plots(self):

        figures = self.create_experiments.plot_doe_experiments()

        self.doe_plot_dialog = DOEPlotsDialog(figures,self)

        self.doe_plot_dialog.show()

    def open_profile_dialog(self):

        dlg = ExperimentProfileDialog(
            self.experiment_service.exp_par,
            self
        )

        if dlg.exec():
            self.log_message(
                "Experiment profile dialog opened/updated.",
                "INFO"
            )

    def define_doe(self):

        if self.CONTROLLER is None:
            QMessageBox.warning(self, "Error", "Please define environment variables first!")
            self.log_message("Environment variables not set. Set environment variables first.","ERROR",)
            return


        self.define_command()

        self.settings_page.show_doe_generating()

        QApplication.processEvents()

        dict_for_strategy_selection = {

            "Greedy minimizataion upon sobol sampling"         : "GREEDY" ,
            "Latin Hypercube Sampling"                         : "LHS"    ,
            "Latin Hypercube Sampling + Lloyd-Max optimization": "LHS_OPT",
            "Random sampling"                                  : "RAND"   ,
            "Sobol sampling"                                   : "SOB"    ,
            "Full Factorial DoE generation"                    : "FACT"   ,
        }

        SAMPLING_STRATEGY_SELECTED = (self.settings_page.get_sampling_strategy())
        strategy_code = dict_for_strategy_selection.get(SAMPLING_STRATEGY_SELECTED, "GREEDY")

        # Create experiments generator
        self.create_experiments = (
            self.experiment_service.generate_initial_doe(
                self.CONTROLLER.initial_exp_count,
                strategy_code
            )
        )

        self.execution.set_experiment_creator(
            self.create_experiments
        )

        self.CONTROLLER.exp_num(
            len(self.create_experiments.exp_storage.data)
        )

        if self.create_experiments.DoE_val.keys():

            displayed_text = ( f"{',\n'.join(self.create_experiments.DoE_val.keys())}" )

            added_comment = ""

            if strategy_code == "FACT":
                added_comment = (
                    "*\n"
                    "(*Full Fact. sample number\n"
                    "is calculated by the\n"
                    "number of dimensions.)"
                )

        else:

            displayed_text = (
                "0 dimensions are selected."
            )

            added_comment = (
                " identical sample(s)\n"
                "had been taken."
            )

        self.settings_page.set_doe_info( displayed_text, self.create_experiments.dimensions, f"{self.CONTROLLER.initial_exp_count} {added_comment}")

        self.log_message(f"{self.CONTROLLER.initial_exp_count} samples in the initial DoE.", "INFO")
        self.log_message( f"{self.create_experiments.dimensions} changed dimensions in the initial DoE.","INFO",)
        self.log_message( f"Sampling strategy selected: {self.create_experiments.TYPE}.", "INFO",)

        self.settings_page.reset_plots(self.dark_mode)

        self.settings_page.render_doe_plots(self.create_experiments,self.dark_mode)

        self.log_message("Initial DoE generated.", "SUCCESS")
        self.settings_page.update_plot_theme(self.dark_mode)

        self.settings_page.set_generate_success()


    def log_message(self, message, level="INFO"):
        if hasattr(self, "dashboard_page"):
            self.dashboard_page.add_log(message, level)

    def set_buttons_enabled(self, enabled):
        buttons = [
            self.dashboard_page.start_button,
            self.dashboard_page.reset_button,
            self.quick_start_btn,
            self.quick_reset_btn,
        ]

        for button in buttons:
            button.setEnabled(enabled)

        self.settings_page.set_controls_enabled( enabled)

    @Slot(dict)
    def update_system_cards(self, data):
        # Extract values directly from the dictionary emitted by the worker
        cpu  = data["cpu" ]
        ram  = data["ram" ]
        down = data["down"]
        up   = data["up"  ]
        self.log_message(f"CPU:{cpu}%, RAM:{ram}%, Net Down:{down:.2f} MB/s")

    @Slot(dict)
    def on_omni_completed(self, result):
        """
        Update GUI after OMNI completion.
        """

        self.dashboard_page.set_omni_count(
            result["omni_run_count"]
        )

        self.log_message(
            f"OMNI stage finished - "
            f"experimental iteration no. "
            f"{result['omni_run_count']}. "
            f"Experiments are stored in: "
            f"{self.CONTROLLER.current_omni_folder}",
            "SUCCESS",
        )

    @Slot(dict)
    def on_model_completed(self, result):

        self.dashboard_page.set_model_count(result["model_run_count"])    
        self.log_message(f"Model output no. {result['model_run_count']} stored in: {self.CONTROLLER.current_model_folder}","INFO")    
        self.log_message(f"MODEL stage finished - iteration no. {result['model_run_count']}","SUCCESS")

    @Slot(object)
    def on_stage_changed(self, stage):

        if stage == Stage.OMNI:

            self.dashboard_page.set_omni_status("Running")
            self.dashboard_page.set_model_status("Inactive")

        elif stage == Stage.MODEL:

            self.dashboard_page.set_omni_status("Inactive")
            self.dashboard_page.set_model_status("Running")

        else:
            # Stop/Reset handles the individual status labels.
            self.dashboard_page.set_current_stage("None")
            return

        self.dashboard_page.set_current_stage(stage.value)
        self.dashboard_page.set_total_runs( self.execution.omni_run_count + self.execution.model_run_count)


    def define_command(self):

        self.settings_page.set_message()

        folder_loc, folder_MODEL_loc, no_exp = (
            self.settings_page.get_environment_inputs()
        )

        if not folder_loc or not folder_MODEL_loc or not no_exp:
            self.settings_page.set_message(
                message="Please fill all fields"
            )
            return

        if not os.path.isdir(folder_loc):
            self.settings_page.set_message(
                error="Invalid OMNI folder"
            )
            return

        if not os.path.isdir(folder_MODEL_loc):
            self.settings_page.set_message(
                error="Invalid MODEL folder"
            )
            return

        try:
            no_exp = int(no_exp)

            if no_exp <= 0:
                raise ValueError

        except ValueError:
            self.settings_page.set_message(
                error="Enter a positive integer"
            )
            return

        self.settings_page.set_message()

        self.CONTROLLER = Controller(
            folder_loc,
            folder_MODEL_loc,
            no_exp,
            self.omni_filename,
            self.MODEL_filename,
        )

        # Give the coordinator the SAME controller instance.
        self.execution.set_controller(self.CONTROLLER)

        self.settings_page.set_define_success()

        self.log_message(
            "🌎 Environment variables set.",
            "SUCCESS",
        )

    def start_command(self):

        if self.CONTROLLER is None:
            QMessageBox.warning(self,"Error", "Please define environment variables first!")
            self.log_message("Controller not initialized. Define environment variables first.","ERROR")
            return

        # Check for existing OMNI experiment directories
        if any(
            entry.is_dir()
            for entry in os.scandir(self.CONTROLLER.omni_folder)
        ):
            QMessageBox.warning(
                self,
                "Error",
                "OMNI folder already contains experiment directories"
            )
            self.log_message(
                "OMNI root folder is not empty.",
                "ERROR"
            )
            return

        # Check for existing MODEL directories
        if any(
            entry.is_dir()
            for entry in os.scandir(self.CONTROLLER.model_folder)
        ):
            QMessageBox.warning(
                self,
                "Error",
                "MODEL folder already contains experiment directories"
            )
            self.log_message(
                "MODEL root folder is not empty.",
                "ERROR"
            )
            return

        if self.create_experiments is None:
            QMessageBox.warning(  self,   "Error",  "Generate the initial DoE first!" )
            self.log_message( "Initial DoE not defined.",  "ERROR" )
            return

        if self.CONTROLLER.process_stop:
            QMessageBox.warning(
                self,
                "Reset Required",
                "The previous execution was stopped. "
                "Please Reset and configure a new experiment "
                "before starting again."
            )
            return

        self.set_buttons_enabled(False)

        self.DATAPROCESSOR = DataPlotter( self.omni_filename,  self.MODEL_filename )

        self.execution.set_data_processor(self.DATAPROCESSOR)

        self.execution.start_execution(
            omni_root=self.CONTROLLER.omni_folder,
            initial_exp_count=self.CONTROLLER.initial_exp_count
        )

        self.dashboard_page.set_monitoring_state( True )

        self.log_message( "Experiment started", "INFO", )



        self.switch_page(0)

    def stop_command(self):

        if self.CONTROLLER is None:
            return

        self.execution.stop_execution()

        # ----------------------------------------------------------
        # GUI ONLY
        # ----------------------------------------------------------

        self.set_buttons_enabled(True)

        self.dashboard_page.set_omni_status("Stopped")

        self.dashboard_page.set_model_status("Stopped")

        self.dashboard_page.set_monitoring_state(False)

        self.log_message( "Experiment process stopped by user.", "WARNING")



    def reset_command(self):

        # ----------------------------------------------------------
        # BACKEND TEARDOWN
        # ----------------------------------------------------------

        self.execution.reset()

        self.experiment_service.reset()

        # ----------------------------------------------------------
        # MAIN WINDOW REFERENCES
        # ----------------------------------------------------------

        self.create_experiments = None
        self.CONTROLLER = None
        self.DATAPROCESSOR = None


        # ----------------------------------------------------------
        # GUI RESET
        # ----------------------------------------------------------

        self.dashboard_page.reset_status()

        self.settings_page.reset_ui()

        self.settings_page.reset_plots(self.dark_mode)

        while self.plots_page.plot_container_layout.count():
            item = (self.plots_page.plot_container_layout.takeAt(0))

            widget = item.widget()

            if widget:
                widget.deleteLater()

        self.log_message("Reset settings","SUCCESS",)

    def _build_page_stack(self):
        self.stack = QStackedWidget()

        # ----------------------------------------------------------
        # CREATE PAGES
        # ----------------------------------------------------------

        self.dashboard_page = DashboardPage(dark_mode=self.dark_mode,spacing_1=self.SPACING_1,spacing_2=self.SPACING_2,parent=self)

        self.experiment_page    = ExperimentPage( parent=self)    
        self.plots_page         = PlotsPage(parent=self)
        self.brain_page         = BrainPage(parent=self)    
        self.modeling_page      = ModelingPage(parent=self)    
        self.control_panel_page = ControlPanelPage(parent=self)
        self.control_panel_page.control_panel.set_theme(
            self.dark_mode
        )
        self.control_panel = (self.control_panel_page.control_panel)
        self.control_panel.dashboard_host = self.embed_experiment_dashboard

        self.hardware = (self.control_panel_page.hardware)

        self.settings_page = SettingsPage(parent=self)

        # ----------------------------------------------------------
        # ALLOW PAGES TO SHRINK
        # ----------------------------------------------------------

        pages = [
            self.dashboard_page,
            self.experiment_page,
            self.plots_page,
            self.brain_page,
            self.modeling_page,
            self.control_panel_page,
            self.settings_page,
        ]

        for page in pages:
            page.setMinimumWidth(0)
            page.setMinimumHeight(0)
            self.stack.addWidget(page)

        self.stack.setMinimumWidth(0)
        self.stack.setMinimumHeight(0)

        return self.stack

    def embed_experiment_dashboard(self, dashboard):

        dashboard.set_theme(self.dark_mode)

        self.experiment_page.attach_dashboard(dashboard)

        self.switch_page(1)

    def end_experiment_session(self):
        """
        End the embedded ExperimentDashboard session safely.

        The dashboard remains attached if its experiment
        worker cannot be stopped.
        """

        page = self.experiment_page
        dashboard = page.dashboard

        if dashboard is None:
            return

        panel = self.control_panel_page.control_panel

        print("[MAIN WINDOW] Ending experiment session...")

        # --------------------------------------------------
        # 1. Stop worker and close dashboard
        # --------------------------------------------------

        try:
            closed = dashboard.close()

        except RuntimeError as error:
            QMessageBox.warning(
                self,
                "Session Cleanup Error",
                f"Could not close ExperimentDashboard:\n\n{error}"
            )
            return

        if not closed:
            QMessageBox.warning(
                self,
                "Experiment Still Running",
                "The ExperimentDashboard could not finish "
                "its shutdown.\n\n"
                "The session remains open. "
                "Wait for the worker to stop and try again."
            )
            return

        # --------------------------------------------------
        # 2. Verify experiment ownership has been released
        # --------------------------------------------------

        manager = panel.hardware_manager

        try:
            manager.require_idle()

        except HardwareBusyError as error:
            QMessageBox.warning(
                self,
                "Hardware Still Busy",
                "The experiment dashboard closed, but "
                "hardware ownership is still active.\n\n"
                f"{error}"
            )
            return

        # --------------------------------------------------
        # 3. Remove dashboard from Experiment page
        # --------------------------------------------------

        page.detach_dashboard()

        if panel.experiment_window is dashboard:
            panel.experiment_window = None

        print(
            "[MAIN WINDOW] Experiment session ended. "
            "Hardware connections remain available "
            "to Control Panel."
        )

    def _connect_page_signals(self):

        # Dashboard
        self.dashboard_page.start_requested.connect(self.start_command)

        self.dashboard_page.stop_requested.connect(self.stop_command)

        self.dashboard_page.reset_requested.connect(self.reset_command)

        self.dashboard_page.plot_requested.connect(self.plot_expts)

        # Settings
        self.settings_page.define_requested.connect(self.define_command)

        self.settings_page.generate_requested.connect(self.define_doe)

        self.settings_page.profile_requested.connect(self.open_profile_dialog)

        self.settings_page.plot_doe_requested.connect(self.show_doe_plots)

        # Experiment
        self.experiment_page.end_session_button.clicked.connect(self.end_experiment_session)

        # control panel
        self.control_panel.connection_status_changed.connect(
            self.update_hardware_status
        )

        # Initialize from the existing device state.
        panel = self.control_panel

        states = [
            widget.is_connected()
            for widget in panel.device_widgets.values()
        ]

        self.update_hardware_status(
            bool(states) and all(states),
            sum(states),
            len(states)
        )

    def _build_sidebar(self):
        sidebar_widget = QWidget()
        sidebar_widget.setObjectName("sidebar")

        sidebar_widget.setMinimumWidth(218)
        sidebar_widget.setMaximumWidth(218)

        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(self.SPACING_2)

        # ----------------------------------------------------------
        # NAVIGATION
        # ----------------------------------------------------------

        self.btn_dashboard = QPushButton("🏠  Dashboard")
        self.btn_experiment = QPushButton("🧪  Experiment")
        self.btn_plots = QPushButton("📊  Data & Plots")
        self.btn_brain = QPushButton("🧠  Brain")
        self.btn_modeling = QPushButton("📐  Modeling")
        self.btn_control_panel = QPushButton("🎛  Control Panel")
        self.btn_settings = QPushButton("⚙️  Settings")

        self.nav_buttons = [
            self.btn_dashboard,
            self.btn_experiment,
            self.btn_plots,
            self.btn_brain,
            self.btn_modeling,
            self.btn_control_panel,
            self.btn_settings,
        ]

        for btn in self.nav_buttons:
            btn.setProperty("sidebarCompact", True)
            btn.setCheckable(True)
            btn.setFixedHeight(36)
            sidebar_layout.addWidget(btn)

        sidebar_layout.addSpacing(8)

        # ----------------------------------------------------------
        # QUICK ACTIONS
        # ----------------------------------------------------------

        quick_label = QLabel("Quick Actions")
        quick_label.setStyleSheet(
            "color: #64748b;"
            "font-size: 11px;"
            "font-weight: bold;"
            "background: transparent;"
        )

        sidebar_layout.addWidget(quick_label)

        self.quick_start_btn = QPushButton("▶ Start")
        self.quick_start_btn.setObjectName("start_button")

        self.quick_stop_btn = QPushButton("⏹ Stop")
        self.quick_stop_btn.setObjectName("stop_button")

        self.quick_reset_btn = QPushButton("🔄 Reset")
        self.quick_reset_btn.setObjectName("quick_reset_button")

        self.quick_quit_btn = QPushButton("🔐 Quit")
        self.quick_quit_btn.setObjectName("quit_button")

        quick_actions = [
            self.quick_start_btn,
            self.quick_stop_btn,
            self.quick_reset_btn,
            self.quick_quit_btn,
        ]

        for btn in quick_actions:
            btn.setProperty("sidebarCompact", True)
            btn.setFixedHeight(36)
            sidebar_layout.addWidget(btn)

        self.quick_start_btn.clicked.connect(
            self.start_command
        )

        self.quick_stop_btn.clicked.connect(
            self.stop_command
        )

        self.quick_reset_btn.clicked.connect(
            self.reset_command
        )

        self.quick_quit_btn.clicked.connect(
            self.close
        )

        # ----------------------------------------------------------
        # BOTTOM ICON
        # ----------------------------------------------------------

        self.bottom_icon_label = QLabel()

        self.bottom_icon_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.bottom_icon_label.setStyleSheet(
            "background: transparent;"
        )

        if ICON_PATH.exists():
            pixmap = QPixmap(str(ICON_PATH))

            scaled_pixmap = pixmap.scaledToWidth(
                120,
                Qt.TransformationMode.SmoothTransformation,
            )

            self.bottom_icon_label.setPixmap(
                scaled_pixmap
            )

        sidebar_layout.addWidget(
            self.bottom_icon_label
        )

        sidebar_layout.addStretch()

        # ----------------------------------------------------------
        # SCROLLABLE SIDEBAR
        # ----------------------------------------------------------

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)

        sidebar_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        sidebar_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_scroll.setFixedWidth(220)

        sidebar_scroll.setWidget(sidebar_widget)

        return sidebar_scroll

    def _connect_navigation(self):
        for index, button in enumerate(self.nav_buttons):
            button.clicked.connect(
                lambda checked=False, i=index: self.switch_page(i)
            )

        self.switch_page(0)

    def plot_expts(self):
        if self.DATAPROCESSOR is None:
            QMessageBox.warning(self, "Error", "No data available yet!")
            return

        try:
            self.DATAPROCESSOR.read_data_from = self.CONTROLLER.current_omni_folder
        except Exception as e:
            self.log_message(f"Possibly deleted fodler, error message: {e}", "ERROR")

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
        while self.plots_page.plot_container_layout.count():
            item = self.plots_page.plot_container_layout.takeAt(0)
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


        # Transparent Qt background
        self.canvas_embedded.setStyleSheet("background: transparent;")

        # Add to layout
        self.plots_page.plot_container_layout.addWidget(self.canvas_embedded,1)

        # Draw after adding to layout
        self.canvas_embedded.draw_idle()

        self.switch_page(2)

    def closeEvent(self, event):

        # --------------------------------------------------
        # 0. Block MainWindow close while session is attached
        # --------------------------------------------------

        if (
            hasattr(self, "experiment_page")
            and self.experiment_page.dashboard is not None
        ):

            QMessageBox.warning(
                self,
                "Experiment Session Open",
                "An experiment session is still open.\n\n"
                "Go to the Experiment page and select "
                "'End Session' before closing MainWindow."
            )

            event.ignore()
            return

        # --------------------------------------------------
        # 1. CONTROL PANEL / HARDWARE CLEANUP
        # --------------------------------------------------

        if hasattr(self, "control_panel_page"):

            control_panel = (
                self.control_panel_page.control_panel
            )

            if not control_panel.prepare_shutdown():

                event.ignore()
                return

        # --------------------------------------------------
        # 2. Existing MainWindow cleanup
        # --------------------------------------------------

        self.reset_command()

        if hasattr(self, "dashboard_page"):
            self.dashboard_page.system_monitor.closeEvent(event)

        super().closeEvent(event)



# ========================================================== #
#                       MAIN                                 #
# ========================================================== #
# 1. Set App ID BEFORE creating QApplication instance
if sys.platform == "win32":
    import ctypes
    my_app_id = "crystalloop.controlcenter.1.0"
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_app_id)
    except Exception as e:
        print(f"Failed to set AppUserModelID: {e}")

# Resolve absolute path to avoid relative path resolution bugs
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # 2. Check path & set Application Icon globally
    if LOGO_PATH.exists():
        app_icon = QIcon(str(LOGO_PATH))
        app.setWindowIcon(app_icon)

    window = MainWindow()

    # 3. Explicitly apply icon directly to MainWindow instance
    if LOGO_PATH.exists():
        window.setWindowIcon(QIcon(str(LOGO_PATH)))


    window.show()
    app.exec()



# In[ ]:




