from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
)

from PySide6.QtCore import Qt

class ExperimentPage(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.dashboard = None

        self._build_ui()

    # ==========================================================
    # BUILD UI
    # ==========================================================

    def _build_ui(self):

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(10)

        # ------------------------------------------------------
        # Header
        # ------------------------------------------------------

        header_layout = QHBoxLayout()

        self.title_label = QLabel("Experiment")
        self.title_label.setStyleSheet(
            "font-size: 20px; font-weight: bold;"
        )

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        self.end_session_button = QPushButton("End Session")
        self.end_session_button.setEnabled(False)

        header_layout.addWidget(self.end_session_button)

        self.main_layout.addLayout(header_layout)

        # ------------------------------------------------------
        # Placeholder
        # ------------------------------------------------------

        self.placeholder = QLabel(
            "No experiment session is currently open.\n\n"
            "Configure profiles in Control Panel and "
            "select Start Experiment."
        )

        self.placeholder.setWordWrap(True)

        self.main_layout.addWidget(self.placeholder)

        # ------------------------------------------------------
        # Scrollable dashboard container
        # ------------------------------------------------------
        
        self.dashboard_scroll = QScrollArea()
        self.dashboard_scroll.setWidgetResizable(True)
        self.dashboard_scroll.setFrameShape(
            QScrollArea.Shape.NoFrame
        )
        
        self.dashboard_container = QWidget()
        
        self.dashboard_layout = QVBoxLayout(
            self.dashboard_container
        )
        
        self.dashboard_layout.setContentsMargins(0, 0, 0, 0)
        self.dashboard_layout.setSpacing(0)
        
        self.dashboard_scroll.setWidget(
            self.dashboard_container
        )
        
        self.main_layout.addWidget(
            self.dashboard_scroll,
            1
        )

    # ==========================================================
    # ATTACH DASHBOARD
    # ==========================================================

    def attach_dashboard(self, dashboard):

        if self.dashboard is not None:
            raise RuntimeError(
                "An ExperimentDashboard is already attached."
            )

        self.dashboard = dashboard

        dashboard.setParent(self.dashboard_container)
        dashboard.setWindowFlags(Qt.WindowType.Widget)
        self.dashboard_layout.addWidget(dashboard)

        dashboard.show()

        self.placeholder.hide()
        self.end_session_button.setEnabled(True)

    # ==========================================================
    # DETACH DASHBOARD
    # ==========================================================

    def detach_dashboard(self):

        if self.dashboard is None:
            return

        dashboard = self.dashboard

        self.dashboard_layout.removeWidget(dashboard)

        dashboard.hide()
        dashboard.setParent(None)

        self.dashboard = None

        self.placeholder.show()
        self.end_session_button.setEnabled(False)