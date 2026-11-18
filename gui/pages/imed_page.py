from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from imed_python.control_panel import ControlPanel
from imed_python.simulation import Window


class IMEDPage(QWidget):
    """Host the IMED workflow in the main navigation stack."""

    bo_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("IMED")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        toolbar = QHBoxLayout()
        toolbar.addWidget(title, 1)
        bo = QPushButton("Bayesian optimization")
        bo.clicked.connect(self.bo_requested.emit)
        toolbar.addWidget(bo)
        layout.addLayout(toolbar)

        scroll = QScrollArea(self)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        scroll.setWidgetResizable(True)
        self.panel = ControlPanel(Window, parent=scroll)
        self.panel.setMaximumWidth(640)
        # Use the same primary action colors as Settings in both themes.
        self.panel.body.setStyleSheet("""
            QPushButton[imedPrimary="true"]:enabled {
                background: #1d4ed8;
                border: 1px solid #60a5fa;
                color: white;
                text-align: center;
                font-weight: 700;
            }
            QPushButton[imedPrimary="true"]:hover:enabled { background: #2563eb; }
            QPushButton[imedPrimary="true"]:pressed:enabled { background: #1e40af; }
            QPushButton[imedPrimary="true"]:focus:enabled { border-color: #bfdbfe; }
            QCheckBox { background: transparent; }
        """)
        scroll.setWidget(self.panel)
        layout.addWidget(scroll, 1)
