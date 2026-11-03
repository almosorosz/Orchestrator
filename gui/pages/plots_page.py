from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QScrollArea,
    QFrame,
)
from PySide6.QtCore import Qt


class PlotsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        title_label = QLabel("Data & Plots View")
        title_label.setStyleSheet(
            "font-size: 18px;"
            "font-weight: bold;"
            "color: #f8fafc;"
        )

        layout.addWidget(title_label)

        self.plot_scroll_area = QScrollArea()
        self.plot_scroll_area.setWidgetResizable(True)
        self.plot_scroll_area.setFrameShape(QFrame.NoFrame)

        self.plot_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        self.plot_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )

        self.plot_container = QWidget()

        self.plot_container_layout = QVBoxLayout(
            self.plot_container
        )
        self.plot_container_layout.setContentsMargins(
            5, 5, 5, 5
        )
        self.plot_container_layout.setSpacing(10)

        self.plot_placeholder_label = QLabel(
            "No experiment data plotted yet. "
            "Click 'Plot data' to load."
        )

        self.plot_placeholder_label.setStyleSheet(
            "color: #64748b;"
            "font-size: 13px;"
        )

        self.plot_container_layout.addWidget(
            self.plot_placeholder_label
        )

        self.plot_container_layout.addStretch()

        self.plot_scroll_area.setWidget(
            self.plot_container
        )

        layout.addWidget(
            self.plot_scroll_area,
            1,
        )