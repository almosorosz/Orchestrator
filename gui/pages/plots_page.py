from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QScrollArea,
    QFrame,
)
from PySide6.QtCore import Qt
from gui.campaign_browser import CampaignBrowser


class PlotsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.page_scroll_area = QScrollArea()
        self.page_scroll_area.setWidgetResizable(True)
        self.page_scroll_area.setFrameShape(QFrame.NoFrame)
        self.page_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.page_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.page_content = QWidget()
        # Preserve room for controls, tables and a useful plot viewport. Small
        # windows scroll the page instead of compressing/clipping its contents.
        self.page_content.setMinimumHeight(600)
        layout = QVBoxLayout(self.page_content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.page_scroll_area.setWidget(self.page_content)
        page_layout.addWidget(self.page_scroll_area)

        title_label = QLabel("Data & Plots View")
        title_label.setStyleSheet(
            "font-size: 15px;"
            "font-weight: bold;"
            "color: #f8fafc;"
        )

        layout.addWidget(title_label)
        self.browser = CampaignBrowser(self)
        layout.addWidget(self.browser, 1)
