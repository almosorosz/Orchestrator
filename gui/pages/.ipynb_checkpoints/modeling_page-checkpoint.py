from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
)
class ModelingPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Modeling")
        title.setStyleSheet(
            "font-size: 22px;"
            "font-weight: 700;"
        )

        description = QLabel(
            "Process modeling module under construction."
        )

        description.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addStretch()