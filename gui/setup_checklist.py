"""Clickable, wrapping setup progress shared by Settings, LOOP and IMED."""
from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from gui.flow_layout import FlowLayout


@dataclass(frozen=True)
class SetupItem:
    key: str
    title: str
    ready: bool
    detail: str
    required: bool = True


class SetupChecklist(QFrame):
    activated = Signal(str)

    def __init__(self, title="Setup checklist", parent=None):
        super().__init__(parent)
        self.items = []
        self._selected_key = None
        self.buttons = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        row = QHBoxLayout()
        self.heading = QLabel(title)
        row.addWidget(self.heading, 1)
        self.next_button = QPushButton("Review next item")
        self.next_button.clicked.connect(self.review_next)
        row.addWidget(self.next_button)
        layout.addLayout(row)
        self.flow = FlowLayout(spacing=5)
        layout.addLayout(self.flow)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)

    def set_items(self, items):
        self.items = items
        keys = {item.key for item in items}
        for key in list(self.buttons):
            if key not in keys:
                button = self.buttons.pop(key)
                for index in range(self.flow.count()):
                    if self.flow.itemAt(index).widget() is button:
                        self.flow.takeAt(index)
                        break
                button.deleteLater()
        for item in items:
            if item.key not in self.buttons:
                button = QPushButton(self)
                button.clicked.connect(lambda checked=False, key=item.key: self.activate(key))
                self.buttons[item.key] = button
                self.flow.addWidget(button)
            button = self.buttons[item.key]
            state = "ready" if item.ready else "missing" if item.required else "optional"
            prefix = "OK" if item.ready else "!" if item.required else "Optional:"
            button.setText(f"{prefix} {item.title}")
            button.setToolTip(item.detail)
            button.setAccessibleName(f"{item.title}: {state}. {item.detail}")
            button.setProperty("ready", item.ready)
            color, background = {"ready": ("#ffffff", "#15803d"),
                                 "missing": ("#ffffff", "#854d0e"),
                                 "optional": ("#ffffff", "#475569")}[state]
            button.setStyleSheet(f"QPushButton {{ background: {background}; color: {color}; border: 1px solid {background}; "
                                "border-radius: 5px; padding: 5px 9px; text-align: left; }")
        pending = [item for item in items if item.required and not item.ready]
        self.next_button.setEnabled(bool(pending))
        self.heading.setText(f"Setup checklist · {len(pending)} required item(s) need attention" if pending
                             else "Setup checklist · Required items ready")
        selected = next((item for item in items if item.key == self._selected_key), None)
        self.detail.setText(selected.detail if selected else f"Next: {pending[0].detail}" if pending
                            else "All required items are ready. Click an item to review its values.")
        self.flow.invalidate()
        self.updateGeometry()

    def activate(self, key):
        item = next((item for item in self.items if item.key == key), None)
        if item:
            self._selected_key = key
            self.detail.setText(item.detail)
            self.activated.emit(key)

    def review_next(self):
        pending = next((item for item in self.items if item.required and not item.ready), None)
        if pending:
            self.activate(pending.key)
