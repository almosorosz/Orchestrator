"""Small wrapping layout for toolbars that must fit narrow application pages."""
from PySide6.QtCore import Qt, QRect, QSize
from PySide6.QtWidgets import QLayout, QSpacerItem


class FlowLayout(QLayout):
    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self.items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        return self.items[index] if 0 <= index < len(self.items) else None

    def takeAt(self, index):
        return self.items.pop(index) if 0 <= index < len(self.items) else None

    def addStretch(self, stretch=0):
        pass  # Wrapping rows pack controls to the left.

    def addSpacing(self, spacing):
        self.addItem(QSpacerItem(spacing, 0))

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, width, 0), False)

    def minimumSize(self):
        size = QSize()
        for item in self.items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def sizeHint(self):
        return self.minimumSize()

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, True)

    def _arrange(self, rect, apply):
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, height = area.x(), area.y(), 0
        for item in self.items:
            if item.isEmpty():
                continue
            size = item.sizeHint()
            width = min(size.width(), max(1, area.width()))
            if height and x + width > area.right() + 1:
                x, y, height = area.x(), y + height + self.spacing(), 0
            if apply:
                item.setGeometry(QRect(x, y, width, size.height()))
            x += width + self.spacing()
            height = max(height, size.height())
        return y + height - rect.y() + margins.bottom()
