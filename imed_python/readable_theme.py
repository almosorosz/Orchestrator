"""Explicit colors for Qt controls that otherwise use native subcontrol text."""


def readable_controls(dark):
    surface, text, border, hover, active, active_text, focus, muted = (
        ("#111d31", "#e5e7eb", "#465975", "#162846", "#1d4ed8", "#ffffff", "#93c5fd", "#a5b4ca")
        if dark else
        ("#ffffff", "#243247", "#a4b3c6", "#edf3fc", "#dbeafe", "#1e3a8a", "#2563eb", "#52637a")
    )
    return f"""
        QTabBar::tab {{
            background: {surface};
            color: {text};
            border: 1px solid {border};
            border-radius: 5px;
            margin: 0 3px 4px 0;
            padding: 5px 8px;
            font-weight: 600;
        }}
        QTabBar::tab:!selected:hover {{ background: {hover}; }}
        QTabBar::tab:selected {{
            background: {active};
            color: {active_text};
            border-color: {focus};
        }}
        QTabBar::tab:selected:focus {{ border-bottom: 2px solid {focus}; }}
        QTabBar::tab:disabled {{ color: {muted}; background: {surface}; }}
        QAbstractItemView, QComboBox QAbstractItemView, QPlainTextEdit, QAbstractSpinBox {{
            background-color: {surface};
            color: {text};
            alternate-background-color: {hover};
            selection-background-color: {active};
            selection-color: {active_text};
        }}
        QHeaderView::section {{
            background: {hover};
            color: {text};
            padding: 5px;
            border: 1px solid {border};
        }}
        QComboBox, QMenu, QToolTip {{
            color: {text};
            background-color: {surface};
        }}
        QMenu::item:selected {{ background: {active}; color: {active_text}; }}
        QMenu::item:disabled, QAbstractSpinBox:disabled, QComboBox:disabled, QPushButton:disabled {{
            color: {muted};
        }}
        QToolButton {{
            color: {text};
            background: {surface};
        }}
        QToolButton:hover {{ background: {hover}; }}
    """
