from PySide6.QtWidgets   import QWidget, QTextEdit, QLabel, QVBoxLayout, QHBoxLayout
from PySide6.QtCore      import Qt
from datetime          import datetime

class MetricCard(QWidget):
    def __init__(self, title, value="0", icon="⚡"):
        super().__init__()

        self.setObjectName("metricCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        title_row = QHBoxLayout()

        self.icon_label = QLabel(icon)
        self.icon_label.setStyleSheet(
            "font-size:18px;"
        )

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            "color:#94a3b8;"
            "font-size:12px;"
            "font-weight:600;"
        )

        title_row.addWidget(self.icon_label)
        title_row.addWidget(self.title_label)
        title_row.addStretch()

        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(
            "font-size:16px;"
            "font-weight:700;"
            "color:white;"
        )

        self.status_label = QLabel("Normal")
        self.status_label.setStyleSheet(
            "color:#22c55e;"
            "font-size:12px;"
        )

        layout.addLayout(title_row)
        layout.addWidget(self.value_label)
        layout.addWidget(self.status_label)


    def update_value(self,value,status="Normal",color="#22c55e"):

        self.value_label.setText(str(value))

        self.status_label.setText(status)

        self.status_label.setStyleSheet(
            f"color:{color}; font-size:12px;"
        )

class LogViewer(QTextEdit):

    def __init__(self):
        super().__init__()

        self.setReadOnly(True)

        self.setStyleSheet("""
        QTextEdit {
            background:#050b14;
            color:#dbeafe;
            font-family:Consolas;
            font-size:12px;
            border-radius:10px;
        }
        """)


    def add_log(self,message,level="INFO"):

        colors = {
            "INFO":"#60a5fa",
            "WARNING":"#facc15",
            "ERROR":"#ef4444",
            "SUCCESS":"#22c55e"
        }

        color = colors.get(level,"white")

        timestamp=datetime.now().strftime("%H:%M:%S")

        html=f"""
        <span style='color:#64748b'>
        [{timestamp}]
        </span>

        <span style='color:{color};font-weight:bold'>
        [{level}]
        </span>

        {message}
        """

        self.append(html)

class StatusBadge(QLabel):

    def __init__(self,text="Idle"):
        super().__init__(text)

        self.setAlignment(Qt.AlignCenter)

        self.setFixedHeight(26)

        self.set_status(text)


    def set_status(self,status):

        colors={
            "Running":"#22c55e",
            "Inactive":"#64748b",
            "Stopped":"#ef4444"
        }

        self.setText(status)

        self.setStyleSheet(f"""
        QLabel {{
            background:{colors.get(status,'#64748b')};
            color:white;
            border-radius:13px;
            padding:4px 12px;
            font-weight:700;
        }}
        """)

class LogViewer(QTextEdit):

    def __init__(self):
        super().__init__()

        self.setReadOnly(True)

        self.setStyleSheet("""
        QTextEdit {
            background:#081321;
            border:1px solid #16263d;
            border-radius:10px;
            padding:8px;
            color:#dbe5ef;
            font-family:Consolas;
            font-size:12px;
        }
        """)


    def add_log(self, message, level="INFO"):

        colors = {
            "INFO": "#60a5fa",
            "WARNING": "#facc15",
            "ERROR": "#ef4444",
            "SUCCESS": "#22c55e"
        }

        color = colors.get(level, "white")

        timestamp = datetime.now().strftime("%H:%M:%S")

        self.append(
            f"""
            <span style="color:#64748b">
            [{timestamp}]
            </span>

            <span style="color:{color};font-weight:bold">
            [{level}]
            </span>

            {message}
            """
        )