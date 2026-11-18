import json

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)

from control_panel.control_panel import (
    ControlPanel,
    HARDWARE_TYPES,
)
from app.paths import ASSETS_DIR


class ControlPanelPage(QWidget):
    """
    Embeds the existing hardware Control Panel
    inside the application's page navigation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setObjectName("controlPanelPage")

        # --------------------------------------------------
        # CONFIGURATION
        # --------------------------------------------------

        default_configuration = json.loads(
            (ASSETS_DIR / "control_panel_defaults.json").read_text(encoding="utf-8")
        )
        self.devices = {}

        self.hardware_types = HARDWARE_TYPES

        self.default_devices = default_configuration["devices"]

        # --------------------------------------------------
        # HARDWARE
        # --------------------------------------------------

        self.hardware = {}

        # --------------------------------------------------
        # EMBED EXISTING CONTROL PANEL
        # --------------------------------------------------

        self.control_panel = ControlPanel(
            devices=self.devices,
            hardware_types=self.hardware_types,
            hardware=self.hardware,
            default_devices=self.default_devices,
            parent=self,
            default_configuration=default_configuration,
        )
        self.control_panel.apply_default_configuration()

        # --------------------------------------------------
        # LAYOUT
        # --------------------------------------------------

        layout = QVBoxLayout(self)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self.control_panel)
