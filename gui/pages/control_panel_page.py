from copy import deepcopy

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)

from control_panel.control_panel import (
    ControlPanel,
    HARDWARE_TYPES,
    DEFAULT_DEVICES,
    create_hardware,
)


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

        self.devices = deepcopy(DEFAULT_DEVICES)

        self.hardware_types = HARDWARE_TYPES

        self.default_devices = DEFAULT_DEVICES

        # --------------------------------------------------
        # HARDWARE
        # --------------------------------------------------

        self.hardware = create_hardware(
            self.devices,
            self.hardware_types,
        )

        # --------------------------------------------------
        # EMBED EXISTING CONTROL PANEL
        # --------------------------------------------------

        self.control_panel = ControlPanel(
            devices=self.devices,
            hardware_types=self.hardware_types,
            hardware=self.hardware,
            default_devices=self.default_devices,
            parent=self,
        )

        # --------------------------------------------------
        # LAYOUT
        # --------------------------------------------------

        layout = QVBoxLayout(self)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self.control_panel)