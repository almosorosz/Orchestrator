"""PBM choices shared by the IMED control panel and simulator."""
from PySide6.QtWidgets import QComboBox

from .solver import PBM_MODES


class SolverChoice(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        for mode, label in PBM_MODES.items():
            self.addItem("JAX PBM (Auto)" if mode == "jax" else label, (mode, "auto"))
            if mode == "jax":
                self.addItem("JAX PBM (CPU)", (mode, "cpu"))
                self.addItem("JAX PBM (GPU)", (mode, "gpu"))
        self.setToolTip("JAX Auto uses JAX’s default device. JAX CPU forces the CPU. "
                        "JAX GPU requires a working GPU backend. All three use the same PBM. "
                        "The first solve for an input shape includes compilation. "
                        "On Windows, GPU use requires running IMED in WSL2.")
        self.setWhatsThis(self.toolTip())

    def load_settings(self, settings):
        settings = settings if isinstance(settings, dict) else {}
        mode = settings.get("Numerics", "conservative")
        device = settings.get("JaxDevice", "auto") if mode == "jax" else "auto"
        choice = next((i for i in range(self.count()) if self.itemData(i) == (mode, device)), 0)
        self.setCurrentIndex(choice)

    def apply_settings(self, settings):
        settings = dict(settings) if isinstance(settings, dict) else {}
        mode, device = self.currentData()
        settings["Numerics"] = mode
        if mode == "jax" and device != "auto":
            settings["JaxDevice"] = device
        else:
            settings.pop("JaxDevice", None)
        return settings
