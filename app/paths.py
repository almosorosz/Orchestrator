from pathlib import Path
import sys


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()

ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"

OMNI_DATA_DIR = DATA_DIR / "omni"
OMNI_EXPORT_DIR = DATA_DIR / "omni_export"
PUMP_CALIBRATION_DIR = DATA_DIR / "pump_calibration"

LOGO_PATH = ASSETS_DIR / "LOGO.png"
TEXT_PATH = ASSETS_DIR / "TEXT.png"
ICON_PATH = ASSETS_DIR / "ICON.png"

AS_PUMP_CALIBRATION_FILE = (
    PUMP_CALIBRATION_DIR / "expttsd1_AS.csv"
)

SOLUTION_PUMP_CALIBRATION_FILE = (
    PUMP_CALIBRATION_DIR / "expttsd1_SOLU.csv"
)

SOLVENT_PUMP_CALIBRATION_FILE = (
    PUMP_CALIBRATION_DIR / "expttsd1_SOLV.csv"
)

OMNI_HEADER_FILE = (
    OMNI_EXPORT_DIR / "headers_OMNI.csv"
)