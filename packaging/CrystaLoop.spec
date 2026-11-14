# Build from the repository root: python -m PyInstaller packaging/CrystaLoop.spec
from pathlib import Path
from importlib.util import find_spec
import os
from PyInstaller.utils.hooks import collect_data_files

root = Path(SPECPATH).parent
datas = [(str(root / "assets"), "assets"),
         (str(root / "data" / "pump_calibration"), "data/pump_calibration"),
         (str(root / "imed_python" / "workflow.png"), "imed_python")]
datas += collect_data_files("pyDOE3")
datas += collect_data_files("pyvis")
hiddenimports = ["MAIN_WINDOW", "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_agg", "pyDOE3"]
for optional in ("nidaqmx", "sqlalchemy", "psycopg"):
    if find_spec(optional):
        hiddenimports.append(optional)
# Include the BO runtime together when both optional dependencies are installed.
bo_imports = ("torch", "sklearn")
include_bo = all(find_spec(module) is not None for module in bo_imports)
exclusions = ["PyQt5", "PyQt6", "PySide2", "tkinter", "pytest", "tensorflow", "keras",
              "torchvision", "jax", "jaxlib", "numba", "dask", "notebook"]
if not include_bo:
    exclusions.extend([*bo_imports, "sympy"])
a = Analysis([str(root / "launcher.py")], pathex=[str(root)], binaries=[], datas=datas,
             hiddenimports=hiddenimports, hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=exclusions, noarchive=False)

# The Qt hook can collect linker objects from the QML tree. They are not
# needed at runtime and their nested paths can exceed Windows' path limit.
def is_qml_linker_object(entry):
    path = entry[0].replace("\\", "/").lower()
    return path.startswith("pyside6/qml/") and path.endswith(".obj")

a.binaries = [entry for entry in a.binaries if not is_qml_linker_object(entry)]
a.datas = [entry for entry in a.datas if not is_qml_linker_object(entry)]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="CrystaLoop", debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=os.environ.get("CRYSTALOOP_BUILD_CONSOLE") == "1",
          icon=str(root / "assets" / "CrystaLoop.ico"),
          version=str(root / "packaging" / "windows_version.txt"))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="CrystaLoop")
