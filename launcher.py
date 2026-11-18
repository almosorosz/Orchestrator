"""Frozen entry point with startup diagnostics before application imports."""
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import sys
import traceback


def main():
    multiprocessing.freeze_support()
    try:
        runpy.run_module("MAIN_WINDOW", run_name="__main__")
    except Exception:
        error = traceback.format_exc()
        if "--smoke-test" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--smoke-test") + 1])
            output.write_text(json.dumps({"passed": False, "error": error}, indent=2), encoding="utf-8")
        else:
            root = Path(os.environ.get("CRYSTALOOP_DATA_DIR", str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "CrystaLoop" / "data")))
            report = root / "runtime" / "STARTUP_FAILED.txt"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(error, encoding="utf-8")
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, f"CrystaLoop could not start.\n\nDetails were saved to:\n{report}", "CrystaLoop startup error", 0x10)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
