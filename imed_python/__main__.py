"""Standalone PySide6 port of the IMED workflow."""
import argparse
import sys

from PySide6.QtWidgets import QApplication

from .simulation import Window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", help="CrystalLoop IMED case or MATLAB MAT case to open at startup")
    parser.add_argument("--simulate", action="store_true", help="Open the forward simulator directly")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    from .readable_theme import readable_controls

    app.setStyleSheet(readable_controls(False))
    if args.simulate:
        window = Window()
    else:
        from .control_panel import ControlPanel

        window = ControlPanel(Window)
    if args.case:
        window.open_case(args.case)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
