"""Run the Bayesian optimization window independently of CrystalLoop."""
import argparse
import sys

from PySide6.QtWidgets import QApplication

from imed_python.readable_theme import readable_controls
from .window import BOWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matlab", nargs="?", help="Optional IMED MATLAB file")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(readable_controls(False))
    window = BOWindow()
    if args.matlab:
        window.open_matlab(args.matlab)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
