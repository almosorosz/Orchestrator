"""Application logger with Qt delivery; signals safely cross worker threads."""
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from app.paths import DATA_DIR
from PySide6.QtCore import QObject, Signal


class LogBridge(QObject):
    message = Signal(str, str)


class QtLogHandler(logging.Handler):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    def emit(self, record):
        # Full tracebacks remain available to file/console logging handlers.
        message = record.getMessage() if record.name == "orchestrator.app" else f"{record.name}: {record.getMessage()}"
        self.bridge.message.emit(message, getattr(record, "gui_level", record.levelname))


class OutputLogHandler(RotatingFileHandler):
    """Release the output file between messages for synchronized folders."""
    def emit(self, record):
        try:
            super().emit(record)
        finally:
            if self.stream is not None:
                try:
                    self.stream.close()
                except OSError:
                    self.handleError(record)
                finally:
                    self.stream = None


def detach_output_logging_if_idle():
    logger = logging.getLogger("orchestrator")
    if not any(isinstance(h, QtLogHandler) for h in logger.handlers):
        for handler in list(logger.handlers):
            if isinstance(handler, OutputLogHandler):
                logger.removeHandler(handler)
                handler.close()


def set_output_log_root(folder):
    logger = logging.getLogger("orchestrator")
    try:
        destination = Path(folder).expanduser().resolve() / "application.log"
        existing = [h for h in logger.handlers if isinstance(h, OutputLogHandler)]
        if any(Path(h.baseFilename) == destination for h in existing):
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        handler = OutputLogHandler(destination, maxBytes=5_000_000, backupCount=3, encoding="utf-8", delay=True)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        for old in existing:
            logger.removeHandler(old)
            old.close()
        logger.addHandler(handler)
        logger.info("Application log output: %s", destination)
    except OSError:
        logger.exception("Cannot create application log in output folder %s", folder)


def attach_gui_logging(parent, receiver):
    bridge = LogBridge(parent)
    bridge.message.connect(receiver)
    handler = QtLogHandler(bridge)
    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.INFO)
    if not any(isinstance(item, RotatingFileHandler) for item in logger.handlers):
        directory = DATA_DIR / "runtime"
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(directory / "application.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(file_handler)
    logger.addHandler(handler)
    return bridge, handler
