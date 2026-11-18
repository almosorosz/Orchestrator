import logging
from pathlib import Path
import tempfile
import unittest
from app.logging_config import set_output_log_root, OutputLogHandler


class OutputLogTests(unittest.TestCase):
    def test_log_appends_rotates_and_switches_output_without_keeping_file_open(self):
        logger = logging.getLogger('orchestrator')
        level = logger.level
        logger.setLevel(logging.INFO)
        try:
            with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
                first, second = Path(root) / 'manual', Path(root) / 'loop'
                set_output_log_root(first)
                logger.info('manual event')
                handler = next(h for h in logger.handlers if isinstance(h, OutputLogHandler))
                self.assertIsNone(handler.stream)
                self.assertIn('manual event', (first / 'application.log').read_text())
                handler.maxBytes = 100
                logger.warning('rotate ' + 'x' * 150)
                self.assertTrue((first / 'application.log.1').exists())
                set_output_log_root(second)
                logger.error('loop event')
                self.assertIn('loop event', (second / 'application.log').read_text())
                self.assertNotIn('loop event', (first / 'application.log').read_text())
        finally:
            for handler in list(logger.handlers):
                if isinstance(handler, OutputLogHandler):
                    logger.removeHandler(handler)
                    handler.close()
            logger.setLevel(level)
