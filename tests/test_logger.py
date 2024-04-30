import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kagglehub.auth import _capture_logger_output
from kagglehub.logger import (
    _CONSOLE_BLOCK_KEY,
    EXTRA_CONSOLE_BLOCK,
    EXTRA_FILE_BLOCK,
    _block_logrecord_factory,
    _configure_logger,
)


class TestLoggerConfigurations(unittest.TestCase):
    def test_default_behavior(self) -> None:
        with TemporaryDirectory() as f:
            log_path = Path(f) / "test-log"
            _configure_logger(log_path)
            logger = logging.getLogger(str(f))
            logger.setLevel(logging.DEBUG)
            with _capture_logger_output(logger) as output:
                logger.info("This is an info message")
                logger.error("This is an error message")
            self.assertEqual(output.getvalue(), "This is an info message\nThis is an error message\n")

    def test_console_filter_discards_logrecord(self) -> None:
        with TemporaryDirectory() as f:
            log_path = Path(f) / "test-log"
            _configure_logger(log_path)
            logger = logging.getLogger(str(f))
            logger.setLevel(logging.DEBUG)
            logger.addFilter(_block_logrecord_factory([_CONSOLE_BLOCK_KEY]))
            with _capture_logger_output(logger) as output:
                logger.info("This is an blocked message", extra={**EXTRA_CONSOLE_BLOCK})
                logger.info("This is not blocked message")
            self.assertEqual(output.getvalue(), "This is not blocked message\n")

    def test_kagglehub_console_filter_discards_logrecord(self) -> None:
        with TemporaryDirectory() as f:
            log_path = Path(f) / "test-log"
            _configure_logger(log_path)
            logger = logging.getLogger("kagglehub")
            with _capture_logger_output(logger) as output:
                logger.info("This is an blocked message", extra={**EXTRA_CONSOLE_BLOCK})
                logger.info("This is not blocked message")
            self.assertEqual(output.getvalue(), "This is not blocked message\n")

    def test_kagglehub_file_filter(self) -> None:
        with TemporaryDirectory() as f:
            log_path = Path(f) / "testasdfasf-log"
            _configure_logger(log_path)
            logger = logging.getLogger("kagglehub")
            logger.info("This is a blocked message", extra={**EXTRA_FILE_BLOCK})
            logger.info("This is not a blocked message")
            text = (log_path / "kagglehub.log").read_text()
            self.assertRegex(text, "^.*This is not a blocked message.*$")
            self.assertRegex(text, "^(?!.*This is a blocked message).*$")


if __name__ == "__main__":
    unittest.main()
