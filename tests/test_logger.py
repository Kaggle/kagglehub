import logging
import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from kagglehub.logger import (
    _CONSOLE_BLOCK_KEY,
    EXTRA_CONSOLE_BLOCK,
    EXTRA_FILE_BLOCK,
    KAGGLE_LOGGING_ROOT_DIR_ENV_VAR_NAME,
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
            with self.assertLogs(logger, level="INFO") as cm:
                logger.info("HIDE")
                logger.info("SHOW")
                self.assertEqual(cm.output, [f"INFO:{f}:HIDE", f"INFO:{f}:SHOW"])

    def test_console_filter_discards_logrecord(self) -> None:
        with TemporaryDirectory() as f:
            log_path = Path(f) / "test-log"
            _configure_logger(log_path)
            logger = logging.getLogger(str(f))
            logger.setLevel(logging.DEBUG)
            logger.addFilter(_block_logrecord_factory([_CONSOLE_BLOCK_KEY]))
            with self.assertLogs(logger, level="INFO") as cm:
                logger.info("HIDE", extra={**EXTRA_CONSOLE_BLOCK})
                logger.info("SHOW")
                self.assertEqual(cm.output, [f"INFO:{f}:SHOW"])

    def test_kagglehub_console_filter_discards_logrecord(self) -> None:
        with TemporaryDirectory() as f:
            log_path = Path(f) / "test-log"
            logger = logging.getLogger("kagglehub")
            stream = StringIO()
            with redirect_stdout(stream):
                # reconfigure logger, otherwise streamhandler doesnt use the modified stderr
                _configure_logger(log_path)
                logger.info("HIDE", extra={**EXTRA_CONSOLE_BLOCK})
                logger.info("SHOW")
                self.assertEqual(stream.getvalue(), "SHOW\n")

    def test_kagglehub_child_console_filter_discards_logrecord(self) -> None:
        with TemporaryDirectory() as f:
            log_path = Path(f) / "test-log"
            logger = logging.getLogger("kagglehub.models")
            stream = StringIO()
            with redirect_stdout(stream):
                # reconfigure logger, otherwise streamhandler doesnt use the modified stderr
                _configure_logger(log_path)
                logger.info("HIDE", extra={**EXTRA_CONSOLE_BLOCK})
                logger.info("SHOW")
                self.assertEqual(stream.getvalue(), "SHOW\n")

    def test_kagglehub_file_filter(self) -> None:
        with TemporaryDirectory() as f:
            log_path = Path(f) / "testasdfasf-log"
            _configure_logger(log_path)
            logger = logging.getLogger("kagglehub")
            logger.info("HIDE", extra={**EXTRA_FILE_BLOCK})
            logger.info("SHOW")
            text = (log_path / "kagglehub.log").read_text()
            self.assertRegex(text, "^.*SHOW.*$")
            self.assertRegex(text, "^(?!.*HIDE).*$")

    def test_log_env_variable(self) -> None:
        with TemporaryDirectory() as d:
            root_log_dir = Path(d) / "log"
            with mock.patch.dict(os.environ, {KAGGLE_LOGGING_ROOT_DIR_ENV_VAR_NAME: str(root_log_dir)}):
                _configure_logger()
                logger = logging.getLogger("kagglehub")
                logger.info("goose")
                self.assertTrue((Path(root_log_dir) / ".kaggle/logs/kagglehub.log").exists())


if __name__ == "__main__":
    unittest.main()
