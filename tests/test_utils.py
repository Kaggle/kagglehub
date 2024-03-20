import unittest
from unittest.mock import MagicMock, patch

from integration_tests.test_model_upload import retry


class FunctionToTest:
    def __init__(self):
        self.attempt = 0

    def __call__(self, success_on_attempt: int) -> str:
        """A simple function that raises an exception until it reaches the successful attempt."""
        if self.attempt < success_on_attempt:
            self.attempt += 1
            value_error_message = "Test error"
            raise ValueError(value_error_message)
        return "Success"


class TestRetryDecorator(unittest.TestCase):
    def setUp(self) -> None:
        self.function_to_test = FunctionToTest()

    @patch("integration_tests.test_model_upload.time.sleep", autospec=True)
    @patch("integration_tests.test_model_upload.logger.info", autospec=True)
    def test_retry_success_before_limit(self, mock_logger_info: MagicMock, mock_sleep: MagicMock) -> None:
        decorated = retry(times=3, delay_seconds=1)(self.function_to_test)
        result = decorated(2)
        self.assertEqual(result, "Success")
        self.assertEqual(self.function_to_test.attempt, 2)
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertEqual(mock_logger_info.call_count, 2)

    @patch("integration_tests.test_model_upload.time.sleep", autospec=True)
    @patch("integration_tests.test_model_upload.logger.info", autospec=True)
    def test_retry_reaches_limit_raises_timeout(self, mock_logger_info: MagicMock, mock_sleep: MagicMock) -> None:
        decorated = retry(times=3, delay_seconds=2)(self.function_to_test)
        with self.assertRaises(TimeoutError):
            decorated(4)
        self.assertEqual(self.function_to_test.attempt, 3)
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertEqual(mock_logger_info.call_count, 2)


if __name__ == "__main__":
    unittest.main()
