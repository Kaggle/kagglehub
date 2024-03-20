import unittest
from unittest.mock import MagicMock, patch

from integration_tests.test_model_upload import retry


def function_to_test(success_on_attempt: int) -> str:
    """A simple test function that will raise an exception until it reaches the successful attempt."""
    if function_to_test.attempt < success_on_attempt:
        function_to_test.attempt += 1
        value_error_message = "Test error"
        raise ValueError(value_error_message)
    return "Success"


class TestRetryDecorator(unittest.TestCase):
    def setUp(self) -> None:
        function_to_test.attempt = 0

    @patch("integration_tests.test_model_upload.time.sleep", autospec=True)
    @patch("integration_tests.test_model_upload.logger.info", autospec=True)
    def test_retry_success_before_limit(self, mock_logger_info: MagicMock, mock_sleep: MagicMock) -> None:
        decorated = retry(times=3, delay_seconds=1)(function_to_test)
        result = decorated(2)
        self.assertEqual(result, "Success")
        self.assertEqual(function_to_test.attempt, 2)
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertEqual(mock_logger_info.call_count, 2)

    @patch("integration_tests.test_model_upload.time.sleep", autospec=True)
    @patch("integration_tests.test_model_upload.logger.info", autospec=True)
    def test_retry_reaches_limit_raises_timeout(self, mock_logger_info: MagicMock, mock_sleep: MagicMock) -> None:
        decorated = retry(times=3, delay_seconds=2)(function_to_test)
        with self.assertRaises(TimeoutError):
            decorated(4)
        self.assertEqual(function_to_test.attempt, 3)
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertEqual(mock_logger_info.call_count, 2)


if __name__ == "__main__":
    unittest.main()
