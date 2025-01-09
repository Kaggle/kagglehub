import sys
import unittest

from requests import HTTPError

from kagglehub import utility_script_install

from .utils import assert_files, create_test_cache


class TestUtilityScriptInstall(unittest.TestCase):
    def test_download_utility_script_succeeds(self) -> None:
        with create_test_cache():
            response_path = utility_script_install("integrationtester/utility-script")
            self.response_path = response_path

            expected_files = ["utility_script.py"]
            self.assertIn(response_path, sys.path)
            assert_files(self, response_path, expected_files)

    def test_download_private_utility_script_succeeds(self) -> None:
        with create_test_cache():
            response_path = utility_script_install("integrationtester/private-utility-script")
            self.response_path = response_path

            self.assertIn(response_path, sys.path)
            expected_files = ["private_utility_script.py"]
            assert_files(self, response_path, expected_files)

    def test_download_non_utility_script_sys_path_not_updated(self) -> None:
        with create_test_cache():
            response_path = utility_script_install("alexisbcook/titanic-tutorial")
            self.response_path = response_path

            self.assertNotIn(response_path, sys.path)
            expected_files = ["submission.csv"]
            assert_files(self, response_path, expected_files)

    def test_download_non_existent_utility_script_fails(self) -> None:
        with create_test_cache(), self.assertRaises(HTTPError):
            utility_script_install("integrationtester/i-dont-exist")

    def tearDown(self) -> None:
        if hasattr(self, "response_path") and self.response_path in sys.path:
            sys.path.remove(self.response_path)
