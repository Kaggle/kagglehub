import os
from unittest import mock

import kagglehub
from kagglehub.cache import NOTEBOOKS_CACHE_SUBFOLDER, get_cached_archive_path
from kagglehub.exceptions import UserCancelledError
from kagglehub.handle import parse_package_handle
from tests.fixtures import BaseTestCase

from .server_stubs import notebook_output_download_stub as stub
from .server_stubs import serv
from .utils import clear_imported_kaggle_packages, create_test_cache, login, parameterized

INVALID_ARCHIVE_PACKAGE_HANDLE = "invalid/invalid/invalid/invalid/invalid"
VERSIONED_PACKAGE_HANDLE = "dster/package-test/versions/1"
UNVERSIONED_PACKAGE_HANDLE = "dster/package-test"

EXPECTED_NOTEBOOK_SUBDIR = os.path.join(NOTEBOOKS_CACHE_SUBFOLDER, "dster", "package-test", "output", "versions", "1")

YES_INPUTS = ["yes", "y", "YES", "Y", "YeS"]


class TestHttpPackageImport(BaseTestCase):

    def tearDown(self) -> None:
        clear_imported_kaggle_packages()

    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    @parameterized(*YES_INPUTS)
    def test_package_versioned_succeeds(self, input_value: str) -> None:
        with mock.patch("builtins.input", return_value=input_value) as mock_input:
            with create_test_cache():
                package = kagglehub.package_import(VERSIONED_PACKAGE_HANDLE)

                mock_input.assert_called_once()
                self.assertIn("foo", dir(package))
                self.assertEqual("bar", package.foo())

                archive_path = get_cached_archive_path(parse_package_handle(VERSIONED_PACKAGE_HANDLE))
                self.assertFalse(os.path.exists(archive_path))

    def test_package_versioned_bypass_confirmation_succeeds(self) -> None:
        with create_test_cache():
            package = kagglehub.package_import(VERSIONED_PACKAGE_HANDLE, bypass_confirmation=True)

            self.assertIn("foo", dir(package))
            self.assertEqual("bar", package.foo())

            archive_path = get_cached_archive_path(parse_package_handle(VERSIONED_PACKAGE_HANDLE))
            self.assertFalse(os.path.exists(archive_path))

    def test_package_versioned_user_owned_succeeds(self) -> None:
        login("dster", "some-key")

        with create_test_cache():
            package = kagglehub.package_import(VERSIONED_PACKAGE_HANDLE, bypass_confirmation=True)

            self.assertIn("foo", dir(package))
            self.assertEqual("bar", package.foo())

            archive_path = get_cached_archive_path(parse_package_handle(VERSIONED_PACKAGE_HANDLE))
            self.assertFalse(os.path.exists(archive_path))

    @parameterized(*YES_INPUTS)
    def test_package_unversioned_succeeds(self, input_value: str) -> None:
        with mock.patch("builtins.input", return_value=input_value) as mock_input:
            with create_test_cache():
                package = kagglehub.package_import(UNVERSIONED_PACKAGE_HANDLE)

                mock_input.assert_called_once()
                self.assertIn("foo", dir(package))
                self.assertEqual("baz", package.foo())

                archive_path = get_cached_archive_path(parse_package_handle(UNVERSIONED_PACKAGE_HANDLE))
                self.assertFalse(os.path.exists(archive_path))

    @parameterized("no", "NO", "n", "", "anything but yes")
    def test_package_versioned_aborts(self, input_value: str) -> None:
        with mock.patch("builtins.input", return_value=input_value) as mock_input:
            with create_test_cache():
                with self.assertRaises(UserCancelledError):
                    kagglehub.package_import(VERSIONED_PACKAGE_HANDLE)
                mock_input.assert_called_once()

    def test_notebook_download_bad_archive(self) -> None:
        with create_test_cache():
            with self.assertRaises(ValueError):
                kagglehub.package_import(INVALID_ARCHIVE_PACKAGE_HANDLE)
