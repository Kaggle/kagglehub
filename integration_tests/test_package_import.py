import importlib
import subprocess
import sys
import unittest
from unittest import mock

from kagglehub import package_import
from kagglehub.exceptions import UserCancelledError

from .utils import create_test_cache, parameterized, unauthenticated

UNVERSIONED_HANDLE = "dster/package-test"
VERSIONED_HANDLE = "dster/package-test/versions/1"

YES_INPUTS = ["yes", "y", "YES", "Y", "YeS"]


class TestPackageImport(unittest.TestCase):

    def tearDown(self) -> None:
        names = [name for name in sys.modules if name.startswith("kagglehub_package")]
        for name in names:
            del sys.modules[name]

    @parameterized(*YES_INPUTS)
    def test_package_versioned_succeeds(self, input_value: str) -> None:
        with mock.patch("builtins.input", return_value=input_value) as mock_input:
            with create_test_cache():
                package = package_import(VERSIONED_HANDLE)

                mock_input.assert_called_once()
                self.assertIn("foo", dir(package))
                self.assertEqual("bar", package.foo())

    def test_package_versioned_bypass_confirmation_succeeds(self) -> None:
        with create_test_cache():
            package = package_import(VERSIONED_HANDLE, bypass_confirmation=True)

            self.assertIn("foo", dir(package))
            self.assertEqual("bar", package.foo())

    @parameterized(*YES_INPUTS)
    def test_package_unversioned_succeeds(self, input_value: str) -> None:
        with mock.patch("builtins.input", return_value=input_value) as mock_input:
            with create_test_cache():
                package = package_import(UNVERSIONED_HANDLE)

                mock_input.assert_called_once()
                self.assertIn("Model", dir(package))
                model = package.Model()
                self.assertEqual("this is a test package", model.foo())

    def test_download_private_package_succeeds(self) -> None:
        # Don't require confirmation since we're running as integrationtester.
        with create_test_cache():
            package = package_import("integrationtester/kagglehub-test-private-package")

            self.assertIn("foo", dir(package))
            self.assertEqual("bar", package.foo())

    @parameterized(*YES_INPUTS)
    def test_public_package_with_unauthenticated_succeeds(self, input_value: str) -> None:
        with mock.patch("builtins.input", return_value=input_value) as mock_input:
            with create_test_cache():
                with unauthenticated():
                    package = package_import(UNVERSIONED_HANDLE)

                    mock_input.assert_called_once()
                    self.assertIn("Model", dir(package))
                    model = package.Model()
                    self.assertEqual("this is a test package", model.foo())

    @parameterized("no", "NO", "n", "", "anything but yes")
    def test_package_versioned_aborts(self, input_value: str) -> None:
        with mock.patch("builtins.input", return_value=input_value) as mock_input:
            with create_test_cache():
                with self.assertRaises(UserCancelledError):
                    package_import(VERSIONED_HANDLE)
                mock_input.assert_called_once()

    def test_package_with_dependencies_succeeds(self) -> None:
        with create_test_cache():
            # Make sure pyjokes is not already installed, so we test that the package installs it
            self.assertIsNone(importlib.util.find_spec("pyjokes"))

            package = package_import("dster/joke-teller", bypass_confirmation=True)

            self.assertTrue(len(package.get_joke()) > 0)

            # Uninstall pyjokes now
            del sys.modules["pyjokes"]
            importlib.invalidate_caches()
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "pyjokes", "-y"])  # noqa: S603
            self.assertIsNone(importlib.util.find_spec("pyjokes"))
