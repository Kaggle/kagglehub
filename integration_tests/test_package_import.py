import sys
import unittest

from kagglehub import package_import

from .utils import create_test_cache, unauthenticated

UNVERSIONED_HANDLE = "dster/package-test"
VERSIONED_HANDLE = "dster/package-test/versions/1"


class TestPackageImport(unittest.TestCase):

    def tearDown(self) -> None:
        # Clear any imported packages from sys.modules.
        for name in list(sys.modules.keys()):
            if name.startswith("kagglehub_package"):
                del sys.modules[name]

    def test_package_versioned_succeeds(self) -> None:
        with create_test_cache():
            package = package_import(VERSIONED_HANDLE)

            self.assertIn("foo", dir(package))
            self.assertEqual("bar", package.foo())

    def test_package_unversioned_succeeds(self) -> None:
        with create_test_cache():
            package = package_import(UNVERSIONED_HANDLE)

            self.assertIn("foo", dir(package))
            self.assertEqual("baz", package.foo())

    def test_download_private_package_succeeds(self) -> None:
        with create_test_cache():
            package = package_import("integrationtester/kagglehub-test-private-package")

            self.assertIn("foo", dir(package))
            self.assertEqual("bar", package.foo())

    def test_public_package_with_unauthenticated_succeeds(self) -> None:
        with create_test_cache():
            with unauthenticated():
                package = package_import(UNVERSIONED_HANDLE)

                self.assertIn("foo", dir(package))
                self.assertEqual("baz", package.foo())
