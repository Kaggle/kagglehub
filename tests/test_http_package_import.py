import os
import sys

import kagglehub
from kagglehub.cache import NOTEBOOKS_CACHE_SUBFOLDER, get_cached_archive_path
from kagglehub.handle import parse_package_handle
from tests.fixtures import BaseTestCase

from .server_stubs import notebook_output_download_stub as stub
from .server_stubs import serv
from .utils import create_test_cache

INVALID_ARCHIVE_PACKAGE_HANDLE = "invalid/invalid/invalid/invalid/invalid"
VERSIONED_PACKAGE_HANDLE = "dster/package-test/versions/1"
UNVERSIONED_PACKAGE_HANDLE = "dster/package-test"

EXPECTED_NOTEBOOK_SUBDIR = os.path.join(NOTEBOOKS_CACHE_SUBFOLDER, "dster", "package-test", "output", "versions", "1")


class TestHttpPackageImport(BaseTestCase):

    def tearDown(self) -> None:
        # Clear any imported packages from sys.modules.
        for name in list(sys.modules.keys()):
            if name.startswith("kagglehub_package"):
                del sys.modules[name]

    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_package_versioned_succeeds(self) -> None:
        with create_test_cache():
            package = kagglehub.package_import(VERSIONED_PACKAGE_HANDLE)

            self.assertIn("foo", dir(package))
            self.assertEqual("bar", package.foo())

            archive_path = get_cached_archive_path(parse_package_handle(VERSIONED_PACKAGE_HANDLE))
            self.assertFalse(os.path.exists(archive_path))

    def test_package_unversioned_succeeds(self) -> None:
        with create_test_cache():
            package = kagglehub.package_import(UNVERSIONED_PACKAGE_HANDLE)

            self.assertIn("foo", dir(package))
            self.assertEqual("baz", package.foo())

            archive_path = get_cached_archive_path(parse_package_handle(UNVERSIONED_PACKAGE_HANDLE))
            self.assertFalse(os.path.exists(archive_path))

    def test_notebook_download_bad_archive(self) -> None:
        with create_test_cache():
            with self.assertRaises(ValueError):
                kagglehub.package_import(INVALID_ARCHIVE_PACKAGE_HANDLE)
