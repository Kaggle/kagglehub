import os
from unittest import mock

import requests

import kagglehub
from kagglehub.config import DISABLE_KAGGLE_CACHE_ENV_VAR_NAME
from kagglehub.env import KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME
from tests.fixtures import BaseTestCase
from tests.utils import clear_imported_kaggle_packages

from .server_stubs import jwt_stub as stub
from .server_stubs import serv

INVALID_ARCHIVE_PACKAGE_HANDLE = "invalid/invalid/invalid/invalid/invalid"
VERSIONED_PACKAGE_HANDLE = "alexisbcook/test-package/versions/1"
UNVERSIONED_PACKAGE_HANDLE = "alexisbcook/test-package"


# Test cases for package_import and get_packet_asset_path.
class TestKaggleCachePackageImport(BaseTestCase):

    def tearDown(self) -> None:
        clear_imported_kaggle_packages()

    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app, KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME, "http://localhost:7778")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_unversioned_package_import(self) -> None:
        with stub.create_env():
            package = kagglehub.package_import(UNVERSIONED_PACKAGE_HANDLE)
            self.assertEqual("kaggle", package.foo())
            self.assertEqual("abcd", package.bar())

    def test_versioned_package_import(self) -> None:
        with stub.create_env():
            package = kagglehub.package_import(VERSIONED_PACKAGE_HANDLE)
            self.assertEqual("kaggle", package.foo())
            self.assertFalse(hasattr(package, "bar"))

    def test_kaggle_resolver_skipped(self) -> None:
        with mock.patch.dict(os.environ, {DISABLE_KAGGLE_CACHE_ENV_VAR_NAME: "true"}):
            with stub.create_env():
                # Assert that a ConnectionError is set (uses HTTP server which is not set)
                with self.assertRaises(requests.exceptions.ConnectionError):
                    kagglehub.package_import(UNVERSIONED_PACKAGE_HANDLE)

    def test_versioned_package_import_bad_handle_raises(self) -> None:
        with self.assertRaises(ValueError):
            kagglehub.package_import("bad handle")

    def test_versioned_package_import_returns_same(self) -> None:
        with stub.create_env():
            # Importing the same package a second time returns the same exact package.
            package = kagglehub.package_import(VERSIONED_PACKAGE_HANDLE)
            package2 = kagglehub.package_import(VERSIONED_PACKAGE_HANDLE)
            self.assertEqual(package, package2)

    def test_versioned_package_import_force_download_returns_different(self) -> None:
        with stub.create_env():
            # Re-importing with force_download re-installs anew.
            package = kagglehub.package_import(VERSIONED_PACKAGE_HANDLE)
            package_forced = kagglehub.package_import(VERSIONED_PACKAGE_HANDLE, force_download=True)
            self.assertNotEqual(package, package_forced)

    def test_versioned_package_import_with_force_download_explicitly_false(self) -> None:
        with stub.create_env():
            package = kagglehub.package_import(VERSIONED_PACKAGE_HANDLE, force_download=False)
            self.assertEqual("kaggle", package.foo())
            self.assertFalse(hasattr(package, "bar"))
