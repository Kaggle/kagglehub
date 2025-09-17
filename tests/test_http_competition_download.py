import os
from datetime import datetime, timezone

import requests

import kagglehub
from kagglehub.cache import COMPETITIONS_CACHE_SUBFOLDER, get_cached_archive_path
from kagglehub.handle import parse_competition_handle
from tests.fixtures import BaseTestCase

from .server_stubs import competition_download_stub as stub
from .server_stubs import serv
from .utils import AUTO_COMPRESSED_FILE_NAME, create_test_cache

INVALID_ARCHIVE_COMPETITION_HANDLE = "invalid/invalid"
COMPETITION_HANDLE = "titanic"
TEST_FILEPATH = "foo.txt"
TEST_CONTENTS = "foo"
AUTO_COMPRESSED_CONTENTS = """"shape","degrees","sides","color","date"
"square",360,4,"blue","2024-12-17"
"circle",360,,"red","2023-08-01"
"triangle",180,3,"green","2022-01-05"
"""

EXPECTED_COMPETITION_SUBDIR = os.path.join(COMPETITIONS_CACHE_SUBFOLDER, "titanic")
EXPECTED_COMPETITION_SUBPATH = os.path.join(
    COMPETITIONS_CACHE_SUBFOLDER,
    "titanic",
)


class TestHttpCompetitionDownload(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)
        os.environ["KAGGLE_USERNAME"] = "fakeUser"
        os.environ["KAGGLE_KEY"] = "fakeKaggleKey"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _download_competition_and_assert_downloaded(
        self,
        d: str,
        competition_handle: str,
        expected_subdir_or_subpath: str,
        expected_files: list[str] | None = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        if not expected_files:
            expected_files = ["foo.txt"]

        # Download the full competitions and ensure all files are there.
        competition_path = kagglehub.competition_download(competition_handle, **kwargs)
        archive_path = get_cached_archive_path(parse_competition_handle(competition_handle))

        self.assertEqual(os.path.join(d, expected_subdir_or_subpath), competition_path)
        self.assertEqual(sorted(expected_files), sorted(os.listdir(competition_path)))
        # Assert that the archive file has been deleted
        self.assertFalse(os.path.exists(archive_path))

    def _download_test_file_and_assert_downloaded(
        self,
        d: str,
        competition_handle: str,
        **kwargs,  # noqa: ANN003
    ) -> None:
        competition_path = kagglehub.competition_download(competition_handle, path=TEST_FILEPATH, **kwargs)

        self.assertEqual(os.path.join(d, EXPECTED_COMPETITION_SUBPATH, TEST_FILEPATH), competition_path)
        with open(competition_path) as competition_file:
            self.assertEqual(TEST_CONTENTS, competition_file.readline())

    def _download_test_file_and_assert_downloaded_auto_compressed(
        self,
        d: str,
        competition_handle: str,
        **kwargs,  # noqa: ANN003
    ) -> None:
        competition_path = kagglehub.competition_download(competition_handle, path=AUTO_COMPRESSED_FILE_NAME, **kwargs)
        self.assertEqual(os.path.join(d, EXPECTED_COMPETITION_SUBPATH, AUTO_COMPRESSED_FILE_NAME), competition_path)
        with open(competition_path) as competition_file:
            self.assertEqual(AUTO_COMPRESSED_CONTENTS, competition_file.read())

    def test_competition_download(self) -> None:
        with create_test_cache() as d:
            self._download_competition_and_assert_downloaded(d, COMPETITION_HANDLE, EXPECTED_COMPETITION_SUBDIR)

    def test_competition_targz_archive_download(self) -> None:
        with create_test_cache() as d:
            self._download_competition_and_assert_downloaded(
                d,
                stub.TARGZ_ARCHIVE_HANDLE,
                f"{COMPETITIONS_CACHE_SUBFOLDER}/{stub.TARGZ_ARCHIVE_HANDLE}",
                expected_files=[f"{i}.txt" for i in range(1, 51)],
            )

    def test_competition_download_bad_archive(self) -> None:
        with create_test_cache():
            with self.assertRaises(ValueError):
                kagglehub.competition_download(INVALID_ARCHIVE_COMPETITION_HANDLE)

    def test_competition_full_download_with_file_already_cached(self) -> None:
        with create_test_cache() as d:
            # Download a single file first
            kagglehub.competition_download(COMPETITION_HANDLE, path=TEST_FILEPATH)

            self._download_competition_and_assert_downloaded(d, COMPETITION_HANDLE, EXPECTED_COMPETITION_SUBDIR)

    def test_competition_download_with_force_download(self) -> None:
        with create_test_cache() as d:
            self._download_competition_and_assert_downloaded(
                d, COMPETITION_HANDLE, EXPECTED_COMPETITION_SUBDIR, force_download=True
            )

    def test_competition_download_ignored_cache_when_lastest_is_newer(self) -> None:
        with create_test_cache() as d:
            path = kagglehub.competition_download(COMPETITION_HANDLE)
            # force cached file to be out of date. We set it back to March 02 2000.
            test_date = 951955200
            os.utime(os.path.join(d, EXPECTED_COMPETITION_SUBDIR), (test_date, test_date))
            old_date = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)

            # Latest version is from March 02 2020.
            path = kagglehub.competition_download(COMPETITION_HANDLE)

            # New cache file is current day.
            new_date = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
            self.assertEqual(os.path.join(d, EXPECTED_COMPETITION_SUBDIR), path)
            self.assertGreater(new_date, old_date)

    def test_competition_download_with_path(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_and_assert_downloaded(d, COMPETITION_HANDLE)

    def test_competition_download_with_path_auto_compressed(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_and_assert_downloaded_auto_compressed(d, COMPETITION_HANDLE)


class TestHttpNoInternet(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["KAGGLE_USERNAME"] = "fakeUser"
        os.environ["KAGGLE_KEY"] = "fakeKaggleKey"

    def test_competition_download_already_cached_with_no_internet(self) -> None:
        with create_test_cache() as d:
            server = serv.start_server(stub.app)
            kagglehub.competition_download(COMPETITION_HANDLE)
            server.shutdown()

            path = kagglehub.competition_download(COMPETITION_HANDLE)

            self.assertEqual(os.path.join(d, EXPECTED_COMPETITION_SUBDIR), path)

    def test_competition_download_path_already_cached_with_no_internet(self) -> None:
        with create_test_cache() as d:
            server = serv.start_server(stub.app)
            path = kagglehub.competition_download(COMPETITION_HANDLE, path=TEST_FILEPATH)
            server.shutdown()

            path = kagglehub.competition_download(COMPETITION_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(os.path.join(d, EXPECTED_COMPETITION_SUBPATH, TEST_FILEPATH), path)

    def test_competition_download_already_cached_with_force_download_no_internet(self) -> None:
        with create_test_cache():
            server = serv.start_server(stub.app)
            kagglehub.competition_download(COMPETITION_HANDLE)
            server.shutdown()

            # No internet should throw an error.
            with self.assertRaises(requests.exceptions.ConnectionError):
                kagglehub.competition_download(COMPETITION_HANDLE, force_download=True)

    def test_competition_download_with_path_already_cached_with_force_download_no_internet(self) -> None:
        with create_test_cache():
            server = serv.start_server(stub.app)
            kagglehub.competition_download(COMPETITION_HANDLE, path=TEST_FILEPATH)
            server.shutdown()

            # No internet should throw an error.
            with self.assertRaises(requests.exceptions.ConnectionError):
                kagglehub.competition_download(COMPETITION_HANDLE, path=TEST_FILEPATH, force_download=True)
