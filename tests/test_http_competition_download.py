import os
from typing import Optional

import kagglehub
from kagglehub.cache import COMPETITIONS_CACHE_SUBFOLDER, get_cached_archive_path
from kagglehub.handle import parse_competition_handle
from tests.fixtures import BaseTestCase

from .server_stubs import competition_download_stub as stub
from .server_stubs import serv
from .utils import create_test_cache

INVALID_ARCHIVE_COMPETITION_HANDLE = "invalid/invalid"
COMPETITION_HANDLE = "titanic"
COMPETITION_2_HANDLE = "titanic"
TEST_FILEPATH = "foo.txt"
TEST_CONTENTS = "foo"

EXPECTED_COMPETITION_SUBDIR = os.path.join(COMPETITIONS_CACHE_SUBFOLDER, "titanic")
EXPECTED_COMPETITION_SUBPATH = os.path.join(
    COMPETITIONS_CACHE_SUBFOLDER,
    "titanic",
    TEST_FILEPATH,
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
        expected_files: Optional[list[str]] = None,
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

        self.assertEqual(os.path.join(d, EXPECTED_COMPETITION_SUBPATH), competition_path)
        with open(competition_path) as model_file:
            self.assertEqual(TEST_CONTENTS, model_file.readline())

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

    def test_competition_download_with_path(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_and_assert_downloaded(d, COMPETITION_HANDLE)
