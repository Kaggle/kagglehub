import os
from pathlib import Path
from tempfile import TemporaryDirectory

from kagglehub.cache import (
    MODELS_CACHE_SUBFOLDER,
    Cache,
)
from kagglehub.handle import ModelHandle
from tests.fixtures import BaseTestCase

from .utils import InvalidResourceHandle, create_test_cache

EXPECTED_MODEL_SUBDIR = os.path.join(
    MODELS_CACHE_SUBFOLDER, "google", "bert", "tensorFlow2", "answer-equivalence-bem", "2"
)

EXPECTED_MODEL_SUBPATH = os.path.join(
    MODELS_CACHE_SUBFOLDER,
    "google",
    "bert",
    "tensorFlow2",
    "answer-equivalence-bem",
    "2",
    "foo.txt",
)

TEST_MODEL_HANDLE = ModelHandle(
    owner="google",
    model="bert",
    framework="tensorFlow2",
    variation="answer-equivalence-bem",
    version=2,
)

TEST_FILEPATH = "foo.txt"
TEST_MODEL_VARIABLES_DIR_NAME = "variables"
TEST_MODEL_VARIABLES_FILE_NAME = "variables.txt"


class TestCache(BaseTestCase):
    def test_load_from_cache_miss(self) -> None:
        self.assertEqual(None, Cache().load_from_cache(TEST_MODEL_HANDLE))

    def test_load_from_cache_with_path_miss(self) -> None:
        self.assertEqual(None, Cache().load_from_cache(TEST_MODEL_HANDLE, TEST_FILEPATH))

    def test_load_from_cache_not_complete_miss(self) -> None:
        with create_test_cache():
            cache_path = Cache().get_path(TEST_MODEL_HANDLE)

            # Should be a cache `miss` if the directory exists but not the marker file.
            os.makedirs(cache_path)

            self.assertEqual(None, Cache().load_from_cache(TEST_MODEL_HANDLE))

    def test_load_from_cache_with_path_not_complete_miss(self) -> None:
        with create_test_cache():
            cache_path = Cache().get_path(TEST_MODEL_HANDLE)

            # Should be a cache `miss` if the directory and file exist but not the marker file.
            os.makedirs(cache_path)
            Path(os.path.join(cache_path, TEST_FILEPATH)).touch()  # Create file

            self.assertEqual(None, Cache().load_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH))

    def test_load_from_cache_with_complete_marker_no_files_miss(self) -> None:
        with create_test_cache():
            # Should be a cache `miss` if completion marker file exists but not the files themselves.
            Cache().mark_as_complete(TEST_MODEL_HANDLE)

            self.assertEqual(None, Cache().load_from_cache(TEST_MODEL_HANDLE))

    def test_load_from_cache_with_path_complete_marker_no_files_miss(self) -> None:
        with create_test_cache():
            # Should be a cache `miss` if completion marker file exists but not the file itself.
            Cache().mark_as_complete(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(None, Cache().load_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH))

    def test_cache_hit(self) -> None:
        with create_test_cache() as d:
            cache_path = Cache().get_path(TEST_MODEL_HANDLE)
            os.makedirs(cache_path)
            Cache().mark_as_complete(TEST_MODEL_HANDLE)

            path = Cache().load_from_cache(TEST_MODEL_HANDLE)

            self.assertEqual(
                os.path.join(d, EXPECTED_MODEL_SUBDIR),
                path,
            )

    def test_cache_hit_with_path(self) -> None:
        with create_test_cache() as d:
            cache_path = Cache().get_path(TEST_MODEL_HANDLE)

            os.makedirs(cache_path)
            Path(os.path.join(cache_path, TEST_FILEPATH)).touch()  # Create file
            Cache().mark_as_complete(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            path = Cache().load_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(
                os.path.join(d, EXPECTED_MODEL_SUBPATH),
                path,
            )

    def test_load_from_cache_invalid_handle(self) -> None:
        with self.assertRaises(ValueError):
            Cache().load_from_cache(InvalidResourceHandle())

    def test_model_archive_path(self) -> None:
        with create_test_cache() as d:
            archive_path = Cache().get_archive_path(TEST_MODEL_HANDLE)

            self.assertEqual(
                os.path.join(
                    d,
                    MODELS_CACHE_SUBFOLDER,
                    "google",
                    "bert",
                    "tensorFlow2",
                    "answer-equivalence-bem",
                    "2.archive",
                ),
                archive_path,
            )

    def test_cache_override_miss(self) -> None:
        with TemporaryDirectory() as override_dir:
            self.assertEqual(None, Cache(override_dir).load_from_cache(TEST_MODEL_HANDLE))

    def test_cache_override_dir_hit(self) -> None:
        with TemporaryDirectory() as override_dir:
            cache = Cache(override_dir)
            cache.mark_as_complete(TEST_MODEL_HANDLE)
            path = cache.load_from_cache(TEST_MODEL_HANDLE)

            self.assertEqual(
                override_dir,
                path,
            )

    def test_cache_override_hit_with_path(self) -> None:
        with TemporaryDirectory() as override_dir:
            cache = Cache(override_dir)

            Path(os.path.join(override_dir, TEST_FILEPATH)).touch()  # Create file
            cache.mark_as_complete(TEST_MODEL_HANDLE, path=TEST_FILEPATH)
            path = cache.load_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(
                os.path.join(override_dir, TEST_FILEPATH),
                path,
            )

    def _download_test_model_to_cache(self) -> None:
        cache_path = Cache().get_path(TEST_MODEL_HANDLE)
        model_variable_dir = os.path.join(cache_path, TEST_MODEL_VARIABLES_DIR_NAME)

        os.makedirs(model_variable_dir)
        Path(os.path.join(cache_path, TEST_FILEPATH)).touch()
        Path(os.path.join(model_variable_dir, TEST_MODEL_VARIABLES_FILE_NAME)).touch()

        Cache().mark_as_complete(TEST_MODEL_HANDLE)

    def _download_test_file_to_cache(self) -> None:
        cache_path = Cache().get_path(TEST_MODEL_HANDLE)

        os.makedirs(cache_path)
        Path(os.path.join(cache_path, TEST_FILEPATH)).touch()

        Cache().mark_as_complete(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

    def test_delete_from_cache(self) -> None:
        with create_test_cache() as d:
            self._download_test_model_to_cache()

            deleted_path = Cache().delete_from_cache(TEST_MODEL_HANDLE)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBDIR), deleted_path)
            self.assertFalse(os.path.exists(Cache().get_path(TEST_MODEL_HANDLE)))

    def test_delete_from_cache_with_path(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_to_cache()

            deleted_path = Cache().delete_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBPATH), deleted_path)
            self.assertFalse(os.path.exists(os.path.join(Cache().get_path(TEST_MODEL_HANDLE), TEST_FILEPATH)))

    def test_delete_from_cache_without_files_without_complete_marker(self) -> None:
        with create_test_cache():
            deleted_path = Cache().delete_from_cache(TEST_MODEL_HANDLE)
            self.assertEqual(None, deleted_path)

    def test_delete_from_cache_without_files_without_complete_marker_with_path(self) -> None:
        with create_test_cache():
            deleted_path = Cache().delete_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH)
            self.assertEqual(None, deleted_path)

    def test_delete_from_cache_without_files_with_complete_marker(self) -> None:
        with create_test_cache():
            Cache().mark_as_complete(TEST_MODEL_HANDLE)

            deleted_path = Cache().delete_from_cache(TEST_MODEL_HANDLE)

            # Should not delete anything if only the marker file existed.
            self.assertEqual(None, deleted_path)

    def test_delete_from_cache_without_files_with_complete_marker_with_path(self) -> None:
        with create_test_cache():
            Cache().mark_as_complete(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            deleted_path = Cache().delete_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            # Should not delete anything if only the marker file existed.
            self.assertEqual(None, deleted_path)

    def test_delete_from_cache_with_files_without_complete_marker(self) -> None:
        with create_test_cache() as d:
            self._download_test_model_to_cache()
            Cache().mark_as_incomplete(TEST_MODEL_HANDLE)

            deleted_path = Cache().delete_from_cache(TEST_MODEL_HANDLE)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBDIR), deleted_path)
            self.assertFalse(os.path.exists(Cache().get_path(TEST_MODEL_HANDLE)))

    def test_delete_from_cache_with_files_without_complete_marker_with_path(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_to_cache()
            Cache().mark_as_incomplete(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            deleted_path = Cache().delete_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBPATH), deleted_path)
            self.assertFalse(os.path.exists(os.path.join(Cache().get_path(TEST_MODEL_HANDLE), TEST_FILEPATH)))
