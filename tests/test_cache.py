import os
from pathlib import Path

from kagglehub.cache import (
    MODELS_CACHE_SUBFOLDER,
    delete_from_cache,
    get_cached_archive_path,
    get_cached_path,
    load_from_cache,
    mark_as_complete,
    mark_as_incomplete,
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
        self.assertEqual(None, load_from_cache(TEST_MODEL_HANDLE))

    def test_load_from_cache_with_path_miss(self) -> None:
        self.assertEqual(None, load_from_cache(TEST_MODEL_HANDLE, TEST_FILEPATH))

    def test_load_from_cache_not_complete_miss(self) -> None:
        with create_test_cache():
            cache_path = get_cached_path(TEST_MODEL_HANDLE)

            # Should be a cache `miss` if the directory exists but not the marker file.
            os.makedirs(cache_path)

            self.assertEqual(None, load_from_cache(TEST_MODEL_HANDLE))

    def test_load_from_cache_with_path_not_complete_miss(self) -> None:
        with create_test_cache():
            cache_path = get_cached_path(TEST_MODEL_HANDLE)

            # Should be a cache `miss` if the directory and file exist but not the marker file.
            os.makedirs(cache_path)
            Path(os.path.join(cache_path, TEST_FILEPATH)).touch()  # Create file

            self.assertEqual(None, load_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH))

    def test_load_from_cache_with_complete_marker_no_files_miss(self) -> None:
        with create_test_cache():
            # Should be a cache `miss` if completion marker file exists but not the files themselves.
            mark_as_complete(TEST_MODEL_HANDLE)

            self.assertEqual(None, load_from_cache(TEST_MODEL_HANDLE))

    def test_load_from_cache_with_path_complete_marker_no_files_miss(self) -> None:
        with create_test_cache():
            # Should be a cache `miss` if completion marker file exists but not the file itself.
            mark_as_complete(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(None, load_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH))

    def test_cache_hit(self) -> None:
        with create_test_cache() as d:
            cache_path = get_cached_path(TEST_MODEL_HANDLE)
            os.makedirs(cache_path)
            mark_as_complete(TEST_MODEL_HANDLE)

            path = load_from_cache(TEST_MODEL_HANDLE)

            self.assertEqual(
                os.path.join(d, EXPECTED_MODEL_SUBDIR),
                path,
            )

    def test_cache_hit_with_path(self) -> None:
        with create_test_cache() as d:
            cache_path = get_cached_path(TEST_MODEL_HANDLE)

            os.makedirs(cache_path)
            Path(os.path.join(cache_path, TEST_FILEPATH)).touch()  # Create file
            mark_as_complete(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            path = load_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(
                os.path.join(d, EXPECTED_MODEL_SUBPATH),
                path,
            )

    def test_load_from_cache_invalid_handle(self) -> None:
        with self.assertRaises(ValueError):
            load_from_cache(InvalidResourceHandle())

    def test_model_archive_path(self) -> None:
        with create_test_cache() as d:
            archive_path = get_cached_archive_path(TEST_MODEL_HANDLE)

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

    def _download_test_model_to_cache(self) -> None:
        cache_path = get_cached_path(TEST_MODEL_HANDLE)
        model_variable_dir = os.path.join(cache_path, TEST_MODEL_VARIABLES_DIR_NAME)

        os.makedirs(model_variable_dir)
        Path(os.path.join(cache_path, TEST_FILEPATH)).touch()
        Path(os.path.join(model_variable_dir, TEST_MODEL_VARIABLES_FILE_NAME)).touch()

        mark_as_complete(TEST_MODEL_HANDLE)

    def _download_test_file_to_cache(self) -> None:
        cache_path = get_cached_path(TEST_MODEL_HANDLE)

        os.makedirs(cache_path)
        Path(os.path.join(cache_path, TEST_FILEPATH)).touch()

        mark_as_complete(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

    def test_delete_from_cache(self) -> None:
        with create_test_cache() as d:
            self._download_test_model_to_cache()

            deleted_path = delete_from_cache(TEST_MODEL_HANDLE)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBDIR), deleted_path)
            self.assertFalse(os.path.exists(get_cached_path(TEST_MODEL_HANDLE)))

    def test_delete_from_cache_with_path(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_to_cache()

            deleted_path = delete_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBPATH), deleted_path)
            self.assertFalse(os.path.exists(os.path.join(get_cached_path(TEST_MODEL_HANDLE), TEST_FILEPATH)))

    def test_delete_from_cache_without_files_without_complete_marker(self) -> None:
        with create_test_cache():
            deleted_path = delete_from_cache(TEST_MODEL_HANDLE)
            self.assertEqual(None, deleted_path)

    def test_delete_from_cache_without_files_without_complete_marker_with_path(self) -> None:
        with create_test_cache():
            deleted_path = delete_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH)
            self.assertEqual(None, deleted_path)

    def test_delete_from_cache_without_files_with_complete_marker(self) -> None:
        with create_test_cache():
            mark_as_complete(TEST_MODEL_HANDLE)

            deleted_path = delete_from_cache(TEST_MODEL_HANDLE)

            # Should not delete anything if only the marker file existed.
            self.assertEqual(None, deleted_path)

    def test_delete_from_cache_without_files_with_complete_marker_with_path(self) -> None:
        with create_test_cache():
            mark_as_complete(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            deleted_path = delete_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            # Should not delete anything if only the marker file existed.
            self.assertEqual(None, deleted_path)

    def test_delete_from_cache_with_files_without_complete_marker(self) -> None:
        with create_test_cache() as d:
            self._download_test_model_to_cache()
            mark_as_incomplete(TEST_MODEL_HANDLE)

            deleted_path = delete_from_cache(TEST_MODEL_HANDLE)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBDIR), deleted_path)
            self.assertFalse(os.path.exists(get_cached_path(TEST_MODEL_HANDLE)))

    def test_delete_from_cache_with_files_without_complete_marker_with_path(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_to_cache()
            mark_as_incomplete(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            deleted_path = delete_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBPATH), deleted_path)
            self.assertFalse(os.path.exists(os.path.join(get_cached_path(TEST_MODEL_HANDLE), TEST_FILEPATH)))
