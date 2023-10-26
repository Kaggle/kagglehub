import os
import unittest
from pathlib import Path

from kagglehub.cache import MODELS_CACHE_SUBFOLDER, get_cached_path, load_from_cache, mark_as_complete
from kagglehub.handle import ModelHandle

from .utils import create_test_cache

TEST_MODEL_HANDLE = ModelHandle(
    owner="google",
    model="bert",
    framework="tensorFlow2",
    variation="answer-equivalence-bem",
    version=2,
)

TEST_FILEPATH = "foo.txt"


class TestCache(unittest.TestCase):
    def test_load_from_cache_miss(self):
        ModelHandle(
            owner="google",
            model="bert",
            framework="tensorFlow2",
            variation="answer-equivalence-bem",
            version=2,
        )
        self.assertEqual(None, load_from_cache(TEST_MODEL_HANDLE))

    def test_load_from_cache_with_path_miss(self):
        self.assertEqual(None, load_from_cache(TEST_MODEL_HANDLE, TEST_FILEPATH))

    def test_load_from_cache_not_complete_miss(self):
        with create_test_cache():
            cache_path = get_cached_path(TEST_MODEL_HANDLE)

            # Should be a cache `miss` if the directory exists but not the marker file.
            os.makedirs(cache_path)

            self.assertEqual(None, load_from_cache(TEST_MODEL_HANDLE))

    def test_load_from_cache_with_path_not_complete_miss(self):
        with create_test_cache():
            cache_path = get_cached_path(TEST_MODEL_HANDLE)

            # Should be a cache `miss` if the directory exists but not the marker file.
            os.makedirs(cache_path)
            Path(os.path.join(cache_path, TEST_FILEPATH)).touch()  # Create file

            self.assertEqual(None, load_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH))

    def test_load_from_cache_with_complete_marker_no_files_miss(self):
        with create_test_cache():
            get_cached_path(TEST_MODEL_HANDLE)

            # Should be a cache `miss` if completion marker file exists but not the files themselves.
            mark_as_complete(TEST_MODEL_HANDLE)

            self.assertEqual(None, load_from_cache(TEST_MODEL_HANDLE))

    def test_load_from_cache_with_path_complete_marker_no_files_miss(self):
        with create_test_cache():
            get_cached_path(TEST_MODEL_HANDLE)

            # Should be a cache `miss` if completion marker file exists but not the file itself.
            mark_as_complete(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(None, load_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH))

    def test_cache_hit(self):
        with create_test_cache() as d:
            cache_path = get_cached_path(TEST_MODEL_HANDLE)
            os.makedirs(cache_path)
            mark_as_complete(TEST_MODEL_HANDLE)

            path = load_from_cache(TEST_MODEL_HANDLE)

            self.assertEqual(
                os.path.join(d, MODELS_CACHE_SUBFOLDER, "google", "bert", "tensorFlow2", "answer-equivalence-bem", "2"),
                path,
            )

    def test_cache_hit_with_path(self):
        with create_test_cache() as d:
            cache_path = get_cached_path(TEST_MODEL_HANDLE)

            os.makedirs(cache_path)
            Path(os.path.join(cache_path, TEST_FILEPATH)).touch()  # Create file
            mark_as_complete(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            path = load_from_cache(TEST_MODEL_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(
                os.path.join(
                    d,
                    MODELS_CACHE_SUBFOLDER,
                    "google",
                    "bert",
                    "tensorFlow2",
                    "answer-equivalence-bem",
                    "2",
                    "foo.txt",
                ),
                path,
            )

    def test_load_from_cache_invalid_handle(self):
        with self.assertRaises(ValueError):
            load_from_cache("invalid_handle")
