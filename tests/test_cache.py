import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from kagglehub.cache import MODELS_CACHE_SUBFOLDER, get_cached_path, load_from_cache
from kagglehub.config import CACHE_FOLDER_ENV_VAR_NAME
from kagglehub.handle import ModelHandle

TEST_MODEL_HANDLE = ModelHandle(
    owner="google",
    model="bert",
    framework="tensorFlow2",
    variation="answer-equivalence-bem",
    version=2,
)


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
        self.assertEqual(None, load_from_cache(TEST_MODEL_HANDLE, "foo.txt"))

    def test_cache_hit(self):
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: d}):
                cache_path = get_cached_path(TEST_MODEL_HANDLE)
                os.makedirs(cache_path)

                path = load_from_cache(TEST_MODEL_HANDLE)

                self.assertEqual(
                    os.path.join(
                        d, MODELS_CACHE_SUBFOLDER, "google", "bert", "tensorFlow2", "answer-equivalence-bem", "2"
                    ),
                    path,
                )

    def test_cache_hit_with_path(self):
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: d}):
                cache_path = get_cached_path(TEST_MODEL_HANDLE)

                os.makedirs(cache_path)
                Path(os.path.join(cache_path, "foo.txt")).touch()  # Create file

                path = load_from_cache(TEST_MODEL_HANDLE, path="foo.txt")

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
