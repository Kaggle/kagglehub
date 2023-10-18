import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import kagglehub
from kagglehub.cache import MODELS_CACHE_SUBFOLDER, get_cached_path
from kagglehub.handle import parse_model_handle

VERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b/3"
UNVERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b"


class TestModelDownload(unittest.TestCase):
    def test_unversioned_model_download(self):
        with self.assertRaises(NotImplementedError):
            kagglehub.model_download(UNVERSIONED_MODEL_HANDLE)

    def test_versioned_model_download(self):
        with self.assertRaises(NotImplementedError):
            kagglehub.model_download(VERSIONED_MODEL_HANDLE)

    def test_versioned_model_download_with_path(self):
        with self.assertRaises(NotImplementedError):
            kagglehub.model_download(VERSIONED_MODEL_HANDLE, path="foo.txt")

    def test_versioned_model_download_already_cached(self):
        with TemporaryDirectory() as d:
            cache_path = get_cached_path(parse_model_handle(VERSIONED_MODEL_HANDLE), cache_dir=d)
            os.makedirs(cache_path)

            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, cache_dir=d)

            self.assertEqual(
                os.path.join(d, MODELS_CACHE_SUBFOLDER, "metaresearch", "llama-2", "pyTorch", "13b", "3"), model_path
            )

    def test_versioned_model_download_with_path_already_cached(self):
        with TemporaryDirectory() as d:
            cache_path = get_cached_path(parse_model_handle(VERSIONED_MODEL_HANDLE), cache_dir=d)
            os.makedirs(cache_path)
            Path.touch(os.path.join(cache_path, "foo.txt"))  # Create file

            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, path="foo.txt", cache_dir=d)

            self.assertEqual(
                os.path.join(d, MODELS_CACHE_SUBFOLDER, "metaresearch", "llama-2", "pyTorch", "13b", "3", "foo.txt"),
                model_path,
            )
