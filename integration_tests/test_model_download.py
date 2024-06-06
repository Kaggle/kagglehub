import unittest

from requests import HTTPError

from kagglehub import model_download

from .utils import assert_files, create_test_cache, unauthenticated

HANDLE = "keras/bert/keras/bert_tiny_en_uncased/2"


class TestModelDownload(unittest.TestCase):
    def test_model_versioned_succeeds(self) -> None:
        with create_test_cache():
            actual_path = model_download(HANDLE)

            expected_files = [
                "assets/tokenizer/vocabulary.txt",
                "config.json",
                "metadata.json",
                "model.weights.h5",
                "tokenizer.json",
            ]
            assert_files(self, actual_path, expected_files)

    def test_model_unversioned_succeeds(self) -> None:
        with create_test_cache():
            unversioned_handle = "keras/bert/keras/bert_tiny_en_uncased"
            actual_path = model_download(unversioned_handle)

            expected_files = [
                "assets/tokenizer/vocabulary.txt",
                "config.json",
                "metadata.json",
                "model.weights.h5",
                "tokenizer.json",
            ]
            assert_files(self, actual_path, expected_files)

    def test_public_model_as_unauthenticated_succeeds(self) -> None:
        with create_test_cache():
            with unauthenticated():
                unversioned_handle = "keras/bert/keras/bert_tiny_en_uncased"
                actual_path = model_download(unversioned_handle)

                expected_files = [
                    "assets/tokenizer/vocabulary.txt",
                    "config.json",
                    "metadata.json",
                    "model.weights.h5",
                    "tokenizer.json",
                ]
                assert_files(self, actual_path, expected_files)

    def test_download_private_model_succeeds(self) -> None:
        with create_test_cache():
            actual_path = model_download("integrationtester/test-private-model/pyTorch/b0")

            expected_files = [
                "efficientnet-b0.pth",
            ]

            assert_files(self, actual_path, expected_files)

    def test_download_archive_model_many_files_succeeds(self) -> None:
        with create_test_cache():
            # If the model has > 25 files, we download the archive and uncompress it locally.
            actual_path = model_download("integrationtester/test-private-model/keras/many-files-with-subdirectories/1")

            expected_files = [f"assets/{i}.txt" for i in range(1, 26)] + [
                "config.json",
                "model.keras",
            ]
            assert_files(self, actual_path, expected_files)

    def test_download_multiple_files(self) -> None:
        with create_test_cache():
            file_paths = ["tokenizer.json", "config.json"]
            for p in file_paths:
                actual_path = model_download(HANDLE, path=p)
                assert_files(self, actual_path, [p])

    def test_download_with_incorrect_file_path(self) -> None:
        incorrect_path = "nonexistent/file/path"
        with self.assertRaises(HTTPError):
            model_download(HANDLE, path=incorrect_path)
