import os
import unittest
from typing import List

from requests import HTTPError

from kagglehub import model_download

HANDLE = "keras/bert/keras/bert_tiny_en_uncased/2"


class TestModelDownload(unittest.TestCase):
    def list_files_recursively(self, path: str) -> List[str]:
        """List all files recursively in the given path.
        If the path is a file, return a list containing only that file.
        If the path is a directory, list all files recursively in that directory."""
        if os.path.isfile(path):
            directory = os.path.dirname(path)
            rel_file = os.path.relpath(path, directory)
            return [rel_file]

        files = []
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                # Create a relative path from the directory and add it to the list
                rel_dir = os.path.relpath(root, path)
                rel_file = os.path.join(rel_dir, filename)
                if not rel_file.startswith("./?"):
                    files.append(rel_file)
        return sorted(files)

    def assert_files(self, path: str, expected_files: List[str]) -> None:
        """Assert that all expected files exist and are non-empty."""
        files = self.list_files_recursively(path)
        expected_files_sorted = sorted(expected_files)
        self.assertEqual(files, expected_files_sorted, "Downloaded files did not match expected files")

        # Assert that each file's size is greater than zero
        for file in files:
            if os.path.isfile(path):
                file_path = os.path.join(os.path.dirname(path), file)
            else:
                file_path = os.path.join(path, file)

            self.assertGreater(os.path.getsize(file_path), 0, f"File {file} is empty")

    def test_model_versioned_succeeds(self) -> None:
        actual_path = model_download(HANDLE)

        expected_files = [
            "assets/tokenizer/vocabulary.txt",
            "./config.json",
            "./metadata.json",
            "./model.weights.h5",
            "./tokenizer.json",
        ]
        self.assert_files(actual_path, expected_files)

    def test_model_unversioned_succeeds(self) -> None:
        unversioned_handle = "keras/bert/keras/bert_tiny_en_uncased"
        actual_path = model_download(unversioned_handle)

        expected_files = [
            "assets/tokenizer/vocabulary.txt",
            "./config.json",
            "./metadata.json",
            "./model.weights.h5",
            "./tokenizer.json",
        ]
        self.assert_files(actual_path, expected_files)

    def test_download_private_model_succeeds(self) -> None:
        actual_path = model_download("integrationtester/test-private-model/pyTorch/b0")

        expected_files = [
            "./efficientnet-b0.pth",
        ]

        self.assert_files(actual_path, expected_files)

    def test_download_multiple_files(self) -> None:
        file_paths = ["tokenizer.json", "config.json"]
        for p in file_paths:
            actual_path = model_download(HANDLE, path=p)
            self.assert_files(actual_path, [p])

    def test_download_with_incorrect_file_path(self) -> None:
        incorrect_path = "nonexistent/file/path"
        with self.assertRaises(HTTPError):
            model_download(HANDLE, path=incorrect_path)
