import unittest

from requests import HTTPError

from kagglehub import model_download

HANDLE = "keras/bert/keras/bert_base_en/2"


class TestModelDownload(unittest.TestCase):
    def test_model_versioned_succeeds(self):
        expected_path = "/usr/local/google/home/aminmohamed/.cache/kagglehub/models/keras/bert/keras/bert_base_en/2"
        actual_path = model_download(HANDLE)
        self.assertEqual(actual_path, expected_path, "Model download path did not match expected path")

    def test_download_multiple_files(self):
        file_paths = ["?select=tokenizer.json", "?select=config.json"]
        expected_paths = [
            "/usr/local/google/home/aminmohamed/.cache/kagglehub/models/keras/bert/keras/bert_base_en/2/?select=tokenizer.json",
            "/usr/local/google/home/aminmohamed/.cache/kagglehub/models/keras/bert/keras/bert_base_en/2/?select=config.json",
        ]
        for p, e in zip(file_paths, expected_paths):
            self.assertEqual(model_download(HANDLE, path=p), e, "Path for file did not match expected path")

    def test_validate_downloaded_file_content(self):
        file_path = "?select=tokenizer.json"
        path = model_download(HANDLE, path=file_path)
        with open(path, "rb") as file:
            content = file.read()
        self.assertGreater(len(content), 0, "Downloaded file is empty")

    def test_download_with_incorrect_file_path(self):
        incorrect_path = "nonexistent/file/path"
        with self.assertRaises(HTTPError):
            model_download(HANDLE, path=incorrect_path)
