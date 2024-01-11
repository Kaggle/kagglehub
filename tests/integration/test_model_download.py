import pytest
from requests import HTTPError

from kagglehub import model_download

HANDLE = "keras/bert/keras/bert_base_en/2"


def test_successful_download():
    # Test a successful model download
    expected_path = "/usr/local/google/home/aminmohamed/.cache/kagglehub/models/keras/bert/keras/bert_base_en/2"
    assert model_download(HANDLE) == expected_path, "Model download path did not match expected path"


def test_model_download_invalid_handle():
    with pytest.raises(ValueError):
        model_download("invalid_handle")


def test_download_specific_file():
    file_path = "?select=tokenizer.json"
    expected_full_path = "/usr/local/google/home/aminmohamed/.cache/kagglehub/models/keras/bert/keras/bert_base_en/2/?select=tokenizer.json"  # noqa: E501
    assert (
        model_download(HANDLE, path=file_path) == expected_full_path
    ), "Specific file download path did not match expected path"


def test_download_multiple_files():
    file_paths = ["?select=tokenizer.json", "?select=config.json"]
    expected_paths = [
        "/usr/local/google/home/aminmohamed/.cache/kagglehub/models/keras/bert/keras/bert_base_en/2/?select=tokenizer.json",
        "/usr/local/google/home/aminmohamed/.cache/kagglehub/models/keras/bert/keras/bert_base_en/2/?select=config.json",
    ]
    assert all(
        model_download(HANDLE, path=p) == e for p, e in zip(file_paths, expected_paths)
    ), "Paths for multiple files did not match expected paths"


def test_validate_downloaded_file_content():
    file_path = "?select=tokenizer.json"
    path = model_download(HANDLE, path=file_path)
    with open(path, "rb") as file:
        content = file.read()
    assert len(content) > 0, "Downloaded file is empty"


def test_download_with_incorrect_file_path():
    incorrect_path = "nonexistent/file/path"
    with pytest.raises(HTTPError):
        model_download(HANDLE, path=incorrect_path)
