import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from kagglehub.config import CACHE_FOLDER_ENV_VAR_NAME, get_kaggle_api_endpoint
from kagglehub.handle import ResourceHandle

MOCK_GCS_BUCKET_BASE_PATH = "/mock-gcs-bucket/file-path"


def get_test_file_path(relative_path: str) -> str:
    return os.path.join(Path(__file__).parent, "data", relative_path)


def resolve_endpoint(var_name: str = "KAGGLE_API_ENDPOINT") -> tuple[str, int]:
    endpoint = os.environ.get(var_name, "127.0.0.1:7777")
    address, port = endpoint.replace("http://", "").split(":")
    return address, int(port)


def get_mocked_gcs_signed_url(file_name: str) -> str:
    return f"{get_kaggle_api_endpoint()}{MOCK_GCS_BUCKET_BASE_PATH}/{file_name}?X-Goog-Headers=all-kinds-of-stuff"


@contextmanager
def create_test_cache() -> Generator[str, None, None]:
    with TemporaryDirectory() as d:
        with mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: d}):
            yield d


class InvalidResourceHandle(ResourceHandle):
    def __init__(self):
        self.owner = "invalid"

    def to_url(self) -> str:
        return "invalid"
