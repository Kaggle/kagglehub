import hashlib
import mimetypes
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from flask import Flask, Response
from flask.typing import ResponseReturnValue

from kagglehub.config import CACHE_FOLDER_ENV_VAR_NAME, get_kaggle_api_endpoint
from kagglehub.handle import ResourceHandle
from kagglehub.integrity import GCS_HASH_HEADER, to_b64_digest

MOCK_GCS_BUCKET_BASE_PATH = "/mock-gcs-bucket/file-path"
AUTO_COMPRESSED_FILE_NAME = "shapes.csv"
LOCATION_HEADER = "Location"
CONTENT_LENGTH_HEADER = "Content-Length"


def get_test_file_path(relative_path: str) -> str:
    return os.path.join(Path(__file__).parent, "data", relative_path)


def resolve_endpoint(var_name: str = "KAGGLE_API_ENDPOINT") -> tuple[str, int]:
    endpoint = os.environ.get(var_name, "127.0.0.1:7777")
    address, port = endpoint.replace("http://", "").split(":")
    return address, int(port)


def get_mocked_gcs_signed_url(file_name: str) -> str:
    return f"{get_kaggle_api_endpoint()}{MOCK_GCS_BUCKET_BASE_PATH}/{file_name}?X-Goog-Headers=all-kinds-of-stuff"


# All downloads, regardless of archive or file, happen via GCS signed URLs. We mock the 302 and handle
# the redirect not only to be thorough--without this, the response.url in download_file (clients.py)
# will not pick up on followed redirect URL being different from the originally requested URL.
def get_gcs_redirect_response(file_name: str) -> ResponseReturnValue:
    return (
        Response(
            headers={
                LOCATION_HEADER: get_mocked_gcs_signed_url(file_name),
                CONTENT_LENGTH_HEADER: "0",
            }
        ),
        302,
    )


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


def add_mock_gcs_route(app: Flask) -> None:
    """Adds the mock GCS route for handling signed URL redirects"""

    app.add_url_rule(
        f"{MOCK_GCS_BUCKET_BASE_PATH}/<file_name>",
        endpoint="handle_mock_gcs_redirect",
        view_func=handle_mock_gcs_redirect,
        methods=["get"],
    )


def handle_mock_gcs_redirect(file_name: str) -> ResponseReturnValue:
    test_file_path = get_test_file_path(file_name)

    def generate_file_content() -> Generator[bytes, Any, None]:
        with open(test_file_path, "rb") as f:
            while True:
                chunk = f.read(4096)  # Read file in chunks
                if not chunk:
                    break
                yield chunk

    with open(test_file_path, "rb") as f:
        content = f.read()
        file_hash = hashlib.md5()
        file_hash.update(content)
        return (
            Response(
                generate_file_content(),
                headers={
                    GCS_HASH_HEADER: f"md5={to_b64_digest(file_hash)}",
                    "Content-Length": str(os.path.getsize(test_file_path)),
                    "Content-Type": mimetypes.guess_type(test_file_path)[0] or "application/octet-stream",
                },
            ),
            200,
        )
