import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Type
from unittest import mock
from urllib.parse import urlparse

from kagglehub.clients import (
    KAGGLE_DATA_PROXY_TOKEN_ENV_VAR_NAME,
    KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME,
    KAGGLE_JWT_TOKEN_ENV_VAR_NAME,
)
from kagglehub.config import CACHE_FOLDER_ENV_VAR_NAME, KAGGLE_API_ENDPOINT_ENV_VAR_NAME
from kagglehub.kaggle_cache_resolver import KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME, KAGGLE_NOTEBOOK_ENV_VAR_NAME


def get_test_file_path(relative_path):
    return os.path.join(Path(__file__).parent, "data", relative_path)


@contextmanager
def create_test_cache():
    with TemporaryDirectory() as d:
        with mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: d}):
            yield d


@contextmanager
def create_test_http_server(handler_class: Type[BaseHTTPRequestHandler]):
    endpoint = os.getenv(KAGGLE_API_ENDPOINT_ENV_VAR_NAME)
    test_server_address = urlparse(endpoint)
    if not test_server_address.hostname or not test_server_address.port:
        msg = f"Invalid test server address: {endpoint}. You must specify a hostname & port"
        raise ValueError(msg)
    with HTTPServer((test_server_address.hostname, test_server_address.port), handler_class) as httpd:
        threading.Thread(target=httpd.serve_forever).start()

        try:
            yield httpd
        finally:
            httpd.shutdown()


@contextmanager
def create_test_jwt_http_server(handler_class: Type[BaseHTTPRequestHandler]):
    with TemporaryDirectory() as cache_mount_folder:
        with mock.patch.dict(
            os.environ,
            {
                KAGGLE_NOTEBOOK_ENV_VAR_NAME: "Interactive",
                KAGGLE_JWT_TOKEN_ENV_VAR_NAME: "foo jwt token",
                KAGGLE_DATA_PROXY_TOKEN_ENV_VAR_NAME: "foo proxy token",
                KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME: cache_mount_folder,
            },
        ):
            endpoint = os.getenv(KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME)
            test_server_address = urlparse(endpoint)
            if not test_server_address.hostname or not test_server_address.port:
                msg = f"Invalid JWT test server address: {endpoint}. You must specify a hostname & port"
                raise ValueError(msg)
            with HTTPServer((test_server_address.hostname, test_server_address.port), handler_class) as httpd:
                threading.Thread(target=httpd.serve_forever).start()

                try:
                    yield httpd
                finally:
                    httpd.shutdown()
