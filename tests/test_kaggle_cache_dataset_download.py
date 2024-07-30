import os
from unittest import mock

import requests

import kagglehub
from kagglehub.config import DISABLE_KAGGLE_CACHE_ENV_VAR_NAME
from kagglehub.env import KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME
from tests.fixtures import BaseTestCase

from .server_stubs import jwt_stub as stub
from .server_stubs import serv

INVALID_ARCHIVE_DATASET_HANDLE = "test-owner/test-dataset"
VERSIONED_MODEL_HANDLE = "test-owner/test-dataset/versions/2"
LATEST_DATASET_VERSION = 2
UNVERSIONED_DATASET_HANDLE = "test-owner/test-dataset"
TEST_FILEPATH = "bar.txt"

# Test cases for the DatasetKaggleCacheResolver.
class TestKaggleCacheDatasetDownload(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app, KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME, "http://localhost:7778")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

