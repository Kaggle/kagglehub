import json
import os
import unittest
from tempfile import TemporaryDirectory
from unittest import mock

from kagglehub.config import (
    CACHE_FOLDER_ENV_VAR_NAME,
    CREDENTIALS_FILENAME,
    CREDENTIALS_FOLDER_ENV_VAR_NAME,
    DEFAULT_CACHE_FOLDER,
    KAGGLE_API_ENDPOINT_ENV_VAR_NAME,
    KEY_ENV_VAR_NAME,
    USERNAME_ENV_VAR_NAME,
    get_cache_folder,
    get_kaggle_api_endpoint,
    get_kaggle_credentials,
)


class TestConfig(unittest.TestCase):
    def test_get_cache_folder_default(self):
        self.assertEqual(DEFAULT_CACHE_FOLDER, get_cache_folder())

    def test_get_kaggle_api_endpoint_default(self):
        self.assertEqual("http://localhost:7777", get_kaggle_api_endpoint())

    @mock.patch.dict(os.environ, {KAGGLE_API_ENDPOINT_ENV_VAR_NAME: "http://localhost"})
    def test_get_kaggle_api_endpoint_environment_var_override(self):
        self.assertEqual("http://localhost", get_kaggle_api_endpoint())

    @mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: "/test_cache"})
    def test_get_cache_folder_environment_var_override(self):
        self.assertEqual("/test_cache", get_cache_folder())

    def test_get_kaggle_credentials_not_set_returns_none(self):
        self.assertEqual(None, get_kaggle_credentials())

    @mock.patch.dict(os.environ, {USERNAME_ENV_VAR_NAME: "lastplacelarry"})
    def test_get_kaggle_credentials_missing_key_returns_none(self):
        self.assertEqual(None, get_kaggle_credentials())

    @mock.patch.dict(os.environ, {KEY_ENV_VAR_NAME: "some-key"})
    def test_get_kaggle_credentials_missing_username_returns_none(self):
        self.assertEqual(None, get_kaggle_credentials())

    @mock.patch.dict(os.environ, {USERNAME_ENV_VAR_NAME: "lastplacelarry", KEY_ENV_VAR_NAME: "some-key"})
    def test_get_kaggle_credentials_missing_username_succeeds(self):
        creds = get_kaggle_credentials()

        self.assertEqual("lastplacelarry", creds.username)
        self.assertEqual("some-key", creds.key)

    def test_get_kaggle_credentials_file_succeeds(self):
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {CREDENTIALS_FOLDER_ENV_VAR_NAME: d}):
                with open(os.path.join(d, CREDENTIALS_FILENAME), "x") as creds_file:
                    json.dump({"username": "kerneler", "key": "another-key"}, creds_file)

                creds = get_kaggle_credentials()

                self.assertEqual("kerneler", creds.username)
                self.assertEqual("another-key", creds.key)

    def test_get_kaggle_credentials_invalid_json_file_raises(self):
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {CREDENTIALS_FOLDER_ENV_VAR_NAME: d}):
                with open(os.path.join(d, CREDENTIALS_FILENAME), "x") as creds_file:
                    creds_file.write("invalid json credentials content")

                self.assertRaises(ValueError, get_kaggle_credentials)

    def test_get_kaggle_credentials_json_missing_username_raises(self):
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {CREDENTIALS_FOLDER_ENV_VAR_NAME: d}):
                with open(os.path.join(d, CREDENTIALS_FILENAME), "x") as creds_file:
                    json.dump(
                        {
                            # Missing 'username'
                            "key": "another-key"
                        },
                        creds_file,
                    )

                self.assertRaises(ValueError, get_kaggle_credentials)

    def test_get_kaggle_credentials_json_missing_key_raises(self):
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {CREDENTIALS_FOLDER_ENV_VAR_NAME: d}):
                with open(os.path.join(d, CREDENTIALS_FILENAME), "x") as creds_file:
                    json.dump(
                        {
                            "username": "kerneler",
                            # Missing 'key'
                        },
                        creds_file,
                    )

                self.assertRaises(ValueError, get_kaggle_credentials)
