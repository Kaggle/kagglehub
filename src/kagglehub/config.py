import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE_FOLDER = os.path.join(Path.home(), ".cache", "kagglehub")
DEFAULT_KAGGLE_API_ENDPOINT = "https://www.kaggle.com"
DEFAULT_KAGGLE_CREDENTIALS_FOLDER = os.path.join(Path.home(), ".kaggle")
CREDENTIALS_FILENAME = "kaggle.json"

CACHE_FOLDER_ENV_VAR_NAME = "KAGGLEHUB_CACHE"
KAGGLE_API_ENDPOINT_ENV_VAR_NAME = "KAGGLE_API_ENDPOINT"
USERNAME_ENV_VAR_NAME = "KAGGLE_USERNAME"
KEY_ENV_VAR_NAME = "KAGGLE_KEY"
CREDENTIALS_FOLDER_ENV_VAR_NAME = "KAGGLE_CONFIG_DIR"

CREDENTIALS_JSON_USERNAME = "username"
CREDENTIALS_JSON_KEY = "key"


def get_cache_folder():
    if CACHE_FOLDER_ENV_VAR_NAME in os.environ:
        return os.environ[CACHE_FOLDER_ENV_VAR_NAME]
    return DEFAULT_CACHE_FOLDER


def get_kaggle_api_endpoint():
    if KAGGLE_API_ENDPOINT_ENV_VAR_NAME in os.environ:
        return os.environ[KAGGLE_API_ENDPOINT_ENV_VAR_NAME]
    return DEFAULT_KAGGLE_API_ENDPOINT


def get_kaggle_credentials():
    creds_filepath = _get_kaggle_credentials_file()

    if USERNAME_ENV_VAR_NAME in os.environ and KEY_ENV_VAR_NAME in os.environ:
        return KaggleApiCredentials(username=os.environ[USERNAME_ENV_VAR_NAME], key=os.environ[KEY_ENV_VAR_NAME])
    if os.path.exists(creds_filepath):
        with open(creds_filepath) as creds_json_file:
            try:
                creds_dict = json.load(creds_json_file)
            except json.JSONDecodeError as err:
                msg = f"Invalid Kaggle credentials file at {creds_filepath}"
                raise ValueError(msg) from err

            if CREDENTIALS_JSON_USERNAME not in creds_dict:
                msg = f"Kaggle credentials file at {creds_filepath} is missing '{CREDENTIALS_JSON_USERNAME}' key"
                raise ValueError(msg)

            if CREDENTIALS_JSON_KEY not in creds_dict:
                msg = f"Kaggle credentials file at {creds_filepath} is missing '{CREDENTIALS_JSON_KEY}' key"
                raise ValueError(msg)

            return KaggleApiCredentials(
                username=creds_dict[CREDENTIALS_JSON_USERNAME], key=creds_dict[CREDENTIALS_JSON_KEY]
            )

    return None


def _get_kaggle_credentials_file():
    return os.path.join(_get_kaggle_credentials_folder(), CREDENTIALS_FILENAME)


def _get_kaggle_credentials_folder():
    if CREDENTIALS_FOLDER_ENV_VAR_NAME in os.environ:
        return os.environ[CREDENTIALS_FOLDER_ENV_VAR_NAME]
    return DEFAULT_KAGGLE_CREDENTIALS_FOLDER


@dataclass
class KaggleApiCredentials:
    username: str
    key: str
