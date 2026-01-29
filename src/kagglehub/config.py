"""Retrieve config values that a user may set/override.

For config values specific to a resolver's environment (a user is not expected to override),
add it to the resolver's module.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from kagglesdk.kaggle_env import get_access_token_from_env

from kagglehub.env import is_in_colab_notebook

DEFAULT_CACHE_FOLDER = os.path.join(Path.home(), ".cache", "kagglehub")
DEFAULT_KAGGLE_API_ENDPOINT = "https://www.kaggle.com"
DEFAULT_KAGGLE_CREDENTIALS_FOLDER = os.path.join(Path.home(), ".kaggle")
DEFAULT_LOG_LEVEL = logging.INFO
CREDENTIALS_FILENAME = "kaggle.json"

CACHE_FOLDER_ENV_VAR_NAME = "KAGGLEHUB_CACHE"
USERNAME_ENV_VAR_NAME = "KAGGLE_USERNAME"
KEY_ENV_VAR_NAME = "KAGGLE_KEY"
CREDENTIALS_FOLDER_ENV_VAR_NAME = "KAGGLE_CONFIG_DIR"
LOG_VERBOSITY_ENV_VAR_NAME = "KAGGLEHUB_VERBOSITY"
DISABLE_KAGGLE_CACHE_ENV_VAR_NAME = "DISABLE_KAGGLE_CACHE"
DISABLE_COLAB_CACHE_ENV_VAR_NAME = "DISABLE_COLAB_CACHE"
TBE_RUNTIME_ADDR_ENV_VAR_NAME = "TBE_RUNTIME_ADDR"

CREDENTIALS_JSON_USERNAME = "username"
CREDENTIALS_JSON_KEY = "key"

COLAB_SECRET_USERNAME = "KAGGLE_USERNAME"
COLAB_SECRET_KEY = "KAGGLE_KEY"
COLAB_SECRET_API_TOKEN = "KAGGLE_API_TOKEN"

_kaggle_credentials = None

LOG_LEVELS_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
TRUTHY_VALUES = ["true", "1", "t"]

logger = logging.getLogger(__name__)


@dataclass
class KaggleApiCredentials:
    username: str | None = None
    key: str | None = None
    api_key: str | None = None


def get_cache_folder() -> str:
    if CACHE_FOLDER_ENV_VAR_NAME in os.environ:
        return os.environ[CACHE_FOLDER_ENV_VAR_NAME]
    return DEFAULT_CACHE_FOLDER


def get_kaggle_credentials() -> KaggleApiCredentials | None:
    # Check for credentials in the global variable
    if _kaggle_credentials:
        return _kaggle_credentials

    api_key, _ = get_access_token_from_env()
    if api_key:
        return KaggleApiCredentials(api_key=api_key)

    # Legacy credentials support.
    creds_filepath = _get_kaggle_credentials_file()
    env_var_username = os.getenv(USERNAME_ENV_VAR_NAME)
    env_var_key = os.getenv(KEY_ENV_VAR_NAME)
    if env_var_username and env_var_key:
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
    if is_in_colab_notebook() and (colab_secret := get_colab_credentials()) is not None:
        return colab_secret

    return None


def get_log_verbosity() -> int:
    if LOG_VERBOSITY_ENV_VAR_NAME in os.environ:
        log_level_str = os.environ[LOG_VERBOSITY_ENV_VAR_NAME]
        if log_level_str in LOG_LEVELS_MAP:
            return LOG_LEVELS_MAP[log_level_str]
        else:
            logger.warning(
                f"Unknown verbosity level set with {LOG_VERBOSITY_ENV_VAR_NAME}={log_level_str}, "
                f"Accepted values are: {', '.join(LOG_LEVELS_MAP.keys())}"
            )
    return DEFAULT_LOG_LEVEL


def is_colab_cache_disabled() -> bool:
    return _is_env_var_truthy(DISABLE_COLAB_CACHE_ENV_VAR_NAME)


def is_kaggle_cache_disabled() -> bool:
    return _is_env_var_truthy(DISABLE_KAGGLE_CACHE_ENV_VAR_NAME)


def _get_kaggle_credentials_file() -> str:
    return os.path.join(_get_kaggle_credentials_folder(), CREDENTIALS_FILENAME)


def _get_kaggle_credentials_folder() -> str:
    if CREDENTIALS_FOLDER_ENV_VAR_NAME in os.environ:
        return os.environ[CREDENTIALS_FOLDER_ENV_VAR_NAME]
    return DEFAULT_KAGGLE_CREDENTIALS_FOLDER


def _is_env_var_truthy(env_var_name: str) -> bool:
    return env_var_name in os.environ and os.environ[env_var_name].lower() in TRUTHY_VALUES


def set_kaggle_credentials(username: str, api_key: str) -> None:
    stripped_username = username.strip()
    stripped_api_key = api_key.strip()
    if not stripped_username or not stripped_api_key:
        error_message = "Both username and API key cannot be empty or whitespace"
        raise ValueError(error_message)

    global _kaggle_credentials  # noqa: PLW0603
    _kaggle_credentials = KaggleApiCredentials(username=username, key=api_key)
    logger.info("Kaggle credentials set.")


def set_kaggle_api_token(api_token: str) -> None:
    stripped_token = api_token.strip()
    if not stripped_token:
        error_message = "The API token cannot be empty"
        raise ValueError(error_message)

    global _kaggle_credentials  # noqa: PLW0603
    _kaggle_credentials = KaggleApiCredentials(api_key=api_token)
    logger.info("Kaggle credentials set.")


def clear_kaggle_credentials() -> None:
    global _kaggle_credentials  # noqa: PLW0603
    _kaggle_credentials = None


def _normalize_whitespace(s: str) -> str:
    return s.replace("\r", "").replace("\n", "").strip()


def get_colab_credentials() -> KaggleApiCredentials | None:
    # userdata access is already thread-safe after b/318732641
    try:
        from google.colab import userdata  # type: ignore[import] # noqa: PLC0415
    except ImportError:
        return None

    try:
        api_token = _normalize_whitespace(userdata.get(COLAB_SECRET_API_TOKEN))
        if api_token:
            return KaggleApiCredentials(api_key=api_token)

        # Legacy credentials support.
        username = _normalize_whitespace(userdata.get(COLAB_SECRET_USERNAME))
        key = _normalize_whitespace(userdata.get(COLAB_SECRET_KEY))
        if username and key:
            return KaggleApiCredentials(username=username, key=key)
        return None
    except Exception:
        # fail to get Kaggle credentials from Colab secrets. Skip...
        return None
