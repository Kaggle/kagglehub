import os
from pathlib import Path

DEFAULT_CACHE_FOLDER = os.path.join(Path.home(), ".cache", "kagglehub")
CACHE_FOLDER_ENV_VAR_NAME = "KAGGLEHUB_CACHE"


def get_cache_folder():
    if CACHE_FOLDER_ENV_VAR_NAME in os.environ:
        return os.environ[CACHE_FOLDER_ENV_VAR_NAME]
    return DEFAULT_CACHE_FOLDER
