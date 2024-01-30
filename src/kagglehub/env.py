import logging
import os

KAGGLE_NOTEBOOK_ENV_VAR_NAME = "KAGGLE_KERNEL_RUN_TYPE"
KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME = "KAGGLE_DATA_PROXY_URL"

logger = logging.getLogger(__name__)


def is_in_colab_notebook() -> bool:
    return os.getenv("COLAB_RELEASE_TAG") is not None


def is_in_kaggle_notebook() -> bool:
    return (
        os.getenv(KAGGLE_NOTEBOOK_ENV_VAR_NAME) is not None
        and os.getenv(KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME) is not None
    )


def read_kaggle_build_date() -> str:
    build_date_file = "/etc/build_date"
    try:
        with open(build_date_file) as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.warning(f"Build date file {build_date_file} not found in Kaggle Notebook environment.")
        return "unknown"
