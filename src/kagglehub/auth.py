import logging

import requests

from kagglehub.clients import KaggleApiV1Client
from kagglehub.config import set_kaggle_credentials

logger = logging.getLogger(__name__)

INVALID_CREDENTIALS_ERROR = 403


def login(validate_credentials=True):  # noqa: FBT002
    """Prompt the user for their Kaggle username and API key and save them globally."""

    username = input("Enter your Kaggle username: ")
    api_key = input("Enter your Kaggle API key: ")

    set_kaggle_credentials(username=username, api_key=api_key)

    logger.info("Kaggle credentials set.")

    if not validate_credentials:
        return

    try:
        api_client = KaggleApiV1Client()
        api_client.get("/hello")
        logger.info("Kaggle credentials successfully validated.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == INVALID_CREDENTIALS_ERROR:
            logger.error(
                "Invalid Kaggle credentials. You can check your credentials on the [Kaggle settings page](https://www.kaggle.com/settings/account)."
            )
        else:
            logger.warning("Unable to validate Kaggle credentials at this time.")
