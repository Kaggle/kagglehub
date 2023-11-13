import json
import logging
from kagglehub.config import set_kaggle_credentials

logger = logging.getLogger(__name__)


def login():
    """Prompt the user for their Kaggle username and API key and save them globally."""

    username = input("Enter your Kaggle username: ")
    api_key = input("Enter your Kaggle API key: ")

    set_kaggle_credentials(username=username, api_key=api_key)

    logger.info("You are now logged in to Kaggle Hub.")
