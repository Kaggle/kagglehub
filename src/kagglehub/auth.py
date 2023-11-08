import json
from kagglehub.config import CREDENTIALS_JSON_KEY, CREDENTIALS_JSON_USERNAME, KaggleApiCredentials, _get_kaggle_credentials_file


def login():
    """Prompt the user for their Kaggle username and API key and save them globally."""

    username = input("Enter your Kaggle username: ")
    api_key = input("Enter your Kaggle API key: ")

    set_kaggle_credentials(username=username, api_key=api_key)

    print("You are now logged in to Kaggle Hub.")
