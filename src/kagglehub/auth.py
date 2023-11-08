import json
from kagglehub.config import CREDENTIALS_JSON_KEY, CREDENTIALS_JSON_USERNAME, KaggleApiCredentials, _get_kaggle_credentials_file


def login():
    """Prompt the user for their Kaggle username and API key and save them to a global variable."""

    username = input("Enter your Kaggle username: ")
    api_key = input("Enter your Kaggle API key: ")

    global _kaggle_credentials
    _kaggle_credentials = KaggleApiCredentials(username=username, key=api_key)

    # Save the credentials to a file for persistence
    creds_filepath = _get_kaggle_credentials_file()
    with open(creds_filepath, "w") as creds_json_file:
        json.dump({CREDENTIALS_JSON_USERNAME: username, CREDENTIALS_JSON_KEY: api_key}, creds_json_file)

    print("You are now logged in to Kaggle Hub.")
