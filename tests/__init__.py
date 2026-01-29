import os

from tests.utils import get_test_file_path

# Ensures tests do not try to read your own legacy kaggle credentials.
os.environ["KAGGLE_CONFIG_DIR"] = "/some-missing-directory"

# Ensures tests do not try to read your own API token
os.environ["KAGGLE_API_TOKEN"] = get_test_file_path("empty_access_token")

# All APIs call in tests should go to a local test server.
os.environ["KAGGLE_API_ENVIRONMENT"] = "TEST"
