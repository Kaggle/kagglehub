import os

# Ensures tests do not try to read your own kaggle credentials.
os.environ["KAGGLE_CONFIG_DIR"] = "/some-missing-directory"

# All APIs call in tests should go to a local test server.
os.environ["KAGGLE_API_ENDPOINT"] = "http://localhost:7777"
