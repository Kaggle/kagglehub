import os

# Ensures tests do not try to read your own kaggle credentials.
os.environ["KAGGLE_CONFIG_DIR"] = "/some-missing-directory"
