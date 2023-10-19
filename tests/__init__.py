import os

# Ensures tests are try to read your kaggle credentials.
os.environ["KAGGLE_CONFIG_DIR"] = "/some-missing-directory"
