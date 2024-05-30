import os
from pathlib import Path

# Ensures integration tests do not try to write into regular cache directory.
os.environ["KAGGLEHUB_CACHE"] = os.path.join(Path.home(), ".cache", "kagglehub-integration-tests")
