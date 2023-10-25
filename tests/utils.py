import os
from pathlib import Path


def get_test_file_path(relative_path):
    return os.path.join(Path(__file__).parent, "data", relative_path)
