import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Generator
from unittest import mock

from kagglehub.config import CACHE_FOLDER_ENV_VAR_NAME


@contextmanager
def create_test_cache() -> Generator[str, None, None]:
    with TemporaryDirectory() as d:
        with mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: d}):
            yield d
