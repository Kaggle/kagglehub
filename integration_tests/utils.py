import os
import unittest
from collections.abc import Generator
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from unittest import mock

from kagglehub.config import CACHE_FOLDER_ENV_VAR_NAME, KEY_ENV_VAR_NAME, USERNAME_ENV_VAR_NAME


@contextmanager
def create_test_cache() -> Generator[str, None, None]:
    with TemporaryDirectory() as d:
        with mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: d}):
            yield d


def list_files_recursively(path: str) -> list[str]:
    """List all files recursively in the given path.
    If the path is a file, return a list containing only that file.
    If the path is a directory, list all files recursively in that directory."""
    if os.path.isfile(path):
        return [os.path.basename(path)]

    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            # Create a relative path from the directory and add it to the list
            rel_file = os.path.relpath(os.path.join(root, filename), path)
            if not rel_file.startswith("./?"):
                files.append(rel_file)
    return sorted(files)


def assert_files(test_case: unittest.TestCase, path: str, expected_files: list[str]) -> bool:
    """Assert that all expected files exist and are non-empty."""
    files = list_files_recursively(path)
    expected_files_sorted = sorted(expected_files)
    test_case.assertEqual(files, expected_files_sorted)

    # Assert that each file's size is greater than zero
    for file in files:
        if os.path.isfile(path):
            file_path = os.path.join(os.path.dirname(path), file)
        else:
            file_path = os.path.join(path, file)

        test_case.assertGreater(os.path.getsize(file_path), 0, f"File {file} is empty")


@contextmanager
def unauthenticated() -> Generator[None, None, None]:
    with mock.patch.dict(
        os.environ,
        {
            USERNAME_ENV_VAR_NAME: "",
            KEY_ENV_VAR_NAME: "",
        },
    ):
        yield
