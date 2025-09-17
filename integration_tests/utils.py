import functools
import os
import unittest
from collections.abc import Callable, Generator
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from kagglehub.config import (
    CACHE_FOLDER_ENV_VAR_NAME,
    CREDENTIALS_FOLDER_ENV_VAR_NAME,
    KEY_ENV_VAR_NAME,
    USERNAME_ENV_VAR_NAME,
)


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


def list_columns(path: str) -> list[str]:
    """Assuming the path is a CSV, list all columns sorted lexicographically"""
    with open(path) as file:
        first_line = file.readline().strip()
        return sorted(first_line.split(","))


def assert_files(test_case: unittest.TestCase, path: str, expected_files: list[str]) -> None:
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


def assert_columns(test_case: unittest.TestCase, path: str, expected_columns: list[str]) -> None:
    """Assert that the given path to a CSV has the expected columns."""
    columns = list_columns(path)
    test_case.assertEqual(columns, sorted(expected_columns))


@contextmanager
def unauthenticated() -> Generator[None, None, None]:
    with mock.patch.dict(
        os.environ,
        {USERNAME_ENV_VAR_NAME: "", KEY_ENV_VAR_NAME: "", CREDENTIALS_FOLDER_ENV_VAR_NAME: "/nonexistent"},
    ):
        yield


def parameterized(*parameter_values: Any) -> Callable:  # noqa: ANN401
    """Decorator which parameterizes a unittest test method.

    Currently only supports single arguments but could be extended."""

    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self) -> None:  # noqa: ANN001
            for value in parameter_values:
                if hasattr(self, "setUp"):
                    self.setUp()

                with self.subTest(value):
                    method(self, value)

                if hasattr(self, "tearDown"):
                    self.tearDown()

        return wrapper

    return decorator
