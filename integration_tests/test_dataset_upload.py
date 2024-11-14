import logging
import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

from kagglehub import dataset_upload, datasets_helpers
from kagglehub.clients import KaggleApiV1Client
from kagglehub.config import get_kaggle_credentials
from kagglehub.handle import parse_dataset_handle
from kagglehub.http_resolver import _get_current_version

logger = logging.getLogger(__name__)


ReturnType = TypeVar("ReturnType")


class TestDatasetUpload(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.dummy_files = ["foo.txt", "bar.csv"]
        for file in self.dummy_files:
            with open(os.path.join(self.temp_dir, file), "w") as f:
                f.write("dummy content")
        self.dataset_uuid = str(uuid.uuid4())
        credentials = get_kaggle_credentials()
        if not credentials:
            self.fail("Make sure to set your Kaggle credentials before running the tests")
        self.owner_slug = credentials.username
        self.dataset_slug = f"dataset-{self.dataset_uuid}"
        self.handle = f"{self.owner_slug}/{self.dataset_slug}"

    def test_dataset_upload_and_versioning(self) -> None:
        # Create Dataset
        self.assertTrue(dataset_upload_and_wait(self.handle, self.temp_dir))

        # Create Version
        dataset_upload(self.handle, self.temp_dir, "new version")

        # If delete dataset does not raise an error, then the upload was successful.

    def test_dataset_upload_and_versioning_zip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            for i in range(60):
                test_filepath = Path(temp_dir) / f"temp_test_file_{i}"
                test_filepath.touch()

            # Create Dataset
            self.assertTrue(dataset_upload_and_wait(self.handle, self.temp_dir))

            # Create Version
            # dataset_upload(self.handle, self.temp_dir, "new version")

    def test_dataset_upload_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Create new folder within temp_dir
            inner_folder_path = Path(temp_dir) / "inner_folder"
            inner_folder_path.mkdir()

            for i in range(60):
                # Create a file in the temp_dir
                test_filepath = Path(temp_dir) / f"temp_test_file_{i}"
                test_filepath.touch()

                # Create the same file in the inner_folder
                test_filepath_inner = inner_folder_path / f"temp_test_file_{i}"
                test_filepath_inner.touch()

            # Create Dataset
            self.assertTrue(dataset_upload_and_wait(self.handle, self.temp_dir))

            # Create Version
            dataset_upload(self.handle, self.temp_dir, "new version")

    def test_dataset_upload_directory_structure(self) -> None:
        nested_dir = Path(self.temp_dir) / "nested"
        nested_dir.mkdir()

        with open(Path(self.temp_dir) / "file1.txt", "w") as f:
            f.write("dummy content in nested file")

        # Create dummy files in the nested directory
        nested_dummy_files = ["nested_dataset.txt", "nested_csv.csv"]
        for file in nested_dummy_files:
            with open(nested_dir / file, "w") as f:
                f.write("dummy content in nested file")

        # Call the dataset upload function with the base directory
        dataset_upload(self.handle, self.temp_dir)

    def test_dataset_upload_nested_dir(self) -> None:
        # Create a nested directory within self.temp_dir
        nested_dir = Path(self.temp_dir) / "nested"
        nested_dir.mkdir()

        # Create dummy files in the nested directory
        nested_dummy_files = ["nested_dataset.txt", "nested_csv.csv"]
        for file in nested_dummy_files:
            with open(nested_dir / file, "w") as f:
                f.write("dummy content in nested file")

        # Call the dataset upload function with the base directory
        dataset_upload(self.handle, self.temp_dir)

    def test_single_file_upload(self) -> None:
        single_file_path = Path(self.temp_dir) / "dummy_file.txt"
        with open(single_file_path, "wb") as f:
            f.write(os.urandom(100))

        dataset_upload(self.handle, str(single_file_path))

    def tearDown(self) -> None:
        time.sleep(5)  # hacky. Need to wait until a dataset is ready to be deleted.
        datasets_helpers.dataset_delete(self.owner_slug, self.dataset_slug)


# TODO(b/379171781): Remove waiting logic once uploading a new version while the first version is being processed is
# supported.
def dataset_upload_and_wait(handle: str, local_dataset_dir: str, version_notes: str = "") -> bool:
    dataset_upload(handle, local_dataset_dir, version_notes)
    h = parse_dataset_handle(handle)
    time.sleep(1)
    client = KaggleApiV1Client()

    max_attempts = 10
    for _attempt in range(max_attempts):
        try:
            if _get_current_version(client, h) > 0:
                return True
        except Exception:
            # wait a bit before checking for completion again.
            time.sleep(5)

    return False
