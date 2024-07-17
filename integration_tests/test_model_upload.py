import logging
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

from kagglehub import model_upload, models_helpers
from kagglehub.config import get_kaggle_credentials

LICENSE_NAME = "MIT"

logger = logging.getLogger(__name__)


ReturnType = TypeVar("ReturnType")


class TestModelUpload(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.dummy_files = ["dummy_model.h5", "config.json", "metadata.json"]
        for file in self.dummy_files:
            with open(os.path.join(self.temp_dir, file), "w") as f:
                f.write("dummy content")
        self.model_uuid = str(uuid.uuid4())
        credentials = get_kaggle_credentials()
        if not credentials:
            self.fail("Make sure to set your Kaggle credentials before running the tests")
        self.owner_slug = credentials.username
        self.model_slug = f"model_{self.model_uuid}"
        self.handle = f"{self.owner_slug}/{self.model_slug}/pyTorch/new-variation"

    def test_model_upload_and_versioning(self) -> None:
        # Create Instance
        model_upload(self.handle, self.temp_dir, LICENSE_NAME)

        # Create Version
        model_upload(self.handle, self.temp_dir, LICENSE_NAME)

        # If delete model does not raise an error, then the upload was successful.

    def test_model_upload_and_versioning_zip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            for i in range(60):
                test_filepath = Path(temp_dir) / f"temp_test_file_{i}"
                test_filepath.touch()

            # Create Instance
            model_upload(self.handle, temp_dir, LICENSE_NAME)

            # Create Version
            model_upload(self.handle, temp_dir, LICENSE_NAME)

    def test_model_upload_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Create the new folder within temp_dir
            inner_folder_path = Path(temp_dir) / "inner_folder"
            inner_folder_path.mkdir()

            for i in range(60):
                # Create a file in the temp_dir
                test_filepath = Path(temp_dir) / f"temp_test_file_{i}"
                test_filepath.touch()

                # Create the same file in the inner_folder
                test_filepath_inner = inner_folder_path / f"temp_test_file_{i}"
                test_filepath_inner.touch()

            # Create Instance
            model_upload(self.handle, temp_dir, LICENSE_NAME)

            # Create Version
            model_upload(self.handle, temp_dir, LICENSE_NAME)

    def test_model_upload_directory_structure(self) -> None:
        nested_dir = Path(self.temp_dir) / "nested"
        nested_dir.mkdir()

        with open(Path(self.temp_dir) / "file1.txt", "w") as f:
            f.write("dummy content in nested file")

        # Create dummy files in the nested directory
        nested_dummy_files = ["nested_model.h5", "nested_config.json", "nested_metadata.json"]
        for file in nested_dummy_files:
            with open(nested_dir / file, "w") as f:
                f.write("dummy content in nested file")

        # Call the model upload function with the base directory
        model_upload(self.handle, self.temp_dir, LICENSE_NAME)

    def test_model_upload_nested_dir(self) -> None:
        # Create a nested directory within self.temp_dir
        nested_dir = Path(self.temp_dir) / "nested"
        nested_dir.mkdir()

        # Create dummy files in the nested directory
        nested_dummy_files = ["nested_model.h5", "nested_config.json", "nested_metadata.json"]
        for file in nested_dummy_files:
            with open(nested_dir / file, "w") as f:
                f.write("dummy content in nested file")

        # Call the model upload function with the base directory
        model_upload(self.handle, self.temp_dir, LICENSE_NAME)

    def test_single_file_upload(self) -> None:
        single_file_path = Path(self.temp_dir) / "dummy_file.txt"
        with open(single_file_path, "wb") as f:
            f.write(os.urandom(100))

        model_upload(self.handle, str(single_file_path), LICENSE_NAME)

    def test_model_upload_empty_files(self) -> None:
        # Create a temp file with empty and non-empty files.
        test_empty_dir = Path(self.temp_dir) / "test_empty"
        test_empty_dir.mkdir()
        (test_empty_dir / "empty.json").touch()
        (test_empty_dir / "non_empty.json").write_bytes(b"hello")

        model_upload(self.handle, test_empty_dir)

    def tearDown(self) -> None:
        models_helpers.delete_model(self.owner_slug, self.model_slug)
