import logging
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

from kagglehub import dataset_upload
from kagglehub.config import get_kaggle_credentials

LICENSE_NAME = "MIT"

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
        self.dataset_slug = f"dataset_{self.dataset_uuid}"
        self.handle = f"{self.owner_slug}/{self.dataset_slug}"

    def test_dataset_upload_and_versioning(self) -> None:
        # Create Instance
        dataset_upload(self.handle, self.temp_dir)

        # Create Version
        dataset_upload(self.handle, self.temp_dir, "new version")
