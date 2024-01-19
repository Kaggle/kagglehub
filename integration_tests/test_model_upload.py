import os
import tempfile
import unittest
import uuid

from kagglehub import model_upload, models_helpers
from kagglehub.config import get_kaggle_credentials

LICENSE_NAME = "MIT"


class TestModelUpload(unittest.TestCase):
    def setUp(self):
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

    def test_model_upload_and_versioning(self):
        # Create Instance
        model_upload(self.handle, self.temp_dir, LICENSE_NAME)

        # Create Version
        model_upload(self.handle, self.temp_dir, LICENSE_NAME)

        # If delete model does not raise an error, then the upload was successful.

    def tearDown(self):
        models_helpers.delete_model(self.owner_slug, self.model_slug)
