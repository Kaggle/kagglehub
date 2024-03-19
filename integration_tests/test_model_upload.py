import logging
import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from kagglehub import model_upload, models_helpers
from kagglehub.config import get_kaggle_credentials
from kagglehub.exceptions import BackendError

LICENSE_NAME = "MIT"

logger = logging.getLogger(__name__)


def upload_with_retries(
    handle: str, temp_dir: str, license_name: str, max_retries: int = 5, retry_delay: int = 5
) -> None:
    """
    Tries to upload a model with retries on BackendError indicating the instance slug is already in use.

    Args:
        handle: The model handle.
        temp_dir: Temporary directory where the model is stored.
        license_name: License name for the model.
        max_retries (int): Maximum number of retry attempts.
        retry_delay (int): Delay in seconds between retries.

    Raises:
        TimeoutError: If the maximum number of retries is reached without success.
    """
    for attempt in range(max_retries):
        try:
            model_upload(handle, temp_dir, license_name)
            break
        except BackendError as e:
            if "is already used by another model instance." in str(e):
                logger.info(f"Attempt {attempt + 1!s} failed: {e!s}. Retrying in {retry_delay!s} seconds...")
                time.sleep(retry_delay)
            else:
                raise  # Reraise if it's a different error
    else:
        time_out_message = "Maximum retries reached without success."
        raise TimeoutError(time_out_message)


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
        upload_with_retries(self.handle, self.temp_dir, LICENSE_NAME)

        # If delete model does not raise an error, then the upload was successful.

    def test_model_upload_and_versioning_zip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            for i in range(60):
                test_filepath = Path(temp_dir) / f"temp_test_file_{i}"
                test_filepath.touch()

            # Create Instance
            model_upload(self.handle, temp_dir, LICENSE_NAME)

            # Create Version
            upload_with_retries(self.handle, self.temp_dir, LICENSE_NAME)

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

    def tearDown(self) -> None:
        models_helpers.delete_model(self.owner_slug, self.model_slug)
