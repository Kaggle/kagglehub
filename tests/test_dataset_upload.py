import os
from pathlib import Path
from tempfile import TemporaryDirectory

from kagglehub.datasets import dataset_upload
from kagglehub.gcs_upload import TEMP_ARCHIVE_FILE
from tests.fixtures import BaseTestCase

from .server_stubs import dataset_upload_stub as stub
from .server_stubs import serv

TEMP_TEST_FILE = "temp-test-file"


class TestDatasetUpload(BaseTestCase):
    def setUp(self) -> None:
        stub.reset()

    @classmethod
    def setUpClass(cls):  # noqa: ANN102
        serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):  # noqa: ANN102
        serv.stop_server()

    def test_dataset_upload_with_invalid_handle(self) -> None:
        with self.assertRaises(ValueError):
            with TemporaryDirectory() as temp_dir:
                test_filepath = Path(temp_dir) / TEMP_TEST_FILE
                test_filepath.touch()  # Create a temporary file in the temporary directory
                dataset_upload("invalid/invalid/invalid", temp_dir)


    def test_dataset_upload_with_valid_handle(self) -> None:
         with TemporaryDirectory() as temp_dir:
            test_filepath = Path(temp_dir) / TEMP_TEST_FILE
            test_filepath.touch()  # Create a temporary file in the temporary directory
            dataset_upload("jeward/newDataset", temp_dir)
            self.assertEqual(len(stub.shared_data.files), 1)
            self.assertIn(TEMP_TEST_FILE, stub.shared_data.files)

