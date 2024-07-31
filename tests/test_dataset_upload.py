import os
from pathlib import Path
from tempfile import TemporaryDirectory

from kagglehub.datasets import dataset_upload
from kagglehub.gcs_upload import MAX_FILES_TO_UPLOAD, TEMP_ARCHIVE_FILE
from tests.fixtures import BaseTestCase

from .server_stubs import dataset_upload_stub as stub
from .server_stubs import serv

TEMP_TEST_FILE = "temp-test-file"


class TestDatasetUpload(BaseTestCase):
    def setUp(self) -> None:
        stub.reset()

    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

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
            dataset_upload("jeward/newDataset", temp_dir, "dataset-type")
            self.assertEqual(len(stub.shared_data.files), 1)
            self.assertIn(TEMP_TEST_FILE, stub.shared_data.files)

    def test_dataset_upload_instance_with_nested_directories(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Create a nested directory structure
            nested_dir = Path(temp_dir) / "nested"
            nested_dir.mkdir()
            # Create a temporary file in the nested directory
            test_filepath = nested_dir / TEMP_TEST_FILE
            test_filepath.touch()
            dataset_upload("jeward/newDataset", temp_dir, "dataset_type")
            self.assertEqual(len(stub.shared_data.files), 1)
            self.assertIn(TEMP_TEST_FILE, stub.shared_data.files)

    def test_dataset_upload_with_too_many_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Create more than 50 temporary files in the directory
            for i in range(MAX_FILES_TO_UPLOAD + 1):
                test_filepath = Path(temp_dir) / f"temp_test_file_{i}"
                test_filepath.touch()
            dataset_upload("jeward/newDataset", temp_dir, "dataset_type")
            self.assertEqual(len(stub.shared_data.files), 1)
            self.assertIn(TEMP_ARCHIVE_FILE, stub.shared_data.files)

    def test_dataset_upload_resumable(self) -> None:
        stub.simulate_308(state=True)  # Enable simulation of 308 response for this test
        with TemporaryDirectory() as temp_dir:
            test_filepath = Path(temp_dir) / TEMP_TEST_FILE
            test_filepath.touch()
            with open(test_filepath, "wb") as f:
                f.write(os.urandom(1000))
            dataset_upload("jeward/newDataset", temp_dir, "dataset_type")
            self.assertGreaterEqual(stub.shared_data.blob_request_count, 1)
            self.assertEqual(len(stub.shared_data.files), 1)
            self.assertIn(TEMP_TEST_FILE, stub.shared_data.files)
