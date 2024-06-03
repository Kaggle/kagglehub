import os
from pathlib import Path
from tempfile import TemporaryDirectory

from kagglehub.exceptions import BackendError
from kagglehub.gcs_upload import MAX_FILES_TO_UPLOAD, TEMP_ARCHIVE_FILE
from kagglehub.models import model_upload
from tests.fixtures import BaseTestCase

from .server_stubs import model_upload_stub as stub
from .server_stubs import serv

TEMP_TEST_FILE = "temp_test_file"
APACHE_LICENSE = "Apache 2.0"


class TestModelUpload(BaseTestCase):
    def setUp(self) -> None:
        stub.reset()

    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_model_upload_with_invalid_handle(self) -> None:
        with self.assertRaises(ValueError):
            with TemporaryDirectory() as temp_dir:
                test_filepath = Path(temp_dir) / TEMP_TEST_FILE
                test_filepath.touch()  # Create a temporary file in the temporary directory
                model_upload("invalid/invalid/invalid", temp_dir, APACHE_LICENSE, "model_type")

    def test_model_upload_instance_with_valid_handle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            test_filepath = Path(temp_dir) / TEMP_TEST_FILE
            test_filepath.touch()  # Create a temporary file in the temporary directory
            model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, APACHE_LICENSE, "model_type")
            self.assertEqual(len(stub.shared_data.files), 1)
            self.assertIn(TEMP_TEST_FILE, stub.shared_data.files)

    def test_model_upload_instance_with_nested_directories(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Create a nested directory structure
            nested_dir = Path(temp_dir) / "nested"
            nested_dir.mkdir()
            # Create a temporary file in the nested directory
            test_filepath = nested_dir / TEMP_TEST_FILE
            test_filepath.touch()
            model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, APACHE_LICENSE, "model_type")
            self.assertEqual(len(stub.shared_data.files), 1)
            self.assertIn(TEMP_TEST_FILE, stub.shared_data.files)

    def test_model_upload_version_with_valid_handle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            test_filepath = Path(temp_dir) / TEMP_TEST_FILE
            test_filepath.touch()  # Create a temporary file in the temporary directory
            model_upload("metaresearch/llama-2/pyTorch/7b", temp_dir, APACHE_LICENSE, "model_type")
            self.assertEqual(len(stub.shared_data.files), 1)
            self.assertIn(TEMP_TEST_FILE, stub.shared_data.files)

    def test_model_upload_with_too_many_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            # Create more than 50 temporary files in the directory
            for i in range(MAX_FILES_TO_UPLOAD + 1):
                test_filepath = Path(temp_dir) / f"temp_test_file_{i}"
                test_filepath.touch()
            model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, APACHE_LICENSE, "model_type")
            self.assertEqual(len(stub.shared_data.files), 1)
            self.assertIn(TEMP_ARCHIVE_FILE, stub.shared_data.files)

    def test_model_upload_resumable(self) -> None:
        stub.simulate_308(state=True)  # Enable simulation of 308 response for this test
        with TemporaryDirectory() as temp_dir:
            test_filepath = Path(temp_dir) / TEMP_TEST_FILE
            test_filepath.touch()
            with open(test_filepath, "wb") as f:
                f.write(os.urandom(1000))
            model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, APACHE_LICENSE, "model_type")
            self.assertGreaterEqual(stub.shared_data.blob_request_count, 1)
            self.assertEqual(len(stub.shared_data.files), 1)
            self.assertIn(TEMP_TEST_FILE, stub.shared_data.files)

    def test_model_upload_with_none_license(self) -> None:
        with TemporaryDirectory() as temp_dir:
            test_filepath = Path(temp_dir) / TEMP_TEST_FILE
            test_filepath.touch()  # Create a temporary file in the temporary directory
            model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, None, "model_type")
            self.assertEqual(len(stub.shared_data.files), 1)
            self.assertIn(TEMP_TEST_FILE, stub.shared_data.files)

    def test_model_upload_without_license(self) -> None:
        with TemporaryDirectory() as temp_dir:
            test_filepath = Path(temp_dir) / TEMP_TEST_FILE
            test_filepath.touch()  # Create a temporary file in the temporary directory
            model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, version_notes="model_type")
            self.assertEqual(len(stub.shared_data.files), 1)
            self.assertIn(TEMP_TEST_FILE, stub.shared_data.files)

    def test_model_upload_with_invalid_license_fails(self) -> None:
        with TemporaryDirectory() as temp_dir:
            test_filepath = Path(temp_dir) / TEMP_TEST_FILE
            test_filepath.touch()  # Create a temporary file in the temporary directory
            with self.assertRaises(BackendError):
                model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, "Invalid License")

    def test_single_file_upload(self) -> None:
        with TemporaryDirectory() as temp_dir:
            test_filepath = Path(temp_dir) / "single_dummy_file.txt"
            with open(test_filepath, "wb") as f:
                f.write(os.urandom(100))

            model_upload(
                "metaresearch/new-model/pyTorch/new-variation", str(test_filepath), APACHE_LICENSE, "model_type"
            )

            self.assertEqual(len(stub.shared_data.files), 1)
            self.assertIn("single_dummy_file.txt", stub.shared_data.files)

    def test_model_upload_with_directory_structure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            (base_path / "dir1").mkdir()
            (base_path / "dir2").mkdir()

            (base_path / "file1.txt").touch()

            (base_path / "dir1" / "file2.txt").touch()
            (base_path / "dir1" / "file3.txt").touch()

            (base_path / "dir1" / "subdir1").mkdir()
            (base_path / "dir1" / "subdir1" / "file4.txt").touch()

            model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, APACHE_LICENSE, "model_type")

            self.assertEqual(len(stub.shared_data.files), 4)
            expected_files = {"file1.txt", "file2.txt", "file3.txt", "file4.txt"}
            self.assertTrue(set(stub.shared_data.files).issubset(expected_files))

            # TODO: Add assertions on CreateModelInstanceRequest.Directories and
            # CreateModelInstanceRequest.Files to verify the expected structure
            # is sent.
