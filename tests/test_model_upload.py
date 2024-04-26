import json
import os
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar, List

from kagglehub.exceptions import BackendError
from kagglehub.gcs_upload import MAX_FILES_TO_UPLOAD, TEMP_ARCHIVE_FILE
from kagglehub.models import model_upload
from tests.fixtures import BaseTestCase

from .utils import create_test_http_server

GET_INSTANCE = "/models/metaresearch/llama-2/pyTorch/1/get"
GET_MODEL = "/models/metaresearch/llama-2/get"
CREATE_MODEL = "/models/create/new"
MODEL_HANDLE = "metaresearch/llama-2/pyTorch/1"
TEMP_TEST_FILE = "temp_test_file"
APACHE_LICENSE = "Apache 2.0"
ALLOWED_LICENSE_VALUES = (APACHE_LICENSE, None)


class KaggleAPIHandler(BaseHTTPRequestHandler):
    UPLOAD_BLOB_FILE_NAMES: ClassVar[List[str]] = []

    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == f"/api/v1{GET_MODEL}":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps({"message": "Model exists!"}), "utf-8"))
        elif self.path == f"/api/v1{GET_INSTANCE}":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps({"message": "Instance exists!"}), "utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response_data = {"message": "Some response data"}
            self.wfile.write(bytes(json.dumps(response_data), "utf-8"))

    def do_POST(self) -> None:  # noqa: N802
        instance_or_version = self.path.split("/")[-1]
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode("utf-8"))

        # Extracting the 'name' from the data
        name = data.get("name", None)
        if self.path == f"/api/v1{CREATE_MODEL}":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": "Model created successfully"}).encode("utf-8"))
        elif instance_or_version in ("instance", "version"):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            if data.get("licenseName") not in ALLOWED_LICENSE_VALUES:
                error_message = json.dumps({"error": f"bad: {self.path}"})
                self.wfile.write(bytes(error_message, "utf-8"))
            else:
                response = {"status": "success", "message": "Model Instance/Version created successfully"}
                self.wfile.write(json.dumps(response).encode("utf-8"))
        elif self.path == "/api/v1/blobs/upload":
            KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES.append(name)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "token": "dummy",
                        "createUrl": "http://localhost:7778",
                        "status": "success",
                        "message": "Here is your token and Url",
                    }
                ).encode("utf-8")
            )
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            error_message = json.dumps({"error": f"bad: {self.path}"})
            self.wfile.write(bytes(error_message, "utf-8"))


class GcsAPIHandler(BaseHTTPRequestHandler):
    simulate_308 = False
    put_requests_count = 0

    def do_PUT(self) -> None:  # noqa: N802
        GcsAPIHandler.put_requests_count += 1
        if GcsAPIHandler.simulate_308:
            # Simulate "308 Resume Incomplete" response
            self.send_response(308)
            self.send_header("Content-type", "application/json")
            self.send_header("Range", "bytes=0-499")
            self.end_headers()
            GcsAPIHandler.simulate_308 = False
        else:
            # Simulate successful upload
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": "File uploaded"}).encode("utf-8"))


class TestModelUpload(BaseTestCase):
    def setUp(self) -> None:
        # Resetting shared variables in GcsAPIHandler
        GcsAPIHandler.simulate_308 = False
        GcsAPIHandler.put_requests_count = 0

        # Resetting any other necessary setup for KaggleAPIHandler
        KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES = []

    def test_model_upload_with_invalid_handle(self) -> None:
        with create_test_http_server(KaggleAPIHandler):
            with self.assertRaises(ValueError):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / TEMP_TEST_FILE
                    test_filepath.touch()  # Create a temporary file in the temporary directory
                    model_upload("invalid/invalid/invalid", temp_dir, APACHE_LICENSE, "model_type")
                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 1)
                    self.assertIn(TEMP_TEST_FILE, KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES)

    def test_model_upload_instance_with_valid_handle(self) -> None:
        # execution path: get_model -> create_model -> get_instance -> create_version
        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / TEMP_TEST_FILE
                    test_filepath.touch()  # Create a temporary file in the temporary directory
                    model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, APACHE_LICENSE, "model_type")
                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 1)
                    self.assertIn(TEMP_TEST_FILE, KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES)

    def test_model_upload_instance_with_nested_directories(self) -> None:
        # execution path: get_model -> create_model -> get_instance -> create_version
        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
                with TemporaryDirectory() as temp_dir:
                    # Create a nested directory structure
                    nested_dir = Path(temp_dir) / "nested"
                    nested_dir.mkdir()

                    # Create a temporary file in the nested directory
                    test_filepath = nested_dir / TEMP_TEST_FILE
                    test_filepath.touch()
                    model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, APACHE_LICENSE, "model_type")
                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 1)
                    self.assertIn(TEMP_TEST_FILE, KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES)

    def test_model_upload_version_with_valid_handle(self) -> None:
        # execution path: get_model -> get_instance -> create_instance

        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / TEMP_TEST_FILE
                    test_filepath.touch()  # Create a temporary file in the temporary directory
                    model_upload("metaresearch/llama-2/pyTorch/7b", temp_dir, APACHE_LICENSE, "model_type")
                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 1)
                    self.assertIn(TEMP_TEST_FILE, KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES)

    def test_model_upload_with_too_many_files(self) -> None:
        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
                with TemporaryDirectory() as temp_dir:
                    # Create more than 50 temporary files in the directory
                    for i in range(MAX_FILES_TO_UPLOAD + 1):
                        test_filepath = Path(temp_dir) / f"temp_test_file_{i}"
                        test_filepath.touch()

                    model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, APACHE_LICENSE, "model_type")
                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 1)
                    self.assertIn(TEMP_ARCHIVE_FILE, KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES)

    def test_model_upload_resumable(self) -> None:
        GcsAPIHandler.simulate_308 = True  # Enable simulation of 308 response for this test

        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / TEMP_TEST_FILE
                    test_filepath.touch()
                    with open(test_filepath, "wb") as f:
                        f.write(os.urandom(1000))

                    model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, APACHE_LICENSE, "model_type")

                    # Check that GcsAPIHandler received two PUT requests
                    self.assertEqual(GcsAPIHandler.put_requests_count, 2)
                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 1)
                    self.assertIn(TEMP_TEST_FILE, KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES)

    def test_model_upload_with_none_license(self) -> None:
        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / TEMP_TEST_FILE
                    test_filepath.touch()  # Create a temporary file in the temporary directory
                    model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, None, "model_type")
                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 1)
                    self.assertIn(TEMP_TEST_FILE, KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES)

    def test_model_upload_without_license(self) -> None:
        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / TEMP_TEST_FILE
                    test_filepath.touch()  # Create a temporary file in the temporary directory
                    model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, version_notes="model_type")
                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 1)
                    self.assertIn(TEMP_TEST_FILE, KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES)

    def test_model_upload_with_invalid_license_fails(self) -> None:
        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / TEMP_TEST_FILE
                    test_filepath.touch()  # Create a temporary file in the temporary directory
                    with self.assertRaises(BackendError):
                        model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, "Invalid License")

    def test_single_file_upload(self) -> None:
        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / "single_dummy_file.txt"
                    with open(test_filepath, "wb") as f:
                        f.write(os.urandom(100))

                    model_upload(
                        "metaresearch/new-model/pyTorch/new-variation", str(test_filepath), APACHE_LICENSE, "model_type"
                    )

                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 1)
                    self.assertIn("single_dummy_file.txt", KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES)

    def test_model_upload_with_directory_structure(self) -> None:
        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
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

                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 4)
                    expected_files = {"file1.txt", "file2.txt", "file3.txt", "file4.txt"}
                    self.assertTrue(set(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES).issubset(expected_files))

                    # TODO: Add assertions on CreateModelInstanceRequest.Directories and
                    # CreateModelInstanceRequest.Files to verify the expected structure
                    # is sent.
