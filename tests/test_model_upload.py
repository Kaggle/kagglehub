import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar, List

from kagglehub.gcs_upload import MAX_FILES_TO_UPLOAD, TEMP_ARCHIVE_FILE
from kagglehub.models import model_upload
from tests.fixtures import BaseTestCase

from .utils import create_test_http_server

GET_INSTANCE = "/models/metaresearch/llama-2/pyTorch/1/get"
GET_MODEL = "/models/metaresearch/llama-2/get"
CREATE_MODEL = "/models/create/new"
MODEL_HANDLE = "metaresearch/llama-2/pyTorch/1"
TEMP_TEST_FILE = "temp_test_file"


class KaggleAPIHandler(BaseHTTPRequestHandler):
    UPLOAD_BLOB_FILE_NAMES: ClassVar[List[str]] = []

    def do_HEAD(self):  # noqa: N802
        self.send_response(200)

    def do_GET(self):  # noqa: N802
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

    def do_POST(self):  # noqa: N802
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
    def do_HEAD(self):  # noqa: N802
        self.send_response(200)

    def do_PUT(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "success", "message": "File uploaded"}).encode("utf-8"))


class TestModelUpload(BaseTestCase):
    def setUp(self):
        KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES = []

    def test_model_upload_with_invalid_handle(self):
        with create_test_http_server(KaggleAPIHandler):
            with self.assertRaises(ValueError):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / TEMP_TEST_FILE
                    test_filepath.touch()  # Create a temporary file in the temporary directory
                    model_upload("invalid/invalid/invalid", temp_dir, "Apache 2.0", "model_type")
                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 1)
                    self.assertIn(TEMP_TEST_FILE, KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES)

    def test_model_upload_instance_with_valid_handle(self):
        # execution path: get_model -> create_model -> get_instance -> create_version
        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / TEMP_TEST_FILE
                    test_filepath.touch()  # Create a temporary file in the temporary directory
                    model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, "Apache 2.0", "model_type")
                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 1)
                    self.assertIn(TEMP_TEST_FILE, KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES)

    def test_model_upload_version_with_valid_handle(self):
        # execution path: get_model -> get_instance -> create_instance

        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / TEMP_TEST_FILE
                    test_filepath.touch()  # Create a temporary file in the temporary directory
                    model_upload("metaresearch/llama-2/pyTorch/7b", temp_dir, "Apache 2.0", "model_type")
                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 1)
                    self.assertIn(TEMP_TEST_FILE, KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES)

    def test_model_upload_with_too_many_files(self):
        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
                with TemporaryDirectory() as temp_dir:
                    # Create more than 50 temporary files in the directory
                    for i in range(MAX_FILES_TO_UPLOAD + 1):
                        test_filepath = Path(temp_dir) / f"temp_test_file_{i}"
                        test_filepath.touch()

                    model_upload("metaresearch/new-model/pyTorch/new-variation", temp_dir, "Apache 2.0", "model_type")
                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 1)
                    self.assertIn(TEMP_ARCHIVE_FILE, KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES)
