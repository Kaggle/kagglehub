import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar, List

from kagglehub.gcs_upload import TEMP_ARCHIVE_FILE
from kagglehub.datasets import dataset_upload
from tests.fixtures import BaseTestCase

from .utils import create_test_http_server

GET_DATASET = "/datasets/akankshaaa013/top-grossing-movies-dataset/get"
CREATE_DATASET = "/datasets/create/new"

TEMP_TEST_FILE = "temp_test_file"

class KaggleAPIHandler(BaseHTTPRequestHandler):
    UPLOAD_BLOB_FILE_NAMES: ClassVar[List[str]] = []

    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == f"/api/v1{GET_DATASET}":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps({"message": "Dataset exists!"}), "utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response_data = {"message": "Some response data"}
            self.wfile.write(bytes(json.dumps(response_data), "utf-8"))

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode("utf-8"))

        # Extracting the 'name' from the data
        name = data.get("name", None)
        if self.path == f"/api/v1{CREATE_DATASET}":
            self.send_response(200)
            self.send_header("Content=type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": "Dataset created successfully"}).encode("utf-8"))
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
    def do_PUT(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "success", "message": "File uploaded"}).encode("utf-8"))

class TestDatasetUpload(BaseTestCase):
    def test_dataset_upload_with_invalid_handle(self) -> None:
        with create_test_http_server(KaggleAPIHandler):
            with self.assertRaises(ValueError):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / TEMP_TEST_FILE
                    test_filepath.touch() # Creates a temp file in the temp directory
                    dataset_upload("invalid/invalid/invalid", temp_dir)

    def test_dataset_upload_with_valid_handle(self) -> None:
        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / TEMP_TEST_FILE
                    test_filepath.touch() # Creates a temp file in the temp directory
                    dataset_upload("jeward/newDataset", temp_dir)
                    self.assertEqual(len(KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES), 1)
                    self.assertIn(TEMP_ARCHIVE_FILE, KaggleAPIHandler.UPLOAD_BLOB_FILE_NAMES)
