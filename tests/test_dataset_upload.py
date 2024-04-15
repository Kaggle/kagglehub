import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from tempfile import TemporaryDirectory

from kagglehub.gcs_upload import MAX_FILES_TO_UPLOAD
from kagglehub.datasets import dataset_upload
from tests.fixtures import BaseTestCase

from .utils import create_test_http_server

GET_DATASET = "/datasets/akankshaaa013/top-grossing-movies-dataset/get"
CREATE_DATASET = "/datasets/create/new"

class KaggleAPIHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)

    def do_GET(self):
        if self.path == f"/api/v1{GET_DATASET}":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.write.write(bytes(json.dumps({"message": "Dataset exists!"}), "utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response_data = {"message": "Some response data"}
            self.write.write(bytes(json.dumps(response_data), "utf-8"))

    def do_POST(self):
        if self.path == f"/api/v1{CREATE_DATASET}":
            self.send_response(200)
            self.send_header("Content=type", "application/json")
            self.end_headers()
            self.write.write(json.dumps({"status": "success", "message": "Dataset created successfully"}).encode("utf-8"))
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

class TestDatasetUpload(BaseTestCase):
    def test_dataset_upload_with_invalid_handle(self):
        with create_test_http_server(KaggleAPIHandler):
<<<<<<< HEAD
            with self.assertRaises(ValueError):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / "temp_test_file"
                    test_filepath.touch() # Creates a temp file in the temp directory
                    dataset_upload("invalid/invalid/invalid", temp_dir)
=======
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):                
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / "temp_test_file"
                    test_filepath.touch() # Creates a temp file in the temp directory
                    dataset_upload("invalid/invalid/invalid", temp_dir)

    def test_dataset_upload_with_valid_handle(self):
        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, "http://localhost:7778"):
                with TemporaryDirectory() as temp_dir:
                    test_filepath = Path(temp_dir) / "temp_test_file"
                    test_filepath.touch() # Creates a temp file in the temp directory
                    dataset_upload("akankshaaa013/top-grossing-movies-dataset", temp_dir)

    def test_dataset_upload_with_too_many_files(self):
        with self.assertRaises(ValueError):
            with TemporaryDirectory() as temp_dir:
                # Create more than 50 temporary files in the directory
                for i in range(MAX_FILES_TO_UPLOAD + 1):
                        test_filepath = Path(temp_dir) / f"temp_test_file_{i}"
                        test_filepath.touch()
                
                dataset_upload("owner/valid_name", temp_dir)
>>>>>>> bc95b04 (add dataset upload test)
