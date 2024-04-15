import os
from pathlib import Path
from tempfile import TemporaryDirectory

from kagglehub.datasets import dataset_upload
from kagglehub.gcs_upload import TEMP_ARCHIVE_FILE
from tests.fixtures import BaseTestCase

from .server_stubs import dataset_upload_stub as stub
from .server_stubs import serv

TEMP_TEST_FILE = "temp-test-file"

<<<<<<< HEAD
=======
class KaggleAPIHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)

    def do_GET(self):
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

    def do_POST(self):
        if self.path == f"/api/v1{CREATE_DATASET}":
            self.send_response(200)
            self.send_header("Content=type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": "Dataset created successfully"}).encode("utf-8"))
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
>>>>>>> 66db1ff (add helpers/other files  and modify tests)

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

