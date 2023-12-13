from tests.fixtures import BaseTestCase
from http.server import BaseHTTPRequestHandler
import json

from kagglehub.models import model_upload
from tests.fixtures import BaseTestCase

from .utils import create_test_http_server

GET_INSTANCE = "/models/metaresearch/llama-2/pyTorch/1/get"
GET_MODEL = "/models/metaresearch/llama-2/get"
CREATE_MODEL = "/models/create/new"
MODEL_HANDLE = "metaresearch/llama-2/pyTorch/1"
TEST_FILEPATH = "/usr/local/google/home/aminmohamed/NegBio/images/"

class KaggleAPIHandler(BaseHTTPRequestHandler):
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

    def do_POST(self):
        instance_or_version = self.path.split('/')[-1]
        if self.path == f"/api/v1{CREATE_MODEL}":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": "Model created successfully"}).encode('utf-8'))
        elif instance_or_version == 'instance' or instance_or_version == 'version':
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": "Model Instance created successfully"}).encode('utf-8'))
        elif self.path == '/api/v1/blobs/upload':
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"token": "dummy", "createUrl": "https://dummy", "status": "success", "message": "Here is your token and Url"}).encode('utf-8'))
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
        self.wfile.write(json.dumps({"status": "success", "message": "File uploaded"}).encode('utf-8'))
        
class TestModelUpload(BaseTestCase):
    def test_model_upload_with_invalid_handle(self):
        with create_test_http_server(KaggleAPIHandler):
            with self.assertRaises(Exception):
                model_upload("invalid/invalid/invalid", TEST_FILEPATH, "Apache 2.0")

    def test_model_upload_instance_with_valid_handle(self):
        # exection path: get model -> create_model -> get_instance -> create version
        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, 'http://localhost:7778'):
                try:
                    model_upload("valid/valid/valid/1", TEST_FILEPATH, "Apache 2.0")
                except Exception as e:
                    # If an exception is caught, the test fails
                    self.fail(f"Unexpected exception raised: {e}")

    def test_model_upload_version_with_valid_handle(self):
        # exection path: get model -> get_instance -> create version
        with create_test_http_server(KaggleAPIHandler):
            with create_test_http_server(GcsAPIHandler, 'http://localhost:7778'):
                try:
                    model_upload("metaresearch/llama-2/pyTorch/1", TEST_FILEPATH, "Apache 2.0")
                except Exception as e:
                    # If an exception is caught, the test fails
                    self.fail(f"Unexpected exception raised: {e}")