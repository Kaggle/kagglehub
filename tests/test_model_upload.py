import kagglehub
from tests.fixtures import BaseTestCase
from http.server import BaseHTTPRequestHandler
import json

import kagglehub
from kagglehub.cache import MODELS_CACHE_SUBFOLDER, get_cached_archive_path
from kagglehub.handle import parse_model_handle
from kagglehub.http_resolver import MODEL_INSTANCE_VERSION_FIELD
from tests.fixtures import BaseTestCase

from .utils import create_test_cache, create_test_http_server, get_test_file_path

GET_MODEL = "/models/metaresearch/llama-2/pyTorch/13b/get"
GET_INSTANCE = "/models/metaresearch/llama-2/create/instance"
CREATE_MODEL = "/models/create/new"
CREATE_INSTANCE = "/models/metaresearch/llama-2/create/instance"

class KaggleAPIHandler(BaseHTTPRequestHandler):
    model_created = False
    def do_HEAD(self):  # noqa: N802
        self.send_response(200)

    def do_GET(self):  # noqa: N802
        if self.path == f"/api/v1{GET_MODEL}" and model_created is True:
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps({"message": "Model Created!"}), "utf-8"))
        elif self.path == f"/api/v1{GET_INSTANCE}":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps({"message": "Instance Created!"}), "utf-8"))
        else:
            self.send_response(403)
            self.send_header("Content-type", "application/text")
            self.send_header("Content-Length", 0)
            self.end_headers()
            self.wfile.write(bytes("", "utf-8"))

    def do_POST(self):
        if self.path == CREATE_MODEL:
            model_created = True



class TestModelUpload(BaseTestCase):
    def test_model_upload(self):
        with self.assertRaises(NotImplementedError):
            kagglehub.model_upload()
