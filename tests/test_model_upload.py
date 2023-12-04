import kagglehub
from tests.fixtures import BaseTestCase
from http.server import BaseHTTPRequestHandler
import json

import kagglehub
from kagglehub.cache import MODELS_CACHE_SUBFOLDER, get_cached_archive_path
from kagglehub.handle import parse_model_handle
from kagglehub.http_resolver import MODEL_INSTANCE_VERSION_FIELD
from tests.fixtures import BaseTestCase
<<<<<<< HEAD
from unittest import mock
=======
>>>>>>> c5db66768bb899cf80a6c1499530ba07373c174a

from .utils import create_test_cache, create_test_http_server, get_test_file_path

GET_MODEL = "/models/metaresearch/llama-2/pyTorch/13b/get"
GET_INSTANCE = "/models/metaresearch/llama-2/create/instance"
CREATE_MODEL = "/models/create/new"
CREATE_INSTANCE = "/models/metaresearch/llama-2/create/instance"
<<<<<<< HEAD
CREATE_VERSION = "/models/metaresearch/llama-2/pyTorch/13b/create/version"

class KaggleAPIHandler(BaseHTTPRequestHandler):
    model_created = False
    instance_created = False
    version_created = False
=======

class KaggleAPIHandler(BaseHTTPRequestHandler):
    model_created = False
>>>>>>> c5db66768bb899cf80a6c1499530ba07373c174a
    def do_HEAD(self):  # noqa: N802
        self.send_response(200)

    def do_GET(self):  # noqa: N802
<<<<<<< HEAD
        if self.path == f"/api/v1{GET_MODEL}" and self.model_created is True:
=======
        if self.path == f"/api/v1{GET_MODEL}" and model_created is True:
>>>>>>> c5db66768bb899cf80a6c1499530ba07373c174a
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps({"message": "Model Created!"}), "utf-8"))
<<<<<<< HEAD
        elif self.path == f"/api/v1{GET_INSTANCE}" and self.instance_created is True:
=======
        elif self.path == f"/api/v1{GET_INSTANCE}":
>>>>>>> c5db66768bb899cf80a6c1499530ba07373c174a
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps({"message": "Instance Created!"}), "utf-8"))
<<<<<<< HEAD
        elif self.path == f"/api/v1{CREATE_VERSION}" and self.version_created is True:
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps({"message": "Instance Version Created!"}), "utf-8"))
        else:
            self.send_response(404)
            self.wfile.write(bytes(f"Unhandled path: {self.path}", "utf-8"))

    def do_POST(self):
        body = json.loads(self.rfile.read().decode('utf-8'))
        handle = body['handle']
        local_model_dir = body['local_model_dir']

        # Parse the model handle
        h = parse_model_handle(handle)
        if self.path == CREATE_MODEL:
            self.model_created = True
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": "Model created successfully"}).encode('utf-8'))
        elif self.path == CREATE_INSTANCE:
            self.instance_created = True
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": "Model Instance created successfully"}).encode('utf-8'))
        elif self.path == CREATE_VERSION:
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": "Model instance version created successfully"}).encode('utf-8'))
            self.version_created = True
        else:
            self.send_response(404)
            self.wfile.write(bytes(f"Unhandled path: {self.path}", "utf-8"))

    def tearDown(self):
        self.model_created = False
        self.instance_created = False
        self.version_created = False

=======
        else:
            self.send_response(403)
            self.send_header("Content-type", "application/text")
            self.send_header("Content-Length", 0)
            self.end_headers()
            self.wfile.write(bytes("", "utf-8"))

    def do_POST(self):
        if self.path == CREATE_MODEL:
            model_created = True
>>>>>>> c5db66768bb899cf80a6c1499530ba07373c174a



class TestModelUpload(BaseTestCase):
    def test_model_upload_with_invalid_handle(self):
        with create_test_http_server(KaggleAPIHandler):
            with self.assertRaises(Exception) as cm:
                model_upload("invalid-handle", "/path/to/model")

            self.assertEqual(cm.exception.status, 404)
            self.assertEqual(cm.exception.headers, {"Content-type": "application/json"})
            self.assertEqual(cm.exception.wfile.getvalue(), b'{"message": "Invalid model handle"}')

    def test_model_upload_with_valid_handle(self):
        handle = "valid-handle"
        local_model_dir = "/path/to/model"

        with create_test_http_server(KaggleAPIHandler):
            with mock.patch.object(parse_model_handle, "return_value", mock.Mock(owner="owner", model="model", framework="framework", version="version")):
                with mock.patch.object(get_or_create_model, "return_value") as mock_get_or_create_model:
                    with mock.patch.object(create_model_instance_or_version, "return_value") as mock_create_model_instance_or_version:
                        kagglehub.model_upload(handle, local_model_dir)

                        mock_get_or_create_model.assert_called_once_with("owner", "model")
                        mock_create_model_instance_or_version.assert_called_once_with("owner", "model", "framework", "version")
    
    def test_create_instance_with_invalid_handle(self):
        with create_test_http_server(KaggleAPIHandler):
            with self.assertRaises(Exception) as cm:
                create_model_instance_or_version("invalid-handle", "model", "framework", "version")

            self.assertEqual(cm.exception.status, 400)
            self.assertEqual(cm.exception.headers, {"Content-type": "application/json"})
            self.assertEqual(cm.exception.wfile.getvalue(), b'{"message": "Invalid model handle"}')

    def test_create_instance_with_valid_handle(self):
        handle = "valid-handle"
        with create_test_http_server(KaggleAPIHandler):
            response = create_model_instance_or_version(handle, "model", "framework", "version")

            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers, {"Content-type": "application/json"})
            self.assertEqual(response.wfile.getvalue(), b'{"status": "success", "message": "Model Instance created successfully"}')

    def test_create_version_with_invalid_handle(self):
        with create_test_http_server(KaggleAPIHandler):
            with self.assertRaises(Exception) as cm:
                create_model_instance_or_version("invalid-handle", "model", "framework", "version")

            self.assertEqual(cm.exception.status, 400)
            self.assertEqual(cm.exception.headers, {"Content-type": "application/json"})
            self.assertEqual(cm.exception.wfile.getvalue(), b'{"message": "Invalid model handle"}')

    def test_create_version_with_valid_handle(self):
        handle = "valid-handle"
        with create_test_http_server(KaggleAPIHandler):
            response = create_model_instance_or_version(handle, "model", "framework", "version")

            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers, {"Content-type": "application/json"})
            self.assertEqual(response.wfile.getvalue(), b'{"status": "success", "message": "Model instance version created successfully"}')
