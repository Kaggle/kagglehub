import hashlib
import os
import re
from http.server import BaseHTTPRequestHandler
from tempfile import TemporaryDirectory

from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import DataCorruptionError
from kagglehub.integrity import to_b64_digest
from tests.fixtures import BaseTestCase

from .utils import create_test_http_server, get_test_file_path


class KaggleAPIHandler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)

    def do_GET(self) -> None:  # noqa: N802
        test_file_path = get_test_file_path("foo.txt")
        with open(test_file_path, "rb") as f:
            if self.path.endswith("no-integrity"):
                # Do not set the "x-goog-hash" header which skips integrity check.
                self.send_response(200)
                self.send_header("Content-type", "application/octet-stream")
                self.send_header("Content-Length", str(os.path.getsize(test_file_path)))
                self.end_headers()
                self.wfile.write(f.read())
            if self.path.endswith("corrupted"):
                self.send_response(200)
                self.send_header("Content-type", "application/octet-stream")
                self.send_header("Content-Length", str(os.path.getsize(test_file_path)))
                self.send_header("x-goog-hash", "md5=badhash")
                self.end_headers()
                self.wfile.write(f.read())
            if self.path.endswith("good"):
                start = 0
                if "Range" in self.headers:
                    bytes_range = self.headers["Range"]
                    # See: https://developer.mozilla.org/en-US/docs/Web/HTTP/Range_requests
                    # Format is bytes=<start>-<end>
                    # If <end> is missing, then, it means until the end.
                    # This test server only supports bytes=<start>- format since it is the only one we need.
                    m = re.match("^bytes=([0-9]+)-$", bytes_range)
                    if not m:
                        self.end_headers()
                        self.send_response(400)  # Bad Request
                        return
                    start = int(m.group(1))

                f.seek(start)
                self.send_response(200)
                self.send_header("Content-type", "application/octet-stream")
                self.send_header("Content-Length", str(os.path.getsize(test_file_path) - start))
                self.send_header("Accept-Ranges", "bytes")  # support resumable download

                content = f.read()
                file_hash = hashlib.md5()
                file_hash.update(content)
                # The crc32c hash is set to a random value and is ignored by kagglehub which only checks the md5.
                self.send_header("x-goog-hash", f"crc32c=n03x6A==, md5={to_b64_digest(file_hash)}")
                self.end_headers()
                self.wfile.write(content)

            else:
                self.send_response(404)
                self.wfile.write(bytes(f"Unhandled path: {self.path}", "utf-8"))


class TestKaggleApiV1Client(BaseTestCase):
    def test_download_with_integrity_check(self) -> None:
        with create_test_http_server(KaggleAPIHandler):
            with TemporaryDirectory() as d:
                out_file = os.path.join(d, "out")

                api_client = KaggleApiV1Client()
                api_client.download_file("good", out_file)

                with open(out_file) as f:
                    self.assertEqual("foo", f.read())

    def test_resumable_download_with_integrity_check(self) -> None:
        with create_test_http_server(KaggleAPIHandler):
            with TemporaryDirectory() as d:
                out_file = os.path.join(d, "out")

                # If the out_file already has data, we use the 'Range' header to resume download.
                with open(out_file, "w") as f:
                    f.write("fo")  # Should download the remaining "o".

                api_client = KaggleApiV1Client()
                with self.assertLogs("kagglehub", level="INFO") as cm:
                    api_client.download_file("good", out_file)
                    self.assertIn("INFO:kagglehub.clients:Resuming download from 2 bytes (1 bytes left)...", cm.output)

                with open(out_file) as f:
                    self.assertEqual("foo", f.read())

    def test_download_no_integrity_check(self) -> None:
        with create_test_http_server(KaggleAPIHandler):
            with TemporaryDirectory() as d:
                out_file = os.path.join(d, "out")

                api_client = KaggleApiV1Client()
                api_client.download_file("no-integrity", out_file)

                with open(out_file) as f:
                    self.assertEqual("foo", f.read())

    def test_download_corrupted_file_fail_integrity_check(self) -> None:
        with create_test_http_server(KaggleAPIHandler):
            with TemporaryDirectory() as d:
                out_file = os.path.join(d, "out")

                api_client = KaggleApiV1Client()
                with self.assertRaises(DataCorruptionError):
                    api_client.download_file("corrupted", out_file)

                # Assert the corrupted file has been deleted.
                self.assertFalse(os.path.exists(out_file))
