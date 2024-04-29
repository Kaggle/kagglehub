import hashlib
import os
import re

from flask import Flask, Response, jsonify, request
from flask.typing import ResponseReturnValue

from kagglehub.integrity import to_b64_digest
from tests.utils import get_test_file_path

app = Flask(__name__)


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"code": "404", "error": str(e)}
    return jsonify(data), 200


@app.route("/api/v1/no-integrity", methods=["GET"])
def no_integrity() -> ResponseReturnValue:
    test_file_path = get_test_file_path("foo.txt")
    with open(test_file_path, "rb") as f:
        resp = Response(f.read())
        resp.content_type = "application/octet-stream"
        resp.content_length = os.path.getsize(test_file_path)
        return resp, 200


@app.route("/api/v1/corrupted", methods=["GET"])
def corrupted() -> ResponseReturnValue:
    test_file_path = get_test_file_path("foo.txt")
    with open(test_file_path, "rb") as f:
        resp = Response(f.read())
        resp.content_type = "application/octet-stream"
        resp.content_length = os.path.getsize(test_file_path)
        resp.headers["x-goog-hash"] = "md5=badhash"
        return resp, 200


@app.route("/api/v1/good", methods=["GET"])
def good() -> ResponseReturnValue:
    test_file_path = get_test_file_path("foo.txt")
    with open(test_file_path, "rb") as f:
        start = 0
        if "Range" in request.headers:
            bytes_range = request.headers["Range"]
            # See: https://developer.mozilla.org/en-US/docs/Web/HTTP/Range_requests
            # Format is bytes=<start>-<end>
            # If <end> is missing, then, it means until the end.
            # This test server only supports bytes=<start>- format since it is the only one we need.
            m = re.match("^bytes=([0-9]+)-$", bytes_range)
            if not m:
                return "", 400
            start = int(m.group(1))
        f.seek(start)
        content = f.read()
        file_hash = hashlib.md5()
        file_hash.update(content)
        return (
            Response(
                content,
                headers={
                    "Content-type": "application/octet-stream",
                    "Content-Length": str(os.path.getsize(test_file_path)),
                    "Accept-Ranges": "bytes",
                    "x-goog-hash": f"crc32c=n03x6A==, md5={to_b64_digest(file_hash)}",
                },
            ),
            200,
        )
