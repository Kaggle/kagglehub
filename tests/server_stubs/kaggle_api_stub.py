import hashlib
import os
import re

from flask import Flask, Response, jsonify, make_response, redirect, request
from flask.typing import ResponseReturnValue
from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

from kagglehub.integrity import to_b64_digest
from tests.utils import get_test_file_path

app = Flask(__name__)


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"code": "404", "error": str(e), "message": "server side error"}
    return jsonify(data), 404


# We've arbitrarily chosen to test with the DownloadDataset API. Any resources would've have worked.
@app.route("/api/v1/datasets.DatasetApiService/DownloadDataset", methods=["POST"])
def dataset_download() -> ResponseReturnValue:
    r = ApiDownloadDatasetRequest.from_dict(request.get_json())
    if r.dataset_slug == "content_type_mismatch":
        html = """
        <html>
        <head>
            <title>Test</title>
        </head>
        <body>
            <h1>This is HTML content</h1>
        </body>
        </html>
        """
        resp = make_response(html)
        resp.headers["Content-Type"] = "application/json"
        resp.status_code = 404
        return resp
    if r.dataset_slug == "no-integrity":
        test_file_path = get_test_file_path("foo.txt")
        with open(test_file_path, "rb") as f:
            resp = Response(f.read())
            resp.content_type = "application/octet-stream"
            resp.content_length = os.path.getsize(test_file_path)
            return resp, 200
    if r.dataset_slug == "corrupted":
        test_file_path = get_test_file_path("foo.txt")
        with open(test_file_path, "rb") as f:
            resp = Response(f.read())
            resp.content_type = "application/octet-stream"
            resp.content_length = os.path.getsize(test_file_path)
            resp.headers["x-goog-hash"] = "md5=badhash"
            return resp, 200
    if r.dataset_slug == "good":
        test_file_path = get_test_file_path("foo.txt")
        with open(test_file_path, "rb") as f:
            start = 0
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
    if r.dataset_slug in ("range-ignored", "range-wrong-offset"):
        # Served from a URL whose range handling misbehaves, see `range_test_download`.
        return redirect(f"/range-test/{r.dataset_slug}", code=302)
    else:
        msg = "Unknown dataset"
        raise ValueError(msg)


def _parse_range_start(bytes_range: str) -> int | None:
    # See: https://developer.mozilla.org/en-US/docs/Web/HTTP/Range_requests
    # Format is bytes=<start>-<end>
    # If <end> is missing, then, it means until the end.
    # This test server only supports bytes=<start>- format since it is the only one we need.
    m = re.match("^bytes=([0-9]+)-$", bytes_range)
    if not m:
        return None
    return int(m.group(1))


def _foo_partial_response(start: int) -> ResponseReturnValue:
    """Serves foo.txt from `start` the way GCS does: 206 with a Content-Range header."""
    test_file_path = get_test_file_path("foo.txt")
    total_size = os.path.getsize(test_file_path)
    with open(test_file_path, "rb") as f:
        f.seek(start)
        content = f.read()
    file_hash = hashlib.md5()
    file_hash.update(content)
    return (
        Response(
            content,
            headers={
                "Content-type": "application/octet-stream",
                "Content-Length": str(len(content)),
                "Content-Range": f"bytes {start}-{total_size - 1}/{total_size}",
                "Accept-Ranges": "bytes",
                "x-goog-hash": f"crc32c=n03x6A==, md5={to_b64_digest(file_hash)}",
            },
        ),
        206,
    )


@app.route("/api/v1/datasets.DatasetApiService/DownloadDataset", methods=["GET"])
def dataset_range_download() -> ResponseReturnValue:
    if "Range" not in request.headers:
        msg = "Expecting a range request"
        raise ValueError(msg)

    start = _parse_range_start(request.headers["Range"])
    if start is None:
        return "", 400
    return _foo_partial_response(start)


@app.route("/range-test/<behavior>", methods=["GET"])
def range_test_download(behavior: str) -> ResponseReturnValue:
    """Mocks a server that advertises `Accept-Ranges: bytes` but does not honor `Range` requests properly.

    Without a `Range` header this serves the whole of foo.txt (the initial download). With one:
    - "range-ignored": ignores the header and replies 200 with the whole file again.
    - "range-wrong-offset": replies 206 but always from byte 0, whatever offset was requested.
    """
    test_file_path = get_test_file_path("foo.txt")
    with open(test_file_path, "rb") as f:
        content = f.read()
    file_hash = hashlib.md5()
    file_hash.update(content)
    full_response = (
        Response(
            content,
            headers={
                "Content-type": "application/octet-stream",
                "Content-Length": str(len(content)),
                "Accept-Ranges": "bytes",
                "x-goog-hash": f"crc32c=n03x6A==, md5={to_b64_digest(file_hash)}",
            },
        ),
        200,
    )
    if "Range" not in request.headers or behavior == "range-ignored":
        return full_response
    if behavior == "range-wrong-offset":
        return _foo_partial_response(0)
    return "", 400
