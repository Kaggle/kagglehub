from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue
from kagglesdk.kernels.types.kernels_api_service import (
    ApiDownloadKernelOutputRequest,
    ApiGetKernelResponse,
    ApiKernelMetadata,
)

from tests.utils import (
    AUTO_COMPRESSED_FILE_NAME,
    add_mock_gcs_route,
    get_gcs_redirect_response,
)

app = Flask(__name__)
add_mock_gcs_route(app)

GOOD_CREDENTIALS_USERNAME = "dster"
GOOD_CREDENTIALS_API_KEY = "some-key"

# See https://cloud.google.com/storage/docs/xml-api/reference-headers#xgooghash
GCS_HASH_HEADER = "x-goog-hash"
LAST_MODIFIED = "Last-Modified"
LAST_MODIFIED_DATE = "Thu, 02 Mar 2020 02:17:12 GMT"


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.route("/api/v1/api.v1.DiagnosticsService/Hello", methods=["POST"])
def hello() -> ResponseReturnValue:
    auth = request.authorization
    if auth and auth.username == GOOD_CREDENTIALS_USERNAME and auth.password == GOOD_CREDENTIALS_API_KEY:
        data = {"message": "Hello from test server!", "userName": auth.username}
        return jsonify(data), 200
    else:
        return jsonify({"code": 401}), 200


@app.route("/api/v1/kernels.KernelsApiService/GetKernel", methods=["POST"])
def notebook_get() -> ResponseReturnValue:
    response = ApiGetKernelResponse()
    response.metadata = ApiKernelMetadata()
    response.metadata.current_version_number = 2

    return response.to_json(), 200


@app.route("/api/v1/kernels.KernelsApiService/DownloadKernelOutput", methods=["POST"])
def notebook_output_download() -> ResponseReturnValue:
    r = ApiDownloadKernelOutputRequest.from_dict(request.get_json())

    # First, determine if we're fetching a file or the whole notebook output
    if r.kernel_slug == "package-test":
        test_file_name = f"package-v{r.version_number}.zip"
    elif r.file_path:
        # This mimics behavior for our file downloads, where users request a file, but
        # receive a zipped version of the file from GCS.
        test_file_name = f"{AUTO_COMPRESSED_FILE_NAME}.zip" if r.file_path == AUTO_COMPRESSED_FILE_NAME else r.file_path
    else:
        test_file_name = "foo.txt.zip"

    return get_gcs_redirect_response(test_file_name)


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"message": "Some error response data", "error": str(e)}
    return jsonify(data), 404
