from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue

app = Flask(__name__)


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


GOOD_CREDENTIALS_USERNAME = "lastplacelarry"
GOOD_CREDENTIALS_API_TOKEN = "some-token"


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"code": "404", "error": str(e)}
    return jsonify(data), 200


@app.route("/api/v1/api.v1.DiagnosticsService/Hello", methods=["POST"])
def hello() -> ResponseReturnValue:
    auth = request.authorization
    if auth and auth.token == GOOD_CREDENTIALS_API_TOKEN:
        data = {"message": "Hello from test server!", "userName": GOOD_CREDENTIALS_USERNAME}
        return jsonify(data), 200
    else:
        return jsonify({"code": 401}), 200
