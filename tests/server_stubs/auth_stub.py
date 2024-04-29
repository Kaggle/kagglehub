from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue

app = Flask(__name__)


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


GOOD_CREDENTIALS_USERNAME = "lastplacelarry"
GOOD_CREDENTIALS_API_KEY = "some-key"


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"code": "404", "error": str(e)}
    return jsonify(data), 200


@app.route("/api/v1/hello", methods=["GET"])
def model_create() -> ResponseReturnValue:
    auth = request.authorization
    if auth and auth.username == GOOD_CREDENTIALS_USERNAME and auth.password == GOOD_CREDENTIALS_API_KEY:
        data = {"message": "Hello from test server!"}
        return jsonify(data), 200
    else:
        return jsonify({"code": 401}), 200
