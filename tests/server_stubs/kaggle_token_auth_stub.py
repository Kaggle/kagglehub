from pathlib import Path

from flask import Flask, jsonify
from flask.typing import ResponseReturnValue
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = "super-secret"

jwt = JWTManager(app)


@app.route("/setup/<path>", methods=["POST"])
def setup(path: str) -> ResponseReturnValue:
    token = create_access_token(identity="lastplacelarry")
    with (Path(path) / "/etc/secrets/kaggle/api-v1-token").open("w") as f:
        f.write(token)
        f.write("\n")
        return jsonify({"token" : token}, 200)


@app.route("/", methods=["HEAD"])
@jwt_required()
def head() -> ResponseReturnValue:
    return "", 200


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"code": "404", "error": str(e)}
    return jsonify(data), 200


@app.route("/api/v1/hello", methods=["GET"])
@jwt_required()
def model_create() -> ResponseReturnValue:
    data = {"message": "Hello from test server!"}
    return jsonify(data), 200
