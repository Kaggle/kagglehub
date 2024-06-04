import os
import threading
from typing import Callable
from unittest import mock

from flask import Flask
from werkzeug.serving import make_server

from ..utils import resolve_endpoint


class ServerThread(threading.Thread):
    def __init__(self, app: Flask, host: str, port: int, shutdown_callback: Callable):
        threading.Thread.__init__(self)
        self.server = make_server(host, port, app)
        self.shutdown_callback = shutdown_callback

    def run(self) -> None:
        self.server.serve_forever()

    def shutdown(self) -> None:
        self.server.shutdown()
        self.shutdown_callback()


def start_server(
    app: Flask,
    endpoint_env_var_name: str = "KAGGLE_API_ENDPOINT",
    endpoint_env_var_value: str = "http://localhost:7777",
) -> ServerThread:
    env_var_patch = mock.patch.dict(os.environ, {endpoint_env_var_name: endpoint_env_var_value})
    env_var_patch.start()

    address, port = resolve_endpoint(endpoint_env_var_name)

    def shutdown_callback() -> None:
        env_var_patch.stop()

    server = ServerThread(app, address, port, shutdown_callback=shutdown_callback)
    server.start()

    return server
