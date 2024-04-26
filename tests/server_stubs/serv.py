import threading

from flask import Flask
from werkzeug.serving import make_server

from ..utils import resolve_endpoint


class ServerThread(threading.Thread):
    def __init__(self, app: Flask, host: str, port: int):
        threading.Thread.__init__(self)
        self.server = make_server(host, port, app)

    def run(self) -> None:
        self.server.serve_forever()

    def shutdown(self) -> None:
        self.server.shutdown()


server: ServerThread


def start_server(app: Flask, var_name: str = "KAGGLE_API_ENDPOINT") -> None:
    global server  # noqa: PLW0603
    address, port = resolve_endpoint(var_name)
    server = ServerThread(app, address, port)
    server.start()


def stop_server() -> None:
    global server  # noqa: PLW0602
    server.shutdown()
