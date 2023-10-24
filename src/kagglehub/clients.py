from urllib.parse import urljoin

import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

from kagglehub.config import get_kaggle_api_endpoint, get_kaggle_credentials

CHUNK_SIZE = 1048576
# The `connect` timeout is the number of seconds `requests` will wait for your client to establish a connection.
# The `read` timeout is the number of seconds that the client will wait BETWEEN bytes sent from the server.
# See: https://requests.readthedocs.io/en/stable/user/advanced/#timeouts
DEFAULT_CONNECT_TIMEOUT = 5  # seconds
DEFAULT_READ_TIMEOUT = 15  # seconds


# TODO(b/307576378): When ready, use `kagglesdk` to issue requests.
class KaggleApiV1Client:
    BASE_PATH = "api/v1"

    def __init__(self):
        self.credentials = get_kaggle_credentials()
        self.endpoint = get_kaggle_api_endpoint()

    def download_file(self, path: str, out_file: str):
        url = self._build_url(path)
        # TODO(b/307572374) Support resumable downloads.
        with requests.get(
            url,
            stream=True,
            auth=self._get_http_basic_auth(),
            timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
        ) as response:
            response.raise_for_status()
            size = int(response.headers["Content-Length"])
            size_read = 0
            with tqdm(total=size, initial=size_read, unit="B", unit_scale=True, unit_divisor=1024) as progress_bar:
                with open(out_file, "wb") as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        f.write(chunk)
                        size_read = min(size, size_read + CHUNK_SIZE)
                        progress_bar.update(len(chunk))

    def _get_http_basic_auth(self):
        if self.credentials:
            return HTTPBasicAuth(self.credentials.username, self.credentials.key)
        return None

    def _build_url(self, path):
        return urljoin(self.endpoint, f"{KaggleApiV1Client.BASE_PATH}/{path}")
