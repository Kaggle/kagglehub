import os
from typing import Optional
import tempfile
import tarfile

from tqdm import tqdm
import requests
from requests.auth import HTTPBasicAuth

from kagglehub.cache import load_from_cache, get_cached_path
from kagglehub.config import get_kaggle_credentials, get_kaggle_api_endpoint
from kagglehub.handle import ModelHandle, parse_model_handle
from kagglehub.resolver import Resolver


CHUNK_SIZE = 1048576

class HttpResolver(Resolver):
    def is_supported(self, *_):
        # Downloading files over HTTP is supported in all environments for all handles / path.
        return True

    def __call__(self, handle: str, path: Optional[str] = None):
        model_handle = parse_model_handle(handle)
        model_path = load_from_cache(model_handle, path)
        if model_path:
            return model_path  # Already cached

        out_path = get_cached_path(model_handle, path)

        # Create the intermediary directories
        if path:
            # Downloading a single file.
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            download_file(model_handle, out_path, path)
        else:
            # Downloading the full archived bundle.
            with tempfile.NamedTemporaryFile() as archive_file:
                # First, we download the archive to a temporary location.
                download_file(model_handle, archive_file.name, path)
                           
                if not tarfile.is_tarfile(archive_file.name):
                    raise ValueError("Unsupported archive type.")
                
                # Create the directory to extract the archive to.
                os.makedirs(out_path, exist_ok=True)

                # Extract all files to this directory.
                with tarfile.open(archive_file.name) as f:
                    # Use 'data' filter to prevent dangerous security issues.
                    f.extractall(out_path, filter="data")

        return out_path


# TODO: Move this to the clients.py file. Call the class "KaggleApiClient"
def download_file(handle: ModelHandle, out_file: str, path: Optional[str] = None):
    creds = get_kaggle_credentials()
    api_endpoint = get_kaggle_api_endpoint()
    url = f"{api_endpoint}/api/v1/models/{handle.owner}/{handle.model}/{handle.framework}/{handle.variation}/{handle.version}/download"
    if path:
        url = f"{url}/{path}"
    with requests.get(
        url,
        stream=True,
        auth=HTTPBasicAuth(creds.username, creds.key) if creds else None,
    ) as response:
        response.raise_for_status()
        size = int(response.headers['Content-Length'])
        size_read = 0
        with tqdm(total=size, initial=size_read, unit='B', unit_scale=True, unit_divisor=1024) as progress_bar:
            with open(out_file, 'wb') as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    f.write(chunk)
                    size_read = min(size, size_read + CHUNK_SIZE)
                    progress_bar.update(len(chunk))