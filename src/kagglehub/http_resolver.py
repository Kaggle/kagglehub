import os
import tarfile
import tempfile
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

from kagglehub.cache import get_cached_path, load_from_cache
from kagglehub.config import get_kaggle_api_endpoint, get_kaggle_credentials
from kagglehub.handle import ModelHandle, parse_model_handle
from kagglehub.resolver import Resolver
from kagglehub.clients import KaggleApiV1Client


class HttpResolver(Resolver):
    def is_supported(self, *_):
        # Downloading files over HTTP is supported in all environments for all handles / path.
        return True

    def __call__(self, handle: str, path: Optional[str] = None):
        model_handle = parse_model_handle(handle)
        model_path = load_from_cache(model_handle, path)
        if model_path:
            return model_path  # Already cached

        api_client = KaggleApiV1Client()
        url_path = f"models/{model_handle.owner}/{model_handle.model}/{model_handle.framework}/{model_handle.variation}/{model_handle.version}/download"
        out_path = get_cached_path(model_handle, path)

        # Create the intermediary directories
        if path:
            # Downloading a single file.
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            api_client.download_file(url_path + "/" + path, out_path)
        else:
            # Downloading the full archived bundle.
            with tempfile.NamedTemporaryFile() as archive_file:
                # First, we download the archive to a temporary location.
                api_client.download_file(url_path, archive_file.name)

                if not tarfile.is_tarfile(archive_file.name):
                    msg = "Unsupported archive type."
                    raise ValueError(msg)

                # Create the directory to extract the archive to.
                os.makedirs(out_path, exist_ok=True)

                # Extract all files to this directory.
                with tarfile.open(archive_file.name) as f:
                    # Use 'data' filter to prevent dangerous security issues.
                    f.extractall(out_path, filter="data")

        return out_path
