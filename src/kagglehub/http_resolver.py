import os
import tarfile
import tempfile
from typing import Optional

from kagglehub.cache import get_cached_path, load_from_cache, mark_as_complete
from kagglehub.clients import KaggleApiV1Client
from kagglehub.handle import ModelHandle, parse_model_handle
from kagglehub.resolver import Resolver

MODEL_INSTANCE_VERSION_FIELD = "versionNumber"


class HttpResolver(Resolver):
    def is_supported(self, *_):
        # Downloading files over HTTP is supported in all environments for all handles / path.
        return True

    def __call__(self, handle: str, path: Optional[str] = None):
        h = parse_model_handle(handle)
        api_client = KaggleApiV1Client()

        if not h.is_versioned():
            h.version = _get_current_version(api_client, h)

        model_path = load_from_cache(h, path)
        if model_path:
            return model_path  # Already cached

        url_path = _build_download_url_path(h)
        out_path = get_cached_path(h, path)

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
                    # Model archives are created by Kaggle via the Databundle Worker.
                    f.extractall(out_path)

        mark_as_complete(h, path)
        return out_path


def _get_current_version(api_client: KaggleApiV1Client, h: ModelHandle):
    json_response = api_client.get(_build_get_instance_url_path(h))
    if MODEL_INSTANCE_VERSION_FIELD not in json_response:
        msg = f"Invalid GetModelInstance API response. Expected to include a {MODEL_INSTANCE_VERSION_FIELD} field"
        raise ValueError(msg)

    return json_response[MODEL_INSTANCE_VERSION_FIELD]


def _build_get_instance_url_path(h: ModelHandle):
    return f"models/{h.owner}/{h.model}/{h.framework}/{h.variation}/get"


def _build_download_url_path(h: ModelHandle):
    return f"models/{h.owner}/{h.model}/{h.framework}/{h.variation}/{h.version}/download"
