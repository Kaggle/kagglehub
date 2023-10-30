import os
import tarfile
from typing import Optional

from kagglehub.cache import get_cached_archive_path, get_cached_path, load_from_cache, mark_as_complete
from kagglehub.clients import KaggleApiV1Client
from kagglehub.handle import ModelHandle
from kagglehub.resolver import Resolver

MODEL_INSTANCE_VERSION_FIELD = "versionNumber"


class HttpResolver(Resolver):
    def is_supported(self, *_) -> bool:
        # Downloading files over HTTP is supported in all environments for all handles / path.
        return True

    def __call__(self, h: ModelHandle, path: Optional[str] = None) -> str:
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
            archive_path = get_cached_archive_path(h)
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)

            # First, we download the archive.
            api_client.download_file(url_path, archive_path)

            # Create the directory to extract the archive to.
            os.makedirs(out_path, exist_ok=True)

            if not tarfile.is_tarfile(archive_path):
                msg = "Unsupported archive type."
                raise ValueError(msg)

            # Extract all files to this directory.
            with tarfile.open(archive_path) as f:
                # Model archives are created by Kaggle via the Databundle Worker.
                f.extractall(out_path)

            # Delete the archive
            os.remove(archive_path)

        mark_as_complete(h, path)
        return out_path


def _get_current_version(api_client: KaggleApiV1Client, h: ModelHandle):
    json_response = api_client.get(_build_get_instance_url_path(h))
    if MODEL_INSTANCE_VERSION_FIELD not in json_response:
        msg = f"Invalid GetModelInstance API response. Expected to include a {MODEL_INSTANCE_VERSION_FIELD} field"
        raise ValueError(msg)

    return json_response[MODEL_INSTANCE_VERSION_FIELD]


def _build_get_instance_url_path(h: ModelHandle) -> str:
    return f"models/{h.owner}/{h.model}/{h.framework}/{h.variation}/get"


def _build_download_url_path(h: ModelHandle) -> str:
    return f"models/{h.owner}/{h.model}/{h.framework}/{h.variation}/{h.version}/download"
